//! PROBE (liveness): mem::forget a child outer inside the callback —
//! the for<'n> seal stops it from ESCAPING, but not from being
//! LEAKED. The forgotten outer's InnerHandle holds the child Shared's
//! last Arc: the child never parks and never dies, so the parent's
//! next park (any future checkpoint) waits for it forever.
use hslib::{HandleAble, InnerHandle, PersistentCell, independent_task_root};

trait Trait: Send {
    fn step(&mut self);
}

struct Machine;
struct Leaf;

impl Trait for InnerHandle<Machine> {
    fn step(&mut self) {
        let _ = self.fanout(
            &|_m: &Machine| vec![&LEAF],
            |o| {
                // the seal prevents escape; it cannot prevent a leak
                std::mem::forget(o);
            },
        );
        self.safepoint();
    }
}

static LEAF: std::sync::LazyLock<
    PersistentCell<Box<PersistentCell<dyn HandleAble<dyn Trait>>>>,
> = std::sync::LazyLock::new(|| {
    PersistentCell::new(PersistentCell::from_boxed(
        Box::new(LeafCell) as Box<dyn HandleAble<dyn Trait>>,
    ))
});

struct LeafCell;

impl Trait for InnerHandle<LeafCell> {
    fn step(&mut self) {}
}

fn main() {
    let mut root: PersistentCell<Box<PersistentCell<dyn HandleAble<dyn Trait>>>> =
        PersistentCell::new(PersistentCell::from_boxed(
            Box::new(Machine) as Box<dyn HandleAble<dyn Trait>>,
        ));
    independent_task_root(&mut root, |mut outer, ck| {
        outer.step(); // the fanout forgets the child outer here
        // now try to checkpoint: the parent's park requests every
        // registered child — the forgotten one never parks, never dies
        let ck2 = ck.clone();
        std::thread::scope(|s| {
            let reader = s.spawn(move || {
                let _guard = ck2.request();
                println!("never reached: the child Shared is immortal");
            });
            std::thread::sleep(std::time::Duration::from_millis(300));
            // the mutator idles — the request stays pending forever
            reader.join().unwrap();
        });
    });
}
