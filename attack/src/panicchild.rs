//! PROBE: the first child subtask PANICS (its claims must release on
//! unwind), the second completes and drops its outer early; a reader
//! checkpoints throughout. The join propagates the panic; the world
//! must keep resuming; nothing may dangle.
use std::sync::atomic::{AtomicUsize, Ordering};

use hslib::{HandleAble, InnerHandle, PersistentCell, independent_task_root};

trait Trait: Send {
    fn step(&mut self);
}

struct Machine {
    children: Vec<PersistentCell<Box<PersistentCell<dyn HandleAble<dyn Trait>>>>>,
}
struct Leaf;

impl Trait for InnerHandle<Machine> {
    fn step(&mut self) {
        let calls: &AtomicUsize = &PANIC_ON_FIRST;
        let _ = self.fanout(
            &|m: &Machine| m.children.iter().collect(),
            move |_o| {
                // the FIRST invocation panics (claims release on
                // unwind); the second returns (its outer drops early)
                if calls.fetch_add(1, Ordering::SeqCst) == 0 {
                    panic!("child subtask dies mid-drive");
                }
            },
        );
        self.safepoint();
    }
}

static PANIC_ON_FIRST: AtomicUsize = AtomicUsize::new(0);

impl Trait for InnerHandle<Leaf> {
    fn step(&mut self) {}
}

fn main() {
    let mut root: PersistentCell<Box<PersistentCell<dyn HandleAble<dyn Trait>>>> =
        PersistentCell::new(PersistentCell::from_boxed(Box::new(Machine {
            children: vec![
                PersistentCell::new(PersistentCell::from_boxed(Box::new(Leaf)
                    as Box<dyn HandleAble<dyn Trait>>)),
                PersistentCell::new(PersistentCell::from_boxed(Box::new(Leaf)
                    as Box<dyn HandleAble<dyn Trait>>)),
            ],
        }) as Box<dyn HandleAble<dyn Trait>>));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        independent_task_root(&mut root, |mut outer, ck| {
            let ck2 = ck.clone();
            std::thread::scope(|s| {
                let reader = s.spawn(move || {
                    for _ in 0..3 {
                        let guard = ck2.request();
                        let _ = &****guard;
                        std::thread::sleep(std::time::Duration::from_millis(2));
                    }
                    println!("reader: 3 checkpoints survived");
                });
                let mut steps = 0;
                while !reader.is_finished() {
                    outer.step();
                    steps += 1;
                    if steps > 300 {
                        break;
                    }
                }
                reader.join().unwrap();
            });
        });
    }));
    println!("caught (panicked child propagated): {}", result.is_err());
}
