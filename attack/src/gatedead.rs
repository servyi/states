//! PROBE (liveness): the double-request gate self-deadlock — the
//! same thread requesting while already holding a guard blocks on
//! the gate it itself holds. The finding-4 family, gate variant.
use hslib::{HandleAble, InnerHandle, PersistentCell, independent_task_root};

trait Trait: Send {
    fn step(&mut self);
}

struct Machine;

impl Trait for InnerHandle<Machine> {
    fn step(&mut self) {
        self.safepoint();
    }
}

fn main() {
    let mut root: PersistentCell<Box<PersistentCell<dyn HandleAble<dyn Trait>>>> =
        PersistentCell::new(PersistentCell::from_boxed(Box::new(Machine)
            as Box<dyn HandleAble<dyn Trait>>));
    independent_task_root(&mut root, |mut outer, ck| {
        let ck2 = ck.clone();
        std::thread::scope(|s| {
            let reader = s.spawn(move || {
                let _g1 = ck2.request(); // holds the gate
                // the same thread requests again: the gate is its own
                let _g2 = ck2.request(); // blocks forever on the gate
                println!("never reached");
            });
            let mut steps = 0;
            while !reader.is_finished() {
                outer.step();
                steps += 1;
                if steps > 10_000 {
                    break;
                }
            }
            reader.join().unwrap();
        });
    });
}
