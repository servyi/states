//! PROBE: a NESTED root — a state method on tree A's mutator opens an
//! independent root over tree B (owned by A's state). B's checkpoint
//! must work; A's requester waits until A's thread returns from B
//! (liveness for A, but is anything UB?). The interesting model
//! question: two claim registries, one thread.
use hslib::{HandleAble, InnerHandle, PersistentCell, independent_task_root};

trait Trait: Send {
    fn step(&mut self);
}

struct Machine {
    other: Option<PersistentCell<Box<PersistentCell<dyn HandleAble<dyn Trait>>>>>,
}

impl Trait for InnerHandle<Machine> {
    fn step(&mut self) {
        // drive the nested tree B from inside A's drive
        if let Some(node) = self.other.as_mut() {
            independent_task_root(node, |mut ob, _ckb| {
                ob.step();
            });
        }
        self.safepoint();
    }
}

struct Leaf;

impl Trait for InnerHandle<Leaf> {
    fn step(&mut self) {}
}

fn mk(state: Machine) -> PersistentCell<Box<PersistentCell<dyn HandleAble<dyn Trait>>>> {
    PersistentCell::new(PersistentCell::from_boxed(
        Box::new(state) as Box<dyn HandleAble<dyn Trait>>,
    ))
}

fn main() {
    // tree B: a leaf, owned by A's root state
    let b: PersistentCell<Box<PersistentCell<dyn HandleAble<dyn Trait>>>> =
        PersistentCell::new(PersistentCell::from_boxed(
            Box::new(Leaf) as Box<dyn HandleAble<dyn Trait>>,
        ));
    // tree A: a Machine holding B
    let mut a = mk(Machine { other: Some(b) });

    independent_task_root(&mut a, |mut outer, ck| {
        let ck2 = ck.clone();
        std::thread::scope(|s| {
            let reader = s.spawn(move || {
                let guard = ck2.request(); // A's stop: A parks inside B? never — liveness?
                let _ = &****guard;
                println!("A checkpoint granted");
            });
            let mut steps = 0;
            while !reader.is_finished() {
                outer.step(); // drives nested B each time
                steps += 1;
                if steps > 200 {
                    break;
                }
            }
            reader.join().unwrap();
        });
    });
    println!("nestedroot done");
}
