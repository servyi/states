//! The refactor-check mock: every operation the checker side wants,
//! on the DIRECT SaveAble shape (`K = Box<dyn SaveAble>`, zero
//! client unsafe).
//!
//! The mocked operations:
//! 1. repeated stop-the-world snapshots while the machine steps
//! 2. the saves THROUGH the SaveAble interface, read via the guard
//! 3. a structural invariant check on every snapshot
//! 4. observed progress (distinct states across the saves)
//! 5. a mid-run ROLLBACK: install the checker's first snapshot
//!    through `replace`, then behavior must equal the reference's
//!    first step
//! 6. dynamic tree growth: a split that pushes a fresh child
//! 7. the replay: a second root task restored from the snapshot
//!    must reproduce the reference trace — the refactor verdict

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use hslib::{ChildNode, HandleAble, InnerHandle, independent_task_root};

/// the drive vocabulary: step the machine, observe the behavior
trait Trait: Send {
    fn step(&mut self);
    fn observe(&mut self) -> String;
    fn grow(&mut self);
}

/// the CLIENT's save vocabulary: a subtrait of HandleAble
trait SaveAble: HandleAble<dyn Trait> {
    fn save(&self) -> Snap;
}

/// the snapshot value the saves produce
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Snap {
    Node(Vec<Snap>),
    Leaf(i32),
}

/// K, literally the subtrait box
type SaveSlot = Box<dyn SaveAble>;

struct Machine {
    children: Vec<ChildNode<SaveSlot>>,
    steps: i32,
}
struct Counter {
    val: i32,
}

impl Trait for InnerHandle<Machine, dyn Trait> {
    fn step(&mut self) {
        self.steps += 1;
        self.fanout(
            &|m: &Machine| m.children.iter().collect(),
            |mut o| o.step(),
        );
        // the SAFETY POINT: the checker's requests land here
        self.safepoint();
    }

    fn observe(&mut self) -> String {
        let obs: Vec<String> = self.fanout(
            &|m: &Machine| m.children.iter().collect(),
            |mut o| o.observe(),
        );
        obs.join("+")
    }

    fn grow(&mut self) {
        // the push goes through the handle's own deref_mut (the
        // exclusive access the method has), BEFORE the fanout — the
        // shared split then hands out all children, grown one included
        self.children.push(ChildNode::new(Box::new(Counter { val: 100 })));
        self.fanout(
            &|m: &Machine| m.children.iter().collect(),
            |mut o| o.observe(),
        );
        self.safepoint();
    }
}

impl Trait for InnerHandle<Counter, dyn Trait> {
    fn step(&mut self) {
        self.val += 1;
    }
    fn observe(&mut self) -> String {
        format!("{}", self.val)
    }
    fn grow(&mut self) {}
}

impl SaveAble for Machine {
    fn save(&self) -> Snap {
        Snap::Node(self.children.iter().map(|c| c.save()).collect())
    }
}
impl SaveAble for Counter {
    fn save(&self) -> Snap {
        Snap::Leaf(self.val)
    }
}

/// rebuild a live slot from a snapshot — the mirror of the save
fn restore(s: &Snap) -> SaveSlot {
    match s {
        Snap::Leaf(v) => Box::new(Counter { val: *v }),
        Snap::Node(kids) => Box::new(Machine {
            children: kids.iter().map(|k| ChildNode::new(restore(k))).collect(),
            steps: 0,
        }),
    }
}

fn leaves(s: &Snap) -> Vec<i32> {
    match s {
        Snap::Node(kids) => kids.iter().flat_map(leaves).collect(),
        Snap::Leaf(v) => vec![*v],
    }
}

/// the per-snapshot invariant: at the 2-leaf stage the leaves agree
/// (the machine's step discipline); at the 3-leaf stage the grown
/// child (100) trails an agreeing pair
fn check(s: &Snap) {
    let l = leaves(s);
    assert!(l.len() >= 2, "structural: at least the initial pair");
    if l.len() == 2 {
        assert_eq!(l[0], l[1], "2-leaf stage: both counters agree");
    } else {
        assert_eq!(l[0], l[1], "3-leaf stage: the original pair agrees");
        assert!(l[2] >= 100, "3-leaf stage: the grown child (100, possibly stepped since)");
    }
}

fn main() {
    let mut m: ChildNode<SaveSlot> = ChildNode::new(Box::new(Machine {
        children: vec![
            ChildNode::new(Box::new(Counter { val: 1 })),
            ChildNode::new(Box::new(Counter { val: 1 })),
        ],
        steps: 0,
    }));

    let (trace_ref, snap0, grown_obs, n_snaps, n_distinct) =
        independent_task_root(&mut m, |mut outer, checkpointer| {
            let ck = checkpointer.clone();
            // the BASELINE: a pre-run snapshot from another thread,
            // while the machine idles in observe's parking windows —
            // read-only fan-outs, so the captured state is
            // guaranteed pre-step (a sampling checker's first frame
            // is not). The rollback's and the replay's anchor.
            let snap0 = std::thread::scope(|s| {
                let t = s.spawn({
                    let ck = ck.clone();
                    move || {
                        let guard = ck.request();
                        guard.save()
                    }
                });
                while !t.is_finished() {
                    let _ = outer.observe();
                }
                t.join().unwrap()
            });
            println!("[driver] baseline: {:?}", snap0);
            let snaps = Arc::new(Mutex::new(Vec::<Snap>::new()));
            let stop = Arc::new(AtomicBool::new(false));
            std::thread::scope(|scope| {
                // 1+2+3: the sampling checker — every save goes
                // through the SaveAble vocabulary via the guard
                let checker = scope.spawn({
                    let snaps = snaps.clone();
                    let stop = stop.clone();
                    move || {
                        let mut i = 0;
                        loop {
                            if stop.load(Ordering::Relaxed) {
                                break;
                            }
                            let guard = ck.request();
                            let s = guard.save();
                            check(&s);
                            println!("[checker] snapshot {}: {:?}", i, s);
                            snaps.lock().unwrap().push(s);
                            i += 1;
                            drop(guard);
                            thread::sleep(Duration::from_millis(5));
                        }
                    }
                });
                // 5's other half: the reference run — 3 steps, traced
                let mut trace: Vec<String> = Vec::new();
                for _ in 0..3 {
                    outer.step();
                    thread::sleep(Duration::from_millis(15));
                    trace.push(outer.observe());
                }
                println!("[driver] reference trace: {:?}", trace);
                // 5: MID-RUN ROLLBACK to the baseline snapshot
                outer.replace(restore(&snap0));
                outer.step();
                let after = outer.observe();
                println!("[driver] rollback -> step -> {} (reference step 1: {})", after, trace[0]);
                assert_eq!(after, trace[0], "rollback must replay step 1");
                // 6: GROW — a fresh child through the split
                outer.grow();
                let grown = outer.observe();
                println!("[driver] grown tree: {}", grown);
                // let the checker see the final state, then stop it —
                // and one more step as a parking point so any request
                // already in flight resolves before the join
                thread::sleep(Duration::from_millis(20));
                stop.store(true, Ordering::Relaxed);
                outer.step();
                checker.join().unwrap();
                let mut distinct = snaps.lock().unwrap().clone();
                distinct.sort();
                distinct.dedup();
                (
                    trace,
                    snap0,
                    grown,
                    snaps.lock().unwrap().len(),
                    distinct.len(),
                )
            })
        });

    // 7: THE REPLAY — a second root task restored from the same
    // snapshot must reproduce the reference trace
    let mut m2: ChildNode<SaveSlot> = ChildNode::new(restore(&snap0));
    let trace2 = independent_task_root(&mut m2, |mut outer2, _ck| {
        (0..3)
            .map(|_| {
                outer2.step();
                outer2.observe()
            })
            .collect::<Vec<_>>()
    });
    println!("[replay] trace: {:?}", trace2);
    assert_eq!(trace_ref, trace2, "replay must match the reference");

    // 4: progress — the snapshots must show distinct states
    assert!(n_snaps >= 4, "the checker sampled repeatedly (got {})", n_snaps);
    assert!(n_distinct >= 3, "the snapshots show progress (got {} distinct)", n_distinct);
    assert_eq!(grown_obs, "2+2+100");

    println!(
        "refactor check: replay == reference, {} snapshots consistent ({} distinct states) — PASS",
        n_snaps, n_distinct
    );
}
