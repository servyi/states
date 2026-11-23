//! The #19 sketch's own slot shape, end to end: the ERASED slot —
//! `Box<PersistentCell<dyn HandleAble<dyn Vocab>>>` — which every sketch
//! demo is built on. The library's bounds must keep admitting it: a
//! concrete node (`InnerHandle<Node>: Unsize<V>`) is the other admitted
//! shape, and the crate's state layer and the other integration tests
//! cover that one.
//!
//! A checkpoint's guard READS the leaf through every layer (the box, the
//! inner cell, the erased content) while the mutator is parked, and the
//! mutator writes through its minted handle strictly after that read.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use servyi_states::checkpoint::{
    HandleAble, InnerHandle, PersistentCell, independent_task_root,
};

/// Where a leaf's value starts.
const N_START: u32 = 0;
/// A leaf's value after its first write.
const N_AFTER_FIRST_WRITE: u32 = 1;
/// A leaf's value after its write following the checkpoint.
const N_AFTER_SECOND_WRITE: u32 = 2;
/// Where the REPLACEMENT leaf's value starts.
const REPLACEMENT_START: u32 = 10;
/// The replacement leaf's value after its two writes.
const REPLACEMENT_AFTER_TWO_WRITES: u32 = REPLACEMENT_START + N_AFTER_SECOND_WRITE;
/// How many distinct "snapshot taken" flags the suite needs.
const SNAPSHOT_FLAG_COUNT: usize = 1;
/// How long a cooperating leaf sleeps between safepoints.
const COOPERATE_SLEEP: Duration = Duration::from_millis(1);

/// The drive vocabulary.
trait Vocab: Send {
    fn step(&mut self) -> u32;
}

/// The erased slot content: a state whose handle speaks the vocabulary
/// and that a checkpoint can read.
trait Erased: HandleAble<dyn Vocab> {
    fn read(&self) -> u32;
}

type ErasedSlot = Box<PersistentCell<dyn Erased>>;

/// One flag per cooperating leaf: which "snapshot taken" flag it waits on.
static SNAPSHOT_TAKEN: [AtomicBool; SNAPSHOT_FLAG_COUNT] =
    [const { AtomicBool::new(false) }; SNAPSHOT_FLAG_COUNT];

/// A leaf state. A cooperating leaf keeps reaching safepoints until the
/// checker has taken its snapshot — a root has no parent join loop that
/// would park it — then writes again, strictly after the guard's read.
struct Leaf {
    n: u32,
    flag: Option<usize>,
}

impl Erased for Leaf {
    fn read(&self) -> u32 {
        self.n
    }
}

impl Vocab for InnerHandle<Leaf> {
    fn step(&mut self) -> u32 {
        self.n += 1;
        if let Some(flag) = self.flag {
            while !SNAPSHOT_TAKEN[flag].load(Ordering::SeqCst) {
                self.safepoint();
                std::thread::sleep(COOPERATE_SLEEP);
            }
        }
        self.n += 1;
        self.n
    }
}

/// A fan-out phase of erased leaves.
struct Phase {
    children: Vec<PersistentCell<ErasedSlot>>,
}

impl Erased for Phase {
    fn read(&self) -> u32 {
        self.children.iter().map(|child| (***child).read()).sum()
    }
}

impl Vocab for InnerHandle<Phase> {
    fn step(&mut self) -> u32 {
        let sums: Vec<u32> = self.fanout(&|phase: &Phase| phase.children.iter().collect(), |mut child| child.step());
        sums.into_iter().sum()
    }
}

/// An erased slot over `state`.
fn erased(state: impl Erased + 'static) -> ErasedSlot {
    PersistentCell::from_boxed(Box::new(state))
}

/// Read the erased root slot's content through the guard.
fn read_through_guard(guard: &servyi_states::checkpoint::CheckpointGuard<'_, ErasedSlot>) -> u32 {
    // one typed line per layer the read goes through
    let cell: &PersistentCell<ErasedSlot> = guard;
    let slot: &ErasedSlot = cell;
    let inner: &PersistentCell<dyn Erased> = slot;
    let content: &dyn Erased = &**inner; // stated: the pointee is unsized
    content.read()
}

/// The erased tree, root and child: the checker stops the world and
/// reads the LEAF through the guard; the leaf then writes through its
/// minted handle and both writes are visible.
#[test]
fn erased_root_and_child_run_and_checkpoint() {
    const LEAF_FLAG: usize = 0;
    let mut root = PersistentCell::new(erased(Phase {
        children: vec![PersistentCell::new(erased(Leaf { n: N_START, flag: Some(LEAF_FLAG) }))],
    }));
    let (result, seen) = independent_task_root(&mut root, |mut outer, checkpointer| {
        let checkpointer = checkpointer.clone();
        std::thread::scope(|scope| {
            let checker = scope.spawn(move || {
                let guard = checkpointer.request();
                let seen = read_through_guard(&guard);
                drop(guard);
                SNAPSHOT_TAKEN[LEAF_FLAG].store(true, Ordering::SeqCst);
                seen
            });
            let result = outer.step();
            (result, checker.join().expect("checker panicked"))
        })
    });
    assert_eq!(seen, N_AFTER_FIRST_WRITE, "the guard read the leaf after its first write");
    assert_eq!(result, N_AFTER_SECOND_WRITE, "the leaf wrote again after the checkpoint");
}

/// The successor install on an erased root: `replace` re-mints through
/// the REPLACEMENT's own `HandleAble`, and the fresh handle drives it.
#[test]
fn erased_root_is_replaced_then_driven() {
    let mut root = PersistentCell::new(erased(Leaf { n: N_START, flag: None }));
    let result = independent_task_root(&mut root, |mut outer, _checkpointer| {
        outer.replace(erased(Leaf { n: REPLACEMENT_START, flag: None }));
        outer.step()
    });
    assert_eq!(result, REPLACEMENT_AFTER_TWO_WRITES, "the replacement is what runs");
}
