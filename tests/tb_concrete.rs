//! The concrete-slot write → checkpoint → write ride, from the sketch's
//! own `tb_concrete` test (#21 `1ddd563`).
//!
//! A mint that materializes a `&mut` on its path goes `Unique` on the
//! first write and `Frozen` on the guard's read under Tree Borrows —
//! every later write through the chain's subtree is then forbidden. The
//! mint here projects through raws alone ([`AsPtr`]): this has to ride
//! clean under Stacked Borrows and under Tree Borrows alike.

use servyi_states::checkpoint::{
    AsPtr, InnerHandle, PersistentCell, independent_task_root,
};

/// Where the counter starts.
const START: u32 = 0;
/// The write before the checkpoint — a mint `&mut` would go `Unique`
/// here.
const FIRST_WRITE: u32 = 41;
/// The first value the checkpoint can observe.
const FIRST_OBSERVABLE: u32 = 42;
/// The write AFTER the guard's read — the ride.
const FINAL_WRITE: u32 = 100;
/// The mutator's step while it keeps the checkpoint request answered.
const STEP: u32 = 1;

/// The drive vocabulary.
trait CountDriven: Send {
    fn set(&mut self, v: u32);
    fn get(&mut self) -> u32;
}

/// A CONCRETE slot content — no erasure.
struct Cell32(u32);

// the client's one-line identity impl (a local type — orphan-legal)
// SAFETY: the projection is the identity — the line IS the value itself.
unsafe impl AsPtr for Cell32 {
    type Target = Cell32;

    unsafe fn as_ptr(this: *mut Self) -> *mut Self::Target {
        this
    }
}

impl CountDriven for InnerHandle<Cell32> {
    fn set(&mut self, v: u32) {
        // the drive's checkpoint boundary: park while a checkpoint is
        // pending, then write (no deref borrow crosses a park — I4).
        // The re-check loop is the drive's own polling: the request may
        // land after this drive's previous park.
        while self.is_stop_requested() {
            self.safepoint(); // parks; returns when the world resumes
        }
        self.0 = v;
    }

    fn get(&mut self) -> u32 {
        self.0
    }
}

/// Write, checkpoint-read, write — on a concrete slot.
#[test]
fn concrete_slot_write_checkpoint_write_is_tb_clean() {
    let mut root: PersistentCell<Box<Cell32>> = PersistentCell::new(Box::new(Cell32(START)));
    let final_value = independent_task_root::<Box<Cell32>, dyn CountDriven, u32>(
        &mut root,
        |mut outer, checkpointer| {
            outer.set(FIRST_WRITE); // write 1 — a mint `&mut` would go Unique here
            // the requester runs off-thread: the guard's read is the
            // foreign read that would freeze a mint `&mut`; the mutator
            // keeps its checkpoint loop responsive until the guard has
            // been dropped
            let checkpointer = checkpointer.clone();
            std::thread::scope(|scope| {
                let requester = scope.spawn(move || {
                    let guard = checkpointer.request();
                    let read = guard.0;
                    assert!(read >= FIRST_OBSERVABLE, "the guard reads the last written value");
                    // the guard drops here: the world resumes
                });
                let mut v = FIRST_OBSERVABLE;
                while !requester.is_finished() {
                    outer.set(v); // parks inside while a request is pending
                    v += STEP;
                }
                requester.join().expect("requester panicked");
            });
            outer.set(FINAL_WRITE); // the write AFTER the guard's read: the ride
            outer.get()
        },
    );
    assert_eq!(final_value, FINAL_WRITE);
}
