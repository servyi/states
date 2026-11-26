//! The concrete-slot write-save-write ride: a mint that materializes
//! a `&mut` goes `Unique` on the first write and `Frozen` on the
//! guard's read — every later write through the chain's subtree is
//! then forbidden (Tree Borrows). The raw-only mint must ride clean.

use hslib::{AsPtr, InnerHandle, PersistentCell, independent_task_root};

trait CountDriven: Send {
    fn set(&mut self, v: u32);
    fn get(&mut self) -> u32;
}

/// A CONCRETE slot content (no dyn erasure) — the shape whose mint
/// used to materialize the freezing `&mut` (through `as_mut_k` +
/// `DerefMut::deref_mut` + `as_ptr(&mut self)`).
struct Cell32(u32);

// the client's one-line identity impl (a local type — orphan-legal)
unsafe impl AsPtr for Cell32 {
    type Target = Cell32;

    unsafe fn as_ptr(this: *mut Self) -> *mut Self::Target {
        this
    }
}

impl CountDriven for InnerHandle<Cell32> {
    fn set(&mut self, v: u32) {
        // the drive's checkpoint boundary: park while a checkpoint is
        // pending, then write (borrowck: no deref borrow crosses a
        // park — I4). The re-check loop is the drive's own polling:
        // the request may land after this drive's previous park.
        while self.is_stop_requested() {
            self.safepoint(); // parks; returns when the world resumes
        }
        self.0 = v;
    }
    fn get(&mut self) -> u32 {
        self.0
    }
}

#[test]
fn concrete_slot_write_save_write_is_tb_clean() {
    let mut m: PersistentCell<Box<Cell32>> = PersistentCell::new(Box::new(Cell32(0)));
    independent_task_root::<Box<Cell32>, dyn CountDriven, ()>(&mut m, |mut outer, checkpointer| {
        outer.set(41); // write 1 — a mint `&mut` would go Unique here
        // the requester runs off-thread (the protocol per the demos):
        // the guard's read is the foreign read that freezes a mint
        // `&mut`; the mutator keeps its checkpoint loop responsive
        // until the guard has been dropped
        let ck = checkpointer.clone();
        std::thread::scope(|scope| {
            let requester = scope.spawn(move || {
                let g = ck.request();
                let read = g.0;
                assert!(read >= 42, "the guard reads the last written value");
                // the guard drops here: the world resumes
            });
            let mut v = 42;
            while !requester.is_finished() {
                outer.set(v); // parks inside while a request is pending
                v += 1;
            }
            requester.join().expect("requester");
        });
        outer.set(100); // the write AFTER the guard's read: the freeze ride
        assert_eq!(outer.get(), 100);
    });
}
