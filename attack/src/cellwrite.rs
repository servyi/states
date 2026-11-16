//! PROBE: interior-mutable state — the READER writes through a Cell
//! during the save window, the mutator writes the same Cell after
//! resume. Both ride the IM carve-out in theory; the models decide.

use hslib::{HandleAble, InnerHandle, PersistentCell, independent_task_root};

trait Trait: Send {
    fn step(&mut self);
}

struct Machine {
    cell: std::cell::Cell<i64>,
}

impl Trait for InnerHandle<Machine> {
    fn step(&mut self) {
        // the mutator's write through &mut (deref_mut) onto the Cell
        self.cell.set(self.cell.get() + 1);
        self.safepoint();
    }
}

fn main() {
    let mut root: PersistentCell<Box<PersistentCell<dyn HandleAble<dyn Trait>>>> =
        PersistentCell::new(PersistentCell::from_boxed(Box::new(Machine { cell: std::cell::Cell::new(0) })
            as Box<dyn HandleAble<dyn Trait>>));
    independent_task_root(&mut root, |mut outer, ck| {
        let ck3 = ck.clone();
        let ck3 = ck.clone();
        std::thread::scope(|s| {
            let reader = s.spawn(move || {
                let guard = ck3.request();
                let _ = &****guard;
                // the READER's write during the stop: the save is
                // "read-only" as a guard, but the client's save can
                // legitimately write its own Cell (the downcast is the
                // probe's plumbing; a real client reaches it through
                // their save() vocabulary)
                let d: &dyn HandleAble<dyn Trait> = &****guard;
                let m: &Machine = unsafe { &*(std::ptr::from_ref(d) as *const Machine) };
                m.cell.set(m.cell.get() + 100); // the reader's write
            });
            // the park loop: keep reaching safepoints until the
            // reader is done (the demo's observe-pattern)
            let mut steps = 0;
            while !reader.is_finished() {
                outer.step();
                steps += 1;
                if steps > 10_000 { break; } // bounded generously for the probe
            }
            reader.join().unwrap();
        });
    });
    println!("cellwrite probe done");
}
