//! UnsafeCell experiment: the root state wrapped, both sides cast
//! their borrows from the cell's raw pointer.
use std::cell::UnsafeCell;
use hslib::{Checkpointer, HandleAble, InnerHandle, OuterHandle, independent_task_root};

trait Trait: Send {
    fn step(&mut self);
    fn observe(&mut self) -> String;
}
trait SaveAble: HandleAble<dyn Trait> {
    fn save(&self) -> String;
}

struct Machine;
struct Counter {
    val: i32,
}
/// the wrapped root: everything mutable lives behind the cell
struct Guarded(UnsafeCell<MachineInner>);

struct MachineInner {
    children: Vec<Box<dyn SaveAble>>,
    steps: i32,
}

impl Trait for InnerHandle<Guarded, dyn Trait> {
    fn step(&mut self) {
        // the MUTATOR casts its borrow from the cell's raw
        // SAFETY: the world-stopped handshake serializes vs saves
        let m: &mut MachineInner = unsafe { &mut *self.0.get() };
        m.steps += 1; // <-- was the TB failure site
        self.safepoint();
    }
    fn observe(&mut self) -> String {
        // the parking window ONLY — no read: splits the mutator's own
        // shared read from the checker's save read
        self.safepoint();
        String::new()
    }
}

impl SaveAble for Guarded {
    fn save(&self) -> String {
        // the READER casts its borrow from the cell's raw
        // SAFETY: the world is stopped (the guard's handshake)
        let m: &MachineInner = unsafe { &*self.0.get() };
        let kids: Vec<String> = m.children.iter().map(|c| c.save()).collect();
        format!("M({} steps, [{}])", m.steps, kids.join(","))
    }
}
impl Trait for InnerHandle<Counter, dyn Trait> {
    fn step(&mut self) {}
    fn observe(&mut self) -> String {
        String::new()
    }
}

impl SaveAble for Counter {
    fn save(&self) -> String {
        format!("C({})", self.val)
    }
}

/// a save while parked in the safepoint window
fn park_save(
    outer: &mut OuterHandle<'_, Box<dyn SaveAble>, dyn Trait>,
    ck: &Checkpointer<Box<dyn SaveAble>>,
) -> String {
    std::thread::scope(|s| {
        let t = s.spawn({
            let ck = ck.clone();
            move || {
                let guard = ck.request();
                guard.save()
            }
        });
        while !t.is_finished() {
            // SAFETY: read-only window
            outer.observe(); // the parking windows
        }
        t.join().unwrap()
    })
}

fn main() {
    let mut m: Box<dyn SaveAble> = Box::new(Guarded(UnsafeCell::new(MachineInner {
        children: vec![Box::new(Counter { val: 1 })],
        steps: 0,
    })));
    independent_task_root(&mut m, |mut outer, ck| {
        let snap1 = park_save(&mut outer, &ck); // save #1
        outer.step();                           // WRITE #1 (through the cell cast)
        let snap2 = park_save(&mut outer, &ck); // save #2
        outer.step();                           // WRITE #2 — was TB UB
        println!("snaps: {} / {}", snap1, snap2);
    });
}
