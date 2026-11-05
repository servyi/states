//! The binary half: the vocabulary (`Trait`) and the structs —
//! everything that directly references them — plus the demo.

use hslib::{HandleAble, InstallHandle, independent_task_root};

trait Trait: Send {
    fn bark(&mut self);
}

/// The demo's slot shape: the erased box (no Holder needed — with
/// HandleAble<V> single-param, Box<dyn HandleAble<V>> is not
/// recursive).
type Slot = Box<dyn HandleAble<dyn Trait> + 'static>;

struct MyStruct {
    children: Vec<Slot>,
}
struct YourStruct {
    val: i32,
}

impl Trait for InstallHandle<MyStruct, dyn Trait> {
    fn bark(&mut self) {
        println!("MyStruct");
        self.fanout(
            // &mut K directly — K = Slot; the library does the
            // erased coercion internally. iter_mut().collect() just
            // works now (no unsized coercion needed at the call site).
            &|m: &mut MyStruct| m.children.iter_mut().collect(),
            // the child outers arrive branded with this fanout's 'o —
            // unnameable here, so they cannot be stored anywhere that
            // outlives the call
            |mut o| o.bark(),
        );
        // the SAFETY POINT: park here while a checkpoint runs
        self.safepoint();
    }
}

impl Trait for InstallHandle<YourStruct, dyn Trait> {
    fn bark(&mut self) {
        println!("{}", self.val) // through DerefMut onto YourStruct
    }
}

/// Wrap a LIVE driver-owned slot in the OuterHandle (the library's
/// over_slot does the mint).
fn main() {
    let mut m: Slot = Box::new(MyStruct {
        children: vec![
            Box::new(YourStruct { val: 5 }),
            Box::new(YourStruct { val: 7 }),
        ],
    });
    // the ROOT task: the slot is inaccessible inside (borrowed by the
    // call); the outer handle and its Checkpointer travel together
    independent_task_root(&mut m, |mut outer, checkpointer| {
        // a CHECKER: mid-run, from a SCOPED thread — the 'o brand
        // keeps the checkpointer inside the task, so a detached
        // thread cannot carry it — stop the world: the guard holds
        // the root slot while every mutator parks
        let ck = checkpointer.clone();
        std::thread::scope(|scope| {
            let checker = scope.spawn(move || {
                let guard = ck.request();
                let _root: &Slot = &guard;
                println!("checkpoint: world stopped, root held");
                drop(guard);
                println!("checkpoint: world resumed");
            });
            outer.bark();
            // THE INSTALL: the OuterHandle walks to the slot through the
            // current install handle, swaps it, re-mints, keeps the fresh one
            let successor: Slot = Box::new(YourStruct { val: 42 });
            outer.replace(successor);
            outer.bark();
            checker.join().expect("checker panicked");
        });
    });

    let mut y: Slot = Box::new(YourStruct { val: 5 });
    independent_task_root(&mut y, |mut outer2, _checkpointer| {
        outer2.bark();
    });
}

