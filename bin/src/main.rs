//! The binary half: the vocabulary (`Trait`) and the structs —
//! everything that directly references them — plus the demo.

use hslib::{HandleAble, InnerHandle, OuterHandle, independent_task_root};

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

impl Trait for InnerHandle<MyStruct, dyn Trait> {
    fn bark(&mut self) {
        println!("MyStruct");
        self.fanout(
            // &mut K directly — K = Slot; the library does the
            // erased coercion internally. iter_mut().collect() just
            // works now (no unsized coercion needed at the call site).
            &|m: &mut MyStruct| m.children.iter_mut().collect(),
            |mut o: hslib::OuterHandle<Slot, dyn Trait>| o.bark(),
        );
    }
}

impl Trait for InnerHandle<YourStruct, dyn Trait> {
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
    independent_task_root(&mut m, |mut outer, _checkpointer| {
        outer.bark();
        // THE INSTALL: the OuterHandle walks to the slot through the
        // current inner handle, swaps it, re-mints, keeps the fresh one
        let successor: Slot = Box::new(YourStruct { val: 42 });
        outer.replace(successor);
        outer.bark();
    });

    let mut y: Slot = Box::new(YourStruct { val: 5 });
    independent_task_root(&mut y, |mut outer2, _checkpointer| {
        outer2.bark();
    });
}

