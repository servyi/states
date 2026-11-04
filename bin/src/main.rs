//! The binary half: the vocabulary (`Trait`) and the structs —
//! everything that directly references them — plus the demo.

use hslib::{HandleAble, InnerHandle, OuterHandle};

trait Trait: Send {
    fn bark(&mut self);
}

/// The demo's slot shape: the erased box (no Holder needed — with
/// HandleAble<V> single-param, Box<dyn HandleAble<V>> is not
/// recursive).
type Slot = Box<dyn HandleAble<dyn Trait> + 'static>;

struct MyStruct {
    child: Slot,
}
struct YourStruct {
    val: i32,
}

impl Trait for InnerHandle<MyStruct, dyn Trait> {
    fn bark(&mut self) {
        println!("MyStruct");
        self.fanout(
            &|m: &mut MyStruct| vec![&mut m.child],
            |mut b: Box<dyn Trait>| b.bark(),
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
fn outer_over(slot: &mut Slot) -> OuterHandle<Slot, dyn Trait> {
    // SAFETY: the slot lives for the whole drive (main owns it
    // beside the outer handle).
    unsafe { OuterHandle::over_slot(std::ptr::from_mut(slot)) }
}

fn main() {
    let mut m: Slot = Box::new(MyStruct {
        child: Box::new(YourStruct { val: 5 }),
    });
    let mut outer = outer_over(&mut m);
    outer.bark();
    // THE INSTALL: the OuterHandle walks to the slot through the
    // current inner handle, swaps it, re-mints, keeps the fresh one
    let successor: Slot = Box::new(YourStruct { val: 42 });
    outer.replace(successor);
    outer.bark();
    drop(outer);
    drop(m);

    let mut y: Slot = Box::new(YourStruct { val: 5 });
    let mut outer2 = outer_over(&mut y);
    outer2.bark();
    drop(outer2);
    drop(y);
}
