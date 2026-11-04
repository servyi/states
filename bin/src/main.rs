//! The binary half: the vocabulary (`Trait`) and the structs —
//! everything that directly references them — plus the demo.

use std::ops::{Deref, DerefMut};

use hslib::{HandleAble, InnerHandle};

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

/// Mint over a LIVE slot the driver owns (the slot must outlive the
/// handle — the discipline the OuterHandle will own next step).
fn mint_over(slot: &mut Slot) -> Box<dyn Trait> {
    let holder = (slot
        as &mut dyn DerefMut<Target = dyn HandleAble<dyn Trait>>)
        as *mut dyn DerefMut<Target = dyn HandleAble<dyn Trait>>;
    // SAFETY: the slot is live for the whole drive (the caller owns
    // it beside the handle); the receiver derives from the same
    // borrow as the holder pointer — one borrow stack.
    let erased: &mut dyn HandleAble<dyn Trait> = unsafe { &mut **holder };
    erased.box_innerhandle(holder)
}

fn main() {
    let mut m: Slot = Box::new(MyStruct {
        child: Box::new(YourStruct { val: 5 }),
    });
    let mut h: Box<dyn Trait> = mint_over(&mut m);
    h.bark();
    drop(h);
    drop(m);

    let mut y: Slot = Box::new(YourStruct { val: 5 });
    let mut h2: Box<dyn Trait> = mint_over(&mut y);
    h2.bark();
    drop(h2);
    drop(y);
}
