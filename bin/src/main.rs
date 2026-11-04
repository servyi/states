//! The binary half of the runnable sketch: the vocabulary (`Trait`)
//! and the two structs — everything that directly references them —
//! plus the demo main that reassigns the outer `Box<dyn Trait>`.

use std::ops::{Deref, DerefMut};

use hslib::{HandleAble, InnerHandle};

/// The holder type, Jonas's recursion: K = Box<dyn HandleAble<V, K>>.
/// A raw alias cycles (E0391); the newtype indirection names it —
/// and provides the chain's DerefMut through the inner Box.
struct Holder<V: ?Sized + 'static>(Box<dyn HandleAble<Holder<V>, V> + 'static>);

unsafe impl<V: ?Sized + 'static> hslib::ReplaceAble<Holder<V>, V> for Holder<V> {
    fn new(replacement: Box<dyn HandleAble<Holder<V>, V> + 'static>) -> Self {
        Holder(replacement)
    }
}


impl<V: ?Sized + 'static> Deref for Holder<V> {
    type Target = dyn HandleAble<Holder<V>, V>;
    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}
impl<V: ?Sized + 'static> DerefMut for Holder<V> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut *self.0
    }
}

trait Trait: Send {
    fn bark(&mut self);
}

struct MyStruct {
    child: Holder<dyn Trait>,
}
struct YourStruct {
    val: i32,
}

impl Trait for InnerHandle<Holder<dyn Trait>, MyStruct, dyn Trait> {
    fn bark(&mut self) {
        println!("MyStruct");
        // fan the erased child out: split over &mut MyStruct yields
    }
}

impl Trait for InnerHandle<Holder<dyn Trait>, YourStruct, dyn Trait> {
    fn bark(&mut self) {
        println!("{}", self.val) // through DerefMut onto YourStruct
    }
}

fn main() {
    // the roots are stored erased (the tree's shape): the mint
    // receives the box SLOT, the handle points there
    let mut m: Box<dyn HandleAble<Holder<dyn Trait>, dyn Trait>> =
        Box::new(MyStruct {
            child: Holder(Box::new(YourStruct { val: 5 })),
        });
    // the holder crosses as the fat erased pointer
    // the root's holder is a GENUINE Holder wrapping the erased box
    // (the Holder newtype exists to name the recursive type; no
    // layout puns)
    let mut root = Holder(m);
    let root_holder: *mut Holder<dyn Trait> = &mut root;
    // SAFETY: root exclusively owns the box; the mint's dispatch
    // receiver reborrows through the Holder's DerefMut.
    let erased: &mut dyn HandleAble<Holder<dyn Trait>, dyn Trait> = &mut *root.0;
    let mut h: Box<dyn Trait> = erased.box_innerhandle(root_holder);
    h.bark();

    let mut y: Box<dyn HandleAble<Holder<dyn Trait>, dyn Trait>> = Box::new(YourStruct { val: 5 });
    let mut root_y = Holder(y);
    let y_holder: *mut Holder<dyn Trait> = &mut root_y;
    let erased_y: &mut dyn HandleAble<Holder<dyn Trait>, dyn Trait> = &mut *root_y.0;
    h = erased_y.box_innerhandle(y_holder);
    h.bark();

    box_self_replacement();
}


/// The bypass, exercised: Box<dyn HandleAble<K, V>> replaces itself
/// through the GENERIC library impl — no newtype, no named
/// recursion — wherever a K is in scope. Here K = Holder<dyn Trait>
/// (the named recursion): the tree's holder boxes are
/// Box<dyn HandleAble<Holder<dyn Trait>, dyn Trait>>, and one such
/// box replaces itself with a fresh successor box.
fn box_self_replacement() {
    type ErasedBox = Box<dyn HandleAble<Holder<dyn Trait>, dyn Trait> + 'static>;
    let mut holder: Holder<dyn Trait> = Holder(Box::new(YourStruct { val: 41 }));
    // the successor: a fresh erased box
    let successor: ErasedBox = Box::new(YourStruct { val: 42 });
    // the BOX replaces itself — the generic impl, no Holder::new:
    let inner: &mut ErasedBox = &mut holder.0;
    *inner = <ErasedBox as hslib::ReplaceAble<Holder<dyn Trait>, dyn Trait>>::new(successor);
    // and the holder still works through the chain (the slot first —
    // the E0499 lesson from the root wiring):
    let mut h: Box<dyn Trait> = {
        let slot: *mut Holder<dyn Trait> = &mut holder;
        let erased: &mut dyn HandleAble<Holder<dyn Trait>, dyn Trait> = &mut *holder.0;
        erased.box_innerhandle(slot)
    };
    h.bark(); // 42 — the replacement took
}
