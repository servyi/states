//! ATTACK 1 (use-after-free from a purely safe client):
//!
//! `fanout`'s `F: Fn(OuterHandle<'o, K, V>) -> R` lets `R` BE the
//! child handle itself — `|o| o` — so the child outers ride the
//! `Vec<R>` back into the parent's state code. The `'o` brand keeps
//! them inside the root task, but the PARENT'S OWN CHAIN reaches the
//! very same slots through `DerefMut` onto its node: nothing ties
//! the escaped handles' validity to the node's layout. Clearing the
//! children frees the heap the escaped outers still point into.

#![forbid(unsafe_code)]

use hslib::{HandleAble, InnerHandle, OuterHandle, independent_task_root};

trait Trait: Send {
    fn bark(&mut self) -> i32;
}

type Slot = Box<dyn HandleAble<dyn Trait> + 'static>;

struct MyStruct {
    children: Vec<Slot>,
}
struct YourStruct {
    val: i32,
}

impl Trait for InnerHandle<MyStruct, dyn Trait> {
    fn bark(&mut self) -> i32 {
        // THE ESCAPE: each child's OuterHandle is its own R
        let escaped: Vec<OuterHandle<'_, Slot, dyn Trait>> = self.fanout(
            &|m: &mut MyStruct| m.children.iter_mut().collect(),
            |o| o,
        );

        // the parent's own chain, same slots, no lifetime tie
        let m: &mut MyStruct = self;
        m.children.clear(); // drops the Boxes -> frees the children

        // drive an escaped outer: deref into freed memory
        let mut total = 0;
        for mut o in escaped {
            total += o.bark();
        }
        total
    }
}

impl Trait for InnerHandle<YourStruct, dyn Trait> {
    fn bark(&mut self) -> i32 {
        self.val
    }
}

fn main() {
    let mut m: Slot = Box::new(MyStruct {
        children: vec![
            Box::new(YourStruct { val: 5 }),
            Box::new(YourStruct { val: 7 }),
        ],
    });
    independent_task_root(&mut m, |mut outer, _checkpointer| {
        let total = outer.bark();
        println!("total from freed children: {}", total);
    });
}
