//! ROUND 3, PROBE B (expected REFUSED — the invariant brand seal):
//!
//! Stash the child outer into an environment-typed cell instead of
//! returning it. The cell's type mentions a FIXED lifetime from the
//! environment; the callback receives OuterHandle<'n> for the
//! UNIVERSAL 'n — invariance must refuse the unification.

#![forbid(unsafe_code)]

use std::cell::RefCell;
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

thread_local! {
    static STASH: RefCell<Option<OuterHandle<'static, Slot, dyn Trait>>> = const { RefCell::new(None) };
}

impl Trait for InnerHandle<MyStruct, dyn Trait> {
    fn bark(&mut self) -> i32 {
        let sums: Vec<i32> = self.fanout(
            &|m: &mut MyStruct| m.children.iter_mut().collect(),
            |o| {
                // stash the handle instead of returning it
                STASH.with(|s| *s.borrow_mut() = Some(o));
                0
            },
        );
        sums.iter().sum()
    }
}

impl Trait for InnerHandle<YourStruct, dyn Trait> {
    fn bark(&mut self) -> i32 {
        self.val
    }
}

fn main() {
    let mut m: Slot = Box::new(MyStruct {
        children: vec![Box::new(YourStruct { val: 5 })],
    });
    independent_task_root(&mut m, |mut outer, _ck| {
        let _ = outer.bark();
    });
}
