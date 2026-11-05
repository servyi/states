//! ATTACK 4 (self-request deadlock, purely safe):
//!
//! The mutator thread itself calls `request()`: the handshake waits
//! for the mutator to park, but the mutator is the waiter — it can
//! never park. (Same shape as main's `with_checkpoint_pair`, so this
//! one may be a known/accepted hazard; recorded for completeness.)

#![forbid(unsafe_code)]

use hslib::{HandleAble, InnerHandle, independent_task_root};

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
        self.children.len() as i32
    }
}

impl Trait for InnerHandle<YourStruct, dyn Trait> {
    fn bark(&mut self) -> i32 {
        self.val
    }
}

fn main() {
    let mut m: Slot = Box::new(MyStruct { children: vec![] });
    independent_task_root(&mut m, |mut outer, checkpointer| {
        outer.bark();
        // the mutator requests its own checkpoint: waits for itself
        let _guard = checkpointer.request();
        println!("never reached");
    });
}
