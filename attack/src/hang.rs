//! ATTACK 2 (checkpoint deadlock from a purely safe client):
//!
//! The same escape leaves the children's `Shared`s alive in the
//! parent's hands (the escaped outers own their status `Arc`s) with
//! NO thread that will ever park them. The root's `park_self`
//! requests every registered child first — the escaped child never
//! parks, never dies -> the request loops forever -> the guard never
//! comes -> the world never stops and the root never resumes.

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
        let escaped: Vec<hslib::OuterHandle<'_, Slot, dyn Trait>> = self.fanout(
            &|m: &mut MyStruct| m.children.iter_mut().collect(),
            |o| o,
        );
        // the children's Shareds stay alive here, unparkable
        self.safepoint(); // <- parks: requests children first... forever
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
    independent_task_root(&mut m, |mut outer, checkpointer| {
        let ck = checkpointer.clone();
        std::thread::scope(|scope| {
            let checker = scope.spawn(move || {
                let _guard = ck.request(); // never granted
                println!("checkpoint: acquired (unreachable)");
            });
            outer.bark(); // parks forever inside
            checker.join().expect("checker");
        });
    });
}
