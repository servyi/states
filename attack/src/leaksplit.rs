//! ROUND 3, PROBE A (expected SOUND — clearance evidence):
//!
//! The split can hand the fanout OUT-OF-TREE slots: a `Box::leak`'d
//! `&'static mut K` captured by the split closure satisfies the
//! HRTB (only 'static captures can). The fanout mints children over
//! foreign-but-valid memory; the leaked state is never freed, so
//! there is no use-after-free — this should run MIRI-CLEAN (bizarre
//! topology, sound memory).

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
        // two LEAKED slots, out-of-tree, handed to the fanout
        let a: &'static mut Slot = Box::leak(Box::new(Box::new(YourStruct { val: 11 })));
        let b: &'static mut Slot = Box::leak(Box::new(Box::new(YourStruct { val: 13 })));
        // move-capture: owning the leaked refs keeps the closure Sync
        // (a shared capture would borrow through *a, and the state is
        // !Sync)
        // Fn (not FnOnce): the returns are REBORROWS of the captured
        // leaked refs, so the closure stays callable
        let split = move |_m: &mut MyStruct| -> Vec<&mut Slot> { vec![&mut *a, &mut *b] };
        let sums: Vec<i32> = self.fanout(&split, |mut o| o.bark());
        // the node's real children were never touched
        let m: &mut MyStruct = self;
        m.children.len() as i32 + sums.iter().sum::<i32>()
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
        println!("leaksplit total: {}", outer.bark());
    });
}
