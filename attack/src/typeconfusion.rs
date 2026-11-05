//! ATTACK 3 (type confusion via the checkpoint guard, purely safe):
//!
//! `CheckpointGuard: DerefMut -> K` lets the checker WRITE the root
//! slot while the tree is parked. The slot is type-erased
//! (`Box<dyn HandleAble<V>>`), so the write can install a state of a
//! DIFFERENT type than the one the parked mutator's handle was minted
//! over. On resume, the chain casts the erased pointee back to its
//! original `T` — now a lie. (On main the guard derefs to the node
//! type itself, so a write swaps values of the SAME type; the erasure
//! is what makes the write unsound here.)

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
        let sums: Vec<i32> = self.fanout(
            &|m: &mut MyStruct| m.children.iter_mut().collect(),
            |mut o| o.bark(),
        );
        // park here: while we are parked, the checker swaps the root
        // state under us (see main)
        self.safepoint();
        // resume: the chain still believes T = MyStruct, but the slot
        // now holds a 4-byte YourStruct — reading `children` (a Vec)
        // from it is out-of-bounds/type-confused
        let m: &mut MyStruct = self;
        m.children.clear();
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
        children: vec![
            Box::new(YourStruct { val: 5 }),
            Box::new(YourStruct { val: 7 }),
        ],
    });
    independent_task_root(&mut m, |mut outer, checkpointer| {
        let ck = checkpointer.clone();
        std::thread::scope(|scope| {
            let checker = scope.spawn(move || {
                let mut guard = ck.request(); // world stops, root parked
                // SAFE WRITE: install a different state TYPE under the
                // parked mutator's handle
                *guard = Box::new(YourStruct { val: 99 });
                // guard drops: the world resumes over the swapped slot
            });
            outer.bark(); // parks in safepoint, resumes confused
            checker.join().expect("checker");
        });
    });
}
