//! ROUND 3, PROBE C (leak-class, expected HANG not UB):
//!
//! mem::forget(guard) is safe: the guard leaks, its Drop never
//! resumes the world — the parked mutator waits forever and the
//! gate stays locked. Leak/deadlock by forget is traditionally
//! safe-code behavior (cf. MutexGuard); recorded, not a soundness
//! break.

#![forbid(unsafe_code)]

use std::mem::forget;
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
        self.safepoint(); // parks; nobody will ever resume it
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
    independent_task_root(&mut m, |mut outer, ck| {
        let ck2 = ck.clone();
        std::thread::scope(|scope| {
            let checker = scope.spawn(move || {
                let guard = ck2.request();
                forget(guard); // the world never resumes
                println!("guard forgotten");
            });
            outer.bark(); // parked forever
            checker.join().expect("checker");
        });
    });
}
