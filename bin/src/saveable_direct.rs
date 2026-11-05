//! The DIRECT SaveAble shape: K = Box<dyn SaveAble> — literally.
//!
//! `SaveAble: HandleAble<dyn Trait>`, so every `&mut dyn SaveAble`
//! IS a `&mut dyn HandleAble<dyn Trait>` (stable upcasting) — the
//! upcast just had to be NAMED (`AsErased`) because trait objects
//! are invariant in their parameters: `dyn DerefMut<Target =
//! dyn SaveAble>` does not coerce to `dyn DerefMut<Target = dyn
//! HandleAble<V>>`. The crate blanket covers exact-Target Derefs
//! (plain boxes and safe wrappers); a subtrait box implements the
//! hook itself — ONE small impl, and it must be honest:
//!
//! The guard then speaks the vocabulary directly: `guard.save()`.

use std::ops::DerefMut;
use hslib::{AsErased, HandleAble, InnerHandle, independent_task_root};

trait Trait: Send {
    fn bark(&mut self) -> i32;
}

/// the client's save vocabulary: a subtrait of HandleAble
trait SaveAble: HandleAble<dyn Trait> {
    fn save(&self) -> String;
}

/// K, LITERALLY the subtrait box
type SaveSlot = Box<dyn SaveAble>;

// SAFETY: deref honesty — both methods return exactly what a plain
// DerefMut/Deref through the box yields, upcast at this concrete
// site; nothing foreign is ever produced.
unsafe impl AsErased<dyn Trait> for Box<dyn SaveAble> {
    fn as_erased(&mut self) -> &mut dyn HandleAble<dyn Trait> {
        self.deref_mut() // &mut dyn SaveAble -> &mut dyn HandleAble
    }
    fn as_erased_shared(&self) -> &dyn HandleAble<dyn Trait> {
        &**self // &dyn SaveAble -> &dyn HandleAble
    }
}

struct MyStruct {
    children: Vec<SaveSlot>,
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
        self.safepoint();
        sums.iter().sum()
    }
}

impl Trait for InnerHandle<YourStruct, dyn Trait> {
    fn bark(&mut self) -> i32 {
        self.val
    }
}

impl SaveAble for MyStruct {
    fn save(&self) -> String {
        let kids: Vec<String> = self.children.iter().map(|c| c.save()).collect();
        format!("My({})", kids.join(","))
    }
}
impl SaveAble for YourStruct {
    fn save(&self) -> String {
        format!("{}", self.val)
    }
}

fn main() {
    let mut m: SaveSlot = Box::new(MyStruct {
        children: vec![
            Box::new(YourStruct { val: 5 }),
            Box::new(YourStruct { val: 7 }),
        ],
    });
    independent_task_root(&mut m, |mut outer, checkpointer| {
        let ck = checkpointer.clone();
        std::thread::scope(|scope| {
            let checker = scope.spawn(move || {
                let guard = ck.request();
                // the guard DIRECTLY speaks the client vocabulary —
                // auto-deref through the box onto dyn SaveAble
                println!("checkpoint snapshot: {}", guard.save());
            });
            let first = outer.bark();
            println!("sum: {}", first);
            let successor: SaveSlot = Box::new(YourStruct { val: 42 });
            outer.replace(successor);
            let second = outer.bark();
            println!("sum: {}", second);
            checker.join().expect("checker");
        });
    });
}
