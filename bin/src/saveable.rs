//! The SaveAble shape: the client's own subtrait vocabulary on the
//! slot, read through the checkpoint guard.
//!
//! Directly `K = Box<dyn SaveAble>` fails the library's
//! `DerefMut<Target = dyn HandleAble<V>>` (Target equality — no
//! generic dyn-upcast on stable). The working shape: a thin wrapper
//! whose SAFE Deref/DerefMut impls upcast at a concrete coercion
//! site, plus `replace(val: K)` (plain K-cell assignment — the
//! identity pass-through that made ReplaceAble deletable).

use std::ops::{Deref, DerefMut};
use hslib::{PersistentCell, HandleAble, InnerHandle, independent_task_root};

/// the drive vocabulary (as in the main demo)
trait Trait: Send {
    fn bark(&mut self) -> i32;
}

/// the CLIENT's save vocabulary: a subtrait of HandleAble
trait SaveAble: HandleAble<dyn Trait> {
    fn save(&self) -> String;
}

/// the slot: the wrapper whose safe Derefs upcast to the erased form
struct SaveSlot(Box<dyn SaveAble>);

impl Deref for SaveSlot {
    type Target = dyn HandleAble<dyn Trait>;
    fn deref(&self) -> &Self::Target {
        &*self.0 // &dyn SaveAble -> &dyn HandleAble (stable upcast)
    }
}
impl DerefMut for SaveSlot {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut *self.0
    }
}

// SAFETY: the pointer is the inner builtin box's stored line (the
// raw borrow of the builtin deref place — no reference materialized,
// not the wrapper's Deref return); the wrapper cannot delete or move
// the store except through &mut (dropping/moving the box).
unsafe impl hslib::AsPtr for SaveSlot {
    fn as_ptr(&mut self) -> *mut Self::Target {
        &raw mut *self.0
    }
}

struct MyStruct {
    children: Vec<PersistentCell<SaveSlot>>,
}
struct YourStruct {
    val: i32,
}

impl Trait for InnerHandle<MyStruct, dyn Trait> {
    fn bark(&mut self) -> i32 {
        let sums: Vec<i32> = self.fanout(
            &|m: &MyStruct| m.children.iter().collect(),
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
        let kids: Vec<String> = self.children.iter().map(|c| c.0.save()).collect();
        format!("My({})", kids.join(","))
    }
}
impl SaveAble for YourStruct {
    fn save(&self) -> String {
        format!("{}", self.val)
    }
}

fn main() {
    let mut m = PersistentCell::new(SaveSlot(Box::new(MyStruct {
        children: vec![
            PersistentCell::new(SaveSlot(Box::new(YourStruct { val: 5 }))),
            PersistentCell::new(SaveSlot(Box::new(YourStruct { val: 7 }))),
        ],
    })));
    independent_task_root(&mut m, |mut outer, checkpointer| {
        // THE SNAPSHOT READS CLIENT VOCABULARY through the guard's &K
        let ck = checkpointer.clone();
        std::thread::scope(|scope| {
            let checker = scope.spawn(move || {
                let guard = ck.request();
                let snapshot: String = guard.0.save(); // <-- the gap closed
                println!("checkpoint snapshot: {}", snapshot);
            });
            let first = outer.bark();
            println!("sum: {}", first);
            // the install, K directly:
            let successor = SaveSlot(Box::new(YourStruct { val: 42 }));
            outer.replace(successor);
            let second = outer.bark();
            println!("sum: {}", second);
            checker.join().expect("checker");
        });
    });
}
