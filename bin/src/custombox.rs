//! A container that is NOT Box: CustomBox, with its own Drop — the
//! `*slot = val` place assignment must dispose the old value through
//! the container's own drop glue and move the new one in.

use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicU32, Ordering};
use hslib::{PersistentCell, HandleAble, InnerHandle, independent_task_root};

trait Trait: Send {
    fn bark(&mut self) -> i32;
}

static DROPS: AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// the client's own container: a counted-Drop wrapper
struct CustomBox(Box<dyn HandleAble<dyn Trait>>);

impl Drop for CustomBox {
    fn drop(&mut self) {
        DROPS.fetch_add(1, Ordering::SeqCst);
        println!("custom drop");
    }
}

// plain Deref/DerefMut to the erased target — the crate's AsErased
// blanket covers this K with zero unsafe
impl Deref for CustomBox {
    type Target = dyn HandleAble<dyn Trait>;
    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}
impl DerefMut for CustomBox {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut *self.0
    }
}

// SAFETY: the pointer is the inner builtin box's stored line (the
// raw borrow of the builtin deref place); the store can only be
// deleted by dropping the box, which needs exclusive access.
unsafe impl hslib::AsPtr for CustomBox {
    fn as_ptr(&mut self) -> *mut Self::Target {
        &raw mut *self.0
    }
}

struct MyStruct {
    children: Vec<PersistentCell<CustomBox>>,
}
struct YourStruct {
    val: i32,
}

impl Trait for InnerHandle<MyStruct, dyn Trait> {
    fn bark(&mut self) -> i32 {
        let sums: Vec<i32> = self.fanout(
            &|m: &MyStruct| m.children.iter().collect(),
            |mut o| {
                // the child installs a successor built with the
                // CLIENT'S constructor — the library never constructs
                let successor = CustomBox(Box::new(YourStruct { val: 99 }));
                o.replace(successor);
                o.bark()
            },
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

fn main() {
    let mut m = PersistentCell::new(CustomBox(Box::new(MyStruct {
        children: vec![
            PersistentCell::new(CustomBox(Box::new(YourStruct { val: 5 }))),
            PersistentCell::new(CustomBox(Box::new(YourStruct { val: 7 }))),
        ],
    })));
    independent_task_root(&mut m, |mut outer, _ck| {
        let first = outer.bark();
        println!("sum: {} (children replaced themselves)", first);
        // the ROOT install with the client's constructor
        let successor = CustomBox(Box::new(YourStruct { val: 42 }));
        outer.replace(successor);
        let second = outer.bark();
        println!("sum: {}", second);
        println!("custom drops so far: {}", DROPS.load(Ordering::SeqCst));
    });
}
