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
struct CustomBox(Box<PersistentCell<dyn HandleAble<dyn Trait>>>);

impl Drop for CustomBox {
    fn drop(&mut self) {
        DROPS.fetch_add(1, Ordering::SeqCst);
        println!("custom drop");
    }
}

// the container: derefs to its boxed cell — the cell impl covers the
// AsPtr bound with zero client unsafe
impl Deref for CustomBox {
    type Target = PersistentCell<dyn HandleAble<dyn Trait>>;
    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}
impl DerefMut for CustomBox {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut *self.0
    }
}

struct MyStruct {
    children: Vec<PersistentCell<CustomBox>>,
}
struct YourStruct {
    val: i32,
}

impl Trait for InnerHandle<MyStruct> {
    fn bark(&mut self) -> i32 {
        let sums: Vec<i32> = self.fanout(
            &|m: &MyStruct| m.children.iter().collect(),
            |mut o| {
                // the child installs a successor built with the
                // CLIENT'S constructor — the library never constructs
                let successor = CustomBox(PersistentCell::from_boxed(Box::new(YourStruct { val: 99 }) as Box<dyn HandleAble<dyn Trait>>));
                o.replace(successor);
                o.bark()
            },
        );
        self.safepoint();
        sums.iter().sum()
    }
}

impl Trait for InnerHandle<YourStruct> {
    fn bark(&mut self) -> i32 {
        self.val
    }
}

fn main() {
    let mut m = PersistentCell::new(CustomBox(PersistentCell::from_boxed(
        Box::new(MyStruct {
            children: vec![
                PersistentCell::new(CustomBox(PersistentCell::from_boxed(
                    Box::new(YourStruct { val: 5 }) as Box<dyn HandleAble<dyn Trait>>))),
                PersistentCell::new(CustomBox(PersistentCell::from_boxed(
                    Box::new(YourStruct { val: 7 }) as Box<dyn HandleAble<dyn Trait>>))),
            ],
        }) as Box<dyn HandleAble<dyn Trait>>,
    )));
    independent_task_root(&mut m, |mut outer, _ck| {
        let first = outer.bark();
        println!("sum: {} (children replaced themselves)", first);
        // the ROOT install with the client's constructor
        let successor = CustomBox(PersistentCell::from_boxed(Box::new(YourStruct { val: 42 }) as Box<dyn HandleAble<dyn Trait>>));
        outer.replace(successor);
        let second = outer.bark();
        println!("sum: {}", second);
        println!("custom drops so far: {}", DROPS.load(Ordering::SeqCst));
    });
}
