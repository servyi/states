//! A container that is NOT Box: CustomBox, with its own Drop — the
//! `*slot = val` place assignment must dispose the old value through
//! the container's own drop glue and move the new one in.

use std::sync::atomic::{AtomicU32, Ordering};
use hslib::{AsPtr, PersistentCell, HandleAble, InnerHandle, independent_task_root};

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

// the container: a wrapper around a boxed cell. Its AsPtr impl does
// NOT go through a Deref (a Deref would hand out a fresh reference):
// the raw borrow of the builtin box-deref place on the wrapper's own
// field loads the box's STORED pointer — the line a Deref would hand
// out, minus the reference.
// SAFETY: the backing store lives until the counted Drop runs — and
// dropping requires the `&mut` the minter's discipline already holds,
// so no minted raw dangles early.
unsafe impl AsPtr for CustomBox {
    type Target = PersistentCell<dyn HandleAble<dyn Trait>>;

    unsafe fn as_ptr(this: *mut Self) -> *mut Self::Target {
        // SAFETY: `this` is a live `CustomBox` under the minter's
        // exclusivity; the field projection is the wrapper's own
        // storage, then the builtin box deref loads the stored line.
        let boxed: *mut Box<PersistentCell<dyn HandleAble<dyn Trait>>> =
            unsafe { std::ptr::addr_of_mut!((*this).0) };
        // SAFETY: as above — the builtin box deref place, raw only.
        unsafe { &raw mut **boxed }
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
