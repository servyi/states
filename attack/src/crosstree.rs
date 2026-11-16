//! ATTACK (INVALID — retracted): cross-tree shared content.
//!
//! Jonas's ruling: the container itself is unsound, independent of
//! the library. The root takes &mut (or ownership) of the tree, so
//! every content is soundly reachable only as a &mut borrow of that
//! chain; an arena handing the same memory to two slots lets even a
//! plain safe user derive two conflicting &mut borrows WITHOUT the
//! library — verified: two ArenaSlots with one index, &mut *a and
//! &mut *b, aliasing &mut — Miri UB (SB: dead tag; TB likewise), no
//! hslib involved. The DerefMut exclusivity contract is the
//! container author's to uphold; a Copy slot with shared indices
//! violates it by construction. The cross-tree 'finding' therefore
//! says nothing about the library — kept as the record of the
//! retraction.

use std::cell::UnsafeCell;
use std::ops::{Deref, DerefMut};
use std::sync::Mutex;

use hslib::{HandleAble, InnerHandle, PersistentCell, independent_task_root};

trait Trait: Send {
    fn bump(&mut self);
}

/// the mini-registry (as in the arena demo)
#[derive(Clone, Copy)]
struct Erased {
    data: *mut (),
    vtable: *const (),
}
unsafe impl Send for Erased {}

static ARENA: Mutex<Vec<Erased>> = Mutex::new(Vec::new());

/// a COPY container: two values may carry the same index — the same
/// registered content
#[derive(Clone, Copy)]
struct ArenaSlot {
    idx: usize,
}

impl ArenaSlot {
    fn new<T: HandleAble<dyn Trait> + 'static>(val: T) -> ArenaSlot {
        let boxed: Box<dyn HandleAble<dyn Trait>> = Box::new(val);
        let cell = PersistentCell::from_boxed(boxed);
        let fat: *mut PersistentCell<dyn HandleAble<dyn Trait>> = Box::into_raw(cell);
        // SAFETY: fat-pointer components
        let (data, vtable) = unsafe { std::mem::transmute(fat) };
        let idx = ARENA.lock().unwrap().len();
        ARENA.lock().unwrap().push(Erased { data, vtable });
        ArenaSlot { idx }
    }

    fn fat(&self) -> *mut PersistentCell<dyn HandleAble<dyn Trait>> {
        let e = ARENA.lock().unwrap()[self.idx];
        // SAFETY: reassemble the registered fat pointer
        unsafe { std::mem::transmute(e) }
    }
}

impl Deref for ArenaSlot {
    type Target = PersistentCell<dyn HandleAble<dyn Trait>>;
    fn deref(&self) -> &Self::Target {
        // SAFETY: the registered entry lives forever
        unsafe { &*self.fat() }
    }
}

impl DerefMut for ArenaSlot {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: per the container's own exclusivity discipline
        unsafe { &mut *self.fat() }
    }
}

struct Counter {
    val: i64,
}

impl Trait for InnerHandle<Counter> {
    fn bump(&mut self) {
        self.val += 1;
    }
}


fn main() {
    // ONE registered state, TWO container values (the same index)
    let a = ArenaSlot::new(Counter { val: 0 });
    let b = a; // Copy: same content, distinct cells

    // two concurrent trees over the same content — plain safe usage
    let t1 = std::thread::spawn(move || {
        let mut root = PersistentCell::new(a);
        independent_task_root(&mut root, |mut outer, _ck| {
            for _ in 0..200 {
                outer.bump();
            }
        });
    });
    let t2 = std::thread::spawn(move || {
        let mut root2 = PersistentCell::new(b);
        independent_task_root(&mut root2, |mut outer2, _ck| {
            for _ in 0..200 {
                outer2.bump();
            }
        });
    });
    t1.join().unwrap();
    t2.join().unwrap();
    let final_val = {
        let cell: &PersistentCell<dyn HandleAble<dyn Trait>> = &*a;
        // read through the safe deref chain (the trait is drive-only;
        // read via a fresh tree)
        let mut root3 = PersistentCell::new(a);
        independent_task_root(&mut root3, |mut outer3, _ck| {
            let _ = &*cell;
            outer3.bump();
        });
    };
    let _ = final_val;
    println!("cross-tree drive done (both trees wrote the same state)");
}
