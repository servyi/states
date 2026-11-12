//! A custom container NOT based on Box: an arena. States live in
//! raw std::alloc allocations registered in a global registry; the
//! container is an index. The container's unsafe is the standard
//! data-structure-author kind (alloc + fat-pointer plumbing); the
//! states and the driver are plain client code.

use std::alloc::{Layout, alloc};
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use hslib::{Checkpointer, ChildNode, HandleAble, InnerHandle, OuterHandle, independent_task_root};

trait Trait: Send {
    fn step(&mut self);
    fn observe(&mut self) -> String;
}
trait SaveAble: HandleAble<dyn Trait> {
    fn save(&self) -> String;
}

/// the erased fat pointer, stored as its two components (read back
/// through the transmutes in / — the dead_code lint can't
/// see through those)
#[allow(dead_code)]
#[derive(Clone, Copy)]
struct Erased {
    data: *mut (),
    vtable: *const (),
}
// SAFETY: the pointees are states (T: SaveAble: HandleAble: Send);
// the raw crosses threads exactly as a Box<dyn ...> payload would.
unsafe impl Send for Erased {}

static ARENA: Mutex<Vec<Erased>> = Mutex::new(Vec::new());

/// proof the container's DerefMut/Deref are actually exercised
static DEREF_MUT_CALLS: AtomicUsize = AtomicUsize::new(0);
static AS_PTR_CALLS: AtomicUsize = AtomicUsize::new(0);
static DEREF_CALLS: AtomicUsize = AtomicUsize::new(0);

/// the container: an index into the arena. Sized, Copy, Send.
#[derive(Clone, Copy)]
struct ArenaSlot {
    idx: usize,
}

impl ArenaSlot {
    /// allocate the state's storage with std::alloc (no Box), write
    /// it, register the fat pointer. Entries are never removed — the
    /// "do not get rid of the returned pointer" obligation.
    fn new<T: SaveAble + 'static>(val: T) -> ArenaSlot {
        let layout = Layout::for_value(&val);
        // SAFETY: the layout of a real T is nonzero
        let mem: *mut u8 = unsafe { alloc(layout) };
        assert!(!mem.is_null(), "arena allocation failed");
        // SAFETY: fresh valid allocation of T's layout
        unsafe { std::ptr::write(mem as *mut T, val) };
        // SAFETY: T: SaveAble, so &mut T erases to the fat pointer
        let fat: *mut dyn SaveAble = unsafe { &mut *(mem as *mut T) };
        // SAFETY: pointer-to-pointer components of a fat pointer
        let (data, vtable): (*mut (), *const ()) = unsafe { std::mem::transmute(fat) };
        let idx = ARENA.lock().unwrap().len();
        ARENA.lock().unwrap().push(Erased { data, vtable });
        ArenaSlot { idx }
    }

    fn fat(&self) -> *mut dyn SaveAble {
        let e = ARENA.lock().unwrap()[self.idx];
        // SAFETY: reassemble the registered fat pointer
        unsafe { std::mem::transmute(e) }
    }
}

impl Deref for ArenaSlot {
    type Target = dyn SaveAble;
    fn deref(&self) -> &Self::Target {
        DEREF_CALLS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: the registered entry is live for the program's life
        unsafe { &*self.fat() }
    }
}

impl DerefMut for ArenaSlot {
    fn deref_mut(&mut self) -> &mut Self::Target {
        DEREF_MUT_CALLS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: as Deref; the exclusivity is the slot's own
        // (the driver owns the only handle chain over this index)
        unsafe { &mut *self.fat() }
    }
}

// SAFETY: the registered fat pointer IS the stored line the Derefs
// spring from (its data component carries the allocation's
// provenance — no freshly minted reference); and entries are never
// removed, so the backing store outlives every handle.
unsafe impl hslib::AsPtr for ArenaSlot {
    fn as_ptr(&mut self) -> *mut Self::Target {
        AS_PTR_CALLS.fetch_add(1, Ordering::Relaxed);
        self.fat()
    }
}

struct Machine {
    children: Vec<ChildNode<ArenaSlot>>,
    steps: i32,
}
struct Counter {
    val: i32,
}

impl Trait for InnerHandle<Machine, dyn Trait> {
    fn step(&mut self) {
        self.steps += 1; // the historical TB failure site
        // the children WRITE too — every arena line takes the
        // write-save-write ride, not just the root's
        self.fanout(
            &|m: &Machine| m.children.iter().collect(),
            |mut o: OuterHandle<'_, ArenaSlot, dyn Trait>| o.step(),
        );
        self.safepoint();
    }
    fn observe(&mut self) -> String {
        self.safepoint(); // the parking window
        let kids: Vec<String> = self
            .fanout(
                &|m: &Machine| m.children.iter().collect(),
                |mut o| o.observe(),
            )
            .into_iter()
            .collect();
        format!("M({}, [{}])", self.steps, kids.join(","))
    }
}

impl Trait for InnerHandle<Counter, dyn Trait> {
    fn step(&mut self) {
        self.val += 1;
    }
    fn observe(&mut self) -> String {
        format!("C({})", self.val)
    }
}

impl SaveAble for Machine {
    fn save(&self) -> String {
        let kids: Vec<String> = self.children.iter().map(|c| c.save()).collect();
        format!("M({}, [{}])", self.steps, kids.join(","))
    }
}
impl SaveAble for Counter {
    fn save(&self) -> String {
        format!("C({})", self.val)
    }
}

/// a save while parked in the safepoint windows
fn park_save(
    outer: &mut OuterHandle<'_, ArenaSlot, dyn Trait>,
    ck: &Checkpointer<ArenaSlot>,
) -> String {
    std::thread::scope(|s| {
        let t = s.spawn({
            let ck = ck.clone();
            move || {
                let guard = ck.request();
                guard.save()
            }
        });
        while !t.is_finished() {
            outer.observe();
        }
        t.join().unwrap()
    })
}

fn main() {
    let m = ChildNode::new(ArenaSlot::new(Machine {
        children: vec![
            ChildNode::new(ArenaSlot::new(Counter { val: 1 })),
            ChildNode::new(ArenaSlot::new(Counter { val: 1 })),
        ],
        steps: 0,
    }));
    independent_task_root(&mut { m }, |mut outer, ck| {
        let mut snaps = Vec::new();
        for _ in 0..3 {
            snaps.push(park_save(&mut outer, &ck)); // the save
            outer.step(); // every node writes (root + children)
        }
        snaps.push(park_save(&mut outer, &ck));
        for s in &snaps {
            println!("snap: {}", s);
        }
        // progress: every round's snapshot is distinct (writes visible)
        let mut distinct = snaps.clone();
        distinct.sort();
        distinct.dedup();
        assert_eq!(distinct.len(), snaps.len(), "snapshots must show progress");
        // the container's derefs were actually exercised
        let ap = AS_PTR_CALLS.load(Ordering::Relaxed);
        let d = DEREF_CALLS.load(Ordering::Relaxed);
        assert!(ap > 0, "as_ptr was never called (the mints bypass it?)");
        assert!(d > 0, "Deref was never called");
        println!("AsPtr calls: {}, Deref calls: {}", ap, d);
        println!("final: {}", outer.observe());
    });
}
