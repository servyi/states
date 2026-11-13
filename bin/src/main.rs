//! The binary half: the vocabulary (`Trait`) and the structs —
//! everything that directly references them — plus the demo.

use hslib::{PersistentCell, HandleAble, InnerHandle, independent_task_root};

trait Trait: Send {
    fn bark(&mut self) -> i32;
    fn dup_probe(&mut self);
}

/// The variant-B slot: a box OF A CELL — the cell covers the state
/// (the IM line), the box is the container K.
type Slot = Box<PersistentCell<dyn HandleAble<dyn Trait>>>;

struct MyStruct {
    children: Vec<PersistentCell<Slot>>,
}
struct YourStruct {
    val: i32,
}

impl Trait for InnerHandle<MyStruct> {
    fn bark(&mut self) -> i32 {
        println!("MyStruct");
        let sums: Vec<i32> = self.fanout(
            // &mut K directly — K = Slot; the library does the
            // erased coercion internally. iter_mut().collect() just
            // works now (no unsized coercion needed at the call site).
            &|m: &MyStruct| m.children.iter().collect(),
            // the child outers arrive branded with this fanout's 'o —
            // unnameable here, so they cannot be stored anywhere that
            // outlives the call; each child's bark result rides the
            // Vec<R> back
            |mut o| o.bark(),
        );
        // the SAFETY POINT: park here while a checkpoint runs
        self.safepoint();
        sums.iter().sum()
    }

    fn dup_probe(&mut self) {
        // the duplicate-split proof: the same child twice -> the
        // library drops the duplicate -> ONE result, one mint
        let results: Vec<i32> = self.fanout(
            &|m: &MyStruct| vec![&m.children[0], &m.children[0]],
            |mut o| o.bark(),
        );
        // duplicates now SERIALIZE on the global claims registry:
        // both subtasks run, one after the other
        assert_eq!(results.len(), 2, "duplicates serialize, both run");
        println!("duplicate split: {} results (serialized)", results.len());
    }
}

impl Trait for InnerHandle<YourStruct> {
    fn bark(&mut self) -> i32 {
        println!("{}", self.val); // through DerefMut onto YourStruct
        self.val
    }
    fn dup_probe(&mut self) {}
}

/// Wrap a LIVE driver-owned slot in the OuterHandle (the library's
/// over_slot does the mint).
fn main() {
    let mut m: PersistentCell<Slot> = PersistentCell::new(PersistentCell::from_boxed(
        Box::new(MyStruct {
            children: vec![
                PersistentCell::new(PersistentCell::from_boxed(Box::new(YourStruct { val: 5 })
                    as Box<dyn HandleAble<dyn Trait>>)),
                PersistentCell::new(PersistentCell::from_boxed(Box::new(YourStruct { val: 7 })
                    as Box<dyn HandleAble<dyn Trait>>)),
            ],
        }) as Box<dyn HandleAble<dyn Trait>>,
    ));
    // the ROOT task: the slot is inaccessible inside (borrowed by the
    // call); the outer handle and its Checkpointer travel together
    independent_task_root(&mut m, |mut outer, checkpointer| {
        // a CHECKER: mid-run, from a SCOPED thread — the 'o brand
        // keeps the checkpointer inside the task, so a detached
        // thread cannot carry it — stop the world: the guard holds
        // the root slot while every mutator parks
        let ck = checkpointer.clone();
        std::thread::scope(|scope| {
            let checker = scope.spawn(move || {
                let guard = ck.request();
                let _root: &Slot = &guard;
                println!("checkpoint: world stopped, root held");
                drop(guard);
                println!("checkpoint: world resumed");
            });
            let first = outer.bark();
            println!("sum: {}", first);
            // THE INSTALL: the OuterHandle walks to the slot through the
            // current inner handle, swaps it, re-mints, keeps the fresh one
            let successor: Slot =
                PersistentCell::from_boxed(Box::new(YourStruct { val: 42 }));
            outer.replace(successor);
            let second = outer.bark();
            println!("sum: {}", second);
            checker.join().expect("checker panicked");
        });
    });

    let mut y: PersistentCell<Slot> = PersistentCell::new(PersistentCell::from_boxed(
        Box::new(MyStruct {
            children: vec![PersistentCell::new(PersistentCell::from_boxed(Box::new(YourStruct { val: 5 })
                as Box<dyn HandleAble<dyn Trait>>))],
        }) as Box<dyn HandleAble<dyn Trait>>,
    ));
    independent_task_root(&mut y, |mut outer2, _checkpointer| {
        outer2.dup_probe();
    });
}

