//! The SaveAble shape: the client's own subtrait vocabulary on the
//! slot, read through the checkpoint guard.
//!
//! Directly `K = Box<dyn SaveAble>` fails the library's
//! `DerefMut<Target = dyn HandleAble<V>>` (Target equality — no
//! generic dyn-upcast on stable). The working shape: a thin wrapper
//! whose SAFE Deref/DerefMut impls upcast at a concrete coercion
//! site, plus `replace(val: K)` (plain K-cell assignment — the
//! identity pass-through that made ReplaceAble deletable).

use hslib::{ Checkpointer,OuterHandle, PersistentCell, HandleAble, InnerHandle, independent_task_root};

/// the drive vocabulary (as in the main demo)
trait Trait: Send {
    fn bark(&mut self) -> i32;
}

/// the CLIENT's save vocabulary: a subtrait of HandleAble
trait SaveAble: HandleAble<dyn Trait> {
    fn save(&self) -> String;
}

/// the variant-B slot: a box of a cell over the erased SaveAble
type SaveSlot = Box<PersistentCell<dyn SaveAble>>;

struct MyStruct {
    children: Vec<PersistentCell<SaveSlot>>,
}
struct YourStruct {
    val: i32,
}

impl Trait for InnerHandle<MyStruct> {
    fn bark(&mut self) -> i32 {
        let sums: Vec<i32> = self.fanout(
            &|m: &MyStruct| m.children.iter().collect(),
            |mut o| o.bark(),
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

impl SaveAble for MyStruct {
    fn save(&self) -> String {
        let kids: Vec<String> = self.children.iter().map(|c| { (***c).save() }).collect();
        format!("My({})", kids.join(","))
    }
}
impl SaveAble for YourStruct {
    fn save(&self) -> String {
        format!("{}", self.val)
    }
}

fn main() {
    let mut m = PersistentCell::new(PersistentCell::from_boxed(
        Box::new(MyStruct {
            children: vec![
                PersistentCell::new(PersistentCell::from_boxed(
                    Box::new(YourStruct { val: 5 }) as Box<dyn SaveAble>)),
                PersistentCell::new(PersistentCell::from_boxed(
                    Box::new(YourStruct { val: 7 }) as Box<dyn SaveAble>)),
            ],
        }) as Box<dyn SaveAble>,
    ));
    independent_task_root(&mut m, |mut outer: OuterHandle<dyn SaveAble, SaveSlot, dyn Trait>, checkpointer: Checkpointer<SaveSlot>| {
        // THE SNAPSHOT READS CLIENT VOCABULARY through the guard's &K
        let ck = checkpointer.clone();
        std::thread::scope(|scope| {
            let checker = scope.spawn(move || {
                let guard = ck.request();
                let snapshot: String = { (***guard).save() };
                println!("checkpoint snapshot: {}", snapshot);
            });
            let first = outer.bark();
            println!("sum: {}", first);
            // the install, K directly:
            let successor: SaveSlot = PersistentCell::from_boxed(Box::new(YourStruct { val: 42 }) as Box<dyn SaveAble>);
            outer.replace(successor);
            let second = outer.bark();
            println!("sum: {}", second);
            checker.join().expect("checker");
        });
    });
}
