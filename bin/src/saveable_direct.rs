//! The DIRECT SaveAble shape: K = Box<dyn SaveAble> — literally,
//! with ZERO client unsafe.
//!
//! The K bound is Target-generic: `K::Target: Unsize<dyn
//! HandleAble<V>>` — trait-object upcasting expressed generically,
//! so `dyn SaveAble`'s target qualifies. The library's mints upcast
//! `as_ptr`'s raw to `dyn HandleAble<V>` through that judgement; the
//! builtin `Box` provides `AsPtr`.
//!
//! The guard then speaks the vocabulary directly: `guard.save()`.

use hslib::{ChildNode, HandleAble, InnerHandle, independent_task_root};

trait Trait: Send {
    fn bark(&mut self) -> i32;
}

/// the client's save vocabulary: a subtrait of HandleAble
trait SaveAble: HandleAble<dyn Trait> {
    fn save(&self) -> String;
}

/// K, LITERALLY the subtrait box
type SaveSlot = Box<dyn SaveAble>;

struct MyStruct {
    children: Vec<ChildNode<SaveSlot>>,
}
struct YourStruct {
    val: i32,
}

impl Trait for InnerHandle<MyStruct, dyn Trait> {
    fn bark(&mut self) -> i32 {
        let sums: Vec<i32> = self.fanout(
            &|m: &MyStruct| m.children.iter().collect(),
            |mut o| {
                // THE CHILD-CALLBACK INSTALL: the callback owns its
                // outer, constructs the successor K at the call site,
                // and swaps its own subtree from its own thread
                let successor: SaveSlot = Box::new(YourStruct { val: 99 });
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
    let mut m: ChildNode<SaveSlot> = ChildNode::new(Box::new(MyStruct {
        children: vec![
            ChildNode::new(Box::new(YourStruct { val: 5 })),
            ChildNode::new(Box::new(YourStruct { val: 7 })),
        ],
    }));
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
            println!("sum: {} (99+99: the children replaced themselves)", first);
            let successor: SaveSlot = Box::new(YourStruct { val: 42 });
            outer.replace(successor);
            let second = outer.bark();
            println!("sum: {}", second);
            checker.join().expect("checker");
        });
    });
}
