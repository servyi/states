//! The binary half of the runnable sketch: the vocabulary (`Trait`)
//! and the two structs — everything that directly references them —
//! plus the demo main that reassigns the outer `Box<dyn Trait>`.

use hslib::{HandleAble, InnerHandle};

trait Trait {
    fn bark(&mut self);
}

struct MyStruct {
    child: Box<dyn HandleAble<dyn Trait>>,
}
struct YourStruct {
    val: i32,
}

impl Trait for InnerHandle<MyStruct> {
    fn bark(&mut self) {
        println!("MyStruct")
    }
}

impl Trait for InnerHandle<YourStruct> {
    fn bark(&mut self) {
        println!("{}", self.val) // through DerefMut onto YourStruct
    }
}

fn main() {
    let mut m = Box::new(MyStruct {
        child: Box::new(YourStruct { val: 5 }),
    });
    let mut h: Box<dyn Trait> = (*m).box_innerhandle();
    h.bark();

    let mut y = Box::new(YourStruct { val: 5 });
    h = (*y).box_innerhandle();
    h.bark();
}
