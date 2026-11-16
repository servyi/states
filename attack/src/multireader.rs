//! PROBE: many concurrent readers during ONE guard — the guard's
//! &PersistentCell<K> is Sync (K: Send), so the save can fan out its
//! own reading. Plus: the double-request gate self-deadlock (the
//! finding-4 family, gate variant).
use hslib::{HandleAble, InnerHandle, PersistentCell, independent_task_root};

trait Trait: Send {
    fn step(&mut self);
}

struct Machine {
    val: std::sync::atomic::AtomicI64,
}

impl Trait for InnerHandle<Machine> {
    fn step(&mut self) {
        self.val.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.safepoint();
    }
}

fn main() {
    let mut root: PersistentCell<Box<PersistentCell<dyn HandleAble<dyn Trait>>>> =
        PersistentCell::new(PersistentCell::from_boxed(Box::new(Machine {
            val: std::sync::atomic::AtomicI64::new(0),
        }) as Box<dyn HandleAble<dyn Trait>>));
    independent_task_root(&mut root, |mut outer, ck| {
        let ck2 = ck.clone();
        std::thread::scope(|s| {
            let reader = s.spawn(move || {
                let guard = ck2.request();
                // the guard's &PersistentCell is Sync — fan out the
                // reading across four threads during the one stop
                let cell: &PersistentCell<Box<PersistentCell<dyn HandleAble<dyn Trait>>>> = &*guard;
                std::thread::scope(|inner| {
                    for _ in 0..4 {
                        inner.spawn(move || {
                            let d: &dyn HandleAble<dyn Trait> = &****cell;
                            let m: &Machine = unsafe {
                                &*(std::ptr::from_ref(d) as *const Machine)
                            };
                            let _ = m.val.load(std::sync::atomic::Ordering::SeqCst);
                        });
                    }
                });
            });
            let mut steps = 0;
            while !reader.is_finished() {
                outer.step();
                steps += 1;
                if steps > 10_000 {
                    break;
                }
            }
            reader.join().unwrap();
        });
    });
    println!("multireader probe done");
}
