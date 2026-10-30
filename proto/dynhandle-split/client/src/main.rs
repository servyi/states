//! The client subcrate — ZERO unsafe. The chain vocabulary (Env,
//! State, Outcome), the Context with its two entry pointers, the
//! states, the run loop.

use dynstate::{ContextData, DynHandle, Unwrap, install};
use handle::with_pair;

pub struct Env;

pub trait State {
    fn execute(&mut self, env: &Env) -> Result<Outcome, String>;
    fn save(&self) -> String;
}

pub enum Outcome {
    Pending,
    Done(u64),
}

/// The chain's data: its two entry pointers inline (the per-type
/// words, branded 'o — no 'static table).
pub struct Context<'o> {
    execute: for<'x> fn(&'x mut DynHandle<'o, Context<'o>>, &Env) -> Result<Outcome, String>,
    save: for<'x> fn(&'x mut DynHandle<'o, Context<'o>>) -> String,
}

/// The client's secondary impl — generic over the chain's state
/// type, with the client's OWN bound here (the middle crate names no
/// state trait; the equality `Unwrap<'o, Type = S>` does the tying).
impl<'o, S: State + 'static> ContextData<'o, S> for Context<'o> {
    fn new<U: Unwrap<'o, Type = S>>() -> Self {
        let execute: for<'x> fn(
            &'x mut DynHandle<'o, Context<'o>>,
            &Env,
        ) -> Result<Outcome, String> = |dh, env| {
            let typed = U::unwrap_mut(dh);
            let state: &mut S = &mut **typed; // Handle's DerefMut
            S::execute(state, env)
        };
        let save: for<'x> fn(&'x mut DynHandle<'o, Context<'o>>) -> String = |dh| {
            let typed = U::unwrap_mut(dh);
            let state: &S = &*typed; // Handle's Deref
            S::save(state)
        };
        Context { execute, save }
    }
}


/// The 23:38 exercise: leak U::unwrap_mut itself as a bare fn
/// pointer, then try every way of applying it to a foreign handle.
mod leak_unwrap {
    use super::*;
    use dynstate::{ContextData, DynHandle, Unwrap, install};

    /// The extraction itself WORKS — the pointer is nameable:
    ///
    ///   let f: for<'x> fn(&'x mut DynHandle<'o, Context<'o>>)
    ///       -> &'x mut handle::Handle<'o, S> = U::unwrap_mut;
    ///
    /// Its TYPE bakes in BOTH the brand 'o and the state type S.
    /// Everything then fails on that type:
    ///
    /// 1. cross-PAIR application (f over another pair's DynHandle):
    ///    error[E0308] — DynHandle<'o_B, ..> vs DynHandle<'o_A, ..>;
    ///    'o is invariant, the brands never unify. (Verified in the
    ///    attack branch: E0521 even before that — a DynHandle cannot
    ///    escape its with_pair closure, so the attack must nest.)
    /// 2. brand ERASURE (Box<dyn for<'any> Fn(&mut DynHandle<'any, ..>)>):
    ///    does not compile — the coercion would need U to serve every
    ///    'any while it is pinned to the one 'o of its install.
    /// 3. same-brand foreign NODE: no second DynHandle of one brand
    ///    exists today (with_pair mints one handle; construction is
    ///    install-only; fields crate-private). The fanout caveat
    ///    stands: when children share a brand, unwrap_mut must key on
    ///    per-node identity, not the brand alone.
    #[allow(dead_code)]
    fn documented() {}
}

struct Countdown {
    left: u32,
}

impl State for Countdown {
    fn execute(&mut self, _env: &Env) -> Result<Outcome, String> {
        if self.left == 0 {
            return Ok(Outcome::Done(42));
        }
        self.left -= 1;
        Ok(Outcome::Pending)
    }
    fn save(&self) -> String {
        format!("countdown:{}", self.left)
    }
}

const START: u32 = 3;

fn main() {
    let result = with_pair(Countdown { left: START }, |h| {
        let mut dh: DynHandle<'_, Context<'_>> = install::<Countdown, Context<'_>>(h);
        loop {
            let execute = dh.execute; // fn ptr: Copy
            match execute(&mut dh, &Env).expect("transition") {
                Outcome::Pending => continue,
                Outcome::Done(k) => return k,
            }
        }
    });
    println!("result: {result}");
    with_pair(Countdown { left: START }, |h| {
        let mut dh: DynHandle<'_, Context<'_>> = install::<Countdown, Context<'_>>(h);
        let save = dh.save;
        println!("save: {}", save(&mut dh));
        0u64
    });
}
