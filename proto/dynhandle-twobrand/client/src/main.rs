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
pub struct Context<'o, 'c> {
    execute: for<'x> fn(
        &'x mut DynHandle<'o, 'c, Context<'o, 'c>>,
        &Env,
    ) -> Result<Outcome, String>,
    save: for<'x> fn(&'x mut DynHandle<'o, 'c, Context<'o, 'c>>) -> String,
}

/// The client's secondary impl — generic over the chain's state
/// type, with the client's OWN bound here (the middle crate names no
/// state trait; the equality `Unwrap<'o, Type = S>` does the tying).
impl<'o, 'c, S: State + 'static> ContextData<'o, 'c, S> for Context<'o, 'c> {
    fn new<U: Unwrap<'o, 'c, Type = S>>() -> Self {
        let execute: for<'x> fn(
            &'x mut DynHandle<'o, 'c, Context<'o, 'c>>,
            &Env,
        ) -> Result<Outcome, String> = |dh, env| {
            let typed = U::unwrap_mut(dh);
            let state: &mut S = &mut **typed; // Handle's DerefMut
            S::execute(state, env)
        };
        let save: for<'x> fn(&'x mut DynHandle<'o, 'c, Context<'o, 'c>>) -> String = |dh| {
            let typed = U::unwrap_mut(dh);
            let state: &S = &*typed; // Handle's Deref
            S::save(state)
        };
        Context { execute, save }
    }
}


/// The cross-child attack under the two-brand identity, attempted
/// for real: a fanout action that tries to STASH the handle from one
/// child's call and use it during another's. The action must be
/// `for<'c> Fn(..)`; storing a `'c`-branded value across calls would
/// require the closure to hold a borrow valid for EVERY 'c at once —
mod cross_child {
    use super::*;
    use handle::{Handle, fanout_two_children};

    pub fn attempt() {
        let mut a = Countdown { left: 1 };
        let mut b = Countdown { left: 2 };
        // THE ATTACK (all four probe rounds verified by compiler):
        // stashing one child's handle and applying it during another
        // child's call. The wall, layer by layer:
        //  - as an `Fn` action with a stash: E0282/E0596 (Fn cannot
        //    mutate the stash — fanout actions are `Fn` by contract);
        //  - as a fully annotated `FnMut` probe:
        //      *stashed = Some(h);
        //    error[E0521]: borrowed data escapes outside of closure —
        //    `h` (brand 'c, closure-local) can never be stored into
        //    anything outliving the call. The child's brand CANNOT
        //    leave its action call — by construction, not discipline.
        let _ = fanout_demo(&mut a, &mut b);
    }

    /// The compiling shape: each action call sees only its own child.
    fn fanout_demo(a: &mut Countdown, b: &mut Countdown) {
        handle::fanout_two_children_quiet(a, b, |_h| {
            // per-call: h is Handle<'_, 'c, Countdown> with 'c
            // instantiated by the implementation for THIS child.
        });
    }
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
        let mut dh: DynHandle<'_, '_, Context> =
            install::<Countdown, Context>(h);
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
        let mut dh: DynHandle<'_, '_, Context> =
            install::<Countdown, Context>(h);
        let save = dh.save;
        println!("save: {}", save(&mut dh));
        0u64
    });
}
