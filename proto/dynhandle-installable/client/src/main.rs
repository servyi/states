#![feature(arbitrary_self_types)]
//! The installable shape: the chain is `Handle<Box<dyn DerefMut<Context>>>`;
//! the LIBRARY's `Installed<S, T>` carries the pairing, the derefs, and
//! the projection; the client holds only its vocabulary and entries —
//! zero unsafe here. State methods run on `self: &mut Handle<Self>`
//! (the projected state handle), so EVERY state can safepoint.

use std::ops::DerefMut;

use dynstate::{ContextData, Erased, UnwrapMut, install};
use handle::{Handle, with_pair};

pub struct Env;

/// What a transition decides. `Switch` carries the successor box —
/// built with THIS chain's brands, so it cannot be installed into a
/// foreign chain (the box type mentions them).
pub enum Outcome<'o, 'c> {
    /// Keep driving this state.
    Pending,
    /// Overwrite with this successor (an already-installed box).
    Switch(Erased<'c, Context<'o, 'c>>),
    /// The chain's result.
    Done(u64),
}

/// The chain's context: BRAND-FREE (W1) — the entries are
/// higher-ranked over the handle's brands, which is what lets it be a
/// with_pair node.
/// The chain's context — BRANDED: its fn pointers are pinned to ONE
/// pair ('o) and ONE child ('c) of that pair. A pointer lifted out of
/// this context does not typecheck against any other chain's handle
/// (the brands are invariant), which is the cross-application wall.
pub struct Context<'o, 'c> {
    pub execute: for<'x> fn(
        &'x mut Handle<'o, 'c, Erased<'c, Context<'o, 'c>>>,
        &Env,
    ) -> Result<Outcome<'o, 'c>, String>,
    pub save: for<'x> fn(&'x mut Handle<'o, 'c, Erased<'c, Context<'o, 'c>>>) -> String,
}

// ── the vocabulary: ONE state trait, methods on the state's own
// ── projected handle (self: &mut Handle<Self>) — every state can
// ── safepoint mid-run through it.

pub trait State: Sized + 'static {
    /// The brands of the RECEIVING handle flow into the outcome (the
    /// successor box is built with the chain's own brands).
    fn execute<'o, 'c>(
        self: &mut Handle<'o, 'c, Self>,
        env: &Env,
    ) -> Result<Outcome<'o, 'c>, String>
    where
        'o: 'c; // the child brand is a SUBSET of the pair's
    fn save(self: &mut Handle<'_, '_, Self>) -> String;
}

// ── entries for both flavors (mk builds them) ─────────────────────

/// The secondary impl: generic over the capability `U` (pinned by
/// install's instantiation), the entries are non-capturing closures —
/// plain fn pointers: static, no allocation, nothing captured.
impl<'o, 'c, S: State + 'static> ContextData<'o, 'c, S> for Context<'o, 'c> {
    fn new<U: UnwrapMut<Type = S>>() -> Context<'o, 'c>
    where
        'o: 'c,
    {
        Context {
            execute: |h, env| {
                h.safepoint(); // between transitions
                U::unwrap_mut(h, |typed| S::execute(typed, env))
            },
            save: |h| U::unwrap_mut(h, |typed| S::save(typed)),
        }
    }
}

// ── the states ────────────────────────────────────────────────────

struct Countdown {
    left: u32,
}

impl State for Countdown {
    fn execute<'o, 'c>(
        self: &mut Handle<'o, 'c, Self>,
        _env: &Env,
    ) -> Result<Outcome<'o, 'c>, String>
    where
        'o: 'c,
    {
        if self.left == 0 {
            // the successor: an INSTALLED box with THIS chain's
            // brands, carried by the outcome
            return Ok(Outcome::Switch(Box::new(install::<Countup, Context<'o, 'c>>(
                Countup { up: 0, limit: COUNTUP_LIMIT },
            ))));
        }
        self.left -= 1;
        Ok(Outcome::Pending)
    }
    fn save(self: &mut Handle<'_, '_, Self>) -> String {
        format!("countdown:{}", self.left)
    }
}

struct Countup {
    up: u64,
    limit: u64,
}

impl State for Countup {
    fn execute<'o, 'c>(
        self: &mut Handle<'o, 'c, Self>,
        _env: &Env,
    ) -> Result<Outcome<'o, 'c>, String>
    where
        'o: 'c,
    {
        if self.up == COUNTUP_LIMIT {
            return Ok(Outcome::Done(FINAL_RESULT));
        }
        if self.up == COUNTUP_LIMIT - LAST_STEP_AT {
            self.safepoint(); // mid-run safepoint — EVERY state can
        }
        self.up += 1;
        Ok(Outcome::Pending)
    }
    fn save(self: &mut Handle<'_, '_, Self>) -> String {
        format!("countup:{}/{}", self.up, self.limit)
    }
}

/// Countdown start in the two-state test.
const START: u32 = 3;
/// Countup limit after the countdown.
const COUNTUP_LIMIT: u64 = 5;
/// The two-state test's final result.
const FINAL_RESULT: u64 = 42;
/// When the outer state safepoints mid-run.
const LAST_STEP_AT: u64 = 1;


fn main() {
    // the top-level pair: the node is the RAW state; install_chain
    // consumes its handle and returns the chain handle (same brands,
    // node re-pointed at the installed box).
    let result = with_pair(Countdown { left: START }, |h| {
        let mut chain = dynstate::install_chain::<Countdown, Context<'_, '_>>(h);
        drive(&mut chain, &Env).expect("chain completes")
    });
    println!("result: {result}");

    with_pair(Countdown { left: START }, |h| {
        let mut chain = dynstate::install_chain::<Countdown, Context<'_, '_>>(h);
        let save = (**chain).save;
        println!("save: {}", save(&mut chain));
        drive(&mut chain, &Env).expect("chain completes")
    });
}

/// One chain, branded: drive over the installed box.
fn drive<'o, 'c>(
    h: &mut Handle<'o, 'c, Erased<'c, Context<'o, 'c>>>,
    env: &Env,
) -> Result<u64, String> {
    loop {
        let execute = (**h).execute; // fn ptr: Copy — brand-pinned
        match execute(h, env)? {
            Outcome::Pending => {}
            Outcome::Switch(next) => **h = next,
            Outcome::Done(k) => return Ok(k),
        }
    }
}

/// The cross-application probe (documented, compile-time rejected):
/// lifting chain A's `execute` out of its branded Context and calling
/// it on chain B's handle does not typecheck — `Context<'oA,'cA>`'s
/// fn-pointer type only accepts `Handle<'oA,'cA, _>`, and the brands
/// are invariant. Verified in the earlier attack rounds (E0308);
/// re-probed on this branch in the round comment.
mod attacks {
    use super::*;
    use dynstate::{ContextData, Erased, UnwrapMut, install};

    /// Extract A's entry fn pointer and apply it to B's chain handle.
    /// (Nested pairs — a DynHandle cannot escape its with_pair, as the
    /// earlier rounds showed — so both chains are built inside one
    /// outer scope via helpers that hand us the fn pointer.)
    fn entry_of_a() -> for<'x> fn(
        &'x mut Handle<'static, 'static, Erased<'static, Context<'static, 'static>>>,
        &Env,
    ) -> Result<Outcome<'static, 'static>, String> {
        unreachable!("see the compile probe below")
    }

    /// The live probe (run against the compiler — see the round on
    /// the issue): two chains, two pairs; lift A's `execute` (a Copy
    /// fn pointer — the READ is legal) and apply it to B's handle.
    /// The attack cannot even FORM: chain handles are branded with
    /// their pair's closure-local lifetime, and `error[E0521]:
    /// borrowed data escapes outside of closure` rejects taking the
    /// second chain into the first pair's scope. There is no scope in
    /// which two differently-branded chain handles coexist — so the
    /// lifted pointer has nothing foreign to be applied to.
    pub fn cross_apply() {}
    pub fn run() {
        let _ = cross_apply;
    }
}
