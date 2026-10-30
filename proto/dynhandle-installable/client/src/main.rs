//! The client subcrate — ZERO unsafe. The "installable" exploration:
//! the chain's erased state is a `Box<dyn InnerState>` (an
//! object-safe trait, inner execute: no handle, so no mid-run
//! safepoint); the child holds `Handle<Box<dyn InnerState>>`, whose
//! deref makes the box OVERWRITABLE — successors install as plain box
//! writes, no crate successor machinery.

use dynstate::{ContextData, DynHandle, Unwrap, install};
use handle::with_pair;

pub struct Env;

/// The object-safe inner vocabulary: execute over `&mut self` only —
/// an inner state CANNOT call `h.safepoint()` (it never sees a
/// handle). Safepoints happen between transitions, at the entry.
pub trait InnerState {
    fn execute(&mut self, env: &Env) -> Result<Outcome, String>;
    fn save(&self) -> String;
}

/// The erased state: the box whose content is overwritten on switch.
pub type Erased = Box<dyn InnerState>;

/// What a transition decides. `Switch` carries the successor as a
/// fresh boxed inner state — the OVERWRITE material.
pub enum Outcome {
    /// Keep driving this state.
    Pending,
    /// Overwrite the box with this successor.
    Switch(Erased),
    /// The chain's result.
    Done(u64),
}

/// The chain's data: its two entry pointers inline.
pub struct Context<'o, 'c> {
    execute: for<'x> fn(
        &'x mut DynHandle<'o, 'c, Context<'o, 'c>>,
        &Env,
    ) -> Result<Outcome, String>,
    save: for<'x> fn(&'x mut DynHandle<'o, 'c, Context<'o, 'c>>) -> String,
}

/// ONE Context for every inner state: the pairing is with the BOX
/// (`Erased`), not with each concrete state — the two-state wall of
/// the direct-installs dissolves.
impl<'o, 'c> ContextData<'o, 'c, Erased> for Context<'o, 'c> {
    fn new<U: Unwrap<'o, 'c, Type = Erased>>() -> Self {
        let execute: for<'x> fn(
            &'x mut DynHandle<'o, 'c, Context<'o, 'c>>,
            &Env,
        ) -> Result<Outcome, String> = |dh, env| {
            let typed = U::unwrap_mut(dh);
            typed.safepoint(); // between transitions — the entry CAN
            let boxed: &mut Erased = &mut **typed; // Handle -> Box
            boxed.execute(env)
        };
        let save: for<'x> fn(&'x mut DynHandle<'o, 'c, Context<'o, 'c>>) -> String = |dh| {
            let typed = U::unwrap_mut(dh);
            let boxed: &Erased = &**typed;
            boxed.save()
        };
        Context { execute, save }
    }
}

// ── the states: inner, composable, overwrite-installed ────────────

struct Countdown {
    left: u32,
}

impl InnerState for Countdown {
    fn execute(&mut self, _env: &Env) -> Result<Outcome, String> {
        if self.left == 0 {
            // the successor: boxed, carried by the outcome; the driver
            // overwrites the box — no re-pairing, no crate machinery
            return Ok(Outcome::Switch(Box::new(Countup { up: 0, limit: COUNTUP_LIMIT })));
        }
        self.left -= 1;
        Ok(Outcome::Pending)
    }
    fn save(&self) -> String {
        format!("countdown:{}", self.left)
    }
}

struct Countup {
    up: u64,
    limit: u64,
}

impl InnerState for Countup {
    fn execute(&mut self, _env: &Env) -> Result<Outcome, String> {
        if self.up == self.limit {
            return Ok(Outcome::Done(FINAL_RESULT));
        }
        self.up += 1;
        Ok(Outcome::Pending)
    }
    fn save(&self) -> String {
        format!("countup:{}/{}", self.up, self.limit)
    }
}

/// Countdown start in the two-state test.
const START: u32 = 3;
/// Countup limit after the countdown.
const COUNTUP_LIMIT: u64 = 5;
/// The two-state test's final result.
const FINAL_RESULT: u64 = 42;

/// The driver: the entry dispatches; on Switch, the box is
/// OVERWRITTEN through the same unwrap — the installable successor.
fn drive<'o, 'c>(dh: &mut DynHandle<'o, 'c, Context<'o, 'c>>, env: &Env) -> Result<u64, String> {
    loop {
        let execute = dh.execute; // fn ptr: Copy
        match execute(dh, env)? {
            Outcome::Pending => {}
            Outcome::Switch(next) => {
                // the overwrite: Handle<Box<dyn InnerState>> derefs to
                // the box; *boxed = next installs the successor.
                // the entries live in the CONTEXT (the data), which the
                // overwrite does not touch — only the box swaps.
                let typed =
                    <dynstate::Unwrapper<'o, 'c, Erased> as Unwrap<'o, 'c>>::unwrap_mut(dh);
                **typed = next;
            }
            Outcome::Done(k) => return Ok(k),
        }
    }
}

fn main() {
    let result = with_pair(
        Box::new(Countdown { left: START }) as Erased,
        |h| {
            let mut dh: DynHandle<'_, '_, Context> = install::<Erased, Context>(h);
            drive(&mut dh, &Env).expect("chain completes")
        },
    );
    println!("result: {result}");

    // save through the erased box, mid-chain
    with_pair(
        Box::new(Countdown { left: START }) as Erased,
        |h| {
            let mut dh: DynHandle<'_, '_, Context> = install::<Erased, Context>(h);
            let save = dh.save;
            println!("save: {}", save(&mut dh));
            drive(&mut dh, &Env).expect("chain completes")
        },
    );
}

// ── the OUTER flavor probes (walls documented by compiler) ────────
//
// Jonas's note: outer states are "State-like structs with outer
// execute(Handle<??? Self>)" that CAN call h.safepoint. The ??? is
// the question. Probes:
mod outer_probes {
    use super::*;
    use dynstate::{ContextData, DynHandle, Unwrap};
    use handle::Handle;

    // WALL 1 — one Context cannot serve both flavors. The box client
    // needs `impl ContextData<'o,'c, Erased> for Context`; the direct
    // flavor needs `impl<S: ..> ContextData<'o,'c, S> for Context`.
    // `Erased` is `'static`, so the generic impl already covers it:
    //
    // impl<'o, 'c, S: 'static> ContextData<'o, 'c, S> for Context<'o, 'c> { .. }
    // impl<'o, 'c> ContextData<'o, 'c, Erased> for Context<'o, 'c> { .. }
    //
    // error[E0119]: conflicting implementations — the second is covered
    // by the first. No negative bounds exist to exclude `Erased`.

    // WALL 2 — two Context types split the chain. Give the direct
    // flavor its own data type and both compile independently:
    pub struct DirectContext<'o, 'c> {
        pub execute: for<'x> fn(
            &'x mut DynHandle<'o, 'c, DirectContext<'o, 'c>>,
            &Env,
        ) -> Result<u64, String>,
    }

    impl<'o, 'c, S: 'static> ContextData<'o, 'c, S> for DirectContext<'o, 'c> {
        fn new<U: Unwrap<'o, 'c, Type = S>>() -> Self {
            DirectContext {
                execute: |dh, env| {
                    // THE OUTER EXECUTE: over the state's own type —
                    // this flavor CAN safepoint mid-run
                    let typed = U::unwrap_mut(dh);
                    typed.safepoint();
                    let state: &mut S = probe_ctx(typed);
                    let _ = (state, env);
                    unreachable!("probe shape only")
                },
            }
        }
    }

    // The direct flavor needs its state typed; for the probe, the
    // vocabulary would live here (the twobrand client's State trait).
    fn probe_ctx<'x, 'o, 'c, S>(_typed: &'x mut Handle<'o, 'c, S>) -> &'x mut S {
        unreachable!("probe shape only")
    }

    // WALL 3 — the flavors cannot interleave in one chain: a chain is
    // `DynHandle<'o, 'c, Ctx>` for ONE Ctx. A box-flavor Switch
    // carries `Erased`; a direct-flavor switch would need a fresh
    // pairing (`install` of a concrete `Handle<S>`) which no inner
    // state can obtain — it holds no handle to re-install from.
    // Concretely: `Outcome::Switch(Erased)` cannot express a direct
    // successor, and `install` requires a `Handle<S>` that only
    // exists at `with_pair`/`fanout` mint points.

    // WALL 4 — inner states structurally cannot safepoint mid-run:
    // the object-safe method is `execute(&mut self, ..)` — no handle
    // is in scope inside the box. The entry safepoints BETWEEN
    // transitions (verified in the main flow); a long-running inner
    //  execute delays the park until it returns.
}
