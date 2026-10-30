#![feature(arbitrary_self_types)]
//! The installable shape: the chain is `Handle<Box<dyn DerefMut<Context>>>`;
//! the LIBRARY's `Installed<S, T>` carries the pairing, the derefs, and
//! the projection; the client holds only its vocabulary and entries —
//! zero unsafe here. State methods run on `self: &mut Handle<Self>`
//! (the projected state handle), so EVERY state can safepoint.

use dynstate::{Erased, Proj, Unwrap, install};
use handle::{Handle, with_pair};

pub struct Env;

/// What a transition decides. `Switch` carries the successor box.
pub enum Outcome {
    /// Keep driving this state.
    Pending,
    /// Overwrite with this successor (an already-installed box).
    Switch(Box<dyn dynstate::Erasable<Context>>),
    /// The chain's result.
    Done(u64),
}

/// The chain's context: BRAND-FREE (W1) — the entries are
/// higher-ranked over the handle's brands, which is what lets it be a
/// with_pair node.
pub struct Context {
    pub execute: for<'x, 'o, 'c> fn(
        &'x mut Handle<'o, 'c, Boxed>,
        &Env,
    ) -> Result<Outcome, String>,
    pub save: for<'x, 'o, 'c> fn(&'x mut Handle<'o, 'c, Boxed>) -> String,
}

/// The chain's erased node: a box that derefs to the Context.
pub type Boxed = Erased<Context>;

// ── the vocabulary: ONE state trait, methods on the state's own
// ── projected handle (self: &mut Handle<Self>) — every state can
// ── safepoint mid-run through it.

pub trait State: Sized + 'static {
    fn execute(self: &mut Handle<'_, '_, Self>, env: &Env) -> Result<Outcome, String>;
    fn save(self: &mut Handle<'_, '_, Self>) -> String;
}

// ── entries for both flavors (mk builds them) ─────────────────────

fn execute_entry<S: State>(h: &mut Handle<'_, '_, Boxed>, env: &Env) -> Result<Outcome, String> {
    h.safepoint(); // between transitions
    // the temporary Handle<S> (chain's brands) lives inside the
    // closure; the state's execute runs on it
    Proj::<S, Context>::unwrap_mut(h, |typed| S::execute(typed, env))
}

fn save_entry<S: State>(h: &mut Handle<'_, '_, Boxed>) -> String {
    Proj::<S, Context>::unwrap_mut(h, |typed| S::save(typed))
}

/// mk receives the capability (per the sketch); the entries are its
/// fn pointers — the zero-sized value itself is unused (its TYPE is
/// the pairing).
pub fn entries<S: State>(_pair: Proj<S, Context>) -> Context {
    Context { execute: execute_entry::<S>, save: save_entry::<S> }
}

// ── the states ────────────────────────────────────────────────────

struct Countdown {
    left: u32,
}

impl State for Countdown {
    fn execute(self: &mut Handle<'_, '_, Self>, _env: &Env) -> Result<Outcome, String> {
        if self.left == 0 {
            // the successor: an INSTALLED box, carried by the outcome
            return Ok(Outcome::Switch(Box::new(install(
                Countup { up: 0, limit: COUNTUP_LIMIT },
                entries::<Countup>,
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
    fn execute(self: &mut Handle<'_, '_, Self>, _env: &Env) -> Result<Outcome, String> {
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

/// The hardening probe: a hostile/mistaken client applies Countdown's
/// capability to a COUNTUP chain — the pairing check must catch it
/// (loud failure), not mis-type memory.
fn misuse_probe() {
    use dynstate::{Proj as P, Unwrap as _};
    // The capability is still NAMEABLE (a public associated fn —
    // that cannot be prevented), so misapplication must be CAUGHT:
    // unwrap_mut now downcasts the box, and a Proj::<Countdown, _>
    // call over a Countup box panics with "pairing violated" instead
    // of re-typing memory. The loud-failure round runs on the branch.
    let _misapplied: fn(&mut Handle<'_, '_, Erased<Context>>, &mut dyn FnMut(&mut Handle<'_, '_, Countdown>) -> u64) -> u64 =
        |h, f| <P<Countdown, Context> as dynstate::Unwrap<Countdown, Context>>::unwrap_mut(h, |t| f(t));
    let _ = _misapplied;
}

/// The loud-failure round, executed: build a Countup chain, apply
/// Countdown's capability to it — must PANIC with "pairing violated"
/// (run via `cargo run` flag below).
fn misuse_run() {
    let caught = std::panic::catch_unwind(|| {
        with_pair(
            Box::new(install(Countup { up: 0, limit: COUNTUP_LIMIT }, entries::<Countup>)) as Boxed,
            |mut h| {
                // THE MISAPPLICATION: Countdown's capability over a
                // Countup chain. Before the hardening this re-typed
                // memory (UB); now the downcast fails loudly.
                let r = <dynstate::Proj<Countdown, Context> as dynstate::Unwrap<Countdown, Context>>::unwrap_mut(&mut h, |_t| 7u64);
                r
            },
        )
    });
    assert!(caught.is_err(), "the misapplied capability must be caught");
    println!("misuse: caught loudly (pairing violated)");
}

/// The driver: the entry dispatches through the deref chain; on
/// Switch, the box is swapped wholesale — the installed successor.
fn drive(h: &mut Handle<'_, '_, Boxed>, env: &Env) -> Result<u64, String> {
    loop {
        let execute = (**h).execute; // Handle -> Box -> Context (deref)
        match execute(h, env)? {
            Outcome::Pending => {}
            Outcome::Switch(next) => **h = next,
            Outcome::Done(k) => return Ok(k),
        }
    }
}

fn main() {
    let result = with_pair(
        Box::new(install(Countdown { left: START }, entries::<Countdown>)) as Boxed,
        |mut h| drive(&mut h, &Env).expect("chain completes"),
    );
    println!("result: {result}");
    misuse_run();

    with_pair(
        Box::new(install(Countdown { left: START }, entries::<Countdown>)) as Boxed,
        |mut h| {
            let save = (**h).save;
            println!("save: {}", save(&mut h));
            drive(&mut h, &Env).expect("chain completes")
        },
    );
}

// WALLS (noted while building):
//
// W1 — the Context must be BRAND-FREE. with_pair instantiates the
//      node type BEFORE the brands exist (`for<'o> FnOnce(Handle<'o,
//      'o, T>)` with T fixed), so a Context<'o, 'c> naming the brands
//      cannot be the node. Hence the entries are higher-ranked over
//      the brands instead — workable, but it means the entries cannot
//      name their own pair statically (they see the brands only per
//      call).
//
// W2 — the unwrapper's fat->thin re-type lands on the Installed<S, T>
//      the LIBRARY's install built, then takes an ordinary field
//      borrow — no offset assumptions (moved into dynstate with the
//      rest of the reusable parts; the client has zero unsafe).
//
// W3 — flavor mixing at SWITCH works only INTO boxes: the outcome
//      carries `Box<dyn DerefMut<Context>>`. An outer state cannot
//      switch to a "direct" (unboxed) successor — but nothing needs
//      one: both flavors install into boxes, and the outer flavor's
//      handle access survives (W1's entries pass the chain handle).
//
// W6 — outer states cannot receive (handle, &mut Self) together:
//      the state is projected THROUGH the handle, so the two borrows
//      alias (E0499). The outer flavor therefore takes the handle
//      alone and projects per use — each state borrow ends before the
//      handle is used again. The ??? of "outer execute(Handle<???
//      Self>)" resolves to: the BOX handle, with Self reached by
//      projection inside.
//
// W5 — the entries are fn POINTERS, so a closure capturing the
//      unwrapper VALUE cannot coerce into them (E0308). The sketch's
//      `mk(unwrapper)` therefore rides as a zero-sized TYPE (Proj<S>)
//      whose impl is the unwrapper — reachable statically by the
//      monomorphized entry fns, no capture. The unwrapper-as-value
//      exists only inside install::<S>.
//
// W4 — the box swap (`**h = next`) replaces the WHOLE installed pair
//      (state + entries together). That is the modular successor —
//      but it also means the entries of the outgoing state are
//      dropped with it; per-state entry tables are not shared.
