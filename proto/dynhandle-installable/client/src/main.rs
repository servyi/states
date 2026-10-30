//! The installable shape per the sketch — NO DynHandle: the chain is
//! `Handle<Box<dyn DerefMut<Context>>>`; install pairs a state with
//! its Context and returns the DerefMut-impl to box; the unwrapper
//! projects the state out of the box (offset, repr(C)). Client-only:
//! neither subcrate is touched.

use std::ops::{Deref, DerefMut};

use handle::{Handle, with_pair};

pub struct Env;

/// What a transition decides. `Switch` carries the successor box.
pub enum Outcome {
    /// Keep driving this state.
    Pending,
    /// Overwrite with this successor (an already-installed box).
    Switch(Box<dyn DerefMut<Target = Context>>),
    /// The chain's result.
    Done(u64),
}

/// The chain's erased node: a box that derefs to the Context. The
/// Context itself is BRAND-FREE — the entries are higher-ranked over
/// the handle's brands (that is what lets it be a with_pair node; a
/// branded Context cannot, see the wall notes at the bottom).
pub type Boxed = Box<dyn DerefMut<Target = Context>>;

pub struct Context {
    pub execute: for<'x, 'o, 'c> fn(
        &'x mut Handle<'o, 'c, Boxed>,
        &Env,
    ) -> Result<Outcome, String>,
    pub save: for<'x, 'o, 'c> fn(&'x mut Handle<'o, 'c, Boxed>) -> String,
}

/// The installed pair: state first (the unwrapper projects it at
/// offset 0), then the Context.
#[repr(C)]
pub struct Installed<S> {
    state: S,
    ctx: Context,
}

impl<S> Deref for Installed<S> {
    type Target = Context;
    fn deref(&self) -> &Context {
        &self.ctx
    }
}

impl<S> DerefMut for Installed<S> {
    fn deref_mut(&mut self) -> &mut Context {
        &mut self.ctx
    }
}

/// The projection: the S-flavored unwrapper, as a zero-sized type so
/// the (fn-pointer) entries reach it statically without capturing a
/// value. The only re-typing — created solely by [`install`],
/// monomorphized in `S`.
pub struct Proj<S>(std::marker::PhantomData<fn() -> S>);

pub trait UnwrapMut<S> {
    fn unwrap_mut<'x>(h: &'x mut Handle<'_, '_, Boxed>) -> &'x mut S;
}

impl<S: 'static> UnwrapMut<S> for Proj<S> {
    fn unwrap_mut<'x>(h: &'x mut Handle<'_, '_, Boxed>) -> &'x mut S {
        // Handle -> Box (its node), Box -> the erased object
        let b: &mut Boxed = &mut *h;
        let boxed: &mut dyn DerefMut<Target = Context> = &mut **b;
        // SAFETY: every box this projects from was built by
        // install::<S> — an Installed<S> (repr(C), state at offset 0);
        // this impl exists only for that S, so the fat->thin re-type
        // hits exactly that layout. The borrow is exclusive.
        unsafe {
            &mut *(boxed as *mut dyn DerefMut<Target = Context>
                as *mut Installed<S>
                as *mut S)
        }
    }
}

/// install(S, mk) -> the DerefMut-impl to box. mk builds the Context
/// for this S (the sketch's `mk(unwrapper)` — see W5 for why the
/// unwrapper rides as a TYPE, not a value).
pub fn install<S: 'static>(state: S, mk: impl FnOnce() -> Context) -> Installed<S> {
    Installed { ctx: mk(), state }
}

// ── the vocabulary: INNER states see no handle; OUTER see the chain
// ── handle alongside their state, so they CAN safepoint mid-run.

/// Inner: object-safe, handle-less (cannot safepoint).
pub trait InnerState: 'static {
    fn execute(&mut self, env: &Env) -> Result<Outcome, String>;
    fn save(&self) -> String;
}

/// Outer: sees the chain handle; reaches its state by per-use
/// projection (handle and &mut Self cannot coexist — the projection
/// goes through the handle, E0499; see W6).
pub trait OuterState: Sized + 'static {
    fn execute(h: &mut Handle<'_, '_, Boxed>, env: &Env) -> Result<Outcome, String>;
    fn save(h: &mut Handle<'_, '_, Boxed>) -> String;
}

// ── entries for both flavors (mk builds them) ─────────────────────

fn inner_execute<S: InnerState>(h: &mut Handle<'_, '_, Boxed>, env: &Env) -> Result<Outcome, String> {
    h.safepoint(); // between transitions — the entry CAN
    let state = Proj::<S>::unwrap_mut(h);
    state.execute(env)
}

fn inner_save<S: InnerState>(h: &mut Handle<'_, '_, Boxed>) -> String {
    let state = Proj::<S>::unwrap_mut(h);
    state.save()
}

pub fn inner_entries<S: InnerState>() -> Context {
    Context { execute: inner_execute::<S>, save: inner_save::<S> }
}

fn outer_execute<S: OuterState>(h: &mut Handle<'_, '_, Boxed>, env: &Env) -> Result<Outcome, String> {
    h.safepoint();
    S::execute(h, env)
}

fn outer_save<S: OuterState>(h: &mut Handle<'_, '_, Boxed>) -> String {
    S::save(h)
}

pub fn outer_entries<S: OuterState>() -> Context {
    Context { execute: outer_execute::<S>, save: outer_save::<S> }
}

// ── the states ────────────────────────────────────────────────────

struct Countdown {
    left: u32,
}

impl InnerState for Countdown {
    fn execute(&mut self, _env: &Env) -> Result<Outcome, String> {
        if self.left == 0 {
            // the successor: an INSTALLED box, carried by the outcome
            return Ok(Outcome::Switch(Box::new(install(
                Countup { up: 0, limit: COUNTUP_LIMIT },
                outer_entries::<Countup>,
            ))));
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

/// Countup demonstrates the OUTER flavor: it safepoints mid-run
/// (before its last increment) through the chain handle, reaching its
/// state by per-use projection.
impl OuterState for Countup {
    fn execute(h: &mut Handle<'_, '_, Boxed>, _env: &Env) -> Result<Outcome, String> {
        // per-use projection: each borrow of the state ends before
        // the handle is used again
        if Proj::<Self>::unwrap_mut(h).up == COUNTUP_LIMIT {
            return Ok(Outcome::Done(FINAL_RESULT));
        }
        let near_end = Proj::<Self>::unwrap_mut(h).up == COUNTUP_LIMIT - LAST_STEP_AT;
        if near_end {
            h.safepoint(); // mid-run safepoint — the outer flavor CAN
        }
        Proj::<Self>::unwrap_mut(h).up += 1;
        Ok(Outcome::Pending)
    }
    fn save(h: &mut Handle<'_, '_, Boxed>) -> String {
        let state = Proj::<Self>::unwrap_mut(h);
        format!("countup:{}/{}", state.up, state.limit)
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
        Box::new(install(Countdown { left: START }, inner_entries::<Countdown>)) as Boxed,
        |mut h| drive(&mut h, &Env).expect("chain completes"),
    );
    println!("result: {result}");

    with_pair(
        Box::new(install(Countdown { left: START }, inner_entries::<Countdown>)) as Boxed,
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
// W2 — the unwrapper's fat->thin re-type reaches the STATE at offset
//      0 of the installed box; the Context (entries) sits AFTER it.
//      That layout is install's own construction (repr(C)), so it is
//      sound — but it is the one unsafe in the client, and it is the
//      reason `Installed` can never be constructed any other way
//      (its fields are private; the DerefMut impl hides the state).
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
