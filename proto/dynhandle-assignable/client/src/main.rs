//! The client for the two-piece shape: the vocabulary rides on the
//! INNER handles (`impl State for InnerHandle<Countdown>`), the chain
//! node is an Assignable (value + tree membership) whose set() mints
//! the successor's inner handle with the SAME membership.
#![feature(unsize)]

use std::marker::Unsize;

use dynstate::Assignable;
use handle::{InnerHandle, Shared};

pub struct Env;

/// What a transition decides. `Switch` carries the successor VALUE —
/// the chain installs it through the Assignable.
pub enum Outcome {
    /// Keep driving.
    Pending,
    /// Install this successor value.
    Switch(Box<dyn Next>),
    /// The chain's result.
    Done(u64),
}

/// The vocabulary trait — implemented for the INNER handles, so every
/// state method runs on the handle and CAN safepoint mid-run.
pub trait State {
    fn execute(&mut self, env: &Env) -> Result<Outcome, String>;
    fn save(&mut self) -> String;
}

/// The successor payload: a value that knows how to install itself
/// into the assignable (type-erased set).
pub trait Next {
    // SAFETY-free client seam: install into the assignable, return
    // the erased inner handle.
    fn install(self: Box<Self>, assignable: &mut Assignable) -> Box<dyn State>;
}

/// The blanket: any state whose inner handle erases to dyn State can
/// be a successor. This is the sketch's HandleAble discipline, moved
/// to the client side where the vocabulary lives.
struct Wrap<T>(T);

impl<T: 'static> Next for Wrap<T>
where
    InnerHandle<T>: State + Unsize<dyn State>,
{
    fn install(self: Box<Self>, assignable: &mut Assignable) -> Box<dyn State> {
        let Wrap(value) = *self;
        let inner: InnerHandle<T> = assignable.set(value);
        dynstate::erase(inner)
    }
}

fn next<T: 'static>(value: T) -> Box<dyn Next>
where
    InnerHandle<T>: State + Unsize<dyn State>,
{
    Box::new(Wrap(value))
}

// ── the states: methods on the inner handles ──────────────────────

struct Countdown {
    left: u32,
}

impl State for InnerHandle<Countdown> {
    fn execute(&mut self, _env: &Env) -> Result<Outcome, String> {
        if self.left == 0 {
            return Ok(Outcome::Switch(next(Countup {
                up: 0,
                limit: COUNTUP_LIMIT,
            })));
        }
        self.left -= 1;
        Ok(Outcome::Pending)
    }
    fn save(&mut self) -> String {
        format!("countdown:{}", self.left)
    }
}

struct Countup {
    up: u64,
    limit: u64,
}

impl State for InnerHandle<Countup> {
    fn execute(&mut self, _env: &Env) -> Result<Outcome, String> {
        if self.up == self.limit {
            return Ok(Outcome::Done(FINAL_RESULT));
        }
        if self.up == self.limit - LAST_STEP_AT {
            // mid-run safepoint: the inner handle CARRIES the identity
            self.safepoint();
        }
        self.up += 1;
        Ok(Outcome::Pending)
    }
    fn save(&mut self) -> String {
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

/// The chain: the Assignable IS the node (value + membership); the
/// current inner handle (erased) is what the driver dispatches on.
struct Chain {
    assignable: Assignable,
    current: Box<dyn State>,
}

impl Chain {
    fn start(value: Countdown) -> Self {
        let root = Shared::new();
        let (assignable, inner) = Assignable::new(value, &root);
        Chain {
            assignable,
            current: dynstate::erase(inner),
        }
    }

    fn drive(&mut self, env: &Env) -> Result<u64, String> {
        loop {
            match self.current.execute(env)? {
                Outcome::Pending => {}
                Outcome::Switch(nx) => {
                    // the ONLY door: the successor installs through the
                    // assignable — value swapped, membership KEPT, a
                    // fresh inner handle minted over the new value
                    self.current = nx.install(&mut self.assignable);
                }
                Outcome::Done(k) => return Ok(k),
            }
        }
    }
}

fn main() {
    let mut chain = Chain::start(Countdown { left: START });
    let result = chain.drive(&Env).expect("chain completes");
    println!("result: {result}");

    let mut chain = Chain::start(Countdown { left: START });
    println!("save: {}", chain.current.save());
    let result = chain.drive(&Env).expect("chain completes");
    println!("result: {result}");
    variant_b::demo();
}

// ── VARIANT B: the sketch-literal path ─────────────────────────────
// The HandleAble blanket: any sized state produces its own inner
// handle, erased to dyn State — the runnable box the fanout thread
// receives. DIFFERENCE vs the Assignable path: the minted handle's
// membership is a FRESH root (unconnected) — nothing parks it from a
// parent; the sketch's "missing Assignable in the tree" is exactly
// this gap, which variant A closes by minting over the assignable's
// storage with its registered membership.
mod variant_b {
    use super::*;
    use dynstate::HandleAble;

    fn run() {
        let mut m = Countdown { left: 1 };
        // the sketch's own demo lines, typed:
        let mut h: Box<dyn State> = m.box_innerhandle();
        // parks on its OWN fresh root when a state method safepoints
        let _ = h.save();
        let mut y = Countup { up: 0, limit: 2 };
        let mut h2: Box<dyn State> = y.box_innerhandle();
        let _ = h2.save();
        // swapping the outer Box<V> wholesale — the sketch's
        // reassignment; the old inner handle + its root Shared die
        // together (no tree to notify)
        h = h2;
        let _ = h;
    }

    pub fn demo() {
        run();
        println!("variant B: box_innerhandle swap runs (unconnected)");
    }
}

