//! The client for the two-piece shape, following the 16:38 sketch
//! literally: the Assignable is **a Box a new value can be stored
//! into** — a STABLE CELL that inner handles point at, whose content
//! swaps on `set`. The outer `Box<dyn State>` holds the (erased)
//! inner handle; vocabulary methods run on the inner handles.
#![feature(unsize)]

use std::marker::Unsize;

use dynstate::Assignable;
use handle::{InnerHandle, Shared};

pub struct Env;

/// What a transition decides. `Switch` carries the successor install:
/// the value plus the typed re-mint over the cell (the sketch's set
/// with T known at the call site).
pub enum Outcome {
    /// Keep driving.
    Pending,
    /// Store this successor into the cell.
    Switch(Box<dyn Install>),
    /// The chain's result.
    Done(u64),
}

/// The typed successor: stores its value into the cell and re-mints
/// the typed handle over it (steps 1-4 of the sketch's set).
pub trait Install {
    fn install(
        self: Box<Self>,
        status: &std::sync::Arc<Shared>,
        cell: &Assignable<dyn StateAble>,
    ) -> Box<dyn State>;
}

struct InstallImpl<T>(T);

impl<T: 'static> Install for InstallImpl<T>
where
    InnerHandle<T>: State + Unsize<dyn State>,
{
    fn install(
        self: Box<Self>,
        status: &std::sync::Arc<Shared>,
        cell: &Assignable<dyn StateAble>,
    ) -> Box<dyn State> {
        // (1) store the new value into the assignable; (2) the old
        // typed view is forgotten here (the chain replaces the outer
        // with what this returns); (3) clone the membership; (4) the
        // new typed handle over the SAME cell.
        cell.store(Box::new(self.0) as Box<dyn StateAble>);
        // SAFETY: the cell now holds the new T.
        let inner: InnerHandle<T> = unsafe { dynstate::mint(status.clone(), cell) };
        dynstate::erase(inner)
    }
}

fn install<T: 'static>(value: T) -> Box<dyn Install>
where
    InnerHandle<T>: State + Unsize<dyn State>,
{
    Box::new(InstallImpl(value))
}

/// The vocabulary — on the INNER handles: every state method runs on
/// its handle and CAN safepoint mid-run.
pub trait State {
    fn execute(&mut self, env: &Env) -> Result<Outcome, String>;
    fn save(&mut self) -> String;
}

/// The cell's erasure: any value whose handle is a State (the
/// sketch's `dyn HandleAble<V>`).
pub trait StateAble {}
impl<T> StateAble for T where InnerHandle<T>: State {}

/// The chain — the sketch's OuterHandle plus its cell:
/// - `cell` holds the CURRENT VALUE (erased) — the swappable content;
/// - `outer` holds the current INNER HANDLE (erased), pointing at the
///   cell — what the driver dispatches on;
/// - `status` is the shared membership the handles carry.
struct Chain {
    /// The cell lives at a stable heap address (the sketch's
    /// `Vec<Box<K>>`), held as a RAW pointer: handles point at it, so
    /// it must never move, and its borrow stack must never be churned
    /// by Box re-tags (Miri-taught).
    cell: *mut Assignable<dyn StateAble>,
    outer: Box<dyn State>,
    status: std::sync::Arc<Shared>,
}

impl Drop for Chain {
    fn drop(&mut self) {
        // SAFETY: the cell was created by Box::into_raw in start and
        // is solely owned here.
        unsafe { drop(Box::from_raw(self.cell)) };
    }
}

impl Chain {
    fn start<T: 'static>(value: T) -> Self
    where
        InnerHandle<T>: State + Unsize<dyn State>,
    {
        let status = Shared::new();
        let cell: *mut Assignable<dyn StateAble> =
            Box::into_raw(Box::new(Assignable::new(Box::new(value) as Box<dyn StateAble>)));
        // SAFETY: the raw is fresh and solely owned; the cell holds
        // the T we just stored.
        let inner: InnerHandle<T> =
            unsafe { dynstate::mint(status.clone(), &*cell) };
        Chain {
            cell,
            outer: dynstate::erase(inner),
            status,
        }
    }

    /// set — the sketch's four steps, verbatim: (1) store the new
    /// value into the assignable; (2) the old typed view is forgotten
    /// (the outer box is replaced below); (3) clone the membership;
    /// (4) make the new typed handle over the SAME cell and store its
    /// box in the outer.
    fn set(&mut self, next: Box<dyn Install>) {
        // the ONE door — the install does the four steps
        // SAFETY: the raw is the chain's own cell.
        self.outer = next.install(&self.status, unsafe { &*self.cell });
    }

    fn drive(&mut self, env: &Env) -> Result<u64, String> {
        loop {
            match self.outer.execute(env)? {
                Outcome::Pending => {}
                Outcome::Switch(next) => {
                    // ONE door: store into the cell, re-mint over it
                    self.set(next);
                }
                Outcome::Done(k) => return Ok(k),
            }
        }
    }
}

// ── the states: methods on the inner handles, values in the cell ──

struct Countdown {
    left: u32,
}

impl State for InnerHandle<Countdown> {
    fn execute(&mut self, _env: &Env) -> Result<Outcome, String> {
        if self.left == 0 {
            // the successor VALUE, erased for the cell
            return Ok(Outcome::Switch(install(Countup {
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
            // mid-run safepoint — the handle CARRIES the identity
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

fn main() {
    let mut chain = Chain::start(Countdown { left: START });
    let result = chain.drive(&Env).expect("chain completes");
    println!("result: {result}");

    let mut chain = Chain::start(Countdown { left: START });
    println!("save: {}", chain.outer.save());
    let result = chain.drive(&Env).expect("chain completes");
    println!("result: {result}");
}
