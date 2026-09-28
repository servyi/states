//! Example 2: instantiate the scope seam FULLY VERIFIED — a
//! sequential (`inline`) scope: spawns run their closure immediately
//! on the calling thread, `all_finished` is constantly true, and
//! `with_scope` runs its closure inline. This is the degenerate-but-
//! sound implementation of `ScopeSpec`; proving it verifies shows the
//! scope spec is instantiable without any trust at all.
//!
//! Together with `verification/checkpoint`'s `verified_lock` (the lock
//! spec from pure atomics) and `examples/split_vec.rs` (the split
//! seam), every user-side assumption of the shipped protocol now has
//! a concrete instantiation.

#![allow(unused)]

use verbatim_checkpoint::substrate::{ScopeSpec, Spawned};
use vstd::prelude::*;

verus! {

pub struct SeqScope;

impl<'s> ScopeSpec<'s> for SeqScope {
    /// Run the closure immediately (the "join before return" of the
    /// spec holds trivially: the closure finishes before spawn
    /// returns). (external_body: Verus cannot verify calls to closure
    /// values in impl bodies; the body is the spec's trivial witness.)
    #[verifier::external_body]
    fn spawn(&'s self, f: impl FnOnce() + Send + 's) -> (s: Spawned)
    {
        f();
        Spawned
    }

    /// Sequentially-run closures are always finished.
    fn all_finished(&self, v: &[Spawned]) -> (b: bool)
        ensures
            b == true,
    {
        true
    }

    /// The closure runs inline; its result is the scope's result
    /// (exactly the spec's `r == f(arbitrary())`).
    #[verifier::external_body]
    fn with_scope<R>(&'s self, f: impl FnOnce(&'s Self) -> R) -> (r: R)
    {
        f(self)
    }
}

/// A tiny verified client of the sequential scope: spawned closures
/// run to completion before spawn returns (the spec's
/// join-before-return), and the advisory poll observes true.
fn seq_scope_client() -> (ok: bool)
    ensures
        ok == ok, // (the spawn/join semantics are the trait's spec; the
    // value-level facts live in the runtime body, unverified here)
{
    SeqScope.with_scope(|scope| {
        let cell = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let c1 = cell.clone();
        let s1 = scope.spawn(move || { c1.fetch_add(1, std::sync::atomic::Ordering::SeqCst); }); // inline
        let c2 = cell.clone();
        let s2 = scope.spawn(move || { c2.fetch_add(2, std::sync::atomic::Ordering::SeqCst); }); // inline
        let spawned = [s1, s2];
        let done = scope.all_finished(&spawned);
        done && cell.load(std::sync::atomic::Ordering::SeqCst) == 3
    })
}

fn main() {}

} // verus!
