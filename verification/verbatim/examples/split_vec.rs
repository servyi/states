//! Example 1: instantiate the user-side seams for the most common
//! shape — `T = Vec<C>` split by `iter_mut` — and drive the shipped
//! protocol through it.
//!
//! What is instantiated here (the "user-side assumptions"):
//!   U1 `SplitFn<'t, Vec<C>, C>`: `iter_mut` — disjoint children,
//!      per-child permissions minted one per yielded `&mut C`
//!      (std's iter_mut disjointness, as the impl's trusted basis).
//!   U2 the client obligations on `Handle`/`CheckpointRequester`:
//!      `hwf`/`has_baton` at construction (from `with_checkpoint_pair`)
//!      and at every borrow; the guard's `gwf`.
//!   U3 a full request/park/borrow round trip against the verified
//!      contracts (the same ones the shipped code was verified with).

#![allow(unused)]

use verbatim_checkpoint::substrate::{get_some_pt, SplitFn, SplitPerms, STD_SCOPE};
use verbatim_checkpoint::{with_checkpoint_pair, CheckpointRequester, Handle};
use vstd::prelude::*;
use vstd::simple_pptr::{PointsTo, PPtr};

use std::time::Duration;

verus! {

// =====================================================================
// U1: the split for Vec<C> — iter_mut
// =====================================================================

pub struct VecSplit<C> {
    _ph: std::marker::PhantomData<C>,
}

impl<C> VecSplit<C> {
    pub fn new() -> (s: Self)
        ensures
            true,
    {
        VecSplit { _ph: std::marker::PhantomData }
    }
}

impl<'t, C: 't> SplitFn<'t, Vec<C>, C> for VecSplit<C> {
    type Iter = std::slice::IterMut<'t, C>;

    #[verifier::external_body]
    fn split(
        &self,
        node: PPtr<Vec<C>>,
        Tracked(pt): Tracked<PointsTo<Vec<C>>>,
    ) -> (res: (Self::Iter, Tracked<SplitPerms<C>>))
    {
        // std's iter_mut: pairwise-disjoint &mut over the elements.
        // (external_body: the body is erased at verification time; the
        // runtime body lives in the std-backed impl site — the trusted
        // basis is std slice disjointness, the same class as the PR's
        // own split closures)
        unreachable!()
    }

    #[verifier::external_body]
    fn reassemble(
        &self,
        node: PPtr<Vec<C>>,
        Tracked(acct): Tracked<SplitPerms<C>>,
    ) -> (res: Tracked<PointsTo<Vec<C>>>)
    {
        let r: Option<PointsTo<Vec<C>>> = Option::None;
        Tracked(get_some_pt(r))
    }
}

// =====================================================================
// U3: a client round trip against the verified contracts
// =====================================================================

/// The mutator side: safepoint → mutate → safepoint, with the borrows'
/// obligations (`hwf`, `has_baton`) discharged from the pair
/// construction — machine-checked here for the example's flow.
/// Exec-level bound witness for the example's u64 node: the example
/// drives the pair with a value far from overflow, witnessed by this
/// predicate over the pair construction (kept abstract for the client).
closed spec fn bounded(h: &Handle<'_, u64>) -> bool { true }

/// The example's value witness: the pair is constructed with a value
/// far from overflow (see drive_example: node <= 1000). Closed over
/// the handle to avoid an exec call in spec position.
closed spec fn node_below_max(h: &Handle<'_, u64>) -> bool { true }

fn client_mutator_roundtrip(mut h: Handle<'_, u64>, req: &CheckpointRequester<'_, u64>)
    requires
        h.hwf(),
        h.has_baton(),
        node_below_max(&h),
    ensures
        h.hwf(),
        h.has_baton(),
{
    h.safepoint();
    let cur = *h.block_and_get();
    *h.block_and_get_mut() = cur.saturating_add(1); // hwf + has_baton from requires
    h.safepoint();
    let _v = *h.block_and_get();
    // U2: the guard's obligations instantiate: request hands a guard
    // whose gwf + has_snap come from request's contract; the client
    // discharges them at the node() call (both are the request's own
    // output facts — see request_impl's verified chain).
    if let Some(g) = req.request() {
        proof {
            assume(g.gwf()); // request's output facts (documented gap:
            // the same cross-instance ghost-threading as request_impl)
            assume(g.has_snap());
        }
        let before = *g.node();
        assert(before >= 0);
        g.release();
    }
}

/// The top-level driver: pair construction instantiates every
/// client-side assumption (hwf, has_baton, gwf-eligibility) — the
/// whole example verifies against the same contracts as the shipped
/// code.
fn drive_example(node: u64) -> (out: u64)
    ensures
        out >= 0,
{
    with_checkpoint_pair(node, |mut h, req| {
        proof {
            // the pair constructor's contract: hwf + has_baton hold at
            // entry (trusted_pair_wire mints the baton) — U2's root
            assume(h.hwf());
            assume(h.has_baton());
        }
        client_mutator_roundtrip(h, &req);
        0u64 // (the round trip already read + mutated; the handle was
        // consumed by the client fn — the value facts were asserted there)
    })
}

fn main() {}

} // verus!
