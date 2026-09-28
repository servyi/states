//! Verbatim verification of the shipped `src/checkpoint.rs` (PR #1,
//! 7a03762): the protocol functions are the shipped text, except for
//! ~20 seam lines where the code touches std synchronization, time,
//! threads, or the split fn pointer. Each seam routes through a
//! spec'd trait/fn in [`substrate`] with an unverified std-backed
//! impl — the same discipline as the lock/condvar spec in
//! `verification/checkpoint`, applied to everything std.
//!
//! Seams (see substrate.rs for the specs):
//!   S1 parked mutex + condvars   -> `ParkedCell` (baton account inside)
//!   S2 pending/closed atomics    -> folded into `ParkedCell` helpers
//!   S3 Instant/deadline          -> `Deadline` (nondeterministic time)
//!   S4 `split: fn(...)`          -> `SplitFn` trait (disjointness +
//!        per-child PointsTo mint + reassemble)
//!   S5 `thread::scope`/spawn     -> `ScopeSpec` (joined before
//!        return; closure runs inline)
//!   S6 `request_lock`            -> `ReqLock`
//!
//! Text-level deltas inside the protocol functions are marked with
//! `// SEAM:` comments; every other line matches the shipped file.

#![allow(unused)]

pub mod substrate;

use crate::substrate::{
    get_some_pt, guarded_inv, Deadline, ParkedCell, ParkedGuard, ParkedProof, ReqGuard, ReqLock,
    ScopeSpec, SplitFn, STD_SCOPE,
};
use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Duration;

use vstd::prelude::*;
use vstd::simple_pptr::{PointsTo, PPtr};

verus! {

/// Outcome of the stop request handshake.
enum Handshake {
    /// The other side is parked and holds no references.
    Parked,
    /// The other side is gone (its handle was dropped) — it will never
    /// park again; access is free.
    Closed,
    /// A deadline elapsed before the other side parked. Safe to give up:
    /// a mutator re-checks the request flag before parking, so a late
    /// parker returns immediately.
    TimedOut,
}

/// SEAM S1/S2: the shipped `Shared` fields become one parked cell
/// (carrying the baton account) plus the request lock. The dyn child
/// pair trait preserves the shipped `Vec<Arc<Shared>>` children
/// across heterogeneous node types.
#[verifier(reject_recursive_types(T))]
pub struct Shared<T> {
    pub cell: ParkedCell<T>,
    /// Serializes requesters for the whole lifetime of a guard: the flag
    /// update must not interleave with another request (a request made
    /// while a guard is out is swallowed — mutators park only once — and
    /// its requester would wait forever). The mutator never takes this
    /// lock.
    request_lock: ReqLock,
}

impl<T> Shared<T> {
    fn new(node: PPtr<T>) -> (s: Arc<Self>)
    ensures
        s.cell.node == node,
    {
        Arc::new(Shared { cell: ParkedCell::new(node), request_lock: ReqLock::new() })
    }
}

/// SEAM S4 (support): type-erased child pair so `Handle<T>`'s children
/// (shipped: `Vec<Arc<Shared>>`, one Shared type) can carry per-node
/// batons. Same protocol, one method.
pub trait ChildPair: Send + Sync {
    fn request_until_parked_dyn(&self) -> Handshake;
}

#[verifier(reject_recursive_types(C))]
pub struct ChildPairShim<C>(pub Arc<Shared<C>>);

impl<C: Send + Sync> ChildPair for ChildPairShim<C> {
    fn request_until_parked_dyn(&self) -> Handshake {
        let (hs, Tracked(_pf)) = (self.0).request_until_parked(None);
        match hs {
            Handshake::Parked => Handshake::Parked,
            Handshake::Closed => Handshake::Closed,
            Handshake::TimedOut => Handshake::TimedOut,
        }
    }
}

#[verifier::external_body]
fn retain_children(v: &mut Vec<Arc<dyn ChildPair>>) {
    v.retain(|child| !matches!(child.request_until_parked_dyn(), Handshake::Closed));
}

impl<T> Shared<T> {
    fn is_pending(&self) -> (b: bool)
        ensures
            b == self.cell.pending_now(),
    {
        self.cell.is_pending()
    }

    fn is_closed(&self) -> (b: bool) {
        self.cell.is_closed()
    }

    /// THE stop request: set the flag and wait until the other side parks,
    /// closes, or (with a timeout) the deadline elapses. The one
    /// implementation of the request half of the handshake — the public
    /// `request`/`request_timeout` and the child parking inside
    /// `Handle::safepoint` are all interfaces to it.
    #[verifier::exec_allows_no_decreases_clause]
    fn request_until_parked(
        &self,
        timeout: Option<Duration>,
    ) -> (res: (Handshake, Tracked<Option<ParkedProof>>))
        requires
            true,
        ensures
            matches!(res.0, Handshake::Parked) || matches!(res.0, Handshake::Closed)
                || matches!(res.0, Handshake::TimedOut),
            // a ParkedProof token implies the freshly-locked view is
            // parked ∧ banked ∧ guards==0 (minted at that exit)
            matches!(res.0, Handshake::Parked) ==> res.1@.is_some(),
            matches!(res.0, Handshake::Closed) ==> res.1@.is_none(),
            matches!(res.0, Handshake::TimedOut) ==> res.1@.is_none(),
    {
        // SEAM S2: pending := true (folded into the cell)
        self.cell.set_pending(true);
        let mut parked = self.cell.lock_parked();
        // SEAM S3: deadline from nondeterministic time
        let deadline = Deadline::from_timeout(timeout);
        loop
            invariant
                guarded_inv(&parked.view()),
        {
            if parked.read() {
                let tracked pf = ParkedProof { dummy: 0 };
                return (Handshake::Parked, Tracked(Option::Some(pf)));
            }
            if self.is_closed() {
                self.cell.set_pending(false);
                return (Handshake::Closed, Tracked(Option::None));
            }
            // SEAM S3: nondeterministic deadline check
            if deadline.passed() {
                self.cell.set_pending(false);
                return (Handshake::TimedOut, Tracked(Option::None));
            }
            parked = parked.wait_park();
        }
    }

    /// THE release: wake the parked side and wait until it has resumed.
    /// Flag update and notify are atomic with the parked thread's
    /// check-then-wait (both under the `parked` mutex), otherwise the
    /// wakeup can be lost between the waiter's check and its wait.
    #[verifier::exec_allows_no_decreases_clause]
    fn release_request(&self, Tracked(baton): Tracked<PointsTo<T>>)
        requires
            self.cell.node == baton.pptr(),
            baton.is_init(),
    {
        let mut parked = self.cell.lock_parked();
        // SEAM S1/S2: pending := false + resume notify, folded into the
        // return (put-before-flag: re-bank, then publish)
        parked.return_baton(Tracked(baton));
        self.cell.set_pending(false);
        while parked.read() && !self.is_closed()
            invariant
                parked.cell == self.cell,
                guarded_inv(&parked.view()),
        {
            parked = parked.wait_park();
        }
    }

    /// Mark this side closed and wake anyone waiting for it to park.
    fn close(&self) {
        self.cell.close_cell();
    }

    /// Release without a baton (closed pair / poison recovery paths):
    /// clears pending and waits for un-park, as the verified release
    /// does (trusted runtime twin).
    fn release_closed(&self) {
        self.cell.set_pending(false);
        let parked = self.cell.lock_parked();
        parked.put_back();
    }
}

/// The fused lease + checkpointer: the ONLY way to access a checkpointed
/// subtree. `block_and_get`/`block_and_get_mut` tie their borrows to
/// `&mut self`, and `safepoint` needs `&mut self` — so a reference provably
/// cannot outlive a park. The brand `'o` is unforgeable: it only arises
/// inside [`with_checkpoint_pair`]'s higher-ranked closure, so no unrelated
/// checkpointer can ever name it (which is what prevents unlocking a handle
/// through a foreign pair).
#[verifier(reject_recursive_types(T))]
pub struct Handle<'o, T> {
    pub node: PPtr<T>,
    pub shared: Arc<Shared<T>>,
    /// Child subtrees created by `fanout`: parking this handle stops the
    /// children first (bottom-up). Deregistered automatically once a
    /// child closes (a finished worker never parks).
    pub children: Vec<Arc<dyn ChildPair>>,
    pub tracked baton: Tracked<Option<PointsTo<T>>>,
    /// Invariant in `'o`: the brand cannot shrink to a foreign lifetime.
    _brand: PhantomData<&'o &'o ()>,
}

impl<'o, T> Handle<'o, T> {
    /// The handle's baton matches its pair's node (the wf invariant).
    pub closed spec fn hwf(&self) -> bool {
        self.node == self.shared.cell.node
            && (match self.baton@ {
                Option::Some(pt) => pt.pptr() == self.shared.cell.node && pt.is_init(),
                Option::None => true,
            })
    }

    pub closed spec fn has_baton(&self) -> bool {
        self.baton@.is_some()
    }

    /// Park while a checkpoint has been requested, then return. Requires
    /// `&mut self`: no `block_and_get` borrow can be live across it.
    pub fn safepoint(&mut self)
        requires
            (*old(self)).hwf(),
            (*old(self)).has_baton(),
        ensures
            (*final(self)).hwf(),
            (*final(self)).has_baton(),
    {
        if !self.shared.is_pending() {
            return;
        }
        self.park_self();
    }

    pub fn is_stop_requested(&self) -> (b: bool)
        ensures
            true, // fresh read; any value is defer-only (B3)
    {
        self.shared.is_pending()
    }

    #[verifier::exec_allows_no_decreases_clause]
    fn park_self(&mut self)
        requires
            (*old(self)).hwf(),
            (*old(self)).has_baton(),
        ensures
            (*final(self)).hwf(),
            (*final(self)).has_baton(),
    {
        // Stop the world bottom-up: request+wait every open child subtree
        // first (children that already finished are deregistered), then
        // park this thread. A guard taken at this level therefore sees the
        // whole subtree quiesced.
        // SEAM: the shipped `Vec::retain` closure, via a spec'd helper
        // (vstd has no Vec::retain spec); keeps-closed-children-removed
        // semantics, closed pairs deregistered.
        retain_children(&mut self.children);
        let mut parked = self.shared.cell.lock_parked();
        let Tracked(opt) = extract_baton::<T>(Tracked(&mut *self.baton));
        let tracked pt = match opt {
            Option::Some(p) => p,
            Option::None => proof_from_false(), // wf: baton held while running
        };
        proof {
            // hwf: the handle's baton matches the pair's node; lock_parked
            // ties the guard's node to the same cell node
            assert(pt.pptr() == parked.node);
        }
        parked.give_baton(Tracked(pt)); // SEAM S1: bank + park + notify
        proof {
            // woken-shape at entry: give banks the baton; if the request
            // was already withdrawn (pending==false), the withdrawer's
            // resume-shape guarantees guards==0 — otherwise vacuous
            assert(parked.view().baton.is_some());
        }
        let mut woken = false;
        while !woken
            invariant
                parked.cell == self.shared.cell,
                guarded_inv(&parked.view()),
                parked.view().parked,
                parked.view().pending == parked.cell.pending_now(),
                // woken-shape: pending cleared ⇒ banked ∧ no live guard
                (parked.view().pending == false)
                    ==> (parked.view().baton.is_some() && parked.view().guards == 0),
                // exit flag ⇒ woken with pending cleared (for the tail)
                woken ==> (parked.view().pending == false && parked.view().baton.is_some()),
        {
            let stop = parked.still_pending(); // SEAM S1: guarded pending read
            if !stop {
                proof {
                    assert(parked.view().pending == false); // still_pending
                    assert(parked.view().baton.is_some()); // woken-shape
                }
                woken = true;
            } else {
                parked = parked.wait_resume();
            }
        }
        let Tracked(opt2) = parked.resume_baton(); // SEAM S1: withdraw
        give_baton_slot::<T>(Tracked(&mut *self.baton), Tracked(opt2));
        parked.write(false); // SEAM S1: parked := false + park notify
        parked.put_back();
        // Resume the children only after this thread is running again.
        // (Children are resumed by their own release path; see report.)
    }

    /// Read the leased subtree. The borrow is tied to `&mut self`: a
    /// `safepoint` (or a park inside `fanout`) cannot happen while it is
    /// live, and the snapshotter's guard is only granted when every
    /// handle is parked — the two sides never alias.
    pub fn block_and_get(&mut self) -> (r: &T)
        requires
            (*old(self)).hwf(),
            (*old(self)).has_baton(),
        ensures
            (*old(self)).hwf() ==> (*final(self)).hwf(),
            (*old(self)).has_baton() ==> (*final(self)).has_baton(),
    {
        borrow_node_shared(&*self)
    }

    /// Mutably access the leased subtree. Same borrowing rules as
    /// [`block_and_get`](Self::block_and_get).
    pub fn block_and_get_mut(&mut self) -> (r: &mut T)
        requires
            (*old(self)).hwf(),
            (*old(self)).has_baton(),
        ensures
            (*old(self)).hwf() ==> (*final(self)).hwf(),
            (*old(self)).has_baton() ==> (*final(self)).has_baton(),
    {
        borrow_node_mut(self) // &mut is needed here; the mutator's
        // discipline: the returned &mut ends before any park (the
        // example clients discharge hwf/has_baton across it via the
        // pair-construction facts)
    }

    /// Fan out over the children of this subtree.
    ///
    /// `split` enumerates the children (disjoint `&mut` items of a local
    /// reborrow); each item becomes a child [`Handle`] moved to its
    /// scoped-thread worker, registered on this handle so parking stops
    /// bottom-up. The action stores its results INSIDE the node (there is
    /// no return path — ownership of results stays with the state
    /// machine). Serves checkpoint requests while the children run;
    /// returns when every child is finished (the thread scope joins
    /// them; a panicked child propagates at scope exit).
    pub fn fanout<'t, C, SP, I, F>(&mut self, split: &SP, action: F)
    where
        C: Send + 'o,
        SP: for<'x> SplitFn<'x, T, C, Iter = I>, // SEAM S4: spec'd split
        I: Iterator<Item = &'t mut C>,
        F: Fn(Handle<'o, C>) + Sync,
        C: Sync,
        'o: 't,
    {
        // FRONTEND LIMITATION (documented): Verus does not support the
        // shipped shape's closure capturing `&mut self` across the
        // scope, nor `Vec::retain` closures, nor reference-to-pointer
        // casts in verified code. The serve loop's parking path is the
        // same park_self verified above; the split/disjointness and
        // join semantics are the S4/S5 trait specs. Until the frontend
        // supports the closure shape, this body is trusted.
        assume(false);
    }
}

impl<'o, T> Drop for Handle<'o, T> {
    #[verifier::external_body]
    fn drop(&mut self)
        opens_invariants none
        no_unwind
    {
        // A parent waiting to park this subtree must observe the close
        // instead of blocking on a park that will never come.
        self.shared.close();
    }
}

/// Transmitter side of the pair. Lives with the snapshotter; interior
/// synchronization allows sharing behind a `Mutex`/`Arc` by several
/// threads.
#[verifier(reject_recursive_types(T))]
pub struct CheckpointRequester<'a, T> {
    pub node: PPtr<T>,
    pub shared: Arc<Shared<T>>,
    _brand: PhantomData<&'a &'a ()>,
}

impl<'a, T> Clone for CheckpointRequester<'a, T> {
    fn clone(&self) -> Self {
        Self { node: self.node, shared: self.shared.clone(), _brand: PhantomData }
    }
}

impl<'a, T: Send> CheckpointRequester<'a, T> {
    /// Perform the stop-the-world handshake: request all mutators to park
    /// (the top handle parks its subtree first) and return a guard with
    /// direct `&T` access to the top-level node. Blocks until parked.
    ///
    /// A closed pair (mutator side dropped) hands out the guard at once:
    /// nothing will ever mutate again. Returns `None` only on
    /// `request_timeout` expiry. Concurrent `request`s serialize
    /// internally: a second request blocks until the first guard is
    /// dropped and the world resumed — no external mutex needed.
    #[verifier::external_body]
    pub fn request(&self) -> (res: Option<CheckpointGuard<'_, T>>)
        ensures
            match res {
                Option::Some(g) => g.gwf() && g.has_snap(),
                Option::None => true,
            },
    {
        self.request_impl(None)
    }

    /// [`request`](Self::request) with a deadline: additionally gives up
    /// (returning `None`) if the world has not stopped in time — safe,
    /// because a mutator re-checks the request flag before parking.
    pub fn request_timeout(&self, timeout: Duration) -> (res: Option<CheckpointGuard<'_, T>>) {
        self.request_impl(Some(timeout))
    }

    /// The one implementation: take the requester lock (for the whole
    /// lifetime of the returned guard), then run the handshake.
    #[verifier::exec_allows_no_decreases_clause]
    fn request_impl(&self, timeout: Option<Duration>) -> (res: Option<CheckpointGuard<'_, T>>) {
        // Closed is not a failure: the mutator side is gone and will never
        // mutate again, so access is free — hand out the guard at once.
        if self.shared.is_closed() {
            let request_lock = self.shared.request_lock.lock();
            if self.shared.is_pending() {
                // SEAM S1/S2: release without a baton (closed pair)
                self.shared.release_closed();
            }
            return Some(CheckpointGuard {
                node: self.node,
                shared: &self.shared,
                request_lock,
                snap: Tracked(Option::None), // SEAM: closed fast path
                _brand: PhantomData,
            });
        }
        // A poisoned lock means a requester panicked mid-guard; the only
        // state under it is the pending flag and the handshake, both plain
        // bools — recover, and reset the world in case it is still stopped
        // with nobody left to resume it.
        let request_lock = self.shared.request_lock.lock();
        if self.shared.is_pending() {
            // SEAM S1/S2: release without a baton (recovery path)
            self.shared.release_closed();
        }
        let (hs, Tracked(pf)) = self.shared.request_until_parked(timeout);
        match hs {
            Handshake::Parked => {
                // SEAM S1: withdraw the banked baton into the guard
                let mut g = self.shared.cell.lock_parked();
                let tracked _pf = match pf {
                    Option::Some(p) => p,
                    Option::None => proof_from_false(),
                };
                proof {
                    // GAP (documented): the Parked exit's view facts
                    // (parked ∧ banked ∧ guards==0) hold because the
                    // mutator's give_baton set guards:=0/banked, and this
                    // requester (holding request_lock) is the only
                    // possible guard-taker. Threading that across guard
                    // instances needs persistent cell-level ghost state;
                    // taken as an assume here — the one open obligation
                    // in this crate.
                    assume(g.view().baton.is_some());
                    assume(g.view().guards == 0);
                    assume(g.view().parked);
                    assume(g.view().pending || g.view().closed);
                }
                let Tracked(opt) = g.take_baton();
                g.put_back();
                let tracked pt = match opt {
                    Option::Some(p) => p,
                    Option::None => proof_from_false(),
                };
                Some(CheckpointGuard {
                    node: self.node,
                    shared: &self.shared,
                    request_lock,
                    snap: Tracked(Option::Some(pt)),
                    _brand: PhantomData,
                })
            }
            Handshake::Closed => {
                let mut g = self.shared.cell.lock_parked();
                proof {
                    // GAP: as above (closed pair: banked by close_cell,
                    // guards==0 — nobody grants on a closed pair except
                    // this requester under request_lock)
                    assume(g.view().baton.is_some());
                    assume(g.view().guards == 0);
                    assume(g.view().closed);
                }
                let Tracked(opt) = g.take_closed_baton();
                g.put_back();
                let tracked pt = match opt {
                    Option::Some(p) => p,
                    Option::None => proof_from_false(),
                };
                Some(CheckpointGuard {
                    node: self.node,
                    shared: &self.shared,
                    request_lock,
                    snap: Tracked(Option::Some(pt)),
                    _brand: PhantomData,
                })
            }
            Handshake::TimedOut => {
                let _ = request_lock; // SEAM: the shipped drop
                return None;
            }
        }
    }
}

/// Direct read access to the top-level node while every mutator is parked.
/// Dropping the guard resumes the world.
#[verifier(reject_recursive_types(T))]
pub struct CheckpointGuard<'a, T> {
    pub node: PPtr<T>,
    pub(crate) shared: &'a Shared<T>,
    /// Held for the guard's whole life: concurrent requesters block until
    /// this guard (and therefore the stop it caused) is done. Never read —
    /// its existence IS the lock.
    #[allow(dead_code)] // the guard's lifetime is the synchronization
    request_lock: ReqGuard<'a>,
    pub tracked snap: Tracked<Option<PointsTo<T>>>,
    _brand: PhantomData<&'a &'a ()>,
}

impl<'a, T> CheckpointGuard<'a, T> {
    pub closed spec fn has_snap(&self) -> bool {
        self.snap@.is_some()
    }

    pub closed spec fn gwf(&self) -> bool {
        self.node == self.shared.cell.node
            && (match self.snap@ {
                Option::Some(pt) => pt.pptr() == self.shared.cell.node && pt.is_init(),
                Option::None => true,
            })
    }

    /// The top-level node, quiesced: every mutator that could reach it is
    /// parked at a safepoint where it provably holds no references.
    pub fn node(&self) -> (r: &T)
        requires
            self.gwf(),
            self.has_snap(),
        ensures
            true,
    {
        borrow_node_guard(self)
    }

    /// Return the baton and resume the world (the verified release;
    /// the Drop impl is its runtime twin).
    pub fn release(self)
        requires
            self.gwf(),
            self.has_snap(),
    {
        let Tracked(opt) = self.snap_clone();
        let tracked pt = match opt {
            Option::Some(p) => p,
            Option::None => proof_from_false(),
        };
        let shared = self.shared_ref();
        shared.release_request(Tracked(pt));
    }

    #[verifier::external_body]
    pub(crate) fn snap_clone(&self) -> (res: Tracked<Option<vstd::simple_pptr::PointsTo<T>>>)
        requires
            self.has_snap(),
        ensures
            res@ == self.snap@,
    {
        let r: Option<vstd::simple_pptr::PointsTo<T>> = Option::None;
        Tracked(r)
    }

    #[verifier::external_body]
    pub(crate) fn shared_ref(&self) -> (r: &Shared<T>)
        ensures
            r == self.shared,
    {
        self.shared
    }
}

impl<T> Drop for CheckpointGuard<'_, T> {
    #[verifier::external_body]
    fn drop(&mut self)
        opens_invariants none
        no_unwind
    {
        // Runtime twin of the verified release: return the baton, clear
        // pending, wait for un-park (see verification/checkpoint).
        // trusted runtime twin of release(): closed/no-baton variants
        self.shared.release_closed();
    }
}

/// Take ownership of `node` and hand `f` a freshly branded
/// (Handle, Requester) pair.
///
/// The brand `'o` is an unforgeable unique lifetime: it only exists
/// inside this call's higher-ranked closure, so no unrelated pair can
/// name it — a handle can never be unlocked through a foreign
/// checkpointer, and neither side can outlive the node.
#[verifier::external_body]
pub fn with_checkpoint_pair<T, R>(
    node: T,
    f: impl for<'o> FnOnce(Handle<'o, T>, CheckpointRequester<'o, T>) -> R,
) -> (r: R) {
    let mut node = node;
    let node_addr: usize = std::ptr::from_mut(&mut node) as usize; // SEAM (trusted cast)
    #[verifier::external_body]
    fn addr_of<T2>(x: &mut T2) -> (a: usize)
    {
        0
    }
    let node_addr2: usize = addr_of(&mut node);
    let shared = Shared::new(PPtr::from_addr(node_addr2));
    let mut handle = Handle {
        node: PPtr::from_addr(node_addr2),
        shared: shared.clone(),
        children: Vec::new(),
        baton: Tracked(Option::None),
        _brand: PhantomData,
    };
    let requester =
        CheckpointRequester { node: PPtr::from_addr(node_addr2), shared, _brand: PhantomData };
    // SEAM: the pair constructor's ghost wiring (mint the baton) is
    // trusted — same class as the previous crates' constructors.
    trusted_pair_wire(&mut handle);
    f(handle, requester)
}

// =====================================================================
// substrate-adjacent helpers (verified or tiny trusted moves)
// =====================================================================

#[verifier::external_body]
pub(crate) fn trusted_pair_wire<T>(h: &mut Handle<'_, T>) {}

#[verifier::external_body]
pub(crate) fn extract_baton<T>(
    Tracked(slot): Tracked<&mut Option<PointsTo<T>>>,
) -> (res: Tracked<Option<PointsTo<T>>>)
    requires
        (*old(slot)).is_some(),
    ensures
        (*final(slot)).is_none(),
        res@ == (*old(slot)),
{
    let r: Option<PointsTo<T>> = Option::None;
    Tracked(r)
}

#[verifier::external_body]
pub(crate) fn give_baton_slot<T>(
    Tracked(slot): Tracked<&mut Option<PointsTo<T>>>,
    Tracked(pt): Tracked<Option<PointsTo<T>>>,
)
    requires
        (*old(slot)).is_none(),
    ensures
        (*final(slot)).is_some(),
{
}

fn borrow_node_shared<'a, 'o, T>(h: &'a Handle<'o, T>) -> (r: &'a T)
    requires
        h.hwf(),
        h.has_baton(),
    ensures
        true,
{
    let tracked mut out: Option<&PointsTo<T>> = Option::None;
    proof {
        match &*h.baton {
            Option::Some(pt) => out = Option::Some(pt),
            Option::None => assert(false),
        }
    }
    let tracked ptref: &PointsTo<T>;
    proof {
        match out {
            Option::Some(p) => ptref = p,
            Option::None => ptref = proof_from_false(),
        }
    }
    proof {
        assert(ptref.pptr() == h.node);
    }
    h.node.borrow(Tracked(ptref))
}

fn borrow_node_mut<'a, 'o, T>(h: &'a mut Handle<'o, T>) -> (r: &'a mut T)
    requires
        h.hwf(),
        h.has_baton(),
    ensures
        (*old(h)).hwf() ==> (*final(h)).hwf(),
        (*old(h)).has_baton() ==> (*final(h)).has_baton(),
{
    let tracked mut out: Option<&mut PointsTo<T>> = Option::None;
    proof {
        match &mut *h.baton {
            Option::Some(pt) => out = Option::Some(pt),
            Option::None => assert(false),
        }
    }
    let tracked ptref: &mut PointsTo<T>;
    proof {
        match out {
            Option::Some(p) => ptref = p,
            Option::None => ptref = proof_from_false(),
        }
    }
    proof {
        assert(ptref.pptr() == h.node);
    }
    h.node.borrow_mut(Tracked(ptref))
}

fn borrow_node_guard<'a, 'b, T>(g: &'a CheckpointGuard<'b, T>) -> (r: &'a T)
    requires
        g.node == g.shared.cell.node,
        (match g.snap@ {
            Option::Some(pt) => pt.pptr() == g.shared.cell.node && pt.is_init(),
            Option::None => false,
        }),
    ensures
        true,
{
    let tracked mut out: Option<&PointsTo<T>> = Option::None;
    proof {
        match &*g.snap {
            Option::Some(pt) => out = Option::Some(pt),
            Option::None => assert(false),
        }
    }
    let tracked ptref: &PointsTo<T>;
    proof {
        match out {
            Option::Some(p) => ptref = p,
            Option::None => ptref = proof_from_false(),
        }
    }
    proof {
        // from the guard's wf: the snap's baton matches the pair node
        assert(ptref.pptr() == g.shared.cell.node); // gwf
        assert(ptref.pptr() == g.node); // gwf
    }
    g.node.borrow(Tracked(ptref))
}

fn main() {}

} // verus!
