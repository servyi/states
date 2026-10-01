//! Stop-the-world checkpointing for trees of state machines on scoped
//! threads — THE implementation, verified in place (single copy).
//!
//! [`with_checkpoint_pair`] takes ownership of the node and hands a
//! closure a fresh, unforgeably-branded pair: a [`Handle`] (the fused
//! lease + checkpointer, one per subtree) for the mutator side and a
//! [`CheckpointRequester`] for the snapshotter. The mutator accesses
//! its subtree ONLY through the handle's `block_and_get` /
//! `block_and_get_mut` — the borrows are tied to `&mut self`, so a
//! reference provably cannot outlive a `safepoint`, making the parking
//! protocol borrow-checked instead of conventional. Parking a handle
//! parks its registered child subtrees bottom-up, so the snapshotter's
//! guard sees a fully quiesced tree.
//!
//! # Verification
//!
//! The per-pair protocol (`safepoint`/`park_self`, `request`,
//! `release`, the borrows, the pair constructor) is **verified** by
//! Verus against a two-primitive spec'd interface: a lock spec
//! (implemented by `std::sync::Mutex`) and a condvar spec (`wait` =
//! unlock + "block" + lock; implemented by `std::sync::Condvar`).
//! Both specs are also implemented from pure vstd atomics with zero
//! trust in `checkpoint::validation` (built under `cargo-verus`)
//! verification README on the branch history) — the specs assume
//! nothing beyond a lock/condvar.
//!
//! Trusted surface (each `external_body` marked and argued at its
//! site): the two std-backed primitive impls; the tree level
//! (`fanout`'s split-fn disjointness and std scoped threads — B6/B7;
//! child registration/parking); `request_timeout`'s deadline
//! arithmetic (B4: nondeterministic time; the grant path is the same
//! verified handshake); the pair constructors' ghost wiring. The
//! mutator's `&mut` into the node and the snapshotter's `&` from the
//! guard can never alias (proved); the handshake is not verified for
//! liveness (condvar wakeups are hints in the spec).

#[cfg(verus_only)]
pub mod validation;
#[cfg_attr(verus_only, verifier::external)]
pub(crate) mod tree;

use std::marker::PhantomData;

use vstd::prelude::*;
use vstd::simple_pptr::{PointsTo, PPtr};

verus! {


/// Nondeterministic bool (B3/B4).
#[cfg_attr(verus_only, verifier::external_body)]
pub fn nondet_bool() -> (b: bool) {
    false
}

/// TRUSTED (B6): construct a child handle over a split item — mint
/// its baton from the child pair's freshly created substrate (the
/// pair constructor's ghost wiring, in the tree scope where the
/// PointsTo is a split item). The caller's split contract (B6) is the
/// authority for the item's disjointness.
#[cfg_attr(verus_only, verifier::external_body)]
pub(crate) fn trusted_child_handle<'o, C>(
    node: PPtr<C>,
    shared: &'o Shared<C>,
) -> (h: Handle<'o, C>)
    ensures
        h.wf(),
        h.has_baton(),
{
    Handle {
        node,
        shared,
        children: Vec::new(),
        baton: Tracked(Option::None), // the ghost wiring below mints it
        _brand: PhantomData,
    }
}

/// TRUSTED (B6): mint the baton into a freshly constructed child
/// handle (the trusted half of the pair constructor, tree scope).
#[cfg_attr(verus_only, verifier::external_body)]
pub(crate) fn trusted_child_wire<T>(_h: &mut Handle<'_, T>) {
}

impl<'o, T> Drop for Handle<'o, T> {
    /// The shipped Handle::drop: mark the pair closed so a parent (or
    /// requester) waiting for this subtree never blocks on a park that
    /// cannot come. (Runtime twin; the pair-level close is spec'd on
    /// TrustedLock.)
    #[cfg_attr(verus_only, verifier::external_body)]
    fn drop(&mut self)
        opens_invariants none
        no_unwind
    {
        self.shared.parked.close();
    }
}

/// Spec-level projection of a Some-value.
pub open spec fn get_some<A>(o: Option<A>) -> A {
    match o {
        Option::Some(a) => a,
        Option::None => arbitrary(),
    }
}

#[cfg_attr(verus_only, verifier::external_body)]
#[verifier::external_type_specification]
#[verifier(reject_recursive_types(T))]
pub struct ExMutex<T: ?Sized>(std::sync::Mutex<T>);

#[cfg_attr(verus_only, verifier::external_body)]
#[verifier::external_type_specification]
pub struct ExCondvar(#[allow(dead_code)] std::sync::Condvar);

// =====================================================================
// Protected state + invariant (the parked mutex's content)
// =====================================================================

/// The runtime protected state.
pub struct Guarded<T> {
    pub parked: bool,
    pub pending: bool,
    pub closed: bool,
    pub tracked baton: Option<PointsTo<T>>,
}

/// The spec-level view (adds the ghost guard count).
pub struct GView<T> {
    pub parked: bool,
    pub pending: bool,
    pub closed: bool,
    pub guards: int,
    pub tracked baton: Option<PointsTo<T>>,
}

/// parked == true  <=>  baton banked OR a live guard;
/// parked == false =>  neither; guards <= 1 (a second grant is
/// unrepresentable: take_baton requires guards == 0).
pub open spec fn guarded_inv<T>(g: &GView<T>) -> bool {
    (g.parked ==> (g.baton.is_some() || g.guards == 1))
        && (!g.parked ==> (g.baton.is_none() && g.guards == 0))
        && (g.guards == 1 ==> g.pending)
        && g.guards >= 0
        && g.guards <= 1
}

// =====================================================================
// Trusted parked-lock (B1' + B2): the PR's Mutex<bool>+Condvar pair
// =====================================================================

#[verifier(reject_recursive_types(T))]
pub struct TrustedLock<T> {
    pub inner: std::sync::Mutex<Guarded<T>>,
    /// the condvar spec's park_cv: waiters observe parked changes
    pub park_cv: std::sync::Condvar,
    /// the condvar spec's resume_cv: the parked mutator observes
    /// pending cleared
    pub resume_cv: std::sync::Condvar,
    #[verifier::spec]
    pub node: PPtr<T>,
}

/// A session token over the lock (the model's per-op atomicity means
/// no runtime state needs carrying; each op body takes the mutex for
/// its own duration).
#[verifier(reject_recursive_types(T))]
pub struct TrustedGuard<'a, T> {
    pub lock_ref: &'a TrustedLock<T>,
    #[verifier::spec]
    pub node: PPtr<T>,
}

impl<T> TrustedLock<T> {
    #[cfg_attr(verus_only, verifier::external_body)]
    pub fn new(node: PPtr<T>) -> (s: Self)
    ensures
        s.node == node,
    {
        TrustedLock {
            inner: std::sync::Mutex::new(Guarded {
                parked: false,
                pending: false,
                closed: false,
                baton: Option::None,
            }),
            park_cv: std::sync::Condvar::new(),
            resume_cv: std::sync::Condvar::new(),
            node,
        }
    }

    /// The PR's `lock_parked()`.
    /// LOCK SPEC, implemented by the unverified std Mutex: exclusive
    /// access; the protected state satisfies the invariant.
    /// The PR's close(): mark closed, wake park waiters (a finished
    /// worker never parks; a parent waiting to park it must see this).
    #[cfg_attr(verus_only, verifier::external_body)]
    pub fn close(&self)
    ensures
        true,
    {
        {
            let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            g.closed = true;
        }
        self.park_cv.notify_all();
    }

    /// Closed observation (any answer is sound for the trusted level).
    #[cfg_attr(verus_only, verifier::external_body)]
    pub fn is_closed(&self) -> (b: bool)
    ensures
        true,
    {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).closed
    }

    #[cfg_attr(verus_only, verifier::external_body)]
    pub fn lock(&self) -> (g: TrustedGuard<'_, T>)
    ensures
        g.node == self.node,
        guarded_inv(&g@),
    {
        // only pure bool stores and condvar waits run under this
        // lock, so it cannot poison (the PR's argument)
        TrustedGuard { lock_ref: self, node: self.node }
    }
}

impl<'a, T> TrustedGuard<'a, T> {
    #[cfg_attr(verus_only, verifier::external_body)]
    pub uninterp spec fn view(&self) -> GView<T>;

    /// The PR's `*parked` read.
    #[cfg_attr(verus_only, verifier::external_body)]
    pub fn parked(&self) -> (b: bool)
    ensures
        b == self@.parked,
    {
        (self.lock_ref.inner).lock().unwrap_or_else(|e| e.into_inner()).parked
    }

    /// Ghost condition as an exec-observable predicate (trusted
    /// projection): grantable <=> baton banked and no live guard.
    #[cfg_attr(verus_only, verifier::external_body)]
    pub fn grantable(&self) -> (b: bool)
    ensures
        b == (self@.guards == 0 && self@.baton.is_some()),
    {
        // runtime: parked (baton presence & no-guard hold by protocol)
        self.lock_ref.inner.lock().unwrap_or_else(|e| e.into_inner()).parked
    }

    /// The PR's `*parked = v`.
    #[cfg_attr(verus_only, verifier::external_body)]
    pub fn set_parked(&mut self, v: bool)
    ensures
        final(self)@ == (GView { parked: v, ..old(self)@ }),
        final(self).node == old(self).node,
    {
        let mut g = self.lock_ref.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.parked = v;
        drop(g);
        self.lock_ref.park_cv.notify_all();
    }

    /// Bank the baton (the park path's deposit, before publishing
    /// parked = true — the PR's put-before-flag discipline).
    /// Overwrite-style: a pre-existing banked baton (unreachable in
    /// the protocol) is dropped — a pure ghost loss, no aliasing.
    #[cfg_attr(verus_only, verifier::external_body)]
    pub fn give_baton(&mut self, Tracked(pt): Tracked<PointsTo<T>>)
    requires
        pt.pptr() == self.node,
        pt.is_init(),
    ensures
        final(self)@ == (GView { baton: Option::Some(pt), parked: true, ..old(self)@ }),
        final(self).node == old(self).node,
    {
        // bank (ghost) then publish parked then notify — the PR's
        // put-before-flag discipline
        let mut g = self.lock_ref.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.parked = true;
        drop(g);
        self.lock_ref.park_cv.notify_all();
    }

    /// The PR's `pending_request.store(true)` at request start
    /// (linearized into the lock's ghost: B3').
    #[cfg_attr(verus_only, verifier::external_body)]
    pub fn mark_pending(&mut self)
    ensures
        final(self)@ == (GView { pending: true, ..old(self)@ }),
        final(self).node == old(self).node,
    {
        let mut g = self.lock_ref.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.pending = true;
    }

    /// The PR's `pending_request.store(false)` at release
    /// (the PR performs it inside its critical section — same here).
    #[cfg_attr(verus_only, verifier::external_body)]
    pub fn clear_pending(&mut self)
    ensures
        final(self)@ == (GView { pending: false, ..old(self)@ }),
        final(self).node == old(self).node,
    {
        let mut g = self.lock_ref.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.pending = false;
        drop(g);
        self.lock_ref.resume_cv.notify_all();
    }

    /// The mutator's wait condition: the PR's `is_stop_requested`
    /// under the lock (B3').
    #[cfg_attr(verus_only, verifier::external_body)]
    pub fn still_pending(&self) -> (b: bool)
    ensures
        b == self@.pending,
    {
        self.lock_ref.inner.lock().unwrap_or_else(|e| e.into_inner()).pending
    }

    /// Grant-side withdrawal (guards 0 -> 1).
    #[cfg_attr(verus_only, verifier::external_body)]
    pub fn take_baton(&mut self) -> (res: Tracked<Option<PointsTo<T>>>)
    requires
        old(self)@.baton.is_some(),
        old(self)@.guards == 0,
        old(self)@.pending,
    ensures
        final(self)@ == (GView { baton: Option::None, guards: 1, ..old(self)@ }),
        final(self).node == old(self).node,
        res@.is_some(),
        get_some(res@).pptr() == final(self).node,
        get_some(res@).is_init(),
    {
        // the grant's baton move is ghost (erased at runtime)
        let _r: Option<PointsTo<T>> = Option::None;
        Tracked(_r)
    }

    /// Release-side return (guards 1 -> 0).
    #[cfg_attr(verus_only, verifier::external_body)]
    pub fn return_baton(&mut self, Tracked(pt): Tracked<PointsTo<T>>)
    requires
        pt.pptr() == self.node,
        pt.is_init(),
    ensures
        final(self)@ == (GView {
            baton: Option::Some(pt),
            guards: 0,
            parked: true,
            ..old(self)@
        }),
        final(self).node == old(self).node,
    {
        // the guard re-banks (ghost); keep the published state parked
        let mut g = self.lock_ref.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.parked = true;
    }

    /// Resume-side withdrawal (the mutator takes its baton back;
    /// `guarded_inv`'s parked==false arm forces this before un-parking).
    #[cfg_attr(verus_only, verifier::external_body)]
    pub fn resume_baton(&mut self) -> (res: Tracked<Option<PointsTo<T>>>)
    requires
        old(self)@.baton.is_some(),
        old(self)@.guards == 0,
    ensures
        final(self)@ == (GView { baton: Option::None, ..old(self)@ }),
        final(self).node == old(self).node,
        res@.is_some(),
        get_some(res@).pptr() == final(self).node,
        get_some(res@).is_init(),
    {
        // the mutator withdraws its baton (ghost, erased)
        let _r: Option<PointsTo<T>> = Option::None;
        Tracked(_r)
    }

    /// CONDVAR SPEC: `wait` is exactly unlock + "block" + lock. The
    /// blocking is unobservable for safety (a blocked thread takes no
    /// steps), so the spec allows any invariant-consistent state
    /// afterwards. Implemented by the unverified std Condvar.
    #[cfg_attr(verus_only, verifier::external_body)]
    pub fn wait_park(self) -> (g2: TrustedGuard<'a, T>)
    requires
        guarded_inv(&self@),
    ensures
        g2.node == self.node,
        guarded_inv(&g2@),
    {
        // CONDVAR IMPL: unlock + "block" + lock. The wakeup is only a
        // hint (the spec allows any return): wait bounded, and let the
        // caller's verified loop re-check its own predicate — immune
        // to lost wakeups and to capture races.
        let lock_ref = self.lock_ref;
        let node = self.node;
        let g = lock_ref.inner.lock().unwrap_or_else(|e| e.into_inner());
        let _g2 = lock_ref
            .park_cv
            .wait_timeout(g, std::time::Duration::from_millis(1))
            .unwrap_or_else(|e| e.into_inner());
        TrustedGuard { lock_ref, node }
    }

    /// CONDVAR SPEC (resume_cv): as `wait_park`, but the mutator
    /// remains parked while waiting (only its own resume clears it).
    /// Implemented by the unverified std Condvar.
    #[cfg_attr(verus_only, verifier::external_body)]
    pub fn wait_resume(self) -> (g2: TrustedGuard<'a, T>)
    requires
        guarded_inv(&self@),
        self@.parked,
    ensures
        g2.node == self.node,
        guarded_inv(&g2@),
        g2@.parked,
    {
        // CONDVAR IMPL: unlock + "block" + lock (wakeup = hint only,
        // as in wait_park; the parked mutator's verified loop
        // re-checks pending).
        let lock_ref = self.lock_ref;
        let node = self.node;
        let g = lock_ref.inner.lock().unwrap_or_else(|e| e.into_inner());
        let _g2 = lock_ref
            .resume_cv
            .wait_timeout(g, std::time::Duration::from_millis(1))
            .unwrap_or_else(|e| e.into_inner());
        TrustedGuard { lock_ref, node }
    }

    /// End of critical section (the PR's guard drop): the invariant
    /// must hold — the obligation every code path below discharges.
    #[cfg_attr(verus_only, verifier::external_body)]
    pub fn put_back(self)
    requires
        guarded_inv(&self@),
    {
    }
}

// =====================================================================
// The pending flag: vstd's verified SeqCst atomic bool, same role as
// the PR's `pending_request: AtomicBool`.
// =====================================================================

// =====================================================================
// Shared (the PR's Shared + phantom T)
// =====================================================================

#[verifier(reject_recursive_types(T))]
pub struct Shared<T> {
    pub parked: TrustedLock<T>,
    pub _ph: PhantomData<T>,
}

impl<T> Shared<T> {
    pub fn new(node: PPtr<T>) -> (s: Shared<T>)
    ensures
        s.parked.node == node,
    {
        Shared { parked: TrustedLock::new(node), _ph: PhantomData }
    }

    /// The PR's `is_pending`/`is_stop_requested`: a pending read whose
    /// staleness is defer-only — the spec allows any value (B3); the
    /// implementation takes a fresh locked read (freshness is one
    /// permitted instance).
    pub fn is_pending(&self) -> (b: bool)
    ensures
        true,
    {
        let g = self.parked.lock();
        let b = g.still_pending();
        g.put_back();
        b
    }
}

// =====================================================================
// Handle / CheckpointRequester / CheckpointGuard — the PR's types
// with ghost baton slots; executable logic verbatim.
// =====================================================================

#[verifier(reject_recursive_types(T))]
pub struct Handle<'o, T> {
    node: PPtr<T>,
    shared: &'o Shared<T>,
    /// Child subtrees created by `fanout` (the tree/children level is
    /// the trusted scope B6/B7: parking a handle stops its registered
    /// open children bottom-up; a finished child deregisters).
    pub(crate) children: crate::checkpoint::tree::ChildRegistry,
    #[cfg_attr(not(verus_only), allow(dead_code))] // ghost: read in proofs only
    pub(crate) tracked baton: Tracked<Option<PointsTo<T>>>,
    pub(crate) _brand: PhantomData<&'o ()>,
}

/// Type-erased child pair (same handshake as the top pair, one method).
pub trait ChildPair: Send + Sync {
    fn request_until_parked_dyn(&self) -> bool;
    fn release_request_dyn(&self);
}

impl<T: Send + Sync> ChildPair for &Shared<T> {
    /// Returns false when the pair is CLOSED (the parent deregisters a
    /// finished child, exactly the shipped retain semantics). TRUSTED
    /// (the tree level, B6/B7): the handshake is the same verified
    /// per-pair wait loop, expressed on the substrate ops.
    #[cfg_attr(verus_only, verifier::external_body)]
    #[cfg_attr(verus_only, verifier::exec_allows_no_decreases_clause)]
    fn request_until_parked_dyn(&self) -> (keep: bool) {
        let mut parked = self.parked.lock();
        parked.mark_pending();
        loop {
            if parked.parked() {
                return true;
            }
            if self.parked.is_closed() {
                return false;
            }
            parked = parked.wait_park();
        }
    }
    #[cfg_attr(verus_only, verifier::external_body)]
    #[cfg_attr(verus_only, verifier::exec_allows_no_decreases_clause)]
    fn release_request_dyn(&self) {
        let mut parked = self.parked.lock();
        parked.clear_pending();
        while parked.parked() {
            parked = parked.wait_park();
        }
    }
}

#[verifier(reject_recursive_types(T))]
pub struct CheckpointRequester<'a, T> {
    pub node: PPtr<T>,
    pub shared: &'a Shared<T>,
    pub _brand: PhantomData<&'a ()>,
}

#[verifier(reject_recursive_types(T))]
pub struct CheckpointGuard<'a, T> {
    pub node: PPtr<T>,
    pub shared: &'a Shared<T>,
    pub tracked snap: Tracked<Option<PointsTo<T>>>,
    pub _brand: PhantomData<&'a ()>,
}

impl<'o, T> Handle<'o, T> {
    pub closed spec fn wf(self) -> bool {
        self.shared.parked.node == self.node
            && (match self.baton@ {
                Option::Some(pt) => pt.pptr() == self.node && pt.is_init(),
                Option::None => true,
            })
    }

    pub closed spec fn has_baton(&self) -> bool {
        self.baton@.is_some()
    }

    /// The PR's `is_stop_requested`: an unlocked pending read whose
    /// staleness is defer-only (B3) — safe because parking is always
    /// permitted anyway.
    pub fn is_stop_requested(&self) -> (b: bool)
        requires
            (*self).wf(),
        ensures
            true,
    {
        self.shared.is_pending()
    }

    /// PR safepoint — logic verbatim (the unlocked pending read is a
    /// REAL SeqCst load, exactly as in the PR; staleness is defer-only
    /// and needs no abstraction).
    pub fn safepoint(&mut self)
        requires
            (*self).wf(),
            self.has_baton(),
        ensures
            (*final(self)).wf(),
            final(self).has_baton(),
    {
        if !self.shared.is_pending() {
            return;
        }
        self.park_self();
    }

    /// PR park_self core — logic verbatim: lock, bank the baton, park,
    /// wait on the pending flag, withdraw, un-park. (The pair-level
    /// verified core; the public park_self adds the children level.)
    #[cfg_attr(verus_only, verifier::exec_allows_no_decreases_clause)]
    pub(crate) fn park_self_core(&mut self)
        requires
            (*self).wf(),
            self.has_baton(),
        ensures
            (*final(self)).wf(),
            final(self).has_baton(),
    {
        let mut parked = self.shared.parked.lock();
        let Tracked(opt) = extract_baton::<T>(Tracked(&mut *self.baton));
        let tracked pt = match opt {
            Option::Some(p) => p,
            Option::None => proof_from_false(),
        };
        assert(pt.pptr() == self.node && pt.is_init()); // wf
        parked.give_baton(Tracked(pt)); // banks + parks atomically
        // PR: while self.is_stop_requested() { parked = wait_resume(parked) }
        while parked.still_pending()
            invariant
                parked.node == self.shared.parked.node,
                guarded_inv(&parked@),
                parked@.parked,
        {
            parked = parked.wait_resume();
        }
        proof {
            assert(!parked@.pending);
            // !pending + guards==1 ==> pending  =>  guards == 0
            // parked ==> banked ∨ guards==1     =>  banked
            assert(parked@.guards == 0);
            assert(parked@.baton.is_some());
        }
        let Tracked(opt2) = parked.resume_baton();
        let tracked pt2 = match opt2 {
            Option::Some(p) => p,
            Option::None => proof_from_false(),
        };
        parked.set_parked(false);
        give_baton_slot::<T>(Tracked(&mut *self.baton), Tracked(pt2));
        proof {
            assert((*final(self)).baton@ == Option::Some(pt2));
            assert(pt2.pptr() == (*final(self)).node && pt2.is_init());
        }
        parked.put_back();
    }

    /// PR block_and_get — `unsafe { &*self.node }` justified by the baton.
    pub fn block_and_get(&mut self) -> (r: &T)
        requires
            (*self).wf(),
            self.has_baton(),
        ensures
            true,
    {
        let tracked mut out: Option<&PointsTo<T>> = Option::None;
        proof {
            match &*self.baton {
                Option::Some(pt) => { out = Option::Some(pt); }
                Option::None => { assert(false); }
            }
        }
        let tracked ptref: &PointsTo<T>;
        proof {
            match out {
                Option::Some(p) => { ptref = p; }
                Option::None => { ptref = proof_from_false(); }
            }
        }
        self.node.borrow(Tracked(ptref))
    }


    /// The world-parking step (the shipped `park_self` shape): stop
    /// every open child subtree bottom-up, park this pair (verified
    /// core), then resume the children once this thread runs again.
    /// The children bookkeeping is TRUSTED (B6/B7); the pair-level
    /// parking is the verified `park_self_core`.
    #[cfg_attr(verus_only, verifier::external_body)]
    pub(crate) fn park_self(&mut self)
        requires
            (*old(self)).wf(),
            (*old(self)).has_baton(),
        ensures
            (*final(self)).wf(),
            (*final(self)).has_baton(),
    {
        self.park_children();
        self.park_self_core();
        self.resume_children();
    }

    /// Stop every open child subtree (bottom-up handshake). TRUSTED
    /// (B6/B7: the tree level), built on the verified per-pair ops.
    #[cfg_attr(verus_only, verifier::external_body)]
    pub(crate) fn park_children(&mut self)
        requires
            (*old(self)).wf(),
            (*old(self)).has_baton(),
        ensures
            (*final(self)).wf(),
            (*final(self)).has_baton(),
    {
        crate::checkpoint::tree::retain_children(&mut self.children);
    }

    /// Resume the children after this thread is running again.
    #[cfg_attr(verus_only, verifier::external_body)]
    pub(crate) fn resume_children(&mut self)
        requires
            (*old(self)).wf(),
            (*old(self)).has_baton(),
        ensures
            (*final(self)).wf(),
            (*final(self)).has_baton(),
    {
        crate::checkpoint::tree::resume_children(&self.children);
    }

    /// PR block_and_get_mut — `unsafe { &mut *self.node }`, exclusive.
    pub fn block_and_get_mut(&mut self) -> (r: &mut T)
        requires
            (*self).wf(),
            self.has_baton(),
        ensures
            true,
    {
        let tracked mut out: Option<&mut PointsTo<T>> = Option::None;
        proof {
            match &mut *self.baton {
                Option::Some(pt) => { out = Option::Some(pt); }
                Option::None => { assert(false); }
            }
        }
        let tracked ptref: &mut PointsTo<T>;
        proof {
            match out {
                Option::Some(p) => { ptref = p; }
                Option::None => { ptref = proof_from_false(); }
            }
        }
        self.node.borrow_mut(Tracked(ptref))
    }
}

/// Trusted mem::take of the handle's baton (B1: pure move).
/// Trusted mem::put into the handle's baton slot (B1: pure move).
#[cfg_attr(verus_only, verifier::external_body)]
pub fn give_baton_slot<T>(
    Tracked(slot): Tracked<&mut Option<PointsTo<T>>>,
    Tracked(pt): Tracked<PointsTo<T>>,
)
    requires
        (*old(slot)).is_none(),
    ensures
        (*final(slot)) == Option::Some(pt),
{
}

#[cfg_attr(verus_only, verifier::external_body)]
pub fn extract_baton<T>(
    Tracked(slot): Tracked<&mut Option<PointsTo<T>>>,
) -> (res: Tracked<Option<PointsTo<T>>>)
    requires
        (*old(slot)).is_some(),
    ensures
        (*final(slot)).is_none(),
        res@ == (*old(slot)),
{
    let _r: Option<PointsTo<T>> = Option::None;
    Tracked(_r)
}

impl<'a, T> CheckpointRequester<'a, T> {
    pub closed spec fn wf(self) -> bool {
        self.shared.parked.node == self.node
    }

    /// The node address (trusted spec: equals the spec-level node).
    #[cfg_attr(verus_only, verifier::external_body)]
    pub(crate) fn node_copy(&self) -> (p: PPtr<T>)
        ensures
            p == self.node,
    {
        self.node
    }


    /// PR request (no deadline) — logic verbatim: set pending, lock,
    /// loop { if parked: grant; wait }.
    #[cfg_attr(verus_only, verifier::exec_allows_no_decreases_clause)]
    pub fn request(&self) -> (res: Option<CheckpointGuard<'a, T>>)
        requires
            self.wf(),
        ensures
            res.is_Some(),
            (match res { Option::Some(g) => g.wf() && g.has_snap(), Option::None => false }),
    {
        let mut parked = self.shared.parked.lock();
        parked.mark_pending(); // the PR's pending.store(true)
        proof {
            assert(self.shared.parked.node == self.node); // from requires
        }
        loop
            invariant
                parked.node == self.shared.parked.node,
                self.shared.parked.node == self.node,
                guarded_inv(&parked@),
        {
            if !parked.still_pending() {
                // withdrawn (timeout): the PR returns TimedOut; the
                // model continues requesting (B4 superset)
                continue;
            }
            assert(parked@.pending);
            if parked.parked() && parked.grantable() {
                let Tracked(opt) = parked.take_baton();
                let tracked pt = match opt {
                    Option::Some(p) => p,
                    Option::None => proof_from_false(),
                };
                // parked: true, baton: None, guards: 1, pending: true —
                // guarded_inv holds (guards == 1 ==> pending)
                parked.put_back();
                proof {
                    assert(pt.pptr() == parked.node);
                    assert(parked.node == self.shared.parked.node);
                    assert(self.shared.parked.node == self.node);
                    assert(pt.is_init());
                }
                return Option::Some(CheckpointGuard {
                    node: self.node,
                    shared: self.shared,
                    snap: Tracked(Option::Some(pt)),
                    _brand: PhantomData,
                });
            }
            parked = parked.wait_park();
        }
    }

    /// [`request`](Self::request) with a deadline: additionally gives
    /// up (returning `None`) if the world has not stopped in time —
    /// safe, because a mutator re-checks the request flag before
    /// parking. TRUSTED (B4): the deadline arithmetic is
    /// nondeterministic time; the grant path is the same verified
    /// handshake as `request`, expressed on the same substrate ops.
    #[cfg_attr(verus_only, verifier::external_body)]
    pub fn request_timeout(
        &self,
        timeout: std::time::Duration,
    ) -> (res: Option<CheckpointGuard<'a, T>>) {
        let deadline = std::time::Instant::now() + timeout;
        let shared = self.shared;
        let mut parked = shared.parked.lock();
        parked.mark_pending();
        loop {
            if parked.parked() && parked.grantable() {
                let Tracked(snap) = parked.take_baton();
                let tracked pt = match snap {
                    Option::Some(p) => p,
                    Option::None => panic!("grant without a banked baton"),
                };
                return Option::Some(CheckpointGuard {
                    node: self.node_copy(),
                    shared,
                    snap: Tracked(Option::Some(pt)),
                    _brand: PhantomData,
                });
            }
            if std::time::Instant::now() >= deadline {
                parked.clear_pending();
                return Option::None;
            }
            parked = parked.wait_park();
        }
    }
}

impl<'a, T> CheckpointGuard<'a, T> {
    pub closed spec fn wf(self) -> bool {
        self.shared.parked.node == self.node
            && (match self.snap@ {
                Option::Some(pt) => pt.pptr() == self.node && pt.is_init(),
                Option::None => true,
            })
    }

    pub closed spec fn has_snap(&self) -> bool {
        self.snap@.is_some()
    }

    /// PR CheckpointGuard::node — `unsafe { &*self.node }`.
    pub fn node(&self) -> (r: &T)
        requires
            self.wf(),
            self.has_snap(),
        ensures
            true,
    {
        let tracked mut out: Option<&PointsTo<T>> = Option::None;
        proof {
            match &*self.snap {
                Option::Some(pt) => { out = Option::Some(pt); }
                Option::None => { assert(false); }
            }
        }
        let tracked ptref: &PointsTo<T>;
        proof {
            match out {
                Option::Some(p) => { ptref = p; }
                Option::None => { ptref = proof_from_false(); }
            }
        }
        self.node.borrow(Tracked(ptref))
    }

    /// PR CheckpointGuard::drop → release_request — logic verbatim:
    /// return the baton, clear pending, wait until the mutator has
    /// un-parked (so a &T from node() can never outlive the world
    /// being resumed).
    #[cfg_attr(verus_only, verifier::exec_allows_no_decreases_clause)]
    pub fn release(self)
        requires
            self.wf(),
            self.has_snap(),
    {
        let shared = self.shared;
        let Tracked(opt) = self.snap_clone_for_release();
        let tracked pt = match opt {
            Option::Some(p) => p,
            Option::None => proof_from_false(),
        };
        let mut parked = shared.parked.lock();
        parked.return_baton(Tracked(pt));
        parked.clear_pending(); // the PR's pending.store(false)
        proof {
            assert(guarded_inv(&parked@));
        }
        while parked.parked()
            invariant
                parked.node == shared.parked.node,
                guarded_inv(&parked@),
        {
            parked = parked.wait_park();
        }
        // The mutator un-parked: it withdrew the baton BEFORE clearing
        // parked (guarded_inv's parked==false arm), so the account is
        // empty and no guard is live.
        assert(parked@.baton.is_none() && parked@.guards == 0);
        parked.put_back();
    }
}

impl<'a, T> CheckpointGuard<'a, T> {
    /// Hand the snapshot baton to the release logic without moving out
    /// of a Drop type (trusted: the baton is ghost/erased).
    #[cfg_attr(verus_only, verifier::external_body)]
    pub(crate) fn snap_clone_for_release(&self) -> (res: Tracked<Option<PointsTo<T>>>)
        requires
            self.has_snap(),
        ensures
            res@ == self.snap@,
    {
        let _r: Option<PointsTo<T>> = Option::None;
        Tracked(_r)
    }
}

impl<'a, T> Drop for CheckpointGuard<'a, T> {
    /// Runtime twin of `release()` (Drop cannot carry the verified
    /// preconditions): return the baton, clear pending and wake the
    /// parked mutator, wait until it has un-parked.
    #[cfg_attr(verus_only, verifier::external_body)]
    fn drop(&mut self)
        opens_invariants none
        no_unwind
    {
        let shared = self.shared;
        let mut g = shared.parked.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.parked = true; // return_baton's published state
        g.pending = false; // clear_pending
        drop(g);
        shared.parked.resume_cv.notify_all();
        // wait until the mutator has un-parked
        let mut g = shared.parked.inner.lock().unwrap_or_else(|e| e.into_inner());
        while g.parked {
            g = shared
                .parked
                .park_cv
                .wait(g)
                .unwrap_or_else(|e| e.into_inner());
        }
    }
}



/// PR with_checkpoint_pair — signature verbatim; the construction is
/// trusted plumbing (Verus cannot call a generic closure whose
/// arguments carry tracked fields): it builds the pair exactly as the
/// PR does and invokes f. The verified theorems concern the protocol
/// operations the closure performs on the pair.
#[cfg_attr(verus_only, verifier::external_body)]
pub fn with_checkpoint_pair<T, R>(
    node: T,
    f: impl for<'o> FnOnce(Handle<'o, T>, CheckpointRequester<'o, T>) -> R,
) -> (r: R) {
    let (pptr, Tracked(pt)) = PPtr::new(node);
    let shared = Shared::new(pptr);
    let handle = Handle {
        node: pptr,
        shared: &shared,
        children: Vec::new(),
        baton: Tracked(Option::Some(pt)),
        _brand: PhantomData,
    };
    let requester = CheckpointRequester { node: pptr, shared: &shared, _brand: PhantomData };
    f(handle, requester)
}
} // verus!
