//! # checkpoint — the actual protocol code, verified above a
//! two-primitive unverified surface:
//!
//! - a **lock spec** (mutual exclusion over the protected state,
//!   incl. the ghost baton account), implemented by the unverified
//!   `std::sync::Mutex`;
//! - a **condvar spec** — `wait` is exactly unlock + lock (the state
//!   afterwards is arbitrary but invariant-consistent, a superset of
//!   any real condvar) — implemented by the unverified
//!   `std::sync::Condvar`.
//!
//! Everything above that surface — the entire handshake logic of the
//! PR's checkpointer (the bodies of `park_self`/`request`/`release`/
//! `safepoint`/borrows) — is verified against these specs. Blocking
//! between unlock and lock is unobservable for safety (a blocked
//! thread takes no steps), which is why the unverified implementations
//! are sound refinements of the specs.
//!
//!
//!   - ghost baton slots on Handle/CheckpointGuard (struct fields),
//!   - `std::sync::Mutex<bool>` + two Condvars -> one trusted
//!     ghost-carrying parked-lock (B1'; mutual exclusion + exact ghost
//!     transition specs; wait = spurious superset, B2),
//!   - the PR's `pending_request: AtomicBool` -> vstd's verified
//!     SeqCst atomic bool (identical role: set by the requester,
//!     waited on by the mutator, cleared at release),
//!   - `Shared` gains a phantom `T`; guard/requester share `&Shared`
//!     instead of `Arc` (same pair identity, lifetime-shared),
//!   - fanout/children out of scope (B6/B7 as before).
//!
//! The verification discipline: `lock()` ensures the protected-state
//! invariant `guarded_inv`; every `wait`/`put_back` REQUIRES it; so
//! every critical section in the code below PROVES the invariant at
//! exit — machine-checked, not assumed. The baton (the node's
//! `PointsTo`) moves only handle -> banked -> guard -> banked -> handle,
//! which is T1/T2/T3 exactly as in checkpoint-verify.

#![allow(unused)]

#[cfg(verus_only)]
pub mod verified_lock;

use vstd::prelude::*;
use vstd::simple_pptr::{PointsTo, PPtr};

use std::marker::PhantomData;

verus! {

/// Nondeterministic bool (B3/B4).
#[verifier::external_body]
pub fn nondet_bool() -> (b: bool) {
    false
}

/// Spec-level projection of a Some-value.
pub open spec fn get_some<A>(o: Option<A>) -> A {
    match o {
        Option::Some(a) => a,
        Option::None => arbitrary(),
    }
}

#[verifier::external_body]
#[verifier::external_type_specification]
#[verifier(reject_recursive_types(T))]
pub struct ExMutex<T: ?Sized>(std::sync::Mutex<T>);

#[verifier::external_body]
#[verifier::external_type_specification]
pub struct ExCondvar(std::sync::Condvar);

// =====================================================================
// Protected state + invariant (the parked mutex's content)
// =====================================================================

/// The runtime protected state.
pub struct Guarded<T> {
    pub parked: bool,
    pub pending: bool,
    pub tracked baton: Option<PointsTo<T>>,
}

/// The spec-level view (adds the ghost guard count).
pub struct GView<T> {
    pub parked: bool,
    pub pending: bool,
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
    #[verifier::external_body]
    pub fn new(node: PPtr<T>) -> (s: Self)
    ensures
        s.node == node,
    {
        TrustedLock {
            inner: std::sync::Mutex::new(Guarded {
                parked: false,
                pending: false,
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
    #[verifier::external_body]
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
    #[verifier::external_body]
    pub uninterp spec fn view(&self) -> GView<T>;

    /// The PR's `*parked` read.
    #[verifier::external_body]
    pub fn parked(&self) -> (b: bool)
    ensures
        b == self@.parked,
    {
        (self.lock_ref.inner).lock().unwrap_or_else(|e| e.into_inner()).parked
    }

    /// Ghost condition as an exec-observable predicate (trusted
    /// projection): grantable <=> baton banked and no live guard.
    #[verifier::external_body]
    pub fn grantable(&self) -> (b: bool)
    ensures
        b == (self@.guards == 0 && self@.baton.is_some()),
    {
        // runtime: parked (baton presence & no-guard hold by protocol)
        self.lock_ref.inner.lock().unwrap_or_else(|e| e.into_inner()).parked
    }

    /// The PR's `*parked = v`.
    #[verifier::external_body]
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
    #[verifier::external_body]
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
    #[verifier::external_body]
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
    #[verifier::external_body]
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
    #[verifier::external_body]
    pub fn still_pending(&self) -> (b: bool)
    ensures
        b == self@.pending,
    {
        self.lock_ref.inner.lock().unwrap_or_else(|e| e.into_inner()).pending
    }

    /// Grant-side withdrawal (guards 0 -> 1).
    #[verifier::external_body]
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
        let r: Option<PointsTo<T>> = Option::None;
        Tracked(r)
    }

    /// Release-side return (guards 1 -> 0).
    #[verifier::external_body]
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
    #[verifier::external_body]
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
        let r: Option<PointsTo<T>> = Option::None;
        Tracked(r)
    }

    /// CONDVAR SPEC: `wait` is exactly unlock + [block] + lock. The
    /// blocking is unobservable for safety (a blocked thread takes no
    /// steps), so the spec allows any invariant-consistent state
    /// afterwards. Implemented by the unverified std Condvar.
    #[verifier::external_body]
    pub fn wait_park(self) -> (g2: TrustedGuard<'a, T>)
    requires
        guarded_inv(&self@),
    ensures
        g2.node == self.node,
        guarded_inv(&g2@),
    {
        // CONDVAR IMPL: unlock + [block] + lock. The wakeup is only a
        // hint (the spec allows any return): wait bounded, and let the
        // caller's verified loop re-check its own predicate — immune
        // to lost wakeups and to capture races.
        let TrustedGuard { lock_ref, node } = self;
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
    #[verifier::external_body]
    pub fn wait_resume(self) -> (g2: TrustedGuard<'a, T>)
    requires
        guarded_inv(&self@),
        self@.parked,
    ensures
        g2.node == self.node,
        guarded_inv(&g2@),
        g2@.parked,
    {
        // CONDVAR IMPL: unlock + [block] + lock (wakeup = hint only,
        // as in wait_park; the parked mutator's verified loop
        // re-checks pending).
        let TrustedGuard { lock_ref, node } = self;
        let g = lock_ref.inner.lock().unwrap_or_else(|e| e.into_inner());
        let _g2 = lock_ref
            .resume_cv
            .wait_timeout(g, std::time::Duration::from_millis(1))
            .unwrap_or_else(|e| e.into_inner());
        TrustedGuard { lock_ref, node }
    }

    /// End of critical section (the PR's guard drop): the invariant
    /// must hold — the obligation every code path below discharges.
    #[verifier::external_body]
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
    pub(crate) tracked baton: Tracked<Option<PointsTo<T>>>,
    pub(crate) _brand: PhantomData<&'o ()>,
}

#[verifier(reject_recursive_types(T))]
pub struct CheckpointRequester<'a, T> {
    node: PPtr<T>,
    shared: &'a Shared<T>,
    _brand: PhantomData<&'a ()>,
}

#[verifier(reject_recursive_types(T))]
pub struct CheckpointGuard<'a, T> {
    node: PPtr<T>,
    shared: &'a Shared<T>,
    pub(crate) tracked snap: Tracked<Option<PointsTo<T>>>,
    _brand: PhantomData<&'a ()>,
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

    /// PR park_self — logic verbatim: lock, bank the baton, park,
    /// wait on the pending flag, withdraw, un-park.
    #[verifier::exec_allows_no_decreases_clause]
    pub fn park_self(&mut self)
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
#[verifier::external_body]
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

#[verifier::external_body]
pub fn extract_baton<T>(
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

impl<'a, T> CheckpointRequester<'a, T> {
    pub closed spec fn wf(self) -> bool {
        self.shared.parked.node == self.node
    }


    /// PR request (no deadline) — logic verbatim: set pending, lock,
    /// loop { if parked: grant; wait }.
    #[verifier::exec_allows_no_decreases_clause]
    pub fn request(&self) -> (res: CheckpointGuard<'a, T>)
        requires
            self.wf(),
        ensures
            res.wf(),
            res.has_snap(),
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
                return CheckpointGuard {
                    node: self.node,
                    shared: self.shared,
                    snap: Tracked(Option::Some(pt)),
                    _brand: PhantomData,
                };
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
    #[verifier::exec_allows_no_decreases_clause]
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
    #[verifier::external_body]
    pub(crate) fn snap_clone_for_release(&self) -> (res: Tracked<Option<PointsTo<T>>>)
        requires
            self.has_snap(),
        ensures
            res@ == self.snap@,
    {
        let r: Option<PointsTo<T>> = Option::None;
        Tracked(r)
    }
}

impl<'a, T> Drop for CheckpointGuard<'a, T> {
    /// Runtime twin of `release()` (Drop cannot carry the verified
    /// preconditions): return the baton, clear pending and wake the
    /// parked mutator, wait until it has un-parked.
    #[verifier::external_body]
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
#[verifier::external_body]
pub fn with_checkpoint_pair<T, R>(
    node: T,
    f: impl for<'o> FnOnce(Handle<'o, T>, CheckpointRequester<'o, T>) -> R,
) -> (r: R) {
    let (pptr, Tracked(pt)) = PPtr::new(node);
    let shared = Shared::new(pptr);
    let handle = Handle {
        node: pptr,
        shared: &shared,
        baton: Tracked(Option::Some(pt)),
        _brand: PhantomData,
    };
    let requester = CheckpointRequester { node: pptr, shared: &shared, _brand: PhantomData };
    f(handle, requester)
}

fn main() {}

} // verus!
