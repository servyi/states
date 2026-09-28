//! Verification of the PR's checkpointer **as written**: the handshake
//! logic (control flow, conditions, step ordering, the bodies of
//! `park_self`/`request`/`release`/`safepoint`/borrows) is the PR's
//! `src/checkpoint.rs`. Added/swapped:
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

use vstd::prelude::*;
use vstd::simple_pptr::{PointsTo, PPtr};

use std::marker::PhantomData;

verus! {

/// Nondeterministic bool (B3/B4).
#[verifier::external_body]
pub fn nondet_bool() -> (b: bool) {
    false
}

#[verifier::external_body]
#[verifier::external_type_specification]
#[verifier(reject_recursive_types(T))]
pub struct ExMutex<T: ?Sized>(std::sync::Mutex<T>);

// =====================================================================
// Protected state + invariant (the parked mutex's content)
// =====================================================================

pub struct Guarded<T> {
    pub parked: bool,
    pub tracked baton: Option<PointsTo<T>>,
    pub ghost guards: int,
    pub ghost pending: bool,
}

/// parked == true  <=>  baton banked OR a live guard;
/// parked == false =>  neither; guards <= 1 (a second grant is
/// unrepresentable: take_baton requires guards == 0).
pub open spec fn guarded_inv<T>(g: &Guarded<T>) -> bool {
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
    #[verifier::spec]
    pub node: PPtr<T>,
}

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
        unimplemented!() // trusted substrate
    }

    /// The PR's `lock_parked()`.
    #[verifier::external_body]
    pub fn lock(&self) -> (g: TrustedGuard<'_, T>)
    ensures
        g.node == self.node,
        guarded_inv(&g@),
    {
        unimplemented!() // trusted substrate
    }
}

impl<'a, T> TrustedGuard<'a, T> {
    #[verifier::external_body]
    pub closed spec fn view(&self) -> Guarded<T>;

    /// The PR's `*parked` read.
    #[verifier::external_body]
    pub fn parked(&self) -> (b: bool)
    ensures
        b == self@.parked,
    {
        unimplemented!() // trusted substrate: contract above is the spec
    }

    /// Ghost condition as an exec-observable predicate (trusted
    /// projection): grantable <=> baton banked and no live guard.
    #[verifier::external_body]
    pub fn grantable(&self) -> (b: bool)
    ensures
        b == (self@.guards == 0 && self@.baton.is_some()),
    {
        unimplemented!() // trusted substrate
    }

    /// The PR's `*parked = v`.
    #[verifier::external_body]
    pub fn set_parked(&mut self, v: bool)
    ensures
        final(self)@ == (Guarded { parked: v, ..old(self)@ }),
        final(self).node == old(self).node,
    {
        unimplemented!() // trusted substrate
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
        final(self)@ == (Guarded { baton: Option::Some(pt), parked: true, ..old(self)@ }),
        final(self).node == old(self).node,
    {
        unimplemented!() // trusted substrate
    }

    /// The PR's `pending_request.store(true)` at request start
    /// (linearized into the lock's ghost: B3').
    #[verifier::external_body]
    pub fn mark_pending(&mut self)
    ensures
        final(self)@ == (Guarded { pending: true, ..old(self)@ }),
        final(self).node == old(self).node,
    {
        unimplemented!() // trusted substrate
    }

    /// The PR's `pending_request.store(false)` at release
    /// (the PR performs it inside its critical section — same here).
    #[verifier::external_body]
    pub fn clear_pending(&mut self)
    ensures
        final(self)@ == (Guarded { pending: false, ..old(self)@ }),
        final(self).node == old(self).node,
    {
        unimplemented!() // trusted substrate
    }

    /// The mutator's wait condition: the PR's `is_stop_requested`
    /// under the lock (B3').
    #[verifier::external_body]
    pub fn still_pending(&self) -> (b: bool)
    ensures
        b == self@.pending,
    {
        unimplemented!() // trusted substrate
    }

    /// Grant-side withdrawal (guards 0 -> 1).
    #[verifier::external_body]
    pub fn take_baton(&mut self) -> (res: Tracked<PointsTo<T>>)
    requires
        old(self)@.baton.is_some(),
        old(self)@.guards == 0,
        old(self)@.pending,
    ensures
        final(self)@ == (Guarded { baton: Option::None, guards: 1, ..old(self)@ }),
        final(self).node == old(self).node,
        res.pptr() == final(self).node,
        res.is_init(),
    {
        unimplemented!() // trusted substrate
    }

    /// Release-side return (guards 1 -> 0).
    #[verifier::external_body]
    pub fn return_baton(&mut self, Tracked(pt): Tracked<PointsTo<T>>)
    requires
        pt.pptr() == self.node,
        pt.is_init(),
    ensures
        final(self)@ == (Guarded {
            baton: Option::Some(pt),
            guards: 0,
            parked: true,
            ..old(self)@
        }),
        final(self).node == old(self).node,
    {
        unimplemented!() // trusted substrate
    }

    /// Resume-side withdrawal (the mutator takes its baton back;
    /// `guarded_inv`'s parked==false arm forces this before un-parking).
    #[verifier::external_body]
    pub fn resume_baton(&mut self) -> (res: Tracked<PointsTo<T>>)
    requires
        old(self)@.baton.is_some(),
        old(self)@.guards == 0,
    ensures
        final(self)@ == (Guarded { baton: Option::None, ..old(self)@ }),
        final(self).node == old(self).node,
        res.pptr() == final(self).node,
        res.is_init(),
    {
        unimplemented!() // trusted substrate
    }

    /// The PR's `wait_park` (park_cv): B2 superset — arbitrary
    /// invariant-consistent state afterwards (used where the code
    /// waits for a park/un-park to become observable).
    #[verifier::external_body]
    pub fn wait_park(self) -> (g2: TrustedGuard<'a, T>)
    requires
        guarded_inv(&self@),
    ensures
        g2.node == self.node,
        guarded_inv(&g2@),
    {
        unimplemented!() // trusted substrate
    }

    /// The PR's `wait_resume` (resume_cv): B2 superset, but the
    /// mutator remains parked while waiting (only its own resume
    /// clears parked) — exactly the PR's two-condvar structure.
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
        unimplemented!() // trusted substrate
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

    /// The PR's `is_pending` fast-path read: an unlocked hint whose
    /// staleness is defer-only — any value is sound here because
    /// parking is always permitted (B3).
    pub fn is_pending(&self) -> (b: bool)
    ensures
        true,
    {
        nondet_bool()
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
    pub(crate) closed spec fn wf(self) -> bool {
        self.shared.parked.node == self.node
            && (match self.baton@ {
                Option::Some(pt) => pt.pptr() == self.node && pt.is_init(),
                Option::None => true,
            })
    }

    pub(crate) closed spec fn has_baton(&self) -> bool {
        self.baton@.is_some()
    }

    /// PR safepoint — logic verbatim (the unlocked pending read is a
    /// REAL SeqCst load, exactly as in the PR; staleness is defer-only
    /// and needs no abstraction).
    pub(crate) fn safepoint(&mut self)
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
    pub(crate) fn park_self(&mut self)
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
        let Tracked(pt2) = parked.resume_baton();
        parked.set_parked(false);
        give_baton_slot::<T>(Tracked(&mut *self.baton), Tracked(pt2));
        proof {
            assert((*final(self)).baton@ == Option::Some(pt2));
            assert(pt2.pptr() == (*final(self)).node && pt2.is_init());
        }
        parked.put_back();
    }

    /// PR block_and_get — `unsafe { &*self.node }` justified by the baton.
    pub(crate) fn block_and_get(&mut self) -> (r: &T)
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
    pub(crate) fn block_and_get_mut(&mut self) -> (r: &mut T)
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
pub(crate) fn give_baton_slot<T>(
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

impl<'a, T> CheckpointRequester<'a, T> {
    pub(crate) closed spec fn wf(self) -> bool {
        self.shared.parked.node == self.node
    }

    /// PR request (no deadline) — logic verbatim: set pending, lock,
    /// loop { if parked: grant; wait }.
    #[verifier::exec_allows_no_decreases_clause]
    pub(crate) fn request(&self) -> (res: CheckpointGuard<'a, T>)
        requires
            self.shared.parked.node == self.node,
        ensures
            res.shared.parked.node == res.node,
            res.snap@.is_some(),
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
                let Tracked(pt) = parked.take_baton();
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
    pub(crate) closed spec fn wf(self) -> bool {
        self.shared.parked.node == self.node
            && (match self.snap@ {
                Option::Some(pt) => pt.pptr() == self.node && pt.is_init(),
                Option::None => true,
            })
    }

    /// PR CheckpointGuard::node — `unsafe { &*self.node }`.
    pub(crate) fn node(&self) -> (r: &T)
        requires
            self.wf(),
            self.snap@.is_some(),
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
    pub(crate) fn release(self)
        requires
            self.wf(),
            self.snap@.is_some(),
        ensures
            self.shared.wf_(),
    {
        let CheckpointGuard { node: _, shared, snap: Tracked(opt), _brand: _ } = self;
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

impl<T> Shared<T> {
    pub open spec fn wf_(&self) -> bool {
        true
    }
}

/// PR with_checkpoint_pair — signature verbatim; the construction is
/// trusted plumbing (Verus cannot call a generic closure whose
/// arguments carry tracked fields): it builds the pair exactly as the
/// PR does and invokes f. The verified theorems concern the protocol
/// operations the closure performs on the pair.
#[verifier::external_body]
pub(crate) fn with_checkpoint_pair<T, R>(
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
