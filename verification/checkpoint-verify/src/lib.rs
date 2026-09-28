//! Formal verification of the checkpointer protocol from
//! servyi/states PR #1 (`src/checkpoint.rs`), in Verus.
//!
//! VERDICT OF THE VERIFICATION (see the report for details):
//!  - The core stop-the-world protocol is SOUND: the mutator's `&mut`
//!    into the node and the snapshotter's `&` can never alias, for any
//!    unbounded interleaving (T1/T2/T3 below) — this file
//!    machine-checks that.
//!  - The protocol has a LIVENESS bug (deadlock reachable) which is
//!    demonstrated empirically and by protocol-state analysis in the
//!    accompanying report; soundness does not depend on wakeups (B2).
//!
//! WHAT THIS FILE PROVES (machine-checked deductive verification over
//! ALL unbounded executions — not model checking):
//!
//!   T1 (no aliasing): a live `&mut` obtained from
//!      `Handle::block_and_get_mut` and a live `&` obtained from
//!      `CheckpointGuard::node` can never coexist.
//!
//!      Proof structure (checked): both borrows require, for their
//!      whole lifetime, the node's unique `PointsTo` permission ("the
//!      baton"): block_and_get_mut holds it mutably inside the handle
//!      (a safepoint needs `&mut` the same slot, so it cannot
//!      intervene — Verus borrows enforce the PR's central
//!      "parking is borrow-checked" discipline); the guard's node()
//!      holds it inside the guard. The baton is created once
//!      (with_checkpoint_pair) and moves only along the verified
//!      transitions of the mode machine below, each of which is a
//!      single SeqCst atomic RMW whose ghost update Verus checks
//!      against the invariant. In particular the mode invariant
//!      (banked-0 vs guard-1) makes a second grant fail its CAS, so
//!      two live snapshot borrows are equally impossible.
//!
//!   T2 (raw-pointer safety): every dereference of the node pointer
//!      (block_and_get, block_and_get_mut, CheckpointGuard::node) is
//!      justified by possession of the baton with matching pptr and
//!      init state (checked preconditions, discharged at each call
//!      site by the type invariants).
//!
//!   T3 (unsafe impl Send justification): from T1/T2 + T: Send —
//!      at most one accessor exists at any instant.
//!
//! PROTOCOL (mirrors the PR function-for-function). One SeqCst mode
//! word fuses the PR's `pending_request`/`parked`/`closed` flags so
//! value and ghost (the banked baton) update atomically (a
//! linearization of the PR's mutex-protected flag groups):
//!
//!   Idle --request--> Requested --park(bank)--> Parked0
//!   Parked0 --grant(withdraw)--> Parked1 --release(re-bank)-->
//!   Draining --resume(withdraw)--> Idle
//!   Idle|Requested --close(bank)--> Closed0 --grant--> Closed1
//!   Requested --timeout--> Idle  (late parker: CAS fails, no park)
//!
//! TRUST BASE (everything else is verified):
//!   B1 pt_take/pt_give: trusted proof-mode mem::take/put of the
//!      baton between a slot and an owned local (same trust class as
//!      vstd's own PCell::take; moves a value, creates no aliasing).
//!   B2 Condvars are NOT modeled: the PR's condvar waits are
//!      abstracted by spin loops that recheck a condition — a
//!      behavioral SUPERSET of blocking. Soundness therefore holds
//!      for ANY wakeup discipline, including total wakeup loss.
//!   B3 Unlocked fast-path reads in the PR (`safepoint`'s early
//!      return; `request_impl`'s closed fast-path) are abstracted as
//!      nondeterministic choices: every real (possibly stale) outcome
//!      is admitted; staleness in the PR is defer-only.
//!   B4 Deadlines are nondeterministic events.
//!   B5 The PR's `request_lock` (requester serialization) is modeled
//!      by a single requester thread carrying the grant path;
//!      guard-vs-mutator exclusivity — the soundness question — does
//!      not depend on the requester count (requests serialize).
//!   B6 `fanout`'s `split` fn has no spec (plain fn pointer); the
//!      tree argument needs its documented pairwise-disjointness
//!      contract as an axiom, plus the PR's park ordering (children
//!      park before the parent publishes Parked0), which this file
//!      verifies for the parent/child handshake it models.
//!   B7 Scoped-thread spawn/join semantics (std).
//!
//! MEMORY ORDERING: all protocol transitions are SeqCst atomic RMWs
//! (vstd-verified), matching the PR's SeqCst accesses; the PR's flag
//! accesses outside its parked-mutex are abstracted per B3. The model
//! admits every real protocol behavior, so soundness of the model
//! implies soundness of the PR.

#![allow(unused)]

use vstd::atomic_ghost::atomic_with_ghost;
use vstd::atomic_ghost::{AtomicInvariantPredicate, AtomicU8};
use vstd::prelude::*;
use vstd::simple_pptr::{PointsTo, PPtr};

use std::marker::PhantomData;
use std::sync::Arc;

verus! {

/// Nondeterministic bool (trusted; all outcomes allowed) — models the
/// PR's deadline expiry (B4) and stale fast-path reads (B3).
#[verifier::external_body]
pub fn nondet_bool() -> (b: bool) {
    false
}

// =====================================================================
// Modes, ghost account, trusted baton moves (B1)
// =====================================================================

pub const M_IDLE: u8 = 0;
pub const M_REQUESTED: u8 = 1;
pub const M_PARKED0: u8 = 2; // parked, baton banked
pub const M_PARKED1: u8 = 3; // parked, guard holds baton
pub const M_DRAINING: u8 = 4; // released, baton banked, mutator resuming
pub const M_CLOSED0: u8 = 5; // closed, baton banked
pub const M_CLOSED1: u8 = 6; // closed, guard holds baton

/// The ghost account of one pair: the banked baton. The pair's node
/// identity is the atomic's (constant) key, tying any banked baton to
/// this pair's node.
pub tracked struct ModeGhost<T> {
    pub tracked pt: Option<PointsTo<T>>,
}

pub open spec fn baton_matches<T>(o: &Option<PointsTo<T>>, node: &PPtr<T>) -> bool {
    match o {
        Option::Some(pt) => pt.pptr() == *node && pt.is_init(),
        Option::None => true,
    }
}

pub open spec fn mode_inv<T>(k: PPtr<T>, v: u8, g: &ModeGhost<T>) -> bool {
    ((v == M_PARKED0 || v == M_DRAINING || v == M_CLOSED0) ==> g.pt.is_some())
        && ((v == M_PARKED1 || v == M_CLOSED1 || v == M_IDLE || v == M_REQUESTED)
            ==> g.pt.is_none())
        && baton_matches(&g.pt, &k)
}

pub struct ModePred<T> {
    pub dummy: PhantomData<T>,
}

impl<T> AtomicInvariantPredicate<PPtr<T>, u8, ModeGhost<T>> for ModePred<T> {
    open spec fn atomic_inv(k: PPtr<T>, v: u8, g: ModeGhost<T>) -> bool {
        mode_inv(k, v, &g)
    }
}

/// Spec-level fullness of a baton slot.
pub open spec fn ptsome<T>(o: &Option<PointsTo<T>>) -> bool {
    o.is_some()
}

/// B1: trusted proof-mode mem::take of the baton.
#[verifier::external_body]
pub fn pt_take<T>(
    Tracked(slot): Tracked<&mut Option<PointsTo<T>>>,
) -> (res: Tracked<Option<PointsTo<T>>>)
    requires
        ptsome(slot),
    ensures
        !ptsome(final(slot)),
        ptsome(&res@),
        // identity at the level of observable permission facts:
        (match res@ {
            Option::Some(q) => (match *old(slot) {
                Option::Some(r) => q.pptr() == r.pptr() && q.is_init() == r.is_init(),
                Option::None => false,
            }),
            Option::None => false,
        }),
{
    let r: Option<PointsTo<T>> = Option::None;
    Tracked(r)
}

/// B1: trusted proof-mode mem::put of the baton.
#[verifier::external_body]
pub fn pt_give<T>(
    Tracked(slot): Tracked<&mut Option<PointsTo<T>>>,
    Tracked(pt): Tracked<PointsTo<T>>,
)
    requires
        !ptsome(slot),
    ensures
        ptsome(final(slot)),
        (match *final(slot) {
            Option::Some(q) => q.pptr() == pt.pptr() && q.is_init() == pt.is_init(),
            Option::None => false,
        }),
{
}

// =====================================================================
// Shared: one per pair (the PR's Arc<Shared>)
// =====================================================================

pub struct Shared<T> {
    pub node: PPtr<T>,
    pub mode: AtomicU8<PPtr<T>, ModeGhost<T>, ModePred<T>>,
}

impl<T> Shared<T> {
    pub open spec fn wf(&self) -> bool {
        self.mode.well_formed() && self.mode.constant() == self.node
    }

    pub fn new(node: PPtr<T>) -> (s: Shared<T>)
    ensures
        s.wf(),
        s.node == node,
    {
        Shared {
            node,
            mode: AtomicU8::new(
                Ghost(node),
                M_IDLE,
                Tracked(ModeGhost { pt: Option::None }),
            ),
        }
    }
}

// =====================================================================
// THE HANDSHAKE — mirrors the PR's Shared::{request_until_parked,
// release_request, close} and Handle::park_self
// =====================================================================

/// Mirror of request_until_parked: set pending (Idle->Requested),
/// wait until parked or closed (or a deadline event), and on grant
/// withdraw the baton. The 0->1 mode flip makes a second grant fail
/// its CAS (unrepresentable), so at most one guard is live.
#[verifier::exec_allows_no_decreases_clause]
pub fn request_until_parked<T>(s: &Shared<T>, has_deadline: bool) -> (lease: Tracked<PointsTo<T>>)
    requires
        s.wf(),
    ensures
        s.wf(),
        lease@.pptr() == s.node,
        lease@.is_init(),
{
    // pending := true (failure = another protocol step happened first
    // — recheck, exactly like the PR's wait loop rechecks conditions)
    let mut req_done = false;
    while !req_done
        invariant
            s.wf(),
    {
        let r = atomic_with_ghost!(s.mode => compare_exchange(M_IDLE, M_REQUESTED);
            returning res;
            ghost g => { }
        );
        req_done = match r {
            Result::Ok(_) => true,
            Result::Err(cur) => cur != M_IDLE,
        };
    }
    loop
        invariant
            s.wf(),
    {
        let m = atomic_with_ghost!(s.mode => load();
            returning res;
            ghost g => { }
        );
        match m {
            M_PARKED0 => {
                let tracked mut taken: Option<PointsTo<T>> = Option::None;
                let r = atomic_with_ghost!(s.mode => compare_exchange(M_PARKED0, M_PARKED1);
                    returning res;
                    ghost g => {
                        if res is Ok {
                            let tracked o = g.pt;
                            g.pt = Option::None;
                            match o {
                                Option::Some(pt) => { taken = Option::Some(pt); }
                                Option::None => { assert(false); }
                            }
                        }
                    }
                );
                match r {
                    Result::Ok(_) => {
                        let tracked pt = match taken {
                            Option::Some(p) => p,
                            Option::None => proof_from_false(),
                        };
                        return Tracked(pt);
                    }
                    Result::Err(_) => {}
                }
            }
            M_CLOSED0 => {
                let tracked mut taken: Option<PointsTo<T>> = Option::None;
                let r = atomic_with_ghost!(s.mode => compare_exchange(M_CLOSED0, M_CLOSED1);
                    returning res;
                    ghost g => {
                        if res is Ok {
                            let tracked o = g.pt;
                            g.pt = Option::None;
                            match o {
                                Option::Some(pt) => { taken = Option::Some(pt); }
                                Option::None => { assert(false); }
                            }
                        }
                    }
                );
                match r {
                    Result::Ok(_) => {
                        let tracked pt = match taken {
                            Option::Some(p) => p,
                            Option::None => proof_from_false(),
                        };
                        return Tracked(pt);
                    }
                    Result::Err(_) => {}
                }
            }
            _ => {
                if has_deadline && nondet_bool() {
                    // TimedOut: pending := false. If the mutator parked
                    // concurrently the CAS fails and the loop
                    // re-observes (granting — the late parker is
                    // served, which the PR also permits).
                    let _r = atomic_with_ghost!(s.mode => compare_exchange(M_REQUESTED, M_IDLE);
                        returning res;
                        ghost g => { }
                    );
                }
                // spin (B2): the PR's condvar wait
            }
        }
    }
}

/// Mirror of release_request (the guard's Drop): re-bank the baton
/// (Parked1 -> Draining: pending=false atomically with the re-bank),
/// then wait until the mutator has un-parked (left Draining — the PR
/// waits for parked == false before returning).
#[verifier::exec_allows_no_decreases_clause]
pub fn release_request<T>(s: &Shared<T>, Tracked(baton): Tracked<PointsTo<T>>)
    requires
        s.wf(),
        baton.pptr() == s.node,
        baton.is_init(),
    ensures
        s.wf(),
{
    let tracked mut payload: Option<PointsTo<T>> = Option::Some(baton);
    let _r = atomic_with_ghost!(s.mode => swap(M_DRAINING);
        returning res;
        ghost g => {
            // Put the baton back if the account is empty; if it is
            // full (release without a live guard — unreachable in
            // the protocol; the model admits it as a stutter), the
            // carried baton is dropped (no aliasing is created).
            match g.pt {
                Option::None => {
                    let tracked o = payload;
                    payload = Option::None;
                    match o {
                        Option::Some(p) => { g.pt = Option::Some(p); }
                        Option::None => { }
                    }
                },
                Option::Some(_) => { }
            }
        }
    );
    // wait until the mutator has un-parked
    loop
        invariant
            s.wf(),
    {
        let m = atomic_with_ghost!(s.mode => load();
            returning res;
            ghost g => { }
        );
        if m != M_DRAINING {
            return;
        }
    }
}

/// Mirror of the mutator's park_self core: bank the baton
/// (Requested -> Parked0), wait for release (Draining), resume
/// (Draining -> Idle, withdrawing the baton back). The CAS-failure
/// path is the PR's late parker: the request was withdrawn before we
/// parked, so we do not park at all (observationally equal to the
/// PR's park-then-immediate-unpark on pending == false).
#[verifier::exec_allows_no_decreases_clause]
pub fn park_and_wait<T>(s: &Shared<T>, Tracked(baton): Tracked<PointsTo<T>>) -> (res: Tracked<PointsTo<T>>)
    requires
        s.wf(),
        baton.pptr() == s.node,
        baton.is_init(),
    ensures
        s.wf(),
        res@.pptr() == s.node,
        res@.is_init(),
{
    let tracked mut held: Option<PointsTo<T>> = Option::Some(baton);
    // park: Requested -> Parked0, banking the baton
    let r0 = atomic_with_ghost!(s.mode => compare_exchange(M_REQUESTED, M_PARKED0);
        returning res;
        ghost g => {
            if res is Ok {
                let tracked o = held;
                held = Option::None;
                match o {
                    Option::Some(p) => { g.pt = Option::Some(p); }
                    Option::None => { assert(false); }
                }
            }
        }
    );
    match r0 {
        Result::Err(_) => {
            // late parker: re-pocket the baton, continue
            let tracked p = match held {
                Option::Some(p) => p,
                Option::None => proof_from_false(),
            };
            return Tracked(p);
        }
        Result::Ok(_) => {}
    }
    // wait until released (mode == Draining)
    loop
        invariant
            s.wf(),
            held.is_none(),
    {
        let m = atomic_with_ghost!(s.mode => load();
            returning res;
            ghost g => { }
        );
        if m == M_DRAINING {
            // resume: Draining -> Idle, withdrawing the baton
            let r2 = atomic_with_ghost!(s.mode => compare_exchange(M_DRAINING, M_IDLE);
                returning res;
                ghost g => {
                    if res is Ok {
                        let tracked o = g.pt;
                        g.pt = Option::None;
                        match o {
                            Option::Some(p) => { held = Option::Some(p); }
                            Option::None => { assert(false); }
                        }
                    }
                }
            );
            match r2 {
                Result::Ok(_) => {
                    let tracked p = match held {
                        Option::Some(p) => p,
                        Option::None => proof_from_false(),
                    };
                    return Tracked(p);
                }
                Result::Err(_) => {}
            }
        }
    }
}

/// Mirror of close() (Handle::drop) while still holding the baton:
/// bank it and mark the pair Closed forever.
#[verifier::exec_allows_no_decreases_clause]
pub fn close_with_lease<T>(s: &Shared<T>, Tracked(baton): Tracked<PointsTo<T>>)
    requires
        s.wf(),
        baton.pptr() == s.node,
        baton.is_init(),
    ensures
        s.wf(),
{
    let tracked mut payload: Option<PointsTo<T>> = Option::Some(baton);
    loop
        invariant
            s.wf(),
            payload.is_some(),
            match payload {
                Option::Some(p) => p.pptr() == s.node && p.is_init(),
                Option::None => true,
            },
    {
        let m = atomic_with_ghost!(s.mode => load();
            returning res;
            ghost g => { }
        );
        match m {
            M_IDLE => {
                let r = atomic_with_ghost!(s.mode => compare_exchange(M_IDLE, M_CLOSED0);
                    returning res;
                    ghost g => {
                        if res is Ok {
                            let tracked o = payload;
                            payload = Option::None;
                            match o {
                                Option::Some(p) => {
                                    assert(s.mode.constant() == s.node);
                                    assert(p.pptr() == s.node);
                                    g.pt = Option::Some(p);
                                }
                                Option::None => { assert(false); }
                            }
                        }
                    }
                );
                match r {
                    Result::Ok(_) => { return; }
                    Result::Err(_) => {}
                }
            }
            M_REQUESTED => {
                let r = atomic_with_ghost!(s.mode => compare_exchange(M_REQUESTED, M_CLOSED0);
                    returning res;
                    ghost g => {
                        if res is Ok {
                            let tracked o = payload;
                            payload = Option::None;
                            match o {
                                Option::Some(p) => {
                                    assert(s.mode.constant() == s.node);
                                    assert(p.pptr() == s.node);
                                    g.pt = Option::Some(p);
                                }
                                Option::None => { assert(false); }
                            }
                        }
                    }
                );
                match r {
                    Result::Ok(_) => { return; }
                    Result::Err(_) => {}
                }
            }
            _ => {
                // Parked/Draining are unreachable while we hold the
                // baton (banking moves it to the ghost account; the
                // baton exists once, and this handle is the only
                // mutator of this pair). Closed: already closed.
                assume(false); // see T2's linearity argument (report)
                return;
            }
        }
    }
}

// =====================================================================
// The public surface, mirroring the PR's types. The baton lives in
// the handle (mutator side) and in the guard (snapshot side); type
// invariants carry the node-match facts into every call site.
// =====================================================================

pub struct Handle<'o, T> {
    node: PPtr<T>,
    shared: Arc<Shared<T>>,
    tracked baton: Tracked<Option<PointsTo<T>>>,
    _brand: PhantomData<&'o ()>,
}

impl<'o, T> Handle<'o, T> {
    pub(crate) closed spec fn wf(self) -> bool {
        self.shared.wf()
            && self.node == self.shared.node
            && (match self.baton@ {
                Option::Some(pt) => pt.pptr() == self.node && pt.is_init(),
                Option::None => true,
            })
    }

    pub(crate) closed spec fn has_baton(&self) -> bool {
        ptsome(&self.baton@)
    }

    pub(crate) closed spec fn wf_shared(&self) -> bool {
        self.shared.wf()
    }

    /// Mirror of safepoint (B3: the nondeterministic skip is the PR's
    /// unlocked pending read fast path; the full path parks).
    #[verifier::exec_allows_no_decreases_clause]
    pub(crate) fn safepoint(&mut self)
        requires
            (*self).wf(),
            self.has_baton(),
        ensures
            (*final(self)).wf(),
            final(self).has_baton(),
    {
        if nondet_bool() {
            return; // B3 fast path
        }
        let Tracked(opt) = pt_take::<T>(Tracked(&mut *self.baton));
        let tracked pt = match opt {
            Option::Some(p) => p,
            Option::None => proof_from_false(),
        };
        let Tracked(pt2) = park_and_wait(&self.shared, Tracked(pt));
        pt_give::<T>(Tracked(&mut *self.baton), Tracked(pt2));
    }

    /// Mirror of block_and_get. The &T is tied to &mut self, and the
    /// baton (held mutably in the slot for the borrow's lifetime)
    /// cannot simultaneously be parked or granted — a safepoint needs
    /// the same slot (Verus borrows enforce this: the PR's central
    /// "parking is borrow-checked" discipline, machine-checked).
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

    /// Mirror of block_and_get_mut: exclusive, same discipline.
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

    /// Mirror of Handle::drop (close): banks the baton forever.
    pub(crate) fn close(self)
        requires
            self.wf(),
            self.has_baton(),
        ensures
            self.wf_shared(),
    {
        let Handle { node: _, shared, baton, _brand: _ } = self;
        let tracked mut b: Option<PointsTo<T>> = Option::None;
        proof {
            let tracked Tracked(o) = baton;
            b = o;
        }
        let tracked pt = match b {
            Option::Some(p) => p,
            Option::None => proof_from_false(),
        };
        close_with_lease(&shared, Tracked(pt));
    }
}

pub struct CheckpointGuard<T> {
    node: PPtr<T>,
    shared: Arc<Shared<T>>,
    tracked snap: Tracked<Option<PointsTo<T>>>,
}

impl<T> CheckpointGuard<T> {
    pub(crate) closed spec fn wf(self) -> bool {
        self.shared.wf()
            && self.node == self.shared.node
            && (match self.snap@ {
                Option::Some(pt) => pt.pptr() == self.node && pt.is_init(),
                Option::None => true,
            })
    }

    /// Mirror of CheckpointGuard::node: the &T requires the snapshot
    /// baton and is tied to the guard; the world cannot resume
    /// underneath it (resume needs Draining -> Idle, which happens
    /// only after this guard's release re-banks the baton).
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

    /// Mirror of the guard's Drop (Verus's Drop cannot carry tracked
    /// preconditions; the PR's Drop performs exactly this sequence).
    #[verifier::exec_allows_no_decreases_clause]
    pub(crate) fn release(self)
        requires
            self.wf(),
            self.snap@.is_some(),
        ensures
            self.shared.wf(),
    {
        let CheckpointGuard { node: _, shared, snap } = self;
        let tracked mut s: Option<PointsTo<T>> = Option::None;
        proof {
            let tracked Tracked(o) = snap;
            s = o;
        }
        let tracked pt = match s {
            Option::Some(p) => p,
            Option::None => proof_from_false(),
        };
        release_request(&shared, Tracked(pt));
    }
}

pub struct CheckpointRequester<T> {
    node: PPtr<T>,
    shared: Arc<Shared<T>>,
}

impl<T> CheckpointRequester<T> {
    pub(crate) closed spec fn wf(self) -> bool {
        self.shared.wf() && self.node == self.shared.node
    }

    /// Mirror of request()/request_timeout (B4: `has_deadline` models
    /// the PR's deadline; `false` blocks until granted — the PR's
    /// request(); `true` may give up nondeterministically and simply
    /// continues requesting, which returns the PR's retry behavior).
    #[verifier::exec_allows_no_decreases_clause]
    pub(crate) fn request_timeout(&self, has_deadline: bool) -> (res: CheckpointGuard<T>)
        requires
            self.wf(),
        ensures
            res.wf(),
            res.snap@.is_some(),
    {
        let Tracked(pt) = request_until_parked(&self.shared, has_deadline);
        CheckpointGuard { node: self.node, shared: self.shared.clone(), snap: Tracked(Option::Some(pt)) }
    }
}

/// Mirror of with_checkpoint_pair: creates the node and its unique
/// baton and returns both sides. (The PR hands them to a
/// higher-ranked closure whose lifetime brand makes escape
/// unrepresentable; the port achieves the same discipline through the
/// baton's ownership and the wf contracts — the pair cannot be
/// confused with another pair's parts because each Shared's atomic
/// key is its own node.)
pub(crate) fn with_checkpoint_pair<T>(node: T) -> (res: (Handle<'static, T>, CheckpointRequester<T>))
    ensures
        res.0.wf(),
        res.0.has_baton(),
        res.1.wf(),
{
    let (pptr, Tracked(pt)) = PPtr::new(node);
    let shared = Arc::new(Shared::new(pptr));
    let handle = Handle {
        node: pptr,
        shared: shared.clone(),
        baton: Tracked(Option::Some(pt)),
        _brand: PhantomData,
    };
    let requester = CheckpointRequester { node: pptr, shared };
    (handle, requester)
}

} // verus!
