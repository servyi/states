//! A **verified implementation of the lock spec** (`TrustedLock`/
//! `TrustedGuard` in `lib.rs`): the same operation contracts, realized
//! by pure vstd atomics — a `PAtomicBool` spin bit plus a `PCell`
//! whose permissions live in one `AtomicInvariant` (the vstd
//! `basic_lock` pattern).
//!
//! This module is the machine-checked witness that the lock spec is
//! *implementable* and *does not assume too much*: mutual exclusion,
//! the `guarded_inv` discipline, and the condvar spec's
//! "unlock + lock" waits are all discharged by verified code with no
//! trusted bodies. The runtime keeps the `std::sync` impl; by the
//! refinement argument (a blocked thread takes no steps), that impl
//! refines this one.
//!
//! Design: the guard **owns** the protected state (the banked payload
//! moves out of the invariant at acquire and back at release), so the
//! guard's view is a real function of owned state — not an
//! uninterpreted snapshot — and two guards cannot coexist, exactly
//! like `std::sync::MutexGuard`. Every state op runs on owned state
//! while held (race-free by ownership). `wait_*` is literally
//! `put_back` + `lock` (the condvar spec); `wait_resume` re-acquires
//! until parked again, realizing its parked-preservation clause.

use vstd::cell::pcell;
use vstd::cell::CellId;
use vstd::invariant::open_atomic_invariant;
use vstd::invariant::{AtomicInvariant, InvariantPredicate};
use vstd::modes::tracked_swap;
use vstd::prelude::*;

use crate::checkpoint::{GView, guarded_inv};
use vstd::simple_pptr::{PointsTo, PPtr};

verus! {

// =====================================================================
// Banked payload + invariant
// =====================================================================

/// Exec-only flags (plain Copy data for the PCell).
#[derive(Clone, Copy)]
pub struct VFlags {
    pub parked: bool,
    pub pending: bool,
}

pub tracked struct Banked<T> {
    pub flags: pcell::PointsTo<VFlags>,
    pub baton: Option<PointsTo<T>>,
    pub guards: int,
}

/// The guard's view: decode the owned payload.
pub open spec fn banked_view<T>(b: &Banked<T>) -> GView<T> {
    GView {
        parked: b.flags.value().parked,
        pending: b.flags.value().pending,
        closed: false,
        guards: b.guards,
        baton: b.baton,
    }
}

pub struct VLockPred<T> {
    pub dummy: std::marker::PhantomData<T>,
}

impl<T>
    InvariantPredicate<
        (vstd::atomic::AtomicCellId, CellId, PPtr<T>),
        (vstd::atomic::PermissionBool, Option<Banked<T>>),
    > for VLockPred<T>
{
    open spec fn inv(
        k: (vstd::atomic::AtomicCellId, CellId, PPtr<T>),
        g: (vstd::atomic::PermissionBool, Option<Banked<T>>),
    ) -> bool {
        let (perm, banked) = g;
        let (aid, cid, node) = k;
        perm.id() == aid && match banked {
            Option::Some(b) => {
                &&& b.flags.id() == cid
                &&& perm.view().value == false
                &&& b.guards >= 0 && b.guards <= 1
                &&& (match b.baton {
                    Option::Some(pt) => pt.pptr() == node && pt.is_init(),
                    Option::None => true,
                })
                &&& guarded_inv(&banked_view(&b))
            }
            Option::None => perm.view().value == true,
        }
    }
}

// =====================================================================
// The verified lock
// =====================================================================

pub struct VLock<T> {
    pub bit: vstd::atomic::PAtomicBool,
    pub cell: pcell::PCell<VFlags>,
    pub inv: Tracked<
        AtomicInvariant<
            (vstd::atomic::AtomicCellId, CellId, PPtr<T>),
            (vstd::atomic::PermissionBool, Option<Banked<T>>),
            VLockPred<T>,
        >,
    >,
    #[verifier::spec]
    pub node: PPtr<T>,
}

impl<T> VLock<T> {
    pub open spec fn wf(&self) -> bool {
        self.inv@.constant() == (self.bit.id(), self.cell.id(), self.node)
    }
    /// LOCK SPEC (create): unlocked and invariant-satisfying.
    pub fn new(node: PPtr<T>) -> (s: Self)
    ensures
        s.node == node,
        s.wf(),
    {
        let (bit, Tracked(bit_perm)) = vstd::atomic::PAtomicBool::new(false);
        let (cell, Tracked(flags_perm)) = pcell::PCell::new(VFlags {
            parked: false,
            pending: false,
        });
        let tracked banked = Banked { flags: flags_perm, baton: Option::None, guards: 0 };
        let tracked inv =
            AtomicInvariant::new((bit.id(), cell.id(), node), (bit_perm, Option::Some(banked)), 0);
        VLock { bit, cell, inv: Tracked(inv), node }
    }

    /// LOCK SPEC (acquire): the returned guard owns the protected
    /// state, which satisfies the invariant.
    #[verifier::exec_allows_no_decreases_clause]
    pub fn lock(&self) -> (g: VGuard<'_, T>)
        requires
            self.wf(),
        ensures
            g.lock_ref == self,
            g.lock_ref.wf(),
            g.wf(),
            guarded_inv(&g@),
    {
        let tracked mut got: Option<Banked<T>> = Option::None;
        loop
            invariant
                self.wf(),
                got == Option::None,
        {
            let res;
            open_atomic_invariant!(self.inv.borrow() => pair => {
                let tracked (mut perm, mut banked) = pair;
                proof {
                    assert(perm.id() == self.bit.id()); // inv + wf
                }
                res = self.bit.compare_exchange(Tracked(&mut perm), false, true);
                proof {
                    assert(got == Option::None);
                    tracked_swap(&mut got, &mut banked);
                    assert(banked == Option::None);
                    assert(perm.view().value == true); // ok: set; err: was true
                    pair = (perm, banked);
                }
            });
            if res.is_ok() {
                let tracked b = got.tracked_unwrap();
                // from the invariant's Some-arm (opened at the CAS):
                // flags id matches the cell, baton node-matches
                return VGuard { lock_ref: self, banked: Tracked(b) };
            }
        }
    }
}

/// The guard: owns the banked payload while held.
pub struct VGuard<'a, T> {
    pub lock_ref: &'a VLock<T>,
    pub tracked banked: Tracked<Banked<T>>,
}

impl<'a, T> VGuard<'a, T> {
    /// The guard's view — real, not uninterpreted.
    pub open spec fn view(&self) -> GView<T> {
        banked_view(&self.banked@)
    }

    /// Guard wf: the lock is wf and the held PointsTo belongs to the
    /// lock's cell (from the invariant at acquire).
    pub open spec fn wf(&self) -> bool {
        self.lock_ref.wf()
            && self.banked@.flags.id() == self.lock_ref.cell.id()
            && (match self.banked@.baton {
                Option::Some(pt) => pt.pptr() == self.lock_ref.node && pt.is_init(),
                Option::None => true,
            })
    }

    /// LOCK SPEC (release): re-bank; the caller must leave the state
    /// invariant-consistent (the put-back discipline).
    pub fn put_back(self)
        requires
            self.wf(),
            guarded_inv(&banked_view(&self.banked@)),
    {
        let VGuard { lock_ref, banked: Tracked(b) } = self;
        let tracked mut b_opt = Option::Some(b);
        open_atomic_invariant!(lock_ref.inv.borrow() => pair => {
            let tracked (mut perm, mut banked) = pair;
            proof {
                assert(perm.id() == lock_ref.bit.id()); // inv + wf
            }
            lock_ref.bit.store(Tracked(&mut perm), false);
            proof {
                tracked_swap(&mut banked, &mut b_opt);
                pair = (perm, banked);
            }
        });
    }

    // ---- reads (fresh: owned state) ----

    pub fn parked(self) -> (res: (bool, Self))
        requires
            self.wf(),
        ensures
            res.0 == self@.parked,
            res.1@ == self@,
            res.1.wf(),
            res.1.lock_ref == self.lock_ref,
    {
        let VGuard { lock_ref, banked: Tracked(mut b) } = self;
        assert(b.flags.id() == lock_ref.cell.id());
        let v = lock_ref.cell.read(Tracked(&b.flags)).parked;
        (v, VGuard { lock_ref, banked: Tracked(b) })
    }

    pub fn still_pending(self) -> (res: (bool, Self))
        requires
            self.wf(),
        ensures
            res.0 == self@.pending,
            res.1@ == self@,
            res.1.wf(),
            res.1.lock_ref == self.lock_ref,
    {
        let VGuard { lock_ref, banked: Tracked(mut b) } = self;
        assert(b.flags.id() == lock_ref.cell.id());
        let v = lock_ref.cell.read(Tracked(&b.flags)).pending;
        (v, VGuard { lock_ref, banked: Tracked(b) })
    }

    pub open spec fn grantable(&self) -> bool {
        self@.guards == 0 && self@.baton.is_some()
    }

    // ---- transitions (owned state; same contracts as the spec) ----

    pub fn set_parked(self, v: bool) -> (g: Self)
        requires
            self.wf(),
            
            self@.guards == 0 || v == self@.parked,
        ensures
            g@ == (GView { parked: v, ..self@ }),
            g.lock_ref == self.lock_ref,
    {
        let VGuard { lock_ref, banked: Tracked(mut b) } = self;
        assert(b.flags.id() == lock_ref.cell.id());
        let f = lock_ref.cell.read(Tracked(&b.flags));
        lock_ref.cell.write(Tracked(&mut b.flags), VFlags { parked: v, pending: f.pending });
        VGuard { lock_ref, banked: Tracked(b) }
    }

    pub fn mark_pending(self) -> (g: Self)
        requires
            self.wf(),
        ensures
            g@ == (GView { pending: true, ..self@ }),
            g.lock_ref == self.lock_ref,
    {
        let VGuard { lock_ref, banked: Tracked(mut b) } = self;
        assert(b.flags.id() == lock_ref.cell.id());
        let f = lock_ref.cell.read(Tracked(&b.flags));
        lock_ref.cell.write(Tracked(&mut b.flags), VFlags { parked: f.parked, pending: true });
        VGuard { lock_ref, banked: Tracked(b) }
    }

    pub fn clear_pending(self) -> (g: Self)
        requires
            self.wf(),
            
            self@.guards == 0,
        ensures
            g@ == (GView { pending: false, ..self@ }),
            g.lock_ref == self.lock_ref,
    {
        let VGuard { lock_ref, banked: Tracked(mut b) } = self;
        assert(b.flags.id() == lock_ref.cell.id());
        let f = lock_ref.cell.read(Tracked(&b.flags));
        lock_ref.cell.write(Tracked(&mut b.flags), VFlags { parked: f.parked, pending: false });
        VGuard { lock_ref, banked: Tracked(b) }
    }

    pub fn give_baton(self, Tracked(pt): Tracked<PointsTo<T>>) -> (g: Self)
        requires
            self.wf(),
            
            pt.pptr() == self.lock_ref.node,
            pt.is_init(),
            self@.parked == false,
            self@.guards == 0,
            self@.baton.is_none(),
        ensures
            g@ == (GView { parked: true, baton: Option::Some(pt), ..self@ }),
            guarded_inv(&g@),
            g.lock_ref == self.lock_ref,
    {
        let VGuard { lock_ref, banked: Tracked(mut b) } = self;
        assert(b.flags.id() == lock_ref.cell.id());
        let f = lock_ref.cell.read(Tracked(&b.flags));
        lock_ref.cell.write(Tracked(&mut b.flags), VFlags { parked: true, pending: f.pending });
        proof {
            b.baton = Option::Some(pt);
        }
        VGuard { lock_ref, banked: Tracked(b) }
    }

    pub fn take_baton(self) -> (res: (Tracked<Option<PointsTo<T>>>, Self))
        requires
            self.wf(),
            
            self@.baton.is_some(),
            self@.guards == 0,
            self@.parked,
            self@.pending,
        ensures
            res.1@ == (GView { baton: Option::None, guards: 1, ..self@ }),
            guarded_inv(&res.1@),
            res.0@.is_some(),
            get_some_baton(res.0@).pptr() == self.lock_ref.node,
            get_some_baton(res.0@).is_init(),
            res.1.lock_ref == self.lock_ref,
    {
        let VGuard { lock_ref, banked: Tracked(mut b) } = self;
        let tracked mut out = Option::None;
        proof {
            out = b.baton;
            b.baton = Option::None;
            b.guards = 1;
        }
        (Tracked(out), VGuard { lock_ref, banked: Tracked(b) })
    }

    pub fn return_baton(self, Tracked(pt): Tracked<PointsTo<T>>) -> (g: Self)
        requires
            self.wf(),
            
            pt.pptr() == self.lock_ref.node,
            pt.is_init(),
            self@.baton.is_none(),
            self@.guards == 1,
        ensures
            g@ == (GView {
                baton: Option::Some(pt),
                guards: 0,
                parked: true,
                ..self@
            }),
            guarded_inv(&g@),
            g.lock_ref == self.lock_ref,
    {
        let VGuard { lock_ref, banked: Tracked(mut b) } = self;
        assert(b.flags.id() == lock_ref.cell.id());
        let f = lock_ref.cell.read(Tracked(&b.flags));
        lock_ref.cell.write(Tracked(&mut b.flags), VFlags { parked: true, pending: f.pending });
        proof {
            b.baton = Option::Some(pt);
            b.guards = 0;
        }
        VGuard { lock_ref, banked: Tracked(b) }
    }

    pub fn resume_baton(self) -> (res: (Tracked<Option<PointsTo<T>>>, Self))
        requires
            self.wf(),
            
            self@.baton.is_some(),
            self@.guards == 0,
            self@.parked,
            !self@.pending,
        ensures
            res.1@ == (GView { baton: Option::None, ..self@ }),
            get_some_baton(res.0@).pptr() == self.lock_ref.node,
            get_some_baton(res.0@).is_init(),
            res.1.lock_ref == self.lock_ref,
    {
        let VGuard { lock_ref, banked: Tracked(mut b) } = self;
        let tracked mut out = Option::None;
        proof {
            out = b.baton;
            b.baton = Option::None;
        }
        (Tracked(out), VGuard { lock_ref, banked: Tracked(b) })
    }

    // ---- the condvar spec: wait = unlock + [block] + lock ----

    #[verifier::exec_allows_no_decreases_clause]
    pub fn wait_park(self) -> (g2: VGuard<'a, T>)
        requires
            self.wf(),
            guarded_inv(&self@),
        ensures
            g2.lock_ref == self.lock_ref,
            guarded_inv(&g2@),
    {
        let VGuard { lock_ref, .. } = self;
        self.put_back();
        lock_ref.lock()
    }

    /// As `wait_park`, but the mutator remains parked: re-acquire
    /// until parked again (realizing the parked-preservation clause;
    /// the real condvar wakes only when pending is cleared, which
    /// never un-parks).
    #[verifier::exec_allows_no_decreases_clause]
    pub fn wait_resume(self) -> (g2: VGuard<'a, T>)
        requires
            self.wf(),
            guarded_inv(&self@),
            self@.parked,
        ensures
            g2.lock_ref == self.lock_ref,
            guarded_inv(&g2@),
            g2@.parked,
    {
        let VGuard { lock_ref, .. } = self;
        self.put_back();
        proof {
            assert(lock_ref == self.lock_ref);
        }
        loop
            invariant
                lock_ref == self.lock_ref,
                lock_ref.wf(),
        {
            let g0 = lock_ref.lock();
            let (parked, g) = g0.parked();
            if parked {
                return g;
            }
            g.put_back();
        }
    }
}

/// Spec projection of a Some-valued baton (as in lib.rs).
pub open spec fn get_some_baton<T>(o: Option<PointsTo<T>>) -> PointsTo<T> {
    match o {
        Option::Some(pt) => pt,
        Option::None => arbitrary(),
    }
}

fn main() {}


// =====================================================================
// VCondvar: a standalone, fully verified condvar implementing the
// assumed condvar spec (the same one the protocol substrate uses):
//
//   SPEC:  wait(guard) = unlock + [block] + lock;  the state
//   afterwards is arbitrary but invariant-consistent (a superset of
//   any real condvar — soundness holds even for total wakeup loss).
//   notify_all is observationally a no-op (wakeups are hints).
//
// Implementation: the obvious one — `put_back` (unlock, discharging
// the spec's precondition into the invariant) followed by `lock`
// (re-acquire, whose ensures IS the spec's postcondition). Every
// obligation is discharged by the two verified lock ops; nothing is
// trusted. [DEVIATION from the obvious *real* condvar: see the
// module notes in the final report — this wait never blocks and
// notify_all does nothing; both are the immediate-spurious-wakeup
// refinement of a futex condvar.]
// =====================================================================

pub struct VCondvar {
    _nothing: (), // the verified model needs no state
}

impl VCondvar {
    pub fn new() -> Self {
        VCondvar { _nothing: () }
    }

    /// THE condvar spec, on the verified lock's guard.
    #[verifier::exec_allows_no_decreases_clause]
    pub fn wait<'a, T>(&self, g: VGuard<'a, T>) -> (g2: VGuard<'a, T>)
        requires
            g.wf(),
            guarded_inv(&g@),
        ensures
            g2.lock_ref == g.lock_ref,
            guarded_inv(&g2@),
            g2.wf(),
    {
        let lock_ref = g.lock_ref; // &'a VLock is Copy
        g.put_back(); // unlock — the spec's [release] half
        lock_ref.lock() // re-acquire — ensures guarded_inv (the spec)
    }

    /// `notify_all`: observationally a no-op (the spec allows any
    /// wakeup timing, including none; liveness is out of scope).
    pub fn notify_all(&self)
    {
    }

    /// CONSUMER-SIDE derivation (not part of the condvar spec): the
    /// substrate's stronger `wait_resume` clause (`ensures parked`)
    /// obtained from the PLAIN condvar spec above by a retry loop —
    /// showing that clause assumes nothing more about the condvar.
    #[verifier::exec_allows_no_decreases_clause]
    pub fn wait_until_parked<'a, T>(&self, g0: VGuard<'a, T>) -> (g2: VGuard<'a, T>)
        requires
            g0.wf(),
            guarded_inv(&g0@),
        ensures
            g2.lock_ref == g0.lock_ref,
            g2.wf(),
            guarded_inv(&g2@),
            g2@.parked,
    {
        let lock_ref = g0.lock_ref; // Copy; the loop preserves it
        let mut g = g0;
        loop
            invariant
                lock_ref == g0.lock_ref,
                g.wf(),
                g.lock_ref == lock_ref,
                guarded_inv(&g@),
        {
            let g1 = self.wait(g); // ONLY the condvar's public spec
            let (parked, g2) = g1.parked();
            if parked {
                // wait: g1.lock_ref == g.lock_ref; invariant: == lock_ref;
                // entry: lock_ref == g0.lock_ref; parked(): g2.lock_ref == g1.lock_ref
                assert(lock_ref == g0.lock_ref);
                assert(g2.lock_ref == g0.lock_ref);
                return g2;
            }
            g = g2;
        }
    }
}

} // verus!
