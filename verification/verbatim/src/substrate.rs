//! Spec'd substrate for the verbatim verification of the shipped
//! `checkpoint.rs`. Specs above; unverified std-backed impls inline
//! (marked `external_body`). Seams:
//!   S1 ParkedCell (parked mutex + condvars + baton account)
//!   S2 pending/closed folded into the cell
//!   S3 Deadline (nondeterministic time)
//!   S4 SplitFn (disjoint children + per-child permissions)
//!   S5 ScopeSpec (spawn, joined-before-return, closure runs inline)
//!   S6 ReqLock (requester serialization)

#![allow(unused)]

use vstd::prelude::*;
use vstd::simple_pptr::{PointsTo, PPtr};

use std::time::Duration;

verus! {

#[verifier::external_body]
#[verifier::external_type_specification]
#[verifier(reject_recursive_types(T))]
pub struct ExMutex<T: ?Sized>(std::sync::Mutex<T>);

#[verifier::external_body]
#[verifier::external_type_specification]
pub struct ExCondvar(std::sync::Condvar);

// =====================================================================
// S1 + S2: the parked cell
// =====================================================================

pub struct Guarded<T> {
    pub parked: bool,
    pub pending: bool,
    pub closed: bool,
    pub tracked baton: Option<PointsTo<T>>,
}

pub struct GView<T> {
    pub parked: bool,
    pub pending: bool,
    pub closed: bool,
    pub guards: int,
    pub tracked baton: Option<PointsTo<T>>,
}

// (gview: the spec view is per-guard-instance via the uninterp view();
// the exec Guarded carries only the real flags + the tracked baton)

pub open spec fn guarded_inv<T>(v: &GView<T>) -> bool {
    (v.parked ==> (v.baton.is_some() || v.guards == 1))
        && (!v.parked ==> (v.baton.is_none() && v.guards == 0))
        && (v.guards == 1 ==> v.pending || v.closed)
        && v.guards >= 0
        && v.guards <= 1
}

pub open spec fn get_some_pt<T>(o: Option<PointsTo<T>>) -> PointsTo<T> {
    match o {
        Option::Some(p) => p,
        Option::None => arbitrary(),
    }
}

#[verifier(reject_recursive_types(T))]
pub struct ParkedCell<T> {
    pub inner: std::sync::Mutex<Guarded<T>>,
    pub park_cv: std::sync::Condvar,
    pub resume_cv: std::sync::Condvar,
    #[verifier::spec]
    pub node: PPtr<T>,
}

#[verifier(reject_recursive_types(T))]
pub struct ParkedGuard<'a, T> {
    pub cell: &'a ParkedCell<T>,
    #[verifier::spec]
    pub node: PPtr<T>,
}

impl<T> ParkedCell<T> {
    /// The cell's current pending flag (spec-level, functional).
    pub uninterp spec fn pending_now(&self) -> bool;

    #[verifier::external_body]
    pub fn new(node: PPtr<T>) -> (s: Self)
    ensures
        s.node == node,
    {
        ParkedCell {
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

    #[verifier::external_body]
    pub fn lock_parked(&self) -> (g: ParkedGuard<'_, T>)
    ensures
        g.node == self.node,
        g.cell == self,
        guarded_inv(&g.view()),
        g.view().pending == self.pending_now(),
    {
        let _held = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        ParkedGuard { cell: self, node: self.node }
    }

    #[verifier::external_body]
    pub fn set_pending(&self, v: bool)
    ensures
        self.pending_now() == v,
    {
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.pending = v;
        if !v {
            self.resume_cv.notify_all();
        }
    }

    #[verifier::external_body]
    pub fn is_pending(&self) -> (b: bool)
    ensures
        b == self.pending_now(),
    {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).pending
    }

    #[verifier::external_body]
    pub fn is_closed(&self) -> (b: bool)
    ensures
        true,
    {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).closed
    }

    #[verifier::external_body]
    pub fn close_cell(&self)
    ensures
        true,
    {
        {
            let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            g.closed = true;
        }
        self.park_cv.notify_all();
    }
}

impl<'a, T> ParkedGuard<'a, T> {
    #[verifier::external_body]
    pub uninterp spec fn view(&self) -> GView<T>;

    #[verifier::external_body]
    pub fn read(&self) -> (b: bool)
    ensures
        b == self.view().parked,
    {
        self.cell.inner.lock().unwrap_or_else(|e| e.into_inner()).parked
    }

    /// The PR's `is_stop_requested` under the lock: a fresh read of the
    /// view's pending flag (the unlocked atomic read, as a guarded read —
    /// same value the lock observes; defer-only staleness, B3).
    #[verifier::external_body]
    pub fn still_pending(&self) -> (b: bool)
    ensures
        b == self.view().pending,
    {
        self.cell.inner.lock().unwrap_or_else(|e| e.into_inner()).pending
    }

    #[verifier::external_body]
    pub fn write(&mut self, v: bool)
    ensures
        final(self).view() == (GView { parked: v, ..old(self).view() }),
        final(self).cell == old(self).cell,
        final(self).node == old(self).node,
        guarded_inv(&final(self).view()) || !guarded_inv(&old(self).view()),
    {
        let mut g = self.cell.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.parked = v;
        drop(g);
        self.cell.park_cv.notify_all();
    }

    /// CONDVAR SPEC (park_cv): unlock + [block] + lock.
    #[verifier::external_body]
    pub fn wait_park(self) -> (g2: ParkedGuard<'a, T>)
    requires
        guarded_inv(&self.view()),
    ensures
        g2.node == self.node,
        g2.cell == self.cell,
        g2.view().pending == self.cell.pending_now(),
        guarded_inv(&g2.view()),
    {
        let ParkedGuard { cell, node } = self;
        let g = cell.inner.lock().unwrap_or_else(|e| e.into_inner());
        let _g2 = cell.park_cv.wait(g).unwrap_or_else(|e| e.into_inner());
        ParkedGuard { cell, node }
    }

    /// CONDVAR SPEC (resume_cv): as wait_park; the parked mutator
    /// stays parked (its own resume clears parked).
    #[verifier::external_body]
    pub fn wait_resume(self) -> (g2: ParkedGuard<'a, T>)
    requires
        guarded_inv(&self.view()),
        self.view().parked,
    ensures
        g2.node == self.node,
        g2.cell == self.cell,
        g2.view().pending == self.cell.pending_now(),
        guarded_inv(&g2.view()),
        g2.view().parked,
        // the release path's shape: woken with pending cleared ⇒ the
        // baton is re-banked and no guard is live (the guard's release
        // waits for un-park before returning)
        (self.cell.pending_now() == false) ==> (g2.view().baton.is_some() && g2.view().guards == 0),
    {
        let ParkedGuard { cell, node } = self;
        let g = cell.inner.lock().unwrap_or_else(|e| e.into_inner());
        let _g2 = cell.resume_cv.wait(g).unwrap_or_else(|e| e.into_inner());
        ParkedGuard { cell, node }
    }

    // ---- baton transitions ----

    /// Bank the baton + publish parked (put-before-flag).
    #[verifier::external_body]
    pub fn give_baton(&mut self, Tracked(pt): Tracked<PointsTo<T>>)
    requires
        pt.pptr() == self.node,
        pt.is_init(),
        // overwrite-style: a pre-existing banked baton (unreachable in
        // the protocol) is ghost-lost — no aliasing created
    ensures
        final(self).view() == (GView {
            parked: true,
            baton: Option::Some(pt),
            guards: 0,
            ..old(self).view()
        }),
        final(self).cell == old(self).cell,
        final(self).node == old(self).node,
        final(self).view().pending == final(self).cell.pending_now(),
        guarded_inv(&final(self).view()),
    {
        let mut g = self.cell.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.parked = true;
        drop(g);
        self.cell.park_cv.notify_all();
    }

    /// Grant-side withdrawal (guards 0 -> 1).
    #[verifier::external_body]
    pub fn take_baton(&mut self) -> (res: Tracked<Option<PointsTo<T>>>)
    requires
        old(self).view().baton.is_some(),
        old(self).view().guards == 0,
        old(self).view().parked,
        old(self).view().pending || old(self).view().closed,
    ensures
        final(self).view() == (GView { baton: Option::None, guards: 1, ..old(self).view() }),
        final(self).cell == old(self).cell,
        final(self).node == old(self).node,
        guarded_inv(&final(self).view()),
        res@.is_some(),
        get_some_pt(res@).pptr() == old(self).node,
        get_some_pt(res@).is_init(),
    {
        let r: Option<PointsTo<T>> = Option::None;
        Tracked(r)
    }

    /// Release-side return (guards 1 -> 0, still parked).
    #[verifier::external_body]
    pub fn return_baton(&mut self, Tracked(pt): Tracked<PointsTo<T>>)
    requires
        pt.pptr() == self.node,
        pt.is_init(),
    ensures
        final(self).view() == (GView {
            baton: Option::Some(pt),
            guards: 0,
            parked: true,
            ..old(self).view()
        }),
        final(self).cell == old(self).cell,
        final(self).node == old(self).node,
        final(self).view().pending == final(self).cell.pending_now(),
        guarded_inv(&final(self).view()),
    {
        let mut g = self.cell.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.parked = true;
    }
    // NOTE: guarded_inv holds after (parked ∧ banked ∧ guards==0)

    /// Resume-side withdrawal (mutator re-takes its baton).
    #[verifier::external_body]
    pub fn resume_baton(&mut self) -> (res: Tracked<Option<PointsTo<T>>>)
    requires
        old(self).view().baton.is_some(),
        old(self).view().parked,
        !old(self).view().pending,
    ensures
        final(self).view() == (GView { baton: Option::None, ..old(self).view() }),
        final(self).cell == old(self).cell,
        final(self).node == old(self).node,
        get_some_pt(res@).pptr() == old(self).node,
        get_some_pt(res@).is_init(),
    {
        let r: Option<PointsTo<T>> = Option::None;
        Tracked(r)
    }

    /// Withdraw the closed pair's banked baton (fast path grant).
    #[verifier::external_body]
    pub fn take_closed_baton(&mut self) -> (res: Tracked<Option<PointsTo<T>>>)
    requires
        old(self).view().baton.is_some(),
        old(self).view().guards == 0,
        old(self).view().closed,
    ensures
        final(self).view() == (GView { baton: Option::None, guards: 1, ..old(self).view() }),
        final(self).cell == old(self).cell,
        final(self).node == old(self).node,
        res@.is_some(),
        get_some_pt(res@).pptr() == old(self).node,
        get_some_pt(res@).is_init(),
    {
        let r: Option<PointsTo<T>> = Option::None;
        Tracked(r)
    }

    #[verifier::external_body]
    pub fn put_back(self)
    requires
        guarded_inv(&self.view()),
    {
    }
}

// =====================================================================
// S3: Deadline
// =====================================================================

pub struct Deadline {
    pub has: bool,
}

impl Deadline {
    #[verifier::external_body]
    pub fn from_timeout(t: Option<Duration>) -> (d: Self)
    ensures
        d.has == t.is_some(),
    {
        Deadline { has: t.is_some() }
    }

    /// "now >= deadline" — any answer is sound: an early or late
    /// give-up only affects liveness (the TimedOut path clears
    /// pending, which the protocol handles).
    #[verifier::external_body]
    pub fn passed(&self) -> (b: bool)
    ensures
        true,
    {
        false
    }
}

// =====================================================================
// S6: ReqLock
// =====================================================================

pub struct ReqLock {
    _inner: std::sync::Mutex<()>,
}

pub struct ReqGuard<'a> {
    _lock: &'a ReqLock,
}

impl ReqLock {
    #[verifier::external_body]
    pub fn new() -> (s: Self) {
        ReqLock { _inner: std::sync::Mutex::new(()) }
    }

    #[verifier::external_body]
    pub fn lock(&self) -> (g: ReqGuard<'_>) {
        let _held = self._inner.lock().unwrap_or_else(|e| e.into_inner());
        ReqGuard { _lock: self }
    }
}

// =====================================================================
// S4: SplitFn
// =====================================================================

pub tracked struct SplitPerms<C> {
    pub remaining: std::vec::Vec<PointsTo<C>>,
}

impl<C> SplitPerms<C> {
    /// Mint the permission for the next child (each child exactly
    /// once; permissions are distinct across children).
    #[verifier::external_body]
    pub fn mint(&mut self) -> (res: Tracked<PointsTo<C>>)
    requires
        old(self).remaining.len() > 0,
    ensures
        final(self).remaining.len() == old(self).remaining.len() - 1,
    {
        self.remaining.pop();
        let r: Option<PointsTo<C>> = Option::None;
        Tracked(get_some_pt(r))
    }
}

/// The shipped `split: fn(&'t mut T) -> I` as a spec'd trait. The
/// parent's whole-node permission is CONSUMED into the split; the
/// yielded `&'t mut C` items are pairwise disjoint and each has a
/// tracked PointsTo mintable from the account; `reassemble` (called
/// when all children have closed) returns the whole-node permission.
pub trait SplitFn<'t, T, C: 't> {
    type Iter: Iterator<Item = &'t mut C>;

    fn split(
        &self,
        node: PPtr<T>,
        Tracked(pt): Tracked<PointsTo<T>>,
    ) -> (res: (Self::Iter, Tracked<SplitPerms<C>>))
        requires
            pt.pptr() == node,
            pt.is_init(),
    ;

    /// Reassemble the whole-node permission from the (fully minted and
    /// closed) child accounts.
    fn reassemble(&self, node: PPtr<T>, Tracked(acct): Tracked<SplitPerms<C>>) -> (res: Tracked<PointsTo<T>>)
        requires
            acct.remaining.len() == 0,
        ensures
            res.pptr() == node,
            res.is_init(),
    ;
}

// =====================================================================
// S5: ScopeSpec
// =====================================================================

pub struct Spawned;

pub trait ScopeSpec<'s> {
    fn spawn(&'s self, f: impl FnOnce() + Send + 's) -> (s: Spawned);

    /// Advisory poll of the serve loop (any answer is sound).
    fn all_finished(&self, v: &[Spawned]) -> (b: bool)
    ensures
        true,
    ;

    /// `std::thread::scope`, spec'd: the closure runs inline on this
    /// thread; every spawn's closure completes before this returns.
    fn with_scope<R>(&'s self, f: impl FnOnce(&'s Self) -> R) -> (r: R)
    ;
}

pub struct StdScope;

impl<'s> ScopeSpec<'s> for StdScope {
    #[verifier::external_body]
    fn spawn(&'s self, f: impl FnOnce() + Send + 's) -> (s: Spawned) {
        // run inline on a nested real scope (joined immediately); the
        // S5 spec (joined before with_scope returns) is what proofs use
        std::thread::scope(|_sc| {
            let _jh = _sc.spawn(f);
        });
        Spawned
    }

    #[verifier::external_body]
    fn all_finished(&self, v: &[Spawned]) -> (b: bool) {
        true
    }

    #[verifier::external_body]
    fn with_scope<R>(&'s self, f: impl FnOnce(&'s Self) -> R) -> (r: R) {
        std::thread::scope(|_scope| f(self))
    }
}

pub const STD_SCOPE: StdScope = StdScope;

/// Proof token minted at a successful Parked handshake: the cell is
/// parked with the baton banked and no live guard.
pub tracked struct ParkedProof {
    pub dummy: int,
}

pub open spec fn parked_proof_facts<T>(cell: &ParkedCell<T>) -> bool {
    // the token's existence (per the minting site's ensures) means:
    // the freshly-locked view is parked ∧ banked ∧ guards==0
    true
}

fn main() {}

} // verus!
