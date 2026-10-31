//! The Handle subcrate with TWO brands: 'o is the PAIR (shared by
//! the whole tree), 'c is the CHILD identity within it (fresh per
//! fanout child; equal to 'o for a top-level chain). Both invariant.

use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

pub struct Shared(());

// ── delta vs main: the second brand + #[repr(C)] ──────────────────
#[repr(C)]
pub struct Handle<'o, 'c, T> {
    pub(crate) node: *mut T,
    pub(crate) shared: &'o Shared,
    /// Invariant in BOTH brands: the pair brand cannot shrink to a
    /// foreign pair; the child brand cannot shrink to a foreign child.
    pub(crate) _brand: PhantomData<fn(&'o (), &'c ()) -> (&'o (), &'c ())>,
}

impl<'o, 'c, T> Deref for Handle<'o, 'c, T> {
    type Target = T;
    fn deref(&self) -> &T {
        // SAFETY: the node is exclusively owned by this handle and
        // not parked — the documented type invariant.
        unsafe { &*self.node }
    }
}

impl<'o, 'c, T> DerefMut for Handle<'o, 'c, T> {
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: as Deref, exclusive.
        unsafe { &mut *self.node }
    }
}

impl<'o, 'c, T> Handle<'o, 'c, T> {
    pub fn safepoint(&mut self) {}
}

impl<'o, 'c, T> Handle<'o, 'c, T> {
    /// The node pointer (crate-family internal: the move-out and the
    /// re-point below are the library's owned-erase steps — public
    /// only to the sibling crates of this workspace).
    #[doc(hidden)]
    pub fn node_ptr(&mut self) -> *mut T {
        self.node
    }

    /// Re-point this (consumed) handle at a fresh boxed node of a
    /// DIFFERENT type: the brands ride with the value.
    ///
    /// # Safety
    ///
    /// The caller must have moved out of (or dropped) the old node,
    /// and must pass sole ownership of `boxed`. The old node must
    /// never be touched again through this handle.
    #[doc(hidden)]
    pub unsafe fn repoint_boxed<U>(
        self,
        boxed: Box<U>,
    ) -> Handle<'o, 'c, U> {
        let node = Box::into_raw(boxed);
        // SAFETY (caller-provided): repr(C) layout identity — the
        // type parameter sits only behind a thin pointer; shared and
        // the brands are untouched by the parameter change.
        let mut h: Handle<'o, 'c, U> = unsafe { std::mem::transmute(self) };
        h.node = node;
        h
    }
}

impl<'o, 'c, T> Handle<'o, 'c, T> {
    /// Lend a handle over a projected node, INSIDE a closure: the
    /// temporary `Handle<'o, 'c, S>` KEEPS this handle's brands (they
    /// are its tree identity — rebranding would open cross-child
    /// application); `f` runs with it; the temporary dies before this
    /// borrow ends, so no ownership is given out.
    ///
    /// # Safety
    ///
    /// `node` must point at a live, exclusively-owned `S` for the
    /// whole call, and this must not create a second concurrent
    /// handle over the same node.
    pub unsafe fn with_projected<S, R>(
        &mut self,
        node: *mut S,
        f: impl FnOnce(&mut Handle<'o, 'c, S>) -> R,
    ) -> R {
        let mut projected = Handle { node, shared: self.shared, _brand: PhantomData };
        f(&mut projected)
    }
}

/// A top-level pair: one brand for both axes (the chain IS the only
/// child of its pair).
pub fn with_pair<T, R>(
    mut node: T,
    f: impl for<'o> FnOnce(Handle<'o, 'o, T>) -> R,
) -> R {
    let shared = Shared(());
    f(Handle {
        node: &mut node,
        shared: &shared,
        _brand: PhantomData,
    })
}

/// The fan-out shape: the parent is 'o-branded; each action call is
/// higher-ranked over the CHILD brand 'c — the implementation
/// instantiates a fresh 'c per child (the discipline), and client
/// code can never name one.
pub fn fanout_two_children_quiet<T>(
    a: &mut T,
    b: &mut T,
    action: impl for<'c> Fn(Handle<'_, 'c, T>),
) {
    // The mint: two children, two instantiations of 'c. Nothing at
    // the type level FORCES these to differ (region inference may
    // unify) — this is the implementation-discipline trust anchor,
    // double-checkable at runtime inside unwrap_mut.
    action(Handle { node: a as *mut T, shared: &Shared(()), _brand: PhantomData });
    action(Handle { node: b as *mut T, shared: &Shared(()), _brand: PhantomData });
}

#[allow(dead_code)]
pub fn fanout_two_children<T, R>(
    a: &mut T,
    b: &mut T,
    action: impl for<'c> Fn(Handle<'_, 'c, T>),
) -> R {
    fanout_two_children_quiet(a, b, action);
    unimplemented!("prototype: kept for the attack documentation")
}
