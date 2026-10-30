//! The dynstate library, rebuilt around `Installed<S, T>` (per the
//! 15:50 comment): the erased-box pairing, generalized over the state
//! `S` and the context `T`. All the reusable parts — the wrapper, its
//! derefs, the projection, install — live here; the client carries
//! only its vocabulary and entries, zero unsafe.

use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

use handle::Handle;

/// The installed pair: the context `T` (what the box derefs to) and
/// the state `S` it is paired with. Built solely by [`install`].
pub struct Installed<S, T> {
    ctx: T,
    state: S,
}

impl<S, T> Deref for Installed<S, T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.ctx
    }
}

impl<S, T> DerefMut for Installed<S, T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.ctx
    }
}

/// The erased box: what a chain's handle wraps. The box derefs to the
/// context `T`.
pub type Erased<T> = Box<dyn DerefMut<Target = T>>;

/// install(S, mk) — the DerefMut-impl to box: mk builds the context
/// (the entries) for this `S`; the pairing leaves together.
pub fn install<S, T>(state: S, mk: impl FnOnce() -> T) -> Installed<S, T> {
    Installed { ctx: mk(), state }
}

/// The projection's carrier: a zero-sized type per `(S, T)`, so the
/// (fn-pointer) entries reach the unwrapper statically without
/// capturing a value.
pub struct Proj<S, T>(PhantomData<fn() -> (S, T)>);

pub trait Unwrap<S, T> {
    /// Recover the state's own HANDLE — `&mut Handle<S>` — from the
    /// chain handle, preserving the tree membership (same shared, same
    /// brands), so the state's methods can run on `self: &mut
    /// Handle<Self>` and safepoint through it.
    fn unwrap_handle<'x>(
        h: &'x mut Handle<'_, '_, Erased<T>>,
    ) -> StateHandle<'x, S>;
}

/// The state's own handle: lends the projected handle.
pub struct StateHandle<'a, S> {
    handle: Handle<'a, 'a, S>,
}

impl<'a, S> std::ops::Deref for StateHandle<'a, S> {
    type Target = Handle<'a, 'a, S>;
    fn deref(&self) -> &Self::Target {
        &self.handle
    }
}

impl<'a, S> std::ops::DerefMut for StateHandle<'a, S> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.handle
    }
}

impl<S: 'static, T: 'static> Unwrap<S, T> for Proj<S, T> {
    fn unwrap_handle<'x>(
        h: &'x mut Handle<'_, '_, Erased<T>>,
    ) -> StateHandle<'x, S> {
        // dig the erased object (safe derefs), then land on this
        // pairing's Installed
        let b: &mut Erased<T> = &mut *h;
        let erased = &mut **b;
        // SAFETY: the re-type lands on the `Installed<S, T>` that
        // install::<S, T> built (this impl's `(S, T)` names exactly
        // that pairing); provenance: only the entries mk built for
        // this `S` reach this impl, and those travel only inside the
        // same `Installed`. The state borrow is exclusive.
        let installed =
            unsafe { &mut *(erased as *mut dyn DerefMut<Target = T> as *mut Installed<S, T>) };
        // SAFETY: the projected node is this Installed's state —
        // live, exclusively owned by the box this handle wraps, and
        // no second handle over it exists; the brands/shared ride
        // with the chain handle. The StateHandle lends it for 'x.
        let handle = unsafe { h.project(&mut installed.state) };
        StateHandle { handle }
    }
}
