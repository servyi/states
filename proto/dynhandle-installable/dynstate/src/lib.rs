//! The dynstate library, rebuilt around `Installed<S, T>` (per the
//! 15:50 comment): the erased-box pairing, generalized over the state
//! `S` and the context `T`. All the reusable parts — the wrapper, its
//! derefs, the projection, install — live here; the client carries
//! only its vocabulary and entries, zero unsafe.

use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

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

pub trait UnwrapMut<S, T> {
    /// Recover `&mut S` from the erased box.
    fn unwrap_mut<'x>(erased: &'x mut dyn DerefMut<Target = T>) -> &'x mut S;
}

impl<S: 'static, T: 'static> UnwrapMut<S, T> for Proj<S, T> {
    fn unwrap_mut<'x>(erased: &'x mut dyn DerefMut<Target = T>) -> &'x mut S {
        // SAFETY: the re-type lands on the `Installed<S, T>` that
        // install::<S, T> built (this impl's `(S, T)` names exactly
        // that pairing), and the field borrow after the cast is an
        // ordinary safe borrow. Provenance: this impl is only
        // reachable from the entries mk built for this `S` — and
        // those entries travel only inside the same `Installed` — so
        // the erased object passed here is always this pairing's own.
        unsafe {
            let installed =
                erased as *mut dyn DerefMut<Target = T> as *mut Installed<S, T>;
            &mut (*installed).state
        }
    }
}

/// Dig the erased object out of a chain handle (the safe deref chain:
/// handle → box → dyn).
pub fn erased_mut<'x, T>(
    h: &'x mut handle::Handle<'_, '_, Erased<T>>,
) -> &'x mut dyn DerefMut<Target = T> {
    let b: &mut Erased<T> = &mut *h;
    &mut **b
}
