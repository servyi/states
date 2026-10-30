//! The Handle subcrate: the EXISTING checkpoint Handle — the delta
//! vs main is exactly one line: `#[repr(C)]` (marked). Everything
//! else is as in checkpoint.rs today.

use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

pub struct Shared(());

// ── DELTA: the one added line ─────────────────────────────────────
#[repr(C)]
// ── end delta ─────────────────────────────────────────────────────
pub struct Handle<'o, T> {
    pub(crate) node: *mut T,
    pub(crate) shared: &'o Shared,
    pub(crate) _brand: PhantomData<fn(&'o ()) -> &'o ()>,
}

impl<'o, T> Deref for Handle<'o, T> {
    type Target = T;
    fn deref(&self) -> &T {
        // SAFETY: the node is exclusively owned by this handle and
        // not parked — the documented type invariant.
        unsafe { &*self.node }
    }
}

impl<'o, T> DerefMut for Handle<'o, T> {
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: as Deref, exclusive.
        unsafe { &mut *self.node }
    }
}

impl<'o, T> Handle<'o, T> {
    pub fn safepoint(&mut self) {}

}

pub fn with_pair<T, R>(mut node: T, f: impl for<'o> FnOnce(Handle<'o, T>) -> R) -> R {
    let shared = Shared(());
    f(Handle { node: &mut node, shared: &shared, _brand: PhantomData })
}
