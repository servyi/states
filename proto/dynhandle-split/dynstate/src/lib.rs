//! The ContextData/Unwrapper subcrate: the erased chain MECHANISM —
//! DynHandle, the branded capability, the symbolic constructor,
//! install. The chain vocabulary (State, Outcome, entries) is client
//! code; this crate names neither.

use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

use handle::Handle;

/// The erased chain node: the original handle (erased to ()) plus
/// the client's data. Derefs to the data.
pub struct DynHandle<'o, T> {
    pub(crate) handle: Handle<'o, ()>,
    pub(crate) data: T,
}

impl<'o, T> Deref for DynHandle<'o, T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.data
    }
}
impl<'o, T> DerefMut for DynHandle<'o, T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.data
    }
}

/// The capability, BRANDED: 'o pins the pair whose handles it may
/// re-type. Separated from install's pairing it is unsound, so the
/// witness never leaves except as a type parameter of
/// ContextData::new.
pub trait Unwrap<'o> {
    type Type: 'static;
    fn unwrap_mut<'x, T>(dh: &'x mut DynHandle<'o, T>) -> &'x mut Handle<'o, Self::Type>;
}

pub struct Unwrapper<'o, S> {
    _private: (),
    _p: PhantomData<fn(&'o (), S)>,
}

impl<'o, S: 'static> Unwrap<'o> for Unwrapper<'o, S> {
    type Type = S;
    fn unwrap_mut<'x, T>(dh: &'x mut DynHandle<'o, T>) -> &'x mut Handle<'o, S> {
        // SAFETY: repr(C) layout identity (Handle's parameter sits
        // only behind a thin pointer) + install's pairing: this impl
        // is only named by ContextData::new::<Unwrapper<'o, S>> at an
        // install that paired it with this node in the same call.
        // The brand pins the pair.
        unsafe { &mut *(&mut dh.handle as *mut Handle<'o, ()> as *mut Handle<'o, S>) }
    }
}

/// The secondary trait on the client's data, per the original design:
/// `Unwrap { Type = S }` — the equality binds the capability's type
/// to the chain's state type WITHOUT this crate naming any state
/// trait; the client's impl carries its own `S: State` bound.
pub trait ContextData<'o, S>: Sized {
    fn new<U: Unwrap<'o, Type = S>>() -> Self;
}

/// install — the full implementation:
pub fn install<'o, S: 'static, T: ContextData<'o, S>>(
    h: Handle<'o, S>,
) -> DynHandle<'o, T> {
    // SAFETY: the owned erase — repr(C) layout identity; the node
    // stays solely owned (now by the DynHandle); the brand moves with
    // the value. The pairing is established in this call: the erased
    // S node and the Unwrapper<'o, S> capability leave together.
    let erased: Handle<'o, ()> = unsafe { std::mem::transmute(h) };
    let data = T::new::<Unwrapper<'o, S>>();
    DynHandle { handle: erased, data }
}
