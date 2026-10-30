//! dynstate with the two-brand identity: 'o (pair) + 'c (child).

use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

use handle::Handle;

/// The erased chain node, branded by pair AND child.
pub struct DynHandle<'o, 'c, T> {
    pub(crate) handle: Handle<'o, 'c, ()>,
    pub(crate) data: T,
}

impl<'o, 'c, T> Deref for DynHandle<'o, 'c, T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.data
    }
}
impl<'o, 'c, T> DerefMut for DynHandle<'o, 'c, T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.data
    }
}

pub trait Unwrap<'o, 'c> {
    type Type: 'static;
    fn unwrap_mut<'x, T>(
        dh: &'x mut DynHandle<'o, 'c, T>,
    ) -> &'x mut Handle<'o, 'c, Self::Type>;
}

pub struct Unwrapper<'o, 'c, S> {
    _private: (),
    _p: PhantomData<fn(&'o (), &'c (), S)>,
}

impl<'o, 'c, S: 'static> Unwrap<'o, 'c> for Unwrapper<'o, 'c, S> {
    type Type = S;
    fn unwrap_mut<'x, T>(
        dh: &'x mut DynHandle<'o, 'c, T>,
    ) -> &'x mut Handle<'o, 'c, S> {
        // SAFETY: repr(C) layout identity + install's pairing + BOTH
        // brands: the pair pins the tree, the child brand pins the
        // node within it — a capability minted for child A does not
        // typecheck against child B's DynHandle.
        unsafe {
            &mut *(&mut dh.handle as *mut Handle<'o, 'c, ()>
                as *mut Handle<'o, 'c, S>)
        }
    }
}

pub trait ContextData<'o, 'c, S>: Sized {
    fn new<U: Unwrap<'o, 'c, Type = S>>() -> Self;
}

pub fn install<'o, 'c, S: 'static, T: ContextData<'o, 'c, S>>(
    h: Handle<'o, 'c, S>,
) -> DynHandle<'o, 'c, T> {
    // SAFETY: the owned erase — repr(C) layout identity; node stays
    // solely owned; both brands move with the value; the pairing
    // (erased S node + Unwrapper<'o, 'c, S>) leaves together.
    let erased: Handle<'o, 'c, ()> = unsafe { std::mem::transmute(h) };
    let data = T::new::<Unwrapper<'o, 'c, S>>();
    DynHandle { handle: erased, data }
}
