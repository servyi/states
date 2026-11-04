//! The library half of the runnable sketch (issue #18, 16:38):
//! the typing discipline — erased outer `Box<K>` reassignment through
//! `Unsize`-witnessed handles — with no vocabulary in it. The client
//! (binary crate) supplies the trait and the structs; everything
//! here is mechanism only.
//!
//! Shortcut kept from the sketch: `InnerHandle` points directly at
//! the value (`ptr: *mut ()` cast on deref).

#![feature(unsize)]

use std::marker::{PhantomData, Unsize};
use std::ops::{Deref, DerefMut};

/// The inner (non-owning) handle: a raw pointer plus the phantom
/// that remembers the pointee's type. Deref/DerefMut land on `T`.
pub struct InnerHandle<T: ?Sized> {
    ptr: *mut (),
    _marker: PhantomData<T>,
}

/// Anything that can be viewed as an erased `K`, by handing out a
/// boxed inner handle to it. The `Unsize` where-clause is the whole
/// engine: it is what lets `InnerHandle<Self>` coerce into the
/// erased `K` without naming a sized type at the call site.
///
/// # Safety
///
/// `box_innerhandle` must return a handle whose pointer is
/// exclusively-live for the handle's use, pointing at `self`'s data
/// (the sketch's shortcut: the pointee IS `&mut self`).
pub unsafe trait HandleAble<K: ?Sized>
where
    InnerHandle<Self>: Unsize<K>,
{
    fn box_innerhandle(&mut self) -> Box<K>;
}

unsafe impl<T, K: ?Sized> HandleAble<K> for T
where
    InnerHandle<T>: Unsize<K>,
{
    fn box_innerhandle(&mut self) -> Box<K> {
        // bind at the sized type, then let the return coerce
        // Box<InnerHandle<T>> -> Box<K> (the Unsize where-clause).
        let boxed: Box<InnerHandle<T>> = Box::new(InnerHandle {
            ptr: self as *mut Self as *mut (),
            _marker: PhantomData,
        });
        boxed
    }
}

impl<T: Sized> Deref for InnerHandle<T> {
    type Target = T;
    fn deref(&self) -> &T {
        // SAFETY: the pointer is exclusively-live for this handle's
        // use and points at a `T` (the HandleAble contract).
        unsafe { &*(self.ptr as *mut T) }
    }
}

impl<T: Sized> DerefMut for InnerHandle<T> {
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: as Deref, exclusive.
        unsafe { &mut *(self.ptr as *mut T) }
    }
}
