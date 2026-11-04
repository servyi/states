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
use std::sync::{Arc, Weak};

use rayon::prelude::*;

/// The parking/membership state shared by a handle and its children
/// (the sketch's `status`): children are weakly registered here —
/// managed like before, deleted = closed. Management itself is NOT
/// part of this step; only the structure.
pub struct Shared {
    children: Vec<Weak<Shared>>,
}

/// The inner (non-owning) handle: a raw pointer plus the phantom
/// that remembers the pointee's type. Deref/DerefMut land on `T`.
pub struct InnerHandle<T: ?Sized> {
    ptr: *mut (),
    status: Arc<Shared>,
    _marker: PhantomData<T>,
}

// The raw pointer is exclusively owned by this handle for its whole
// life (the HandleAble contract) — moving the handle to another
// thread moves the exclusive access with it.
unsafe impl<T: ?Sized> Send for InnerHandle<T> where Arc<Shared>: Send {}

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
/// The `Unsize` discipline lives on the blanket impl only: `T:
/// HandleAble<K>` exists exactly when `InnerHandle<T>: Unsize<K>`.
/// (Not on the trait definition — the erased `Self` at dyn call
/// sites is statically unknowable, so the clause could never be
/// checked there; maintainer ruling, #19.)
pub unsafe trait HandleAble<K: ?Sized> {
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
            status: Arc::new(Shared { children: Vec::new() }),
            _marker: PhantomData,
        });
        boxed
    }
}

impl<T> InnerHandle<T> {
    /// The fan-out (no protocol yet — plain rayon). `split` runs on
    /// `&mut T` (the node's own fields) and yields the erased
    /// children `&mut Box<dyn HandleAble<V>>`; each element is turned
    /// into a `Box<V>` via `box_innerhandle`, and `f` runs on one
    /// rayon task per box.
    pub fn fanout<V: ?Sized + Send, F>(&mut self, split: &(dyn for<'x> Fn(&'x mut T) -> Vec<&'x mut Box<dyn HandleAble<V>>> + Send + Sync), f: F)
    where
        F: Fn(Box<V>) + Send + Sync,
        Box<V>: Send,
    {
        // Collect offsite under the &mut node borrow (the root's
        // structure); the split's references die at the raw
        // conversion, the earliest point.
        let child_refs: Vec<&mut Box<dyn HandleAble<V>>> = {
            let node: &mut T = self;
            split(node)
        };
        // MINT THE BOXES: each erased element becomes a Box<V> via
        // the blanket impl's box_innerhandle (fresh root Shared for
        // each minted InnerHandle, as before).
        let boxes: Vec<Box<V>> = child_refs
            .into_iter()
            .map(|child| (**child).box_innerhandle())
            .collect();
        // THEN TASKS: one rayon task per box.
        boxes.into_par_iter().for_each(|b| f(b));
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
