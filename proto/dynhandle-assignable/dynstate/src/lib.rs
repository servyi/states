#![feature(unsize)]
//! The dynstate library for the two-piece shape (16:38 sketch): the
//! `Assignable` — the swappable storage IN the tree (value + Arc'd
//! membership) — plus `set`, the successor install that re-mints the
//! inner handle over the new value with the SAME tree identity, and
//! the `HandleAble` blanket (the sketch's shortcut path).

use std::marker::Unsize;
use std::sync::Arc;

use handle::{InnerHandle, Shared};

/// The swappable storage: one heap'd value, its tree membership. The
/// value's pointer is what inner handles are minted over.
pub struct Assignable {
    data: *mut (),
    drop_fn: unsafe fn(*mut ()),
    status: Arc<Shared>,
}

impl Drop for Assignable {
    fn drop(&mut self) {
        // SAFETY: drop_fn was made for exactly the stored value.
        unsafe { (self.drop_fn)(self.data) };
    }
}

fn dropper<T: 'static>() -> unsafe fn(*mut ()) {
    unsafe fn drop_value<T: 'static>(p: *mut ()) {
        drop(unsafe { Box::from_raw(p as *mut T) });
    }
    drop_value::<T>
}

impl Assignable {
    /// Wrap a value under a parent: heap it, mint the inner handle
    /// pointing at it, register the fresh membership with the parent.
    pub fn new<T: 'static>(value: T, parent: &Arc<Shared>) -> (Self, InnerHandle<T>) {
        let status = Shared::new();
        parent.register_child(&status);
        let boxed = Box::into_raw(Box::new(value));
        let assignable = Assignable {
            data: boxed as *mut (),
            drop_fn: dropper::<T>(),
            status: status.clone(),
        };
        // SAFETY: the pointer names the value this handle was minted
        // for, owned by the assignable; exclusive.
        let inner = unsafe { InnerHandle::over(boxed, status) };
        (assignable, inner)
    }

    /// The successor install: allocate the new value, re-mint the
    /// inner handle over it with the SAME membership, drop the old
    /// value. The old inner handle (already replaced in the chain)
    /// never touches the dead storage.
    pub fn set<T: 'static>(&mut self, value: T) -> InnerHandle<T> {
        let boxed_new = Box::into_raw(Box::new(value));
        // SAFETY: as new — pointer names the new value, membership
        // carried over by the Arc clone.
        let inner = unsafe { InnerHandle::over(boxed_new, self.status.clone()) };
        let old_data = self.data;
        let old_drop = self.drop_fn;
        self.data = boxed_new as *mut ();
        self.drop_fn = dropper::<T>();
        // SAFETY: the OLD value dies here by the OLD dropper, after
        // every handle to it.
        unsafe { (old_drop)(old_data) };
        inner
    }

    /// The membership this assignable (and its inner handles) park on.
    pub fn status(&self) -> &Arc<Shared> {
        &self.status
    }
}

/// The sketch's `HandleAble`: any sized state can produce its own
/// inner handle, erased to `V`.
///
/// # Safety
///
/// The produced box must point back at the state this was called on,
/// for the box's whole life; only this crate implements the trait.
pub unsafe trait HandleAble<V: ?Sized + 'static>
where
    InnerHandle<Self>: Unsize<V>,
{
    fn box_innerhandle(&mut self) -> Box<V>;
}

unsafe impl<T: 'static, V: ?Sized + 'static> HandleAble<V> for T
where
    InnerHandle<T>: Unsize<V>,
{
    fn box_innerhandle(&mut self) -> Box<V> {
        // SAFETY: the fresh handle points back at this state,
        // exclusively, for the box's whole life. NOTE: unconnected
        // membership (a root Shared) — the tree-connected path is
        // Assignable::new/set; see the client's variant notes.
        let status = Shared::new();
        let raw = Box::into_raw(Box::new(unsafe { InnerHandle::<T>::over(
            self as *mut Self,
            status,
        ) }));
        unsafe { Box::from_raw(raw) }
    }
}

/// Erase a sized inner handle to the vocabulary view (the coercion
/// the sketch gets from `InnerHandle<T>: Unsize<V>`).
pub fn erase<V: ?Sized + 'static, T: 'static>(inner: InnerHandle<T>) -> Box<V>
where
    InnerHandle<T>: Unsize<V>,
{
    // SAFETY: InnerHandle<T>: Unsize<V> is exactly the coercion
    // contract; the box is fresh and solely owned.
    unsafe {
        let raw = Box::into_raw(Box::new(inner));
        let _typed: *mut V = raw;
        Box::from_raw(_typed)
    }
}
