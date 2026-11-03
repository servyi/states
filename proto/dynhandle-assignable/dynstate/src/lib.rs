//! The dynstate library: the Assignable — literally **a Box a new
//! value can be stored into** (a stable cell, swapped content) — plus
//! `set`, the sketch's four-step successor install, and the erased
//! install seam.
#![feature(unsize)]

use std::cell::RefCell;
use std::marker::Unsize;
use std::sync::Arc;

use handle::{Cell, InnerHandle, Shared};

/// A Box a new value can be stored into: ONE stable cell; the content
/// is replaced by `store`. The cell's address never changes — that is
/// what inner handles point at.
pub struct Assignable<V: ?Sized + 'static> {
    slot: RefCell<Option<Box<V>>>,
}

impl<V: ?Sized + 'static> Assignable<V> {
    /// A cell holding the first value.
    pub fn new(first: Box<V>) -> Self {
        Assignable { slot: RefCell::new(Some(first)) }
    }
}

impl<V: ?Sized + 'static> Cell for Assignable<V> {
    fn data_ptr_mut(&self) -> *mut () {
        match &mut *self.slot.borrow_mut() {
            Some(b) => &mut **b as *mut V as *mut (),
            None => std::ptr::null_mut(),
        }
    }
}

/// Mint a typed inner handle over a cell.
///
/// # Safety
///
/// The cell must currently hold a `T`; the handle must not outlive
/// the next store.
pub unsafe fn mint<T: 'static, V: ?Sized + 'static>(
    status: Arc<Shared>,
    cell: &Assignable<V>,
) -> InnerHandle<T> {
    InnerHandle::over_cell(status, cell)
}

/// set — the sketch's four steps: (1) store the new value into the
/// assignable (the old content drops here; the outer box's old typed
/// handle must already be gone), (2) the old typed view is forgotten,
/// (3) the membership is cloned (the same Arc), (4) a fresh typed
/// handle over the SAME cell is returned — the outer box swaps to it.
pub fn set<T: 'static, V: ?Sized + 'static>(
    status: &Arc<Shared>,
    cell: &Assignable<V>,
    new: Box<V>,
) -> InnerHandle<T> {
    cell.store(new);
    // SAFETY: the cell now holds the new T.
    unsafe { mint(status.clone(), cell) }
}

impl<V: ?Sized + 'static> Assignable<V> {
    /// Store a new value into the cell — interior mutability: the
    /// cell itself is only ever borrowed SHAREDLY, so handles' raw
    /// provenance survives stores.
    pub fn store(&self, new: Box<V>) {
        *self.slot.borrow_mut() = Some(new);
    }
}

/// Erase a sized inner handle to the vocabulary view (the Unsize
/// coercion the sketch relies on).
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
