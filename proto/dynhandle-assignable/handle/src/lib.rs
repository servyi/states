//! The handle subcrate: the two-piece shape per the 16:38 sketch.
//! `InnerHandle` points at a **Cell** — a stable address whose content
//! can be swapped (the Assignable: "a Box a new value can be stored
//! into"). Membership is Arc-OWNED (no `&'o`).

use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};

/// A stable cell whose content can be swapped: the assignable. The
/// cell's address never changes; `data_ptr` tracks the content.
pub trait Cell {
    /// The address of the current content.
    ///
    /// # Safety
    ///
    /// Callers may only read the content through a type the cell
    /// currently holds — i.e. a handle minted after the last store.
    fn data_ptr(&self) -> *mut () {
        self.data_ptr_mut()
    }

    /// The address of the current content, derived with write intent
    /// (through the interior mutability).
    ///
    /// # Safety
    ///
    /// As , and the write path must respect the same
    /// mint-after-store discipline.
    fn data_ptr_mut(&self) -> *mut ();
}

#[repr(C)]
pub struct Shared {
    pending: AtomicBool,
    closed: AtomicBool,
    children: Mutex<Vec<Weak<Shared>>>,
}

impl Shared {
    pub fn new() -> Arc<Self> {
        Arc::new(Shared {
            pending: AtomicBool::new(false),
            closed: AtomicBool::new(false),
            children: Mutex::new(Vec::new()),
        })
    }

    pub fn is_pending(&self) -> bool {
        self.pending.load(Ordering::SeqCst)
    }

    pub fn close(&self) {
        self.closed.store(true, Ordering::SeqCst);
    }

    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
    }

    /// Register a child (kept alive by its owner); deleted = closed.
    pub fn register_child(&self, child: &Arc<Shared>) {
        self.children
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(Arc::downgrade(child));
    }

    /// Merge own children into the parent (drop path); Weak skips
    /// dead/closed ones.
    pub fn merge_children_into(&self, parent: &Shared) {
        let mut own = self.children.lock().unwrap_or_else(|e| e.into_inner());
        let mut target = parent.children.lock().unwrap_or_else(|e| e.into_inner());
        for weak in own.drain(..) {
            if weak.upgrade().is_some_and(|c| !c.is_closed()) {
                target.push(weak);
            }
        }
    }
}

/// The INNER piece: points at a Cell, carries its own Arc'd tree
/// membership. No lifetimes.
pub struct InnerHandle<T: ?Sized> {
    cell: *mut (dyn Cell + 'static),
    status: Arc<Shared>,
    _marker: PhantomData<T>,
}

// SAFETY: the cell names the state this handle was minted over
// (exclusively); the Shared is Arc-owned.
unsafe impl<T: ?Sized + Send> Send for InnerHandle<T> {}

impl<T: Sized> InnerHandle<T> {
    /// Mint a handle over a cell.
    ///
    /// # Safety
    ///
    /// The cell must currently hold a `T`, the cell must outlive this
    /// handle, and this handle must not outlive the next store into
    /// the cell.
    pub unsafe fn over_cell(status: Arc<Shared>, cell: &dyn Cell) -> Self {
        // SAFETY: fat-pointer transmute — identical layout; the
        // lifetime is erased because the discipline above keeps the
        // cell alive for the handle's whole life.
        let raw: *mut (dyn Cell + 'static) =
            unsafe { std::mem::transmute(cell as *const dyn Cell) };
        InnerHandle { cell: raw, status, _marker: PhantomData }
    }
}

impl<T: Sized> Deref for InnerHandle<T> {
    type Target = T;
    fn deref(&self) -> &T {
        // SAFETY: the handle was minted while the cell held a T, and
        // no store happened since (discipline); data_ptr names the T.
        unsafe {
            let cell: *const (dyn Cell + 'static) = self.cell;
            let data = (*cell).data_ptr();
            &*(data as *const T)
        }
    }
}

impl<T: Sized> DerefMut for InnerHandle<T> {
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: as Deref, exclusive.
        unsafe {
            let cell: *mut (dyn Cell + 'static) = self.cell;
            let data = (*cell).data_ptr();
            &mut *(data as *mut T)
        }
    }
}

impl<T: ?Sized> InnerHandle<T> {
    /// Park while a checkpoint has been requested — the identity
    /// lives HERE, on the inner piece.
    pub fn safepoint(&mut self) {
        if self.status.is_pending() {
            while self.status.is_pending() {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
    }
}
