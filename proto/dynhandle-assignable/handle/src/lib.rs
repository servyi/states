//! The handle subcrate: the two-piece shape per the 16:38 sketch.
//! Shared is Arc-OWNED (no &'o) with a Weak-children registry, so the
//! inner handles can live anywhere (a fanout thread) without borrows
//! into the parent. Simplified handshake flags; structure is the point.

use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};

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

    /// Register a child (kept alive by its owner); deleted = closed,
    /// filtered on merge.
    pub fn register_child(&self, child: &Arc<Shared>) {
        self.children
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(Arc::downgrade(child));
    }

    /// Merge own children into the parent (drop path); the Weak list
    /// skips dead (closed) ones.
    pub fn merge_children_into(&self, parent: &Shared) {
        let mut own = self
            .children
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut target = parent
            .children
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        for weak in own.drain(..) {
            if weak.upgrade().is_some_and(|c| !c.is_closed()) {
                target.push(weak);
            }
        }
    }
}

/// The INNER piece: what a state's code runs on. Points at the state
/// data (owned elsewhere — e.g. inside the Assignable box), carries
/// its own Arc'd tree membership. No lifetimes at all.
pub struct InnerHandle<T: ?Sized> {
    ptr: *mut (),
    status: Arc<Shared>,
    _marker: PhantomData<T>,
}

// SAFETY: the pointer names the state this inner handle was minted
// for (exclusively, by the Assignable); the Shared is Arc-owned.
unsafe impl<T: ?Sized + Send> Send for InnerHandle<T> {}

impl<T: Sized> Deref for InnerHandle<T> {
    type Target = T;
    fn deref(&self) -> &T {
        // SAFETY: minted by the Assignable for this state.
        unsafe { &*(self.ptr as *mut T) }
    }
}

impl<T: Sized> DerefMut for InnerHandle<T> {
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: as Deref, exclusive.
        unsafe { &mut *(self.ptr as *mut T) }
    }
}

impl<T: ?Sized> InnerHandle<T> {
    /// Mint a handle over `data` with this membership.
    ///
    /// # Safety
    ///
    /// `data` must name a live, exclusively-owned `T`
    /// for the handle's whole life; no second handle may
    /// exist over it.
    pub unsafe fn over(data: *mut T, status: Arc<Shared>) -> Self
    where
        T: Sized,
    {
        InnerHandle { ptr: data as *mut (), status, _marker: PhantomData }
    }

    /// Park while a checkpoint has been requested (the identity now
    /// lives HERE, on the inner piece).
    pub fn safepoint(&mut self) {
        if self.status.is_pending() {
            // simplified handshake: spin until released
            while self.status.is_pending() {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
    }
}
