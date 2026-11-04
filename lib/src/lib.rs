//! The library half of the runnable sketch (issue #18, 16:38):
//! the typing discipline — erased outer `Box<K>` reassignment through
//! `Unsize`-witnessed handles — with no vocabulary in it.
//!
//! Shortcut kept from the sketch (still, per the 00:09 step): the
//! InnerHandle points at the HOLDER (the erased box slot), derefs
//! through the holder chain onto T.

#![feature(unsize)]

use std::marker::{PhantomData, Unsize};
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex, Weak};

/// The parking/membership state shared by a handle and its children
/// (the sketch's `status`): children are weakly registered here —
/// managed like before, deleted = closed.
///
/// # The lock convention
///
/// The children vec lives behind a Mutex, and every user follows the
/// same rule to avoid the insert/check deadlock: **open the mutex
/// only for the immediate operation — insert, or snapshot — and
/// never hold it across anything that can itself insert** (driving a
/// child, parking, minting). Checkers snapshot under the lock,
/// release, then act on the snapshot; mutators insert under the lock
/// and release. A thread holding the lock while another wants to
/// insert right before it reads the children's status would deadlock
/// — the convention above is the guard, and it is easy to maintain
/// because the lock guard never crosses a call.
pub struct Shared {
    children: Mutex<Vec<Weak<Shared>>>,
}

impl Shared {
    fn new() -> Self {
        Shared { children: Mutex::new(Vec::new()) }
    }

    /// Register a child (insert under the lock, release — the
    /// convention). The child stays alive by its owner's Arc.
    fn register_child(&self, child: &Arc<Shared>) {
        self.children
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(Arc::downgrade(child));
    }

    /// The live (not-yet-dead) children, snapshotted under the lock
    /// and released — the checker's half of the convention. A Weak
    /// that no longer upgrades is a closed child (deleted = closed)
    /// and is skipped.
    pub fn live_children(&self) -> Vec<Arc<Shared>> {
        self.children
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter_map(Weak::upgrade)
            .collect()
    }
}

/// Deref/DerefMut land on `T` through the holder chain: the fat
/// pointer -> &mut dyn DerefMut<..> -> derefmut -> &mut dyn
/// HandleAble<V> -> cast -> &mut T.
/// The erased slot any holder presents: derefs to the erased state.
pub type ErasedSlot<V> = dyn DerefMut<Target = dyn HandleAble<V>>;

/// The OUTER handle: purely a type marker over the current inner
/// handle (`Box<V>`, which it knows is an `InnerHandle<_, V>`
/// pointing at a `K`-shaped slot) — it owns NO slot. `replace` is
/// the successor install: walk to the slot through the current
/// inner handle's own pointer, swap the slot via `K: ReplaceAble<V>`,
/// re-mint, and keep the fresh inner handle in the box.
pub struct OuterHandle<K: ?Sized, V: ?Sized> {
    current: Box<V>,
    _slot: PhantomData<fn(*mut K, *mut V)>,
}

pub struct InnerHandle<T: ?Sized, V: ?Sized> {
    ptr: *mut dyn DerefMut<Target = dyn HandleAble<V>>,
    status: Arc<Shared>,
    _marker: PhantomData<fn(*mut T, *mut V)>,
}

// The raw pointer is exclusively owned by this handle for its whole
// life (the HandleAble contract) — moving the handle to another
// thread moves the exclusive access with it. T: Send because the
// pointee crosses with it (the big-PR lesson, applied here).
unsafe impl<T: ?Sized + Send, V: ?Sized> Send for InnerHandle<T, V> {}

// The marker's K crosses threads only as a type; the box's Send is
// the payload that matters.
unsafe impl<K: ?Sized, V: ?Sized + Send> Send for OuterHandle<K, V> {}

impl<K: ?Sized, V: ?Sized> OuterHandle<K, V> {
    /// The current chain's membership (for registry probes).
    pub fn shared(&self) -> &Arc<Shared> {
        // SAFETY: the box is an InnerHandle<_, V> (the invariant) —
        // reading its status field through the erased view.
        unsafe {
            &(*(self.current.as_ref() as *const V as *const InnerHandle<(), V>)).status
        }
    }
}

impl<K: ?Sized, V: ?Sized> std::ops::Deref for OuterHandle<K, V> {
    type Target = V;
    fn deref(&self) -> &V {
        // the box is the load-bearing invariant (always an
        // InnerHandle<_, V> over the K-shaped slot) — deref lands on
        // the V through it, the box itself is never handed out
        // SAFETY: the box holds an InnerHandle<_, V> (the invariant);
        // viewing it as the V it erases is the Unsize correspondence.
        unsafe { &*(self.current.as_ref() as *const V) }
    }
}

impl<K: ?Sized, V: ?Sized> std::ops::DerefMut for OuterHandle<K, V> {
    fn deref_mut(&mut self) -> &mut V {
        // SAFETY: as Deref — exclusive.
        unsafe { &mut *(self.current.as_mut() as *mut V) }
    }
}

impl<K, V: ?Sized + 'static> OuterHandle<K, V>
where
    K: ReplaceAble<V> + DerefMut<Target = dyn HandleAble<V>> + 'static,
{
    /// Mint over a live slot the driver owns (the slot must outlive
    /// every handle minted over it — the driver's discipline).
    ///
    /// # Safety
    ///
    /// `slot` must stay live and exclusively reachable through this
    /// chain for the OuterHandle's whole life.
    pub unsafe fn over_slot(slot: *mut K) -> Self
    where
        K: Sized,
    {
        let erased_slot: *mut dyn DerefMut<Target = dyn HandleAble<V>> = slot;
        // SAFETY: the caller's contract above.
        let erased: &mut dyn HandleAble<V> = unsafe { &mut **erased_slot };
        // the chain's ORIGIN: a fresh root Shared moves in
        let shared = Arc::new(Shared::new());
        let current = erased.box_innerhandle(erased_slot, shared);
        OuterHandle { current, _slot: PhantomData }
    }

    /// The successor install, per the sketch: walk to the slot through
    /// the CURRENT inner handle's own pointer, swap the slot via
    /// `K: ReplaceAble<V>`, then re-mint through the replacement's
    /// `HandleAble` and keep the fresh inner handle in the box.
    pub fn replace(&mut self, val: Box<dyn HandleAble<V> + 'static>) {
        // 1) the current inner handle: the box IS one (the OuterHandle's
        //    invariant); view it as &mut InnerHandle<(), V> to load the
        //    pointer without naming the state type
        // SAFETY: OuterHandle's invariant — current is always an
        // InnerHandle<_, V> minted over a K-shaped slot; viewing it as
        // InnerHandle<(), V> only reads the ptr field (the () never
        // derefs — the handle is replaced, not driven, in this method).
        let current: &mut InnerHandle<(), V> =
            unsafe { &mut *(self.current.as_mut() as *mut V as *mut InnerHandle<(), V>) };
        // the old handle's shared is PASSED ON to the successor — the
        // chain's membership survives the install
        let shared = current.status.clone();
        // 2) load its ptr, cast it to the slot
        // SAFETY: the slot is live and exclusively reachable through
        // this chain (the mint's discipline; the old inner handle is
        // about to be dropped with the box swap).
        let slot: *mut K = current.ptr as *mut K;
        // 3) swap the slot through ReplaceAble — the slot now HOLDS val
        // SAFETY: as above — the write lands in the K-shaped slot the
        // current chain exclusively owns.
        unsafe { *slot = K::new(val) };
        // 4) re-mint: use the HandleAble OF val — reachable through the
        // slot it now lives in (K's own DerefMut) — to create a new
        // InnerHandle pointing at that same slot, and assign it into
        // the own box
        // SAFETY: the mint's discipline over the fresh slot content;
        // the receiver and the holder pointer derive from the same
        // raw — one borrow stack.
        let fresh: Box<V> = unsafe {
            let erased: &mut dyn HandleAble<V> = (*slot).deref_mut();
            erased.box_innerhandle(
                slot as *mut dyn DerefMut<Target = dyn HandleAble<V>>,
                shared,
            )
        };
        self.current = fresh;
    }
}

/// Anything that can be viewed as an erased `K`, by handing out a
/// boxed inner handle to it. The `Unsize` where-clause is the whole
/// (Not on the trait definition — the erased `Self` at dyn call
/// sites is statically unknowable, so the clause could never be
/// checked there; maintainer ruling, #19.)
/// HandleAble<T, K, V>: V is the erased vocabulary, K is the HOLDER
/// type (static again). In the design's own terms:
/// `K = Box<dyn HandleAble<V, K>>` — K occurs in its own definition.
/// Raw aliases cannot express that (E0391 cycle); a NEWTYPE can
/// (the linked-list indirection), so the intended K in practice is
/// the newtype wrapping that Box.
/// # Safety
///
/// `holder` must point at the holder exclusively-live for the minted
/// handle's whole life, holding `Self` at mint time.
pub unsafe trait HandleAble<V: ?Sized>: Send {
    fn box_innerhandle(
        &mut self,
        holder: *mut dyn DerefMut<Target = dyn HandleAble<V>>,
        shared: Arc<Shared>,
    ) -> Box<V>;
}

unsafe impl<T: Send, V: ?Sized> HandleAble<V> for T
where
    InnerHandle<T, V>: Unsize<V>,
{
    fn box_innerhandle(
        &mut self,
        holder: *mut dyn DerefMut<Target = dyn HandleAble<V>>,
        shared: Arc<Shared>,
    ) -> Box<V> {
        // bind at the sized type, then let the return coerce
        // Box<InnerHandle<T, V>> -> Box<V> (the Unsize where-clause).
        let handle: Box<InnerHandle<T, V>> = Box::new(InnerHandle {
            ptr: holder,
            status: shared,
            _marker: PhantomData,
        });
        handle
    }
}

/// A holder that can REPLACE itself with a fresh `T` (the successor
/// install): `new<T: HandleAble<K, V>>(T) -> K`. The K parameter
/// is the holder being replaced (Self's own type).
/// # Safety
///
/// The replacement must be held exclusively by the new Self; the
/// caller must have dropped every handle minted over the old content.
pub unsafe trait ReplaceAble<V: ?Sized> {
    fn new(replacement: Box<dyn HandleAble<V> + 'static>) -> Self
    where
        Self: Sized;
}
// The bypass (2a9957b): K decoupled from Self, so the Box pass-through
// exists generically:
//   impl<K: ?Sized, V: ?Sized> ReplaceAble<K, V> for Box<dyn HandleAble<K, V>>
//       { fn new(replacement: Box<dyn HandleAble<K, V>>) -> Self { replacement } }

unsafe impl<V: ?Sized + 'static> ReplaceAble<V> for Box<dyn HandleAble<V> + 'static> {
    fn new(replacement: Box<dyn HandleAble<V> + 'static>) -> Self {
        replacement
    }
}

impl<T: Sized, V: ?Sized> InnerHandle<T, V> {
    /// The fan-out (no protocol yet — plain scoped threads). `split`
    /// runs on `&mut T` (the node's own fields) and yields the
    /// children as the erased slots (`&mut Box<dyn HandleAble<V>>` in
    /// the demo — any DerefMut-to-dyn holder works). Each element
    /// becomes a `Box<V>` via `box_innerhandle`, and `f` runs on one
    /// scoped thread per box.
    pub fn fanout<K, F>(
        &mut self,
        split: &(dyn for<'x> Fn(&'x mut T) -> Vec<&'x mut K> + Send + Sync),
        f: F,
    ) where
        T: Send,
        V: Send,
        K: DerefMut<Target = dyn HandleAble<V>> + Sized + Send + 'static,
        F: Fn(Box<V>) + Send + Sync,
    {
        // Collect offsite under the &mut node borrow (the root's
        // conversion, the earliest point — the ROOT's pattern, which
        // also solves the tag problem: once only raws exist, both
        // the receiver (`&mut **raw`, a Unique DERIVED from the raw)
        // and the holder pointer (`raw` itself) live in the same
        // borrow stack — no sibling retag ever invalidates the other.
        let child_raws: Vec<*mut K> = {
            let node: &mut T = self;
            split(node).into_iter().map(std::ptr::from_mut).collect()
        };
        // MINT THE BOXES: each slot becomes a Box<V> via the blanket
        // impl — the receiver derives from the raw (one borrow
        // stack), the holder pointer is the raw itself.
        let boxes: Vec<Box<V>> = child_raws
            .into_iter()
            .map(|raw| {
                // the erased SLOT pointer first (raw, no borrow):
                // SAFETY: the split yielded each child exactly
                // once, disjointly; exclusive for this mint.
                let slot_ptr: *mut ErasedSlot<V> = raw as *mut ErasedSlot<V>;
                // the erased receiver: through the slot's own
                // DerefMut — derived from the raw, one borrow stack
                // SAFETY: as above.
                let erased: &mut dyn HandleAble<V> = unsafe { (*raw).deref_mut() };
                // each child gets its OWN Shared (its parking
                // identity), weakly registered into the parent's vec
                // — insert under the lock, release (the convention)
                let child_shared = Arc::new(Shared::new());
                self.status.register_child(&child_shared);
                erased.box_innerhandle(slot_ptr, child_shared)
            })
            .collect();
        // THEN THREADS: one dedicated scoped thread per box (std, per
        // the 23:55 step — rayon left only noise in Miri)
        std::thread::scope(|scope| {
            for b in boxes {
                scope.spawn(|| f(b));
            }
        });
    }
}

impl<T: Sized, V: ?Sized> Deref for InnerHandle<T, V> {
    type Target = T;
    fn deref(&self) -> &T {
        // 1) cast into the holder (shared read path)
        // SAFETY: exclusively-live per the HandleAble contract.
        let holder: &dyn DerefMut<Target = dyn HandleAble<V>> = unsafe { &*self.ptr };
        // 2) deref that into the erased view
        let erased: &dyn HandleAble<V> = &**holder;
        // 3) cast that into &T
        // SAFETY: the HandleAble contract — the holder holds the `T`
        // this handle was minted for, exclusively-live.
        unsafe { &*(erased as *const dyn HandleAble<V> as *const T) }
    }
}

impl<T: Sized, V: ?Sized> DerefMut for InnerHandle<T, V> {
    fn deref_mut(&mut self) -> &mut T {
        // 1) cast into the holder
        // SAFETY: as Deref, exclusive.
        let holder: &mut dyn DerefMut<Target = dyn HandleAble<V>> = unsafe { &mut *self.ptr };
        // 2) derefmut that into the erased view
        let erased: &mut dyn HandleAble<V> = holder.deref_mut();
        // 3) cast that into &mut T
        // SAFETY: as Deref — exclusive, the minted-for T.
        unsafe { &mut *(erased as *mut dyn HandleAble<V> as *mut T) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two live children registered, snapshot sees both.
    #[test]
    fn registry_snapshots_live_children() {
        let parent = Arc::new(Shared::new());
        let a = Arc::new(Shared::new());
        let b = Arc::new(Shared::new());
        parent.register_child(&a);
        parent.register_child(&b);
        assert_eq!(parent.live_children().len(), 2);
    }

    /// A dropped child is closed: its Weak no longer upgrades, the
    /// snapshot skips it (deleted = closed) — no lock is held while
    /// the child dies (the convention).
    #[test]
    fn dropped_child_is_closed() {
        let parent = Arc::new(Shared::new());
        let transient = Arc::new(Shared::new());
        let keeper = Arc::new(Shared::new());
        parent.register_child(&transient);
        parent.register_child(&keeper);
        drop(transient);
        let live = parent.live_children();
        assert_eq!(live.len(), 1);
        assert!(Arc::ptr_eq(&live[0], &keeper));
    }
}
