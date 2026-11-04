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
use std::sync::{Arc, Weak};

/// The parking/membership state shared by a handle and its children
/// (the sketch's `status`): children are weakly registered here —
/// managed like before, deleted = closed. Management itself is NOT
/// part of this step; only the structure.
pub struct Shared {
    children: Vec<Weak<Shared>>,
}

/// Deref/DerefMut land on `T` through the holder chain: the fat
/// pointer -> &mut dyn DerefMut<..> -> derefmut -> &mut dyn
/// HandleAble<V> -> cast -> &mut T.
pub struct InnerHandle<K: ?Sized, T: ?Sized, V: ?Sized>
where
    K: DerefMut<Target = dyn HandleAble<K, V>>,
{
    ptr: *mut K,
    status: Arc<Shared>,
    _marker: PhantomData<fn(*mut T, *mut K, *mut V)>,
}

// The raw pointer is exclusively owned by this handle for its whole
// life (the HandleAble contract) — moving the handle to another
// thread moves the exclusive access with it. T: Send because the
// pointee crosses with it (the big-PR lesson, applied here).
unsafe impl<K: ?Sized, T: ?Sized + Send, V: ?Sized> Send for InnerHandle<K, T, V> where
    K: DerefMut<Target = dyn HandleAble<K, V>>,
{
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
pub unsafe trait HandleAble<K: ?Sized, V: ?Sized>: Send {
    /// # Safety
    ///
    /// `holder` must point at the holder exclusively-live for the
    /// minted handle's whole life, holding `Self` at mint time.
    fn box_innerhandle(&mut self, holder: *mut K) -> Box<V>;
}

/// A holder that can REPLACE itself with a fresh `T` (the successor
/// install): `new<T: HandleAble<K, V>>(T) -> K`. The K parameter
/// is the holder being replaced (Self's own type).
pub unsafe trait ReplaceAble<K: ?Sized, V: ?Sized> {
    /// # Safety
    ///
    /// The replacement must be held exclusively by the new Self; the
    /// caller must have dropped every handle minted over the old
    /// content.
    fn new(replacement: Box<dyn HandleAble<K, V> + 'static>) -> Self
    where
        Self: Sized;
}
// The bypass (2a9957b): K decoupled from Self, so the Box pass-through
// exists generically:
//   impl<K: ?Sized, V: ?Sized> ReplaceAble<K, V> for Box<dyn HandleAble<K, V>>
//       { fn new(replacement: Box<dyn HandleAble<K, V>>) -> Self { replacement } }

unsafe impl<K: ?Sized + 'static, V: ?Sized + 'static> ReplaceAble<K, V>
    for Box<dyn HandleAble<K, V> + 'static>
{
    fn new(replacement: Box<dyn HandleAble<K, V> + 'static>) -> Self {
        replacement
    }
}

unsafe impl<K: ?Sized + DerefMut<Target = dyn HandleAble<K, V>>, T: Send, V: ?Sized>
    HandleAble<K, V> for T
where
    InnerHandle<K, T, V>: Unsize<V>,
{
    fn box_innerhandle(&mut self, holder: *mut K) -> Box<V> {
        // bind at the sized type, then let the return coerce
        // Box<InnerHandle<K, T, V>> -> Box<V> (the Unsize where-clause).
        let handle: Box<InnerHandle<K, T, V>> = Box::new(InnerHandle {
            ptr: holder,
            status: Arc::new(Shared { children: Vec::new() }),
            _marker: PhantomData,
        });
        handle
    }
}

impl<K: ?Sized, T: Sized, V: ?Sized> InnerHandle<K, T, V>
where
    K: DerefMut<Target = dyn HandleAble<K, V>>,
{
    /// The fan-out (no protocol yet — plain scoped threads). `split`
    /// runs on `&mut T` (the node's own fields) and yields the
    /// children `&mut K` — the SAME static holder type as this
    /// handle's (with K static, there is no erasure between the
    /// tree's holders and the minted handles' holders; they must
    /// agree). Each element becomes a `Box<V>` via `box_innerhandle`,
    /// and `f` runs on one scoped thread per box.
    pub fn fanout<F>(&mut self, split: &(dyn for<'x> Fn(&'x mut T) -> Vec<&'x mut K> + Send + Sync), f: F)
    where
        T: Send,
        V: Send,
        K: Sized + Send + 'static,
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
        // MINT THE BOXES: each holder becomes a Box<V> via the
        // blanket impl — the receiver derives from the raw, the
        // holder pointer is the raw, typed K2.
        let boxes: Vec<Box<V>> = child_raws
            .into_iter()
            .map(|raw| {
                let erased: &mut dyn HandleAble<K, V> =
                    // SAFETY: the split yielded each child exactly
                    // once, disjointly; exclusive for this mint.
                    unsafe { &mut **raw };
                let holder: *mut K = raw;
                erased.box_innerhandle(holder)
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

impl<K: ?Sized, T: Sized, V: ?Sized> Deref for InnerHandle<K, T, V>
where
    K: DerefMut<Target = dyn HandleAble<K, V>>,
{
    type Target = T;
    fn deref(&self) -> &T {
        // 1) the holder (shared read path; the same tag family
        //    DerefMut derives from — the mut raw)
        // SAFETY: exclusively-live per the HandleAble contract.
        let holder: &K = unsafe { &*self.ptr };
        // 2) deref that into the erased view
        let erased: &dyn HandleAble<K, V> = &**holder;
        // 3) cast that into &T
        // SAFETY: the HandleAble contract — the holder holds the `T`
        // this handle was minted for, exclusively-live.
        unsafe { &*(erased as *const dyn HandleAble<K, V> as *const T) }
    }
}

impl<K: ?Sized, T: Sized, V: ?Sized> DerefMut for InnerHandle<K, T, V>
where
    K: DerefMut<Target = dyn HandleAble<K, V>>,
{
    fn deref_mut(&mut self) -> &mut T {
        // 1) the holder
        // SAFETY: exclusively-live per the HandleAble contract.
        let holder: &mut K = unsafe { &mut *self.ptr };
        // 2) derefmut that into the erased view
        let erased: &mut dyn HandleAble<K, V> = holder.deref_mut();
        // 3) cast that into &mut T
        // SAFETY: as Deref — exclusive, the minted-for T.
        unsafe { &mut *(erased as *mut dyn HandleAble<K, V> as *mut T) }
    }
}
