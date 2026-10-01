//! The tree level: `fanout` and the child-parking bookkeeping.
//!
//! Plain Rust (not Verus — the frontend rejects fn-pointer parameters
//! and the raw-pointer split dance), calling into the verified core
//! (`park_self`, the substrate ops). Trusted surface B6/B7, exactly as
//! documented in `mod.rs`: the split fn's pairwise-disjointness
//! contract, std scoped threads, and the per-child ghost wiring.

use super::{trusted_child_handle, trusted_child_wire, ChildPair, Handle, Shared};
use std::sync::Arc;
use std::time::Duration;

use vstd::simple_pptr::PPtr;

impl<'o, T> Handle<'o, T> {
    /// Fan out over the children of this subtree.
    ///
    /// `split` enumerates the children (disjoint `&mut` items of a
    /// local reborrow); each item becomes a child [`Handle`] moved to
    /// its scoped-thread worker, registered on this handle so parking
    /// stops bottom-up. The action stores its results INSIDE the node
    /// (there is no return path — ownership of results stays with the
    /// state machine). Serves checkpoint requests while the children
    /// run; returns when every child is finished (the thread scope
    /// joins them; a panicked child propagates at scope exit).
    ///
    /// TRUSTED (B6: the split fn pointer's pairwise-disjointness
    /// contract; B7: std scoped threads; the per-child baton minting).
    /// The serve loop's parking is the verified `park_self`.
    #[cfg_attr(verus_only, verifier::external_body)]
    pub fn fanout<'t, C, I, F>(&mut self, split: fn(&'t mut T) -> I, action: F)
    where
        C: Send + Sync + 'static,
        I: Iterator<Item = &'t mut C>,
        F: Fn(Handle<'o, C>) + Sync,
        'o: 't,
    {
        let action = &action;
        let node: *mut T = self.node.addr() as *mut T;
        let this: *mut Self = self;
        std::thread::scope(move |scope| {
            let mut join_handles: Vec<std::thread::ScopedJoinHandle<'_, ()>> = Vec::new();
            // SAFETY: `self` is mutably borrowed for the whole fan-out
            // (this method holds `&mut self`); deriving the split
            // reborrow from the node pointer is the same borrow.
            let items = split(unsafe { &mut *node });
            for child in items {
                let child_addr = std::ptr::from_mut(child) as usize;
                let child_shared: &'static Shared<C> =
                    Box::leak(Box::new(Shared::new(PPtr::from_addr(child_addr))));
                // SAFETY: `this` is our own `&mut self`, valid for the
                // scope of this method.
                unsafe { &mut *this }.children.push(Arc::new(child_shared) as Arc<dyn ChildPair>);
                // The item's `&mut C` is consumed into the child handle:
                // its borrow ends here; disjoint split items make the
                // concurrent workers sound (B6).
                let mut h: Handle<'o, C> =
                    trusted_child_handle(PPtr::from_addr(child_addr), child_shared);
                trusted_child_wire(&mut h);
                join_handles.push(scope.spawn(move || action(h)));
            }

            loop {
                // SAFETY: as above — our own `&mut self`.
                let me: &mut Self = unsafe { &mut *this };
                if me.is_stop_requested() {
                    me.park_self();
                }
                if join_handles.iter().all(std::thread::ScopedJoinHandle::is_finished) {
                    break;
                }
                std::thread::sleep(Duration::from_millis(1));
            }
        })
    }
}

/// Park every open child (deregistering closed ones) — the shipped
/// `children.retain` semantics. Plain Rust (B6/B7 trusted).
pub(crate) fn retain_children(children: &mut Vec<Arc<dyn ChildPair>>) {
    children.retain(|child| child.request_until_parked_dyn());
}

/// Resume the parked children after the parent runs again.
pub(crate) fn resume_children(children: &[Arc<dyn ChildPair>]) {
    for child in children {
        child.release_request_dyn();
    }
}

/// The trusted tree level's child registry (plain Rust: the dyn trait
/// object is what the Verus frontend cannot parse).
pub(crate) type ChildRegistry = Vec<Arc<dyn ChildPair>>;
