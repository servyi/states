//! Stop-the-world checkpointing for trees of state machines on scoped
//! threads.
//!
//! A [`Checkpointer`] stays with each mutator thread; the mutator calls
//! [`Checkpointer::safepoint`] between transitions. The snapshotter holds
//! the [`CheckpointRequester`] and calls `request()`, which parks every
//! mutator (a fan-out registers its child subtrees on its checkpointer,
//! so parking is bottom-up) and returns a guard reading the node directly
//! while the world stands still.
//!
//! Leases: a fan-out's `split` yields `&'n mut T` items of the node's
//! lifetime; each becomes a [`Handle`] — safe, exclusive access to that
//! child for its worker (see `Handle::get`).

use std::marker::PhantomData;
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// Outcome of the stop request handshake.
enum Handshake {
    /// The other side is parked and holds no references.
    Parked,
    /// The other side is gone (its checkpointer dropped) — it will never
    /// park; there is nothing to wait for.
    Closed,
    /// A deadline elapsed before the other side parked. Safe to give up:
    /// a mutator re-checks the request flag before parking, so a late
    /// parker returns immediately.
    TimedOut,
}

struct Shared {
    pending_request: AtomicBool,
    /// Set when the owning (child) checkpointer is dropped: a child that
    /// finishes never parks, and a parent waiting to park it must see this
    /// instead of blocking forever.
    closed: AtomicBool,
    /// Serializes requesters for the whole lifetime of a guard: the flag
    /// update must not interleave with another request (a request made
    /// while a guard is out is swallowed — mutators park only once — and
    /// its requester would wait forever). The mutator never takes this
    /// lock.
    request_lock: Mutex<()>,
    /// Poisoning invariant: only pure bool stores/loads and condvar waits
    /// run under this lock — no code that can panic ever holds it, so the
    /// lock cannot be poisoned.
    parked: Mutex<bool>,
    park_cv: Condvar,
    resume_cv: Condvar,
}

impl Shared {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            pending_request: AtomicBool::new(false),
            closed: AtomicBool::new(false),
            request_lock: Mutex::new(()),
            parked: Mutex::new(false),
            park_cv: Condvar::new(),
            resume_cv: Condvar::new(),
        })
    }

    /// The handshake lock, with its poisoning invariant in one place: only
    /// pure bool ops and condvar waits run under it — no code that can
    /// panic ever holds it, so the lock cannot be poisoned.
    fn is_pending(&self) -> bool {
        self.pending_request.load(Ordering::SeqCst)
    }

    fn lock_parked(&self) -> MutexGuard<'_, bool> {
        self.parked.lock().expect("parked handshake lock: see lock_parked's invariant")
    }

    /// Wait for the other side to park (see lock_parked's invariant for
    /// why poisoning is impossible).
    fn wait_park<'g>(&self, parked: MutexGuard<'g, bool>) -> MutexGuard<'g, bool> {
        self.park_cv.wait(parked).expect("parked handshake: see lock_parked's invariant")
    }

    /// Wait for the request to be released (see lock_parked's invariant).
    fn wait_resume<'g>(&self, parked: MutexGuard<'g, bool>) -> MutexGuard<'g, bool> {
        self.resume_cv.wait(parked).expect("parked handshake: see lock_parked's invariant")
    }

    /// THE stop request: set the flag and wait until the other side parks,
    /// closes, or (with a timeout) the deadline elapses. This is the only
    /// implementation of the request half of the handshake — the public
    /// `request`/`request_timeout` and the child parking in `park_self`
    /// are all interfaces to it.
    fn request_until_parked(&self, timeout: Option<Duration>) -> Handshake {
        self.pending_request.store(true, Ordering::SeqCst);
        let mut parked = self.lock_parked();
        let deadline = timeout.map(|t| Instant::now() + t);
        loop {
            if *parked {
                return Handshake::Parked;
            }
            if self.closed.load(Ordering::SeqCst) {
                self.pending_request.store(false, Ordering::SeqCst);
                return Handshake::Closed;
            }
            let Some(deadline) = deadline else {
                parked = self.wait_park(parked);
                continue;
            };
            let now = Instant::now();
            if now >= deadline {
                self.pending_request.store(false, Ordering::SeqCst);
                return Handshake::TimedOut;
            }
            let (guard, _) = self
                .park_cv
                .wait_timeout(parked, deadline - now)
                .expect("parked handshake: see lock_parked's invariant");
            parked = guard;
        }
    }

    /// THE release: wake the parked side and wait until it has resumed.
    /// Flag update and notify are atomic with the parked thread's
    /// check-then-wait (both under the `parked` mutex), otherwise the
    /// wakeup can be lost between the waiter's check and its wait. Used by
    /// the guard's drop and by parents releasing their children.
    fn release_request(&self) {
        let mut parked = self.lock_parked();
        self.pending_request.store(false, Ordering::SeqCst);
        self.resume_cv.notify_all();
        while *parked && !self.closed.load(Ordering::SeqCst) {
            parked = self.wait_park(parked);
        }
    }

    /// Mark this side closed and wake anyone waiting for it to park.
    fn close(&self) {
        self.closed.store(true, Ordering::SeqCst);
        let _guard = self.lock_parked();
        self.park_cv.notify_all();
    }
}

use std::sync::MutexGuard;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Mutator side of the pair. Stays with the thread that owns or leases the
/// node. `safepoint` parks the thread while a checkpoint is in flight;
/// requiring `&mut self` proves no references into handled data are live
/// while parked (they are all tied to borrows of the checkpointer).
pub struct Checkpointer {
    shared: Arc<Shared>,
    /// Subtree pairs created by `create_child`: parking this checkpointer
    /// first parks (or deregisters, once closed) every child subtree, so
    /// a guard at this level sees a fully quiesced tree.
    children: Vec<Arc<Shared>>,
}

impl Default for Checkpointer {
    fn default() -> Self {
        Self::new()
    }
}

impl Checkpointer {
    pub fn new() -> Self {
        Self { shared: Shared::new(), children: Vec::new() }
    }

    /// Create a child checkpointer: parking THIS checkpointer stops the
    /// child's subtree first (bottom-up), so a guard at this level sees
    /// the whole tree quiesced. Deregistered automatically once the child
    /// is dropped (a finished worker never parks).
    pub fn create_child(&mut self) -> Checkpointer {
        let child = Self::new();
        self.children.push(Arc::clone(&child.shared));
        child
    }

    /// Park while a checkpoint has been requested, then return. Call only
    /// between transitions, with all references into handled data dropped.
    pub fn safepoint(&mut self) {
        if !self.shared.is_pending() {
            return;
        }
        self.park_self();
    }

    pub fn is_stop_requested(&self) -> bool {
        self.shared.is_pending()
    }

    fn park_self(&mut self) {
        // Stop the world bottom-up: request+wait every open child subtree
        // first (children that already finished are deregistered), then
        // park this thread. A guard taken at this level therefore sees the
        // whole subtree quiesced.
        self.children.retain(|child| {
            !matches!(child.request_until_parked(None), Handshake::Closed)
        });
        let mut parked = self.shared.lock_parked();
        *parked = true;
        self.shared.park_cv.notify_all();
        while self.is_stop_requested() {
            parked = self.shared.wait_resume(parked);
        }
        *parked = false;
        self.shared.park_cv.notify_all();
        // Resume the children only after this thread is running again.
        for child in &self.children {
            child.release_request();
        }
    }
}

impl Drop for Checkpointer {
    fn drop(&mut self) {
        // A parent waiting to park this subtree must observe the close
        // instead of blocking on a park that will never come.
        self.shared.close();
    }
}

/// Transmitter side of the pair. Lives with the snapshotter; interior
/// synchronization allows sharing behind a `Mutex`/`Arc` by several threads.
///
/// Build it with [`checkpoint_pair`]; there is deliberately no constructor
/// from a bare pointer — pairing the requester with the node's actual
/// checkpointer is what makes the guard's direct read sound.
pub struct CheckpointRequester<'a, T> {
    node: *const T,
    shared: Arc<Shared>,
    _brand: PhantomData<fn(&'a ()) -> &'a T>,
}

// SAFETY: the node pointer is only dereferenced through a guard, which
// exists exclusively while every mutator is parked (see `request`);
// moving or sharing the requester itself touches no node data. Concurrent
// `request`s still require external mutual exclusion (see `request` docs).
// SAFETY: the guard reads `T` on the requester's thread while every
// mutator is parked — the data crosses threads, so `T` must be `Send`.
unsafe impl<'a, T: Send> Send for CheckpointRequester<'a, T> {}
// SAFETY: as Send above — the requester itself touches no node data.
// SAFETY: as Send — sharing the requester shares no node data.
unsafe impl<'a, T: Send> Sync for CheckpointRequester<'a, T> {}

impl<'a, T> Clone for CheckpointRequester<'a, T> {
    fn clone(&self) -> Self {
        Self { node: self.node, shared: self.shared.clone(), _brand: PhantomData }
    }
}

impl<'a, T> CheckpointRequester<'a, T> {
    /// The mutator-side bridge: a requester for a node whose mutation
    /// discipline spans a `&mut` owned elsewhere, where [`checkpoint_pair`]
    /// cannot be used (its shared borrow would exclude the mutation).
    ///
    /// # Safety
    ///
    /// - `node` must be valid at this address for all of `'a`.
    /// - THE CONTRACT: whenever any checkpointer of this pair parks (i.e.
    ///   whenever a guard can exist), there must be no live mutable
    ///   references to the node. Calling `safepoint` at high frequency is
    ///   a latency optimization, not a safety requirement — what matters
    ///   is that no `&mut` into the node outlives a park.
    pub unsafe fn from_node_ptr(node: *const T, cp: &Checkpointer) -> Self {
        Self { node, shared: cp.shared.clone(), _brand: PhantomData }
    }

    /// Perform the stop-the-world handshake: request all mutators to park
    /// (the top mutator parks its subtree first) and return a guard with
    /// direct `&T` access to the top-level node. Blocks until parked.
    ///
    /// A closed pair (mutator side dropped) hands out the guard at once:
    /// nothing will ever mutate again. Returns `None` only on
    /// `request_timeout` expiry. Concurrent `request`s serialize
    /// internally: a second request blocks until the first guard is
    /// dropped and the world resumed — no external mutex needed.
    pub fn request(&self) -> Option<CheckpointGuard<'_, T>> {
        self.request_impl(None)
    }

    /// [`request`](Self::request) with a deadline: additionally gives up
    /// (returning `None`) if the world has not stopped in time — safe,
    /// because a mutator re-checks the request flag before parking.
    pub fn request_timeout(&self, timeout: Duration) -> Option<CheckpointGuard<'_, T>> {
        self.request_impl(Some(timeout))
    }

    /// [`request`](Self::request) with a deadline: additionally gives up
    /// (returning `None`) if the world has not stopped in time — safe,
    /// because a mutator re-checks the request flag before parking.
    /// The one implementation: take the requester lock (for the whole
    /// lifetime of the returned guard), then run the handshake.
    fn request_impl(&self, timeout: Option<Duration>) -> Option<CheckpointGuard<'_, T>> {
        // Closed is not a failure: the mutator side is gone and will never
        // mutate again, so access is free — hand out the guard at once.
        if self.shared.closed.load(Ordering::SeqCst) {
            let request_lock = self
                .shared
                .request_lock
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            if self.shared.is_pending() {
                self.shared.release_request();
            }
            return Some(CheckpointGuard {
                node: self.node,
                shared: &self.shared,
                request_lock,
                _brand: PhantomData,
            });
        }
        // A poisoned lock means a requester panicked mid-guard; the only
        // state under it is the pending flag and the handshake, both plain
        // bools — recover, and reset the world in case it is still stopped
        // with nobody left to resume it.
        let request_lock = self
            .shared
            .request_lock
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if self.shared.is_pending() {
            self.shared.release_request();
        }
        let guard = match self.shared.request_until_parked(timeout) {
            Handshake::Parked | Handshake::Closed => CheckpointGuard {
                node: self.node,
                shared: &self.shared,
                request_lock,
                _brand: PhantomData,
            },
            Handshake::TimedOut => {
                drop(request_lock);
                return None;
            }
        };
        Some(guard)
    }
}

/// Direct read access to the top-level node while every mutator is parked.
/// Dropping the guard resumes the world.
pub struct CheckpointGuard<'a, T> {
    node: *const T,
    shared: &'a Shared,
    /// Held for the guard's whole life: concurrent requesters block until
    /// this guard (and therefore the stop it caused) is done.
    request_lock: MutexGuard<'a, ()>,
    _brand: PhantomData<fn(&'a mut ()) -> &'a T>,
}

impl<'a, T> CheckpointGuard<'a, T> {
    /// The top-level node, quiesced: every mutator that could reach it is
    /// parked at a safepoint where it provably holds no references, and
    /// the node outlives the requester's `'a` brand.
    pub fn node(&self) -> &T {
        // SAFETY: the guard exists only while the handshake proved every
        // mutator parked. The reference is tied to the GUARD (not the
        // requester's brand): it cannot outlive the guard, so the world
        // cannot resume underneath it.
        unsafe { &*self.node }
    }
}

impl<T> Drop for CheckpointGuard<'_, T> {
    fn drop(&mut self) {
        self.shared.release_request();
        // request_lock drops with the struct, releasing the requesters.
    }
}

/// Create the receiver/transmitter pair for a top-level node owned by the
/// mutator. The checkpointer stays with the mutator; the requester goes to
/// the snapshotter.
pub fn checkpoint_pair<T>(node: &T) -> (Checkpointer, CheckpointRequester<'_, T>) {
    let cp = Checkpointer::new();
    let requester = CheckpointRequester {
        node: node as *const T,
        shared: Arc::clone(&cp.shared),
        _brand: PhantomData,
    };
    (cp, requester)
}

// -----------------------------------------------------------------------------
// Handles (leases)

/// A lease on one child of the node, branded with the NODE's lifetime `'n`
/// (not the fan-out scope): the handle may outlive the scope closure, but
/// the child is only reachable through this lease, and `get`'s reborrow is
/// tied to `&mut self` — two live exclusive references from one handle are
/// unrepresentable.
pub struct Handle<'n, T> {
    ptr: *mut T,
    _brand: PhantomData<&'n mut T>,
}

// SAFETY: the handle's pointer is only dereferenced by its owning worker
// (one lease per disjoint split item); moving it to that worker touches no
// data, and `T: Send` covers the data itself.
unsafe impl<'n, T: Send> Send for Handle<'n, T> {}

impl<'n, T> Handle<'n, T> {
    /// The leased child, mutably: the exclusive reborrow is tied to
    /// `&mut self` (the handle's test shape) — no `Deref`/`DerefMut`,
    /// which would spread access through coercion and let references
    /// escape into patterns the parking protocol cannot see.
    pub fn get(&mut self) -> &mut T {
        // SAFETY: `ptr` derives from the exclusive `&'n mut T` consumed in
        // `new`; each call hands out one exclusive reborrow, and the
        // handle is the only remaining path to the child.
        unsafe { &mut *self.ptr }
    }
}

impl<'n, T> Handle<'n, T> {
    /// Lease a freshly split child. The `&'n mut T` is consumed here: the
    /// iterator item's borrow ends at this call, and the child is
    /// exclusively reachable through the returned handle for the node's
    /// lifetime.
    pub(crate) fn new(child: &'n mut T) -> Handle<'n, T> {
        Handle { ptr: child as *mut T, _brand: PhantomData }
    }

    /// The leased child. The reborrow is tied to `&mut self`, so borrows
    /// cannot overlap; after the last `get` the handle itself is the only
    /// remaining path to the child.
    // Deref/DerefMut instead of a `get`: `*h`, `h.field`, and `&mut *h`
    // all reborrow the lease, and two live exclusive references from one
    // handle stay unrepresentable.

    /// Give the lease back as the full `&'n mut T`: consuming the handle
    /// makes the exclusive borrow available for the node's lifetime again
    /// (used to re-enter a nested fan-out, which needs `'n`-typed split
    /// items rather than a shorter reborrow).
    pub(crate) fn into_node(self) -> &'n mut T {
        // SAFETY: `self` is moved — this was the only remaining path to
        // the child, and the returned reference restores the original
        // exclusive borrow with its full lifetime.
        unsafe { &mut *self.ptr }
    }
}

// -----------------------------------------------------------------------------
// Fan-out

/// Nested-fan-out entry point: like [`fanout`], but the node is behind a
/// handle leased to this thread by an enclosing fanout. The handle is
/// consumed: its full `'n` borrow becomes the node the inner fan-out
/// splits over. Aggregate results over this subtree are read one level up
/// (the caller of the enclosing fan-out still owns the node's parent).
pub fn fanout_handle<'n, T, I, C, F>(
    h: Handle<'n, T>,
    cp: &mut Checkpointer,
    split: fn(&'n mut T) -> I,
    action: F,
) where
    I: Iterator<Item = &'n mut C>,
    F: Fn(Handle<'n, C>, Checkpointer) + Sync,
    C: Send + 'n,
{
    let node = h.into_node();
    fanout(node, cp, split, action);
}

/// The action runs per child with the child's exclusive lease and a
/// checkpointer for nested fan-outs and safepoints. Results are NOT
/// returned — the action stores them inside the node (e.g. lease
/// `(&mut R, &Input)` children and write through `get`), which keeps
/// result ownership with the state machine instead of the fan-out.
///
/// Fan out over the children of `node`.
///
/// `split` enumerates children of an arbitrary node type as `&'n mut`
/// items; `action` runs on each child with its [`Handle`] (an exclusive
/// lease for the node's lifetime `'n`) and a child `Checkpointer`.
/// Subtasks are scoped threads, so nothing crosses a thread boundary by
/// ownership and no `'static` is required. After starting all subtasks,
/// the fan-out serves checkpoint requests on its own checkpointer: it
/// parks the whole subtree (children are registered on the checkpointer
/// and park bottom-up) before parking itself, so the snapshotter's guard
/// sees a fully quiesced tree.
///
/// Returns when every subtask is finished; the thread scope joins them
/// (a panicked subtask propagates its unwind there).
pub fn fanout<'n, N, I, T, F>(
    node: &'n mut N,
    cp: &mut Checkpointer,
    split: fn(&'n mut N) -> I,
    action: F,
) where
    I: Iterator<Item = &'n mut T>,
    F: Fn(Handle<'n, T>, Checkpointer) + Sync,
    T: Send + 'n,
{
    let action = &action;
    thread::scope(move |scope| {
        let mut join_handles: Vec<thread::ScopedJoinHandle<'_, ()>> = Vec::new();
        for child in split(node) {
            // The child is leased through the handle; workers from earlier
            // iterations may already be running — sound because split
            // items are disjoint. Parking this checkpointer (the parent's)
            // stops the child subtree first; a finished child closes its
            // checkpointer and is deregistered.
            let h: Handle<'n, T> = Handle::new(child);
            let child_cp = cp.create_child();
            join_handles.push(scope.spawn(move || action(h, child_cp)));
        }

        // Serve checkpoint requests until every subtask is finished; the
        // scope then joins them all.
        loop {
            if cp.is_stop_requested() {
                cp.park_self();
            }
            if join_handles.iter().all(thread::ScopedJoinHandle::is_finished) {
                break;
            }
            thread::sleep(Duration::from_millis(1));
        }
    })
}
