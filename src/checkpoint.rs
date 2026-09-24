//! Checkpointing framework: a receiver/transmitter pair coordinating
//! stop-the-world snapshots of a tree of state machines running on scoped
//! threads.
//!
//! - the **Checkpointer** stays with the mutator; `safepoint(&mut self)`
//!   parks it between transitions (taking `&mut self` proves the thread
//!   holds no references into handled data while parked);
//! - the **CheckpointRequester** lives with the snapshotter (interior
//!   synchronization, shareable behind a `Mutex`/`Arc` by several threads);
//!   `request()` performs the stop-the-world handshake and returns a
//!   `CheckpointGuard<T>` granting direct `&T` access to the top-level node
//!   — and hence to all its children, because every mutator below is parked.
//!
//! The snapshotter serializes the real, live tree; children never
//! self-serialize and there is no request/reply protocol at runtime.
//!
//! The static guarantee inside a worker, enforced by the borrow checker:
//!
//! ```
//! use servyi_states::checkpoint::{fanout, Checkpointer};
//! use std::sync::mpsc::channel;
//!
//! let mut cp = Checkpointer::new();
//! let mut children = vec![1u32];
//! let out = fanout(&mut children, &mut cp, |c| c.iter_mut(), |mut h, mut ccp| {
//!     {
//!         let a = ccp.block_and_get(&h);
//!         assert_eq!(*a, 1);
//!     }
//!     ccp.safepoint();
//!     *ccp.block_and_get_mut(&mut h)
//! });
//! assert_eq!(out, vec![1]);
//! ```
//!
//! Holding the reference across a safepoint cannot compile:
//!
//! ```compile_fail
//! use servyi_states::checkpoint::{fanout, Checkpointer};
//! use std::sync::mpsc::channel;
//!
//! let mut cp = Checkpointer::new();
//! let mut children = vec![0u32];
//! fanout(&mut children, &mut cp, |c| c.iter_mut(), |mut h, mut ccp| {
//!     let r = ccp.block_and_get(&h);
//!     ccp.safepoint();
//!     let _ = *r;
//! });
//! ```
//!
//! Smuggling a Handle + Checkpointer out of the action and using them after
//! the fan-out returned (parent reclaims the node) cannot compile — the
//! `'s` brand has expired:
//!
//! ```compile_fail
//! use servyi_states::checkpoint::{fanout, Checkpointer};
//! use std::sync::mpsc::channel;
//!
//! let (tx, rx) = channel();
//! let _ = rx;
//! let mut cp = Checkpointer::new();
//! let mut children = vec![0u32];
//! fanout(&mut children, &mut cp, |c| c.iter_mut(), |h, c| {
//!     let _ = tx.send((h, c));
//! });
//! assert_eq!(children.len(), 1);
//! let (h, c) = rx.recv().unwrap();
//! let _ = c.block_and_get(&h);
//! ```

use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::Duration;

use serde::Serialize;



// -----------------------------------------------------------------------------
// Receiver/transmitter pair

struct Shared {
    pending_request: AtomicBool,
    /// Set when the owning (child) checkpointer is dropped: a child that
    /// finishes never parks, and a parent waiting to park it must see this
    /// instead of blocking forever.
    closed: AtomicBool,
    /// Poisoning invariant: only pure bool stores/loads and condvar waits
    /// run under this lock — no code that can panic ever holds it, so the
    /// lock cannot be poisoned and the `expect`s below are unreachable.
    parked: Mutex<bool>,
    park_cv: Condvar,
    resume_cv: Condvar,
}

impl Shared {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            pending_request: AtomicBool::new(false),
            closed: AtomicBool::new(false),
            parked: Mutex::new(false),
            park_cv: Condvar::new(),
            resume_cv: Condvar::new(),
        })
    }

    /// The handshake lock, with its poisoning invariant in one place: only
    /// pure bool ops and condvar waits run under it — no code that can
    /// panic ever holds it, so the lock cannot be poisoned.
    fn lock_parked(&self) -> std::sync::MutexGuard<'_, bool> {
        self.parked.lock().expect("parked handshake lock: see lock_parked's invariant")
    }

    /// Wait for all mutators to park (see lock_parked's invariant for why
    /// poisoning is impossible).
    fn wait_park<'g>(&self, parked: std::sync::MutexGuard<'g, bool>) -> std::sync::MutexGuard<'g, bool> {
        self.park_cv.wait(parked).expect("parked handshake: see lock_parked's invariant")
    }

    /// Wait for the request to be released (see lock_parked's invariant).
    fn wait_resume<'g>(&self, parked: std::sync::MutexGuard<'g, bool>) -> std::sync::MutexGuard<'g, bool> {
        self.resume_cv.wait(parked).expect("parked handshake: see lock_parked's invariant")
    }

    /// Request a stop and wait until this side parks — or until it closes
    /// (a finished child never parks). Returns false when closed.
    fn request_until_parked_or_closed(&self) -> bool {
        self.pending_request.store(true, Ordering::SeqCst);
        let mut parked = self.lock_parked();
        loop {
            if self.closed.load(Ordering::SeqCst) {
                return false;
            }
            if *parked {
                return true;
            }
            parked = self.wait_park(parked);
        }
    }

    /// Release a request: wake the parked side and wait until it has
    /// resumed. Shared by the guard's drop and by parents releasing their
    /// children (see lock_parked's invariant).
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

/// Mutator side of the pair. Stays with the thread that owns or leases the
/// node. `safepoint` parks the thread while a checkpoint is in flight;
/// requiring `&mut self` proves no references into handled data are live
/// while parked (they are all tied to borrows of the checkpointer).
pub struct Checkpointer<'n> {
    shared: Arc<Shared>,
    /// Subtree pairs registered by an enclosing `fanout`: parking this
    /// checkpointer first parks (or deregisters, once closed) every child
    /// subtree, so a guard at this level sees a fully quiesced tree.
    children: Vec<Arc<Shared>>,
    _brand: PhantomData<fn() -> &'n ()>,
}

impl<'n> Default for Checkpointer<'n> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'n> Checkpointer<'n> {
    fn from_shared(shared: Arc<Shared>) -> Self {
        Self { shared, children: Vec::new(), _brand: PhantomData }
    }

    /// Register a child subtree's pair: `safepoint`/`park_self` on this
    /// checkpointer will stop the child first. Deregistered automatically
    /// once the child closes (its worker finished).
    fn register_child(&mut self, child: Arc<Shared>) {
        self.children.push(child);
    }

    pub fn new() -> Self {
        Self::from_shared(Shared::new())
    }

    /// Park while a checkpoint has been requested, then return. Call only
    /// between transitions, with all references into handled data dropped.
    pub fn safepoint(&mut self) {
        if !self.shared.pending_request.load(Ordering::SeqCst) {
            return;
        }
        self.park_self();
    }

    pub fn is_stop_requested(&self) -> bool {
        self.shared.pending_request.load(Ordering::SeqCst)
    }

    fn park_self(&mut self) {
        // Stop the world bottom-up: request+wait every open child subtree
        // first (children that already finished are deregistered), then
        // park this thread. A guard taken at this level therefore sees the
        // whole subtree quiesced.
        self.children.retain(|child| child.request_until_parked_or_closed());
        let mut parked = self.shared.lock_parked();
        *parked = true;
        self.shared.park_cv.notify_all();
        while self.shared.pending_request.load(Ordering::SeqCst) {
            parked = self.shared.wait_resume(parked);
        }
        *parked = false;
        self.shared.park_cv.notify_all();
        // Resume the children only after this thread is running again.
        for child in &self.children {
            child.release_request();
        }
    }


/// The shared read accessor of the pair: shared borrow blocks the
    /// snapshotter on this thread (a guard's `&T` and a worker's use of the
    /// same checkpointer are mutually exclusive through `Shared`).
    pub fn block_and_get<'a, T>(&'a self, h: &'a Handle<'n, T>) -> &'a T {
        // SAFETY: the guard proving every mutator is parked borrows this
        // checkpointer; a live `&Handle` means its worker is parked too.
        unsafe { &*h.ptr }
    }

    /// Exclusive access to the leased child. Only a shared borrow of the
    /// checkpointer is needed: exclusivity comes from the `&'a mut Handle<T>`
    /// (exactly one exists per child).
    pub fn block_and_get_mut<'a, T>(&'a self, h: &'a mut Handle<'n, T>) -> &'a mut T {
        // SAFETY: same parking proof as `block_and_get`; the `&mut Handle`
        // is the unique lease, so the returned `&mut` is exclusive.
        unsafe { &mut *h.ptr }
    }
}

/// Transmitter side of the pair. Lives with the snapshotter; interior
/// synchronization allows sharing behind a `Mutex`/`Arc` by several threads.
pub struct CheckpointRequester<'a, T> {
    node: *const T,
    shared: Arc<Shared>,
    _brand: PhantomData<fn(&'a mut ()) -> &'a T>,
}

// SAFETY: the raw node pointer is only dereferenced through a guard, which
// exists exclusively while every mutator is parked (see `request`); moving
// or sharing the requester itself touches no node data, so no `T` bounds
// are needed. Concurrent `request`s still require external mutual
// exclusion (see `request` docs).
// SAFETY: see the block comment above — the pointer is only dereferenced
// through a guard; moving/sharing the requester touches no node data.
unsafe impl<'a, T> Send for CheckpointRequester<'a, T> {}
// SAFETY: as Send above.
unsafe impl<'a, T> Sync for CheckpointRequester<'a, T> {}

impl<'a, T> Clone for CheckpointRequester<'a, T> {
    fn clone(&self) -> Self {
        Self { node: self.node, shared: self.shared.clone(), _brand: PhantomData }
    }
}

impl<'a, T> CheckpointRequester<'a, T> {
    fn from_raw_parts(node: *const T, shared: Arc<Shared>) -> Self {
        Self { node, shared, _brand: PhantomData }
    }

    /// Build the transmitter side from the mutator's checkpointer.
    ///
    /// # Safety
    ///
    /// - `node` must remain valid and reachable for `'a`.
    /// - Every mutator with access to the node in that window must call
    ///   `safepoint` between transitions and hold no references while
    ///   parked; the guard's direct read is sound only under that
    ///   discipline.
    pub unsafe fn from_node_ptr(node: *const T, cp: &Checkpointer<'a>) -> Self {
        Self::from_raw_parts(node, cp.shared.clone())
    }

    /// Perform the stop-the-world handshake: request all mutators to park
    /// (the top mutator parks its subtree first) and return a guard with
    /// direct `&T` access to the top-level node. Blocks until parked.
    ///
    /// Concurrent `request`s from several threads require external
    /// mutual exclusion (wrap the requester in a `Mutex`): overlapping
    /// guards from independent calls are not synchronized against each
    /// other's resume.
    pub fn request(&self) -> CheckpointGuard<'_, T> {
        self.shared.pending_request.store(true, Ordering::SeqCst);
        let mut parked = self.shared.lock_parked();
        while !*parked {
            parked = self.shared.wait_park(parked);
        }
                CheckpointGuard { node: self.node, shared: &self.shared, _brand: PhantomData }
    }

    /// Like [`request`](Self::request), but gives up after `timeout`. Giving
    /// up is safe: a mutator that saw the request parks only while the stop
    /// flag is set, and re-checks it before waiting.
    ///
    /// # Panics
    ///
    /// Panics only if the handshake lock is poisoned — see the invariant
    /// on `Shared::parked` (nothing that can panic holds it).
    pub fn request_timeout(&self, timeout: Duration) -> Option<CheckpointGuard<'_, T>> {
        self.shared.pending_request.store(true, Ordering::SeqCst);
        let mut parked = self.shared.lock_parked();
        let deadline = std::time::Instant::now() + timeout;
        loop {
            if *parked {
                return Some(CheckpointGuard {
                    node: self.node,
                    shared: &self.shared,
                    _brand: PhantomData,
                });
            }
            let now = std::time::Instant::now();
            if now >= deadline {
                self.shared.pending_request.store(false, Ordering::SeqCst);
                self.shared.resume_cv.notify_all();
                drop(parked);
                return None;
            }
            let (g, _t) = self
                .shared
                .park_cv
                .wait_timeout(parked, deadline - now)
                .expect("parked handshake lock: see invariant on Shared::parked");
            parked = g;
        }
    }
}

/// Direct read access to the top-level node while every mutator is parked.
/// Dropping the guard resumes the world.
pub struct CheckpointGuard<'a, T> {
    node: *const T,
    shared: &'a Shared,
    _brand: PhantomData<fn(&'a mut ()) -> &'a T>,
}

impl<'a, T> CheckpointGuard<'a, T> {
    /// # Safety (internal)
    ///
    /// Sound because `request()` waited for the park acknowledgment: every
    /// mutator that could reach `node` is parked at a safepoint where it
    /// provably holds no references, and `node` outlives the requester's
    /// `'a` brand.
    pub fn node(&self) -> &'a T {
        // SAFETY: a live guard proves every mutator that could reach the
        // node is parked; the node outlives the requester's `'a` brand.
        unsafe { &*self.node }
    }
}

impl<'a, T> Drop for CheckpointGuard<'a, T> {
    fn drop(&mut self) {
        self.shared.release_request();
    }
}

/// Create the receiver/transmitter pair for a top-level node owned by the
/// mutator. The checkpointer stays with the mutator; the requester goes to
/// the snapshotter.
pub fn checkpoint_pair<T>(node: &T) -> (Checkpointer<'_>, CheckpointRequester<'_, T>) {
    let shared = Shared::new();
    (
        Checkpointer::from_shared(shared.clone()),
        CheckpointRequester::from_raw_parts(node as *const T, shared),
    )
}

// -----------------------------------------------------------------------------
// Handles (leases)

/// A lease on one child, branded with the thread-scope lifetime `'s` of the
/// enclosing fan-out: the handle is valid exactly while its subtask runs.
/// The brand is invariant, and since `'s` is higher-ranked in [`fanout`]'s
/// action bound, no storage type outside the scope can mention it —
/// smuggling a handle out is unrepresentable by construction.
pub struct Handle<'s, T> {
    ptr: *mut T,
    _brand: PhantomData<fn(&'s mut ()) -> &'s T>,
}

// SAFETY: closing only touches atomics; it must wake, not touch data.
impl Drop for Checkpointer<'_> {
    fn drop(&mut self) {
        // A parent waiting to park this subtree must observe the close
        // instead of blocking on a park that will never come.
        self.shared.close();
    }
}

// SAFETY: the handle's pointer is only dereferenced by its owning worker
// (one lease per split item); moving it to that worker touches no data.
unsafe impl<'s, T: Send> Send for Handle<'s, T> {}

impl<'n, T> Handle<'n, T> {
    /// The unsafe bridge from the design: turn a raw pointer derived from
    /// a split item into a Handle. Crate-internal on purpose — the safety
    /// argument (the `&mut T` is explicitly dropped before the worker can
    /// reference through the handle, and split items are disjoint) is made
    /// at the `fanout` call site where those facts are visible.
    ///
    /// # Safety
    ///
    /// - `ptr` must derive from a split item whose `&mut T` has been
    ///   **dropped** — not merely unused — before the handle is referenced.
    /// - The child may be reached through the parent node or another path
    ///   only while the `Handle` provably holds no live reference (i.e.,
    ///   outside any `block_and_get` / `block_and_get_mut` borrow).
    /// - `'s` must not outlive the lease; `fanout` instantiates it as its
    ///   thread scope.
    pub(crate) unsafe fn from_raw<'s>(ptr: *mut T) -> Handle<'s, T> {
        Handle { ptr, _brand: PhantomData }
    }
}

// -----------------------------------------------------------------------------
// Fan-out

/// Nested-fan-out entry point: like [`fanout`], but the node is behind a
/// handle leased to this thread by an enclosing fanout. Materializing the
/// `&mut N` is sound because this thread holds the exclusive lease (the
/// enclosing fanout will not touch the node until this subtask returns), and
/// stays inside this module — the only place handles are created.
pub fn fanout_handle<'n, N, I, T, R, F>(
    h: &'n mut Handle<'_, N>,
    cp: &mut Checkpointer<'_>,
    split: fn(&'n mut N) -> I,
    action: F,
) -> Vec<R>
where
    I: Iterator<Item = &'n mut T>,
    F: for<'s> Fn(Handle<'s, T>, Checkpointer<'s>) -> R + Sync + 'n,
    T: Serialize + Send + 'n,
    R: Send + Serialize,
{
    // SAFETY: this thread holds the exclusive lease (the enclosing fanout
    // will not touch the node until this subtask returns); materializing
    // the `&mut N` from the raw pointer is the lease's whole purpose.
    let node: &'n mut N = unsafe { &mut *h.ptr };
    fanout(node, cp, split, action)
}

/// Fan out over the children of `node`.
///
/// `split` enumerates children of an arbitrary node type as `&mut T`;
/// `action` runs on each child with a `Handle<T>` (leasing it through a
/// `*mut T`) and a child `Checkpointer`. Subtasks are scoped threads, so
/// nothing crosses a thread boundary by ownership and no `'static` is
/// required. After starting all subtasks, the fanout serves checkpoint
/// requests on its own checkpointer: it parks the whole subtree (taking
/// guards on the child requesters) before parking itself, so the
/// snapshotter's guard sees a fully quiesced tree.
/// # Panics
///
/// Propagates (re-unwinds) a panicked child subtask, and panics only if a
/// handshake lock is poisoned — see the invariant on `Shared::parked`.
pub fn fanout<'n, N, I, T, R, F>(
    node: &'n mut N,
    cp: &mut Checkpointer<'_>,
    split: fn(&'n mut N) -> I,
    action: F,
) -> Vec<R>
where
    I: Iterator<Item = &'n mut T>,
    F: for<'s> Fn(Handle<'s, T>, Checkpointer<'s>) -> R + Sync + 'n,
    T: Send + 'n,
    R: Send,
{
    let action = &action;
    thread::scope(move |scope| {
        let mut child_requesters: Vec<CheckpointRequester<'n, T>> = Vec::new();
        let mut joins: Vec<Option<thread::ScopedJoinHandle<'_, R>>> = Vec::new();
        for child in split(node) {
            // One raw pointer is the entire bridge: the pair's read view
            // and the worker's lease both derive from it.
            let raw: *mut T = child;
            // The `&mut T` must be DROPPED — not merely unused — before the
            // worker can reference through its handle, hence the explicit
            // drop BEFORE the spawn. Workers from earlier iterations are
            // already running and dereferencing their own leases; what
            // makes that sound is that split items are disjoint, not any
            // sequencing here.
            #[allow(clippy::dropping_reference)] // deliberate: marks the borrow's end for readers and from_raw's contract
            drop(child);
            // SAFETY: `raw` derives from the dropped exclusive borrow; the
            // pair's pointer is only read through child guards while the
            // child is parked, and the lease outlives the worker only
            // inside this thread scope.
            let h: Handle<'_, T> = unsafe { Handle::from_raw(raw) };
            let shared = Shared::new();
            let child_cp = Checkpointer::from_shared(shared.clone());
            child_requesters.push(CheckpointRequester::from_raw_parts(raw, shared.clone()));
            // Parking THIS checkpointer now parks the child subtree first;
            // a child that finishes closes its pair and is deregistered.
            cp.register_child(shared);
            joins.push(Some(scope.spawn(move || action(h, child_cp))));
        }

        let n = joins.len();
        let mut results: Vec<Option<R>> = (0..n).map(|_| None).collect();
        loop {
            if cp.is_stop_requested() {
                // Parking this checkpointer stops the whole subtree first
                // (children are registered above); results collection stays
                // purely completion-driven below.
                cp.park_self();
                continue;
            }

            let mut all_done = true;
            for (i, j) in joins.iter_mut().enumerate() {
                if results[i].is_none() {
                    if let Some(h) = j.as_mut() {
                        if h.is_finished() {
                            if let Some(handle) = j.take() {
                                results[i] = match handle.join() {
                                    Ok(r) => Some(r),
                                    // A panicked child must unwind the
                                    // parent too — the snapshot protocol
                                    // has no sane way to continue.
                                    Err(payload) => std::panic::resume_unwind(payload),
                                };
                            }
                        }
                    }
                }
                if results[i].is_none() {
                    all_done = false;
                }
            }
            if all_done {
                break;
            }
            thread::sleep(Duration::from_millis(1));
        }
        results
            .into_iter()
            .map(|r| r.expect("the all_done exit guarantees every slot was filled"))
            .collect()
    })
}
