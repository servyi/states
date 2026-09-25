//! Stop-the-world checkpointing for trees of state machines on scoped
//! threads.
//!
//! [`with_checkpoint_pair`] takes ownership of the node and hands a
//! closure a fresh, unforgeably-branded pair: a [`Handle`] (the fused
//! lease + checkpointer, one per subtree) for the mutator side and a
//! [`CheckpointRequester`] for the snapshotter. The mutator accesses its
//! subtree ONLY through the handle's `block_and_get` / `block_and_get_mut`
//! — the borrows are tied to `&mut self`, so a reference provably cannot
//! outlive a `safepoint`, making the parking protocol borrow-checked
//! instead of conventional. Parking a handle parks its registered child
//! subtrees bottom-up, so the snapshotter's guard sees a fully quiesced
//! tree.

use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread;
use std::time::{Duration, Instant};

/// Outcome of the stop request handshake.
enum Handshake {
    /// The other side is parked and holds no references.
    Parked,
    /// The other side is gone (its handle was dropped) — it will never
    /// park again; access is free.
    Closed,
    /// A deadline elapsed before the other side parked. Safe to give up:
    /// a mutator re-checks the request flag before parking, so a late
    /// parker returns immediately.
    TimedOut,
}

struct Shared {
    pending_request: AtomicBool,
    /// Set when the owning handle is dropped: a finished worker never
    /// parks, and a parent waiting to park it must see this instead of
    /// blocking forever.
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

    fn is_pending(&self) -> bool {
        self.pending_request.load(Ordering::SeqCst)
    }

    /// The handshake lock, with its poisoning invariant in one place: only
    /// pure bool ops and condvar waits run under it — no code that can
    /// panic ever holds it, so the lock cannot be poisoned.
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
    /// closes, or (with a timeout) the deadline elapses. The one
    /// implementation of the request half of the handshake — the public
    /// `request`/`request_timeout` and the child parking inside
    /// `Handle::safepoint` are all interfaces to it.
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
    /// wakeup can be lost between the waiter's check and its wait.
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

/// The fused lease + checkpointer: the ONLY way to access a checkpointed
/// subtree. `block_and_get`/`block_and_get_mut` tie their borrows to
/// `&mut self`, and `safepoint` needs `&mut self` — so a reference into
/// the subtree provably cannot outlive a park. The brand `'o` is
/// unforgeable: it only arises inside [`with_checkpoint_pair`]'s
/// higher-ranked closure, so no unrelated checkpointer can ever name it
/// (which is what prevents unlocking a handle through a foreign pair).
pub struct Handle<'o, T> {
    node: *mut T,
    shared: Arc<Shared>,
    /// Child subtrees created by `fanout`: parking this handle stops the
    /// children first (bottom-up). Deregistered automatically once a
    /// child closes (a finished worker never parks).
    children: Vec<Arc<Shared>>,
    /// Invariant in `'o`: the brand cannot shrink to a foreign lifetime.
    _brand: PhantomData<fn(&'o ()) -> &'o ()>,
}

// SAFETY: the node pointer is only dereferenced through the handle's own
// borrows, and the handle crosses to its (one) owning worker thread —
// `T: Send` covers the data itself.
unsafe impl<'o, T: Send> Send for Handle<'o, T> {}

impl<'o, T> Handle<'o, T> {
    /// Park while a checkpoint has been requested, then return. Requires
    /// `&mut self`: no `block_and_get` borrow can be live across it.
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

    /// Read the leased subtree. The borrow is tied to `&mut self`: a
    /// `safepoint` (or a park inside `fanout`) cannot happen while it is
    /// live, and the snapshotter's guard is only granted when every
    /// handle is parked — the two sides never alias.
    pub fn block_and_get(&mut self) -> &T {
        // SAFETY: exclusive access to the subtree flows only through this
        // handle (the unforgeable brand prevents foreign pairs), and the
        // borrow ends before any park can occur.
        unsafe { &*self.node }
    }

    /// Mutably access the leased subtree. Same borrowing rules as
    /// [`block_and_get`](Self::block_and_get).
    pub fn block_and_get_mut(&mut self) -> &mut T {
        // SAFETY: as `block_and_get`, but exclusive.
        unsafe { &mut *self.node }
    }

    /// Fan out over the children of this subtree.
    ///
    /// `split` enumerates the children (disjoint `&mut` items of a local
    /// reborrow); each item becomes a child [`Handle`] moved to its
    /// scoped-thread worker, registered on this handle so parking stops
    /// bottom-up. The action stores its results INSIDE the node (there is
    /// no return path — ownership of results stays with the state
    /// machine). Serves checkpoint requests while the children run;
    /// returns when every child is finished (the thread scope joins
    /// them; a panicked child propagates at scope exit).
    pub fn fanout<'t, C, I, F>(&mut self, split: fn(&'t mut T) -> I, action: F)
    where
        C: Send + 'o,
        I: Iterator<Item = &'t mut C>,
        F: Fn(Handle<'o, C>) + Sync,
        'o: 't,
    {
        let action = &action;
        // The split runs on a LOCAL reborrow; the item borrows end when
        // each is consumed into its child handle.
        let node: *mut T = self.node;
        let this: *mut Self = self;
        thread::scope(move |scope| {
            let mut join_handles: Vec<thread::ScopedJoinHandle<'_, ()>> = Vec::new();
            // SAFETY: `self` is mutably borrowed for the whole fan-out
            // (this method holds `&mut self`); deriving the split
            // reborrow from the node pointer is the same borrow.
            let items = split(unsafe { &mut *node });
            for child in items {
                let child_shared = Shared::new();
                // SAFETY: `this` is our own `&mut self`, valid for the
                // scope of this method.
                unsafe { &mut *this }.children.push(child_shared.clone());
                // The item's `&mut C` is consumed into the child handle:
                // its borrow ends here, and the child is exclusively
                // reachable through the handle (disjoint split items make
                // concurrent workers sound).
                let h: Handle<'o, C> = Handle {
                    node: std::ptr::from_mut(child),
                    shared: child_shared,
                    children: Vec::new(),
                    _brand: PhantomData,
                };
                join_handles.push(scope.spawn(move || action(h)));
            }

            loop {
                // SAFETY: as above — our own `&mut self`.
                let me: &mut Self = unsafe { &mut *this };
                if me.is_stop_requested() {
                    me.park_self();
                }
                if join_handles.iter().all(thread::ScopedJoinHandle::is_finished) {
                    break;
                }
                thread::sleep(Duration::from_millis(1));
            }
        })
    }
}

impl<'o, T> Drop for Handle<'o, T> {
    fn drop(&mut self) {
        // A parent waiting to park this subtree must observe the close
        // instead of blocking on a park that will never come.
        self.shared.close();
    }
}

/// Transmitter side of the pair. Lives with the snapshotter; interior
/// synchronization allows sharing behind a `Mutex`/`Arc` by several
/// threads.
pub struct CheckpointRequester<'a, T> {
    node: *const T,
    shared: Arc<Shared>,
    _brand: PhantomData<fn(&'a ()) -> &'a T>,
}

// SAFETY: the node pointer is only dereferenced through a guard, which
// exists exclusively while every mutator is parked (see `request`); the
// data crosses threads, so `T` must be `Send`.
unsafe impl<'a, T: Send> Send for CheckpointRequester<'a, T> {}
// SAFETY: as Send — sharing the requester shares no node data.
unsafe impl<'a, T: Send> Sync for CheckpointRequester<'a, T> {}

impl<'a, T> Clone for CheckpointRequester<'a, T> {
    fn clone(&self) -> Self {
        Self { node: self.node, shared: self.shared.clone(), _brand: PhantomData }
    }
}

impl<'a, T: Send> CheckpointRequester<'a, T> {
    /// Perform the stop-the-world handshake: request all mutators to park
    /// (the top handle parks its subtree first) and return a guard with
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
    /// this guard (and therefore the stop it caused) is done. Never read —
    /// its existence IS the lock.
    #[allow(dead_code)] // the guard's lifetime is the synchronization
    request_lock: MutexGuard<'a, ()>,
    _brand: PhantomData<fn(&'a mut ()) -> &'a T>,
}

impl<'a, T> CheckpointGuard<'a, T> {
    /// The top-level node, quiesced: every mutator that could reach it is
    /// parked at a safepoint where it provably holds no references.
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

/// Take ownership of `node` and hand `f` a freshly branded
/// (Handle, Requester) pair.
///
/// The brand `'o` is an unforgeable unique lifetime: it only exists
/// inside this call's higher-ranked closure, so no unrelated pair can
/// name it — a handle can never be unlocked through a foreign
/// checkpointer, and neither side can outlive the node.
pub fn with_checkpoint_pair<T, R>(
    node: T,
    f: impl for<'o> FnOnce(Handle<'o, T>, CheckpointRequester<'o, T>) -> R,
) -> R {
    let mut node = node;
    let node_ptr: *mut T = std::ptr::from_mut(&mut node);
    let shared = Shared::new();
    let handle = Handle {
        node: node_ptr,
        shared: shared.clone(),
        children: Vec::new(),
        _brand: PhantomData,
    };
    let requester = CheckpointRequester {
        node: node_ptr as *const T,
        shared,
        _brand: PhantomData,
    };
    f(handle, requester)
}
