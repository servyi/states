//! The two-piece checkpointing library (#19's design): the typing
//! discipline — erased outer `Box<K>` reassignment through `Unsize`-witnessed
//! handles — with no vocabulary in it.
//!
//! # Soundness invariants
//!
//! **I1 (access paths).** The container `K`'s content may be
//! accessed by client code ONLY through the line the library drives:
//! the `PersistentCell`'s own `Deref`/`DerefMut` and the container's
//! `DerefMut` they lead to. A client knows the concrete `K` and
//! could reach its pointee in other ways — that is forbidden; every
//! content access rides the library's line, which is the line the
//! mint's receiver and the guard's reads share.
//!
//! **I2 (slot lifetime).** The client hands the library a `&` to
//! each `PersistentCell` (the fanout split's children and the root).
//! A sound client can never invalidate that reference: doing so
//! requires a `&mut` to the cell proving the `&` is gone, and the
//! library hands out no such `&mut` to client code until it is done
//! with the subtask. Hence the backing store outlives every handle.
//!
//! **I4 (tree shape of borrows).** While a node is not a leaf node,
//! no `&mut` to it exists. While a node IS a leaf node, a `&mut` to
//! it exists only through borrowing `DerefMut` of the handle.
//! Consequence: while all leaf nodes are in a safepoint, no `&mut`
//! exists to any node in the tree — which lets the guard spring a
//! fresh `&` to the root, from which non-conflicting `&` to every
//! node can be derived (the checkpoint read side).
//!
//! # The slot content's only bound is `HandleAble<V>`
//!
//! The content a slot holds — `<K::Target as AsPtr>::Target` — is
//! bounded by [`HandleAble<V>`] and nothing else: no `Unsize`, no
//! `Sized`, no object lifetime. The mint dispatches
//! `box_innerhandle` through that trait directly — statically for a
//! concrete node, through the vtable for an erased one
//! (`dyn HandleAble<dyn Vocab>`, or `dyn Erased` with
//! `Erased: HandleAble<V>`) — so it never names `dyn HandleAble<V>` as
//! a type. A `: Unsize<dyn Tr>` bound is never needed; `: Tr` does.

use std::cell::UnsafeCell;
use std::marker::{PhantomData, Unsize};
use dashmap::DashMap;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, Weak};
use std::time::{Duration, Instant};

/// The parking/membership state shared by a handle and its children
/// (the sketch's `status`): children are weakly registered here.
/// A child's `Shared` stays alive only while its own handle does —
/// the registry keeps no strong reference, so a dead child is one
/// whose Weak no longer upgrades.
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
///
/// # The handshake protocol (the simplified requester side)
///
/// A requester sets `pending_request` and waits for the mutator to
/// park; a mutator parks at its `safepoint`s (and a fan-out parent in
/// its join loop) — children first, then itself — and resumes its
/// children once it runs again. There is no `closed` flag: the
/// requester side NEVER holds an `Arc`/`&Shared` across a wait — it
/// looks the mutator up through the Weak each round, with every CV
/// wait bounded by [`REQUEST_WAIT_MS`]. A mutator that drops in an
/// inopportune moment (never parking again) therefore does not hang
/// anyone: its drop releases the last strong `Arc`, the next Weak
/// lookup fails, and the requester learns it is gone.
struct Shared {
    children: Mutex<Vec<Weak<Shared>>>,
    /// the tree-wide uniqueness registry (created at the root,
    /// Arc-referenced from every Shared): addresses of slots
    /// currently being driven by some subtask. Same-slot subtasks
    /// anywhere in the tree serialize on it instead of minting
    /// overlapping exclusive chains.
    claims: Arc<DashMap<usize, ()>>,
    /// A checkpoint has been requested: mutators park at the next
    /// safepoint.
    pending_request: std::sync::atomic::AtomicBool,
    /// Poisoning invariant: only pure bool stores/loads and condvar
    /// waits run under this lock — no code that can panic ever holds
    /// it, so the lock cannot be poisoned.
    parked: Mutex<bool>,
    park_cv: Condvar,
    resume_cv: Condvar,
}

/// Releases a global slot claim on unwind or completion — a panicked
/// subtask must not leave its slot claimed (the tree would deadlock).
///
/// The release never waits inside the registry either (see
/// [`try_claim`]): a contended shard is retried — the holder holds it
/// for one map operation.
struct ClaimGuard {
    claims: Arc<DashMap<usize, ()>>,
    key: usize,
}
impl Drop for ClaimGuard {
    fn drop(&mut self) {
        loop {
            match self.claims.try_entry(self.key) {
                // nothing is held under that key: already released
                Some(dashmap::Entry::Vacant(_)) => break,
                Some(dashmap::Entry::Occupied(o)) => {
                    o.remove();
                    break;
                }
                // the shard is contended: the holder lets go within one
                // map operation
                None => std::hint::spin_loop(),
            }
        }
    }
}

/// Take `key` in the claims registry if it is free: `true` when this
/// call claimed it.
///
/// A function on purpose, and NON-BLOCKING on purpose.
///
/// A function: a `dashmap::Entry` owns its registry shard's write lock,
/// and the scrutinee of a `match` lives until the `match` ends — so a
/// waiter that backed off from INSIDE
/// `match claims.entry(key) { .. }` parked (whenever a checkpoint was
/// requested) still holding that shard. Every other waiter with a key
/// in that shard then blocked inside the registry, where it can never
/// reach a safepoint, and the requester — waiting for that waiter to
/// park — never released the first: a deadlock. Three subtasks handed
/// the same cell, plus one checkpoint, were enough. Here the `Entry`
/// is gone before the caller sees the result.
///
/// Non-blocking: a thread must never WAIT inside the registry. A
/// contended shard is reported as "not claimed", and the caller waits
/// in its own backoff — which is where a checkpoint reaches it. With
/// the blocking `entry()` a contending thread instead sat in the
/// registry's lock, able to answer nothing until the holder let go;
/// and the registry's lock parks its waiters through a futex call Miri
/// rejects, which cut off most Miri runs of the contending tests (8 of
/// 8 clean now, both models).
fn try_claim(claims: &DashMap<usize, ()>, key: usize) -> bool {
    for _ in 0..CLAIM_SPIN_ROUNDS {
        match claims.try_entry(key) {
            // another thread is mid-operation on this shard: a spin, not
            // a park — the holder lets go within one map operation
            None => std::hint::spin_loop(),
            Some(dashmap::Entry::Vacant(v)) => {
                let _occupied = v.insert(());
                return true;
            }
            // held by another chain: not claimed. The caller backs off —
            // through the library's own park, where a checkpoint reaches it.
            Some(dashmap::Entry::Occupied(_)) => return false,
        }
    }
    // the shard stayed busy for the whole spin: reported as not claimed,
    // and the caller's backoff (its own park) takes over
    false
}

/// A container slot whose current node a snapshot can read: one
/// `Deref` from the container to the node (the homogeneous-tree
/// shape `Box<PersistentCell<T>>`). The read side's vocabulary —
/// what lets the guard's view serialize without knowing the drive
/// vocabulary.
pub trait SnapshotSlot: DerefMut {
    /// The node this slot currently holds.
    type Node: ?Sized;

    /// The node this slot currently holds, shared.
    fn snapshot_node(&self) -> &Self::Node;
}

impl<T: ?Sized> SnapshotSlot for Box<PersistentCell<T>> {
    type Node = T;

    fn snapshot_node(&self) -> &T {
        self
    }
}

/// The opaque shell for `Arc<Shared>` that crosses the one public
/// signature which needs it (`box_innerhandle`): `Shared` itself is
/// crate-private, and the shell's field is private, so no client can
/// construct or inspect one.
pub struct SharedRef(Arc<Shared>);

/// Outcome of the stop request handshake.
enum Handshake {
    /// The mutator is parked and holds no references.
    Parked,
    /// The Weak no longer upgrades: the mutator side is gone (its
    /// drop released the last strong `Arc`) — it will never park
    /// again; access is free.
    Closed,
}

/// How long the join loop sleeps between finished-checks (main's).
const JOIN_POLL_INTERVAL_MS: u64 = 1;

/// How long any requester-side CV wait may block before the Weak is
/// consulted again — the whole death-detection scheme rides on this
/// being short.
const REQUEST_WAIT_MS: u64 = 10;

/// How long a LEVEL-2 (content) claim may wait before the cycle breaker
/// gives up. Unlike cell claims (which resolve when the owning subtask
/// ends), content waits can form cross-claim cycles and intrathread
/// self-deadlocks — the bounded window turns those into a deterministic
/// [`ContentClaimTimeout`] instead of a silent hang.
const CONTENT_CLAIM_WINDOW: std::time::Duration = std::time::Duration::from_millis(100);

/// The backoff sleep between pending-checks while waiting on a claim.
const PARK_SLEEP: std::time::Duration = std::time::Duration::from_millis(1);

/// How many times a claim re-tries a shard that another thread is in the
/// middle of an operation on. A SPIN, never a park: the holder keeps the
/// shard for one map operation. (Without it a transiently contended shard
/// counts as a failed attempt and costs the caller a full backoff — which,
/// under a slow interpreter, was enough to spend the content claim's
/// whole window on collisions: measured, 6 attempts in 100 ms, the key
/// never held.)
const CLAIM_SPIN_ROUNDS: usize = 4096;

/// The cycle breaker, as an ERROR GATE: a content claim was still held
/// by another chain when `CONTENT_CLAIM_WINDOW` ran out — a
/// cross-claim cycle, or an intrathread self-deadlock.
///
/// The call that returns it minted and installed nothing, so the tree
/// is as it was: the caller can take a snapshot, turn the agent down,
/// and repair it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContentClaimTimeout;

/// What [`ContentClaimTimeout`] says.
const CONTENT_CLAIM_TIMEOUT_MESSAGE: &str =
    "content claim timeout: the state is still driven by another chain \
     (cross-claim cycle or self-deadlock)";

impl std::fmt::Display for ContentClaimTimeout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(CONTENT_CLAIM_TIMEOUT_MESSAGE)
    }
}

impl std::error::Error for ContentClaimTimeout {}

/// THE CONTENT CLAIM (level 2), bounded: take the claim whose key
/// `content_address` yields, waiting while another chain holds it — but
/// only for [`CONTENT_CLAIM_WINDOW`]. Unlike cell claims (which resolve
/// when the owning subtask ends), content waits can form cross-claim
/// cycles and intrathread self-deadlocks, so when the window runs out
/// this gives up — the cycle breaker: the caller gets a
/// [`ContentClaimTimeout`] and nothing has been minted or installed.
///
/// The address is asked for afresh on every attempt: the borrow it is
/// taken through must not live across the park in
/// [`Shared::backoff_checkpoint`] (no `&mut` exists to any node while the
/// world is stopped). The key actually claimed comes back — the caller
/// releases it when it is done.
fn claim_content(
    claims: &DashMap<usize, ()>,
    shared: &Shared,
    content_address: impl Fn() -> usize,
) -> Result<usize, ContentClaimTimeout> {
    let deadline = std::time::Instant::now() + CONTENT_CLAIM_WINDOW;
    loop {
        let key = content_address();
        if try_claim(claims, key) {
            return Ok(key);
        }
        if std::time::Instant::now() >= deadline {
            return Err(ContentClaimTimeout);
        }
        shared.backoff_checkpoint();
    }
}

impl Shared {
    fn new() -> Self {
        Shared {
            children: Mutex::new(Vec::new()),
            claims: Arc::new(DashMap::new()),
            pending_request: std::sync::atomic::AtomicBool::new(false),
            parked: Mutex::new(false),
            park_cv: Condvar::new(),
            resume_cv: Condvar::new(),
        }
    }

    fn is_pending(&self) -> bool {
        self.pending_request.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// The handshake lock, with its poisoning invariant in one place:
    /// only pure bool ops and condvar waits run under it.
    fn lock_parked(&self) -> MutexGuard<'_, bool> {
        self.parked.lock().expect("parked handshake lock: see lock_parked's invariant")
    }

    /// Wait for the request to be released (the MUTATOR side wait —
    /// untimed on purpose: only the requester side must re-look
    /// through the Weak, and only its waits carry the death window).
    fn wait_resume<'g>(&self, parked: MutexGuard<'g, bool>) -> MutexGuard<'g, bool> {
        self.resume_cv.wait(parked).expect("parked handshake: see lock_parked's invariant")
    }

    /// The parking dance (children from the registry): request every
    /// still-live child subtree first (bottom-up), then this thread;
    /// resume the children after it runs again.
    fn park_self(&self) {
        // snapshot the Weaks under the lock, release — the convention —
        // then act; a Weak that stops upgrading mid-dance is a child
        // that died, skipped by the lookup itself
        let children = self.child_weaks();
        for weak in children.iter() {
            let _ = request_until_parked(weak);
        }
        let mut parked = self.lock_parked();
        *parked = true;
        self.park_cv.notify_all();
        while self.is_pending() {
            parked = self.wait_resume(parked);
        }
        *parked = false;
        self.park_cv.notify_all();
        for weak in children.iter() {
            if let Some(child) = weak.upgrade() {
                release_request(&child);
            }
        }
    }

    /// Register a child (insert under the lock, release — the
    /// convention). The child stays alive by its owner's Arc.
    fn register_child(&self, child: &Arc<Shared>) {
        self.children
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(Arc::downgrade(child));
    }

    /// One backoff step for a subtask WAITING on a global claim: the
    /// Shared is preallocated and registered before the claim loop,
    /// so the world must be able to stop while it waits — park on a
    /// pending request (the subtask has no children yet: it has not
    /// minted), then a short sleep before the next attempt.
    fn backoff_checkpoint(&self) {
        if self.is_pending() {
            self.park_self();
        }
        std::thread::sleep(PARK_SLEEP);
    }

    /// A child Shared: the parking identity is its own, the claims
    /// registry is INHERITED from the parent (tree-wide).
    fn new_child(&self) -> Shared {
        Shared::with_claims(Arc::clone(&self.claims))
    }

    fn with_claims(claims: Arc<DashMap<usize, ()>>) -> Self {
        Shared {
            children: Mutex::new(Vec::new()),
            claims,
            pending_request: std::sync::atomic::AtomicBool::new(false),
            parked: Mutex::new(false),
            park_cv: Condvar::new(),
            resume_cv: Condvar::new(),
        }
    }

    /// The registered children as Weaks, cloned under the lock and
    /// released — the requester-side half of the convention: no Arc
    /// leaves this method, so a child can die under our feet and the
    /// next upgrade simply fails.
    fn child_weaks(&self) -> Vec<Weak<Shared>> {
        self.children
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }
}

/// THE stop request, the requester side: set the flag and wait until
/// the mutator parks — or the Weak stops upgrading. Never holds an
/// `Arc`/`&Shared` across a wait: each round looks the mutator up
/// fresh, and every CV wait is bounded by [`REQUEST_WAIT_MS`], so a
/// mutator that drops without parking is noticed on the next lookup
/// instead of hanging the requester forever.
fn request_until_parked(weak: &Weak<Shared>) -> Handshake {
    // the flag first; the mutator keeps its own Arc while it runs,
    // so the Shared cannot die here unless the mutator is already
    // gone (in which case the store was on a Shared the mutator still
    // owned a moment ago — or nobody's, see the lookup below)
    if let Some(shared) = weak.upgrade() {
        shared.pending_request.store(true, std::sync::atomic::Ordering::SeqCst);
    }
    loop {
        let Some(shared) = weak.upgrade() else {
            return Handshake::Closed;
        };
        let parked = shared.lock_parked();
        if *parked {
            // returning with the flag set: the mutator stays parked
            // until OUR release clears it
            return Handshake::Parked;
        }
        let (guard, _) = shared
            .park_cv
            .wait_timeout(parked, Duration::from_millis(REQUEST_WAIT_MS))
            .expect("parked handshake: see lock_parked's invariant");
        // drop the guard AND the Arc before looking again — the death
        // window is exactly this gap
        drop(guard);
        drop(shared);
    }
}

/// [`request_until_parked`] with a deadline: gives up (returning
/// `None`) when the world has not stopped in time — the skip-round
/// behavior the snapshotter needs against busy machines.
fn request_until_parked_timeout(weak: &Weak<Shared>, timeout: Duration) -> Option<Handshake> {
    if let Some(shared) = weak.upgrade() {
        shared.pending_request.store(true, std::sync::atomic::Ordering::SeqCst);
    }
    let deadline = Instant::now() + timeout;
    loop {
        let Some(shared) = weak.upgrade() else {
            return Some(Handshake::Closed);
        };
        let parked = shared.lock_parked();
        if *parked {
            return Some(Handshake::Parked);
        }
        let now = Instant::now();
        if now >= deadline {
            // retract the request: a mutator checking the flag after
            // this store simply does not park
            shared.pending_request.store(false, std::sync::atomic::Ordering::SeqCst);
            return None;
        }
        let (guard, _) = shared
            .park_cv
            .wait_timeout(parked, deadline - now)
            .expect("parked handshake: see lock_parked's invariant");
        drop(guard);
        drop(shared);
    }
}

/// THE release (requester side): wake the parked side. No wait for
/// the resume — waiting would hold the `&Shared` across a wait. The
/// parked side re-checks the flag under its own wait loop, so a
/// follow-up request that re-sets the flag before the parked thread
/// wakes keeps it parked: back-to-back guards hand off seamlessly
/// without the requester ever observing the resume.
fn release_request(shared: &Shared) {
    let parked = shared.lock_parked();
    shared.pending_request.store(false, std::sync::atomic::Ordering::SeqCst);
    shared.resume_cv.notify_all();
    drop(parked);
}

/// The poison recovery for the requester gate: the previous gate
/// holder panicked somewhere between acquiring the gate and dropping
/// its guard. If that happened mid-handshake, `pending_request` is
/// stuck true with no owner — the mutators parked beneath it would
/// wait forever. The gate being FREE here means no live guard can
/// exist, so a pending seen at this point is ownerless by
/// construction; releasing it resumes the world. (A panic that
/// happened mid-guard-life instead unwound the guard's own drop,
/// which released cleanly — the pending is already false, a no-op.)
fn recover_poisoned_gate(weak: &Weak<Shared>) {
    if let Some(shared) = weak.upgrade() {
        if shared.is_pending() {
            release_request(&shared);
        }
    }
}

/// The OUTER handle: owns the TYPED slot pointer (`*mut K` —
/// `replace`'s write target, no casting anywhere), the chain's
/// `Shared` (identity that survives installs), and the current
/// inner handle (`Box<V>`, which it knows is an `InnerHandle<T, V>`
/// minted over this slot) — the dispatch surface for `Deref` to
/// the vocabulary. `replace` is the successor install: write the K
/// cell this handle owns, then re-mint through the replacement's
/// `HandleAble` and keep the fresh handle in the box.
///
/// The `'o` brand is unforgeable: it only arises inside
/// [`independent_task_root`]'s higher-ranked closure, so the handle
/// (and everything minted under it, via `fanout`'s own `'n`)
/// structurally cannot leave that call.
pub struct OuterHandle<'o, K: Sized, V: ?Sized> {
    current: Box<V>,
    /// the PersistentCell cell this chain owns — `replace`'s write
    /// target, typed at the wrapper (the container K lives inside)
    slot: *mut PersistentCell<K>,
    /// the chain's parking identity — passed on across installs
    shared: Arc<Shared>,
    /// Invariant in `'o`: the brand cannot shrink to a foreign
    /// lifetime — the handle stays inside the root task.
    _brand: PhantomData<fn(&'o ()) -> &'o ()>,
}

/// The mutator-side handle: `state` (deref's target — captured at
/// the mint from the receiver itself, so its `T` is the pointed
/// object's own type, fixed by the dyn dispatch that minted it) and
/// the parking `status`. It knows nothing of the slot — that is the
/// `OuterHandle`'s business (the mutator drives, the outer installs).
///
/// No dispatch happens at deref time — K's `DerefMut` is consulted
/// ONCE, at the mint. A lying custom `DerefMut` can therefore only
/// choose WHICH honest handle is minted (whatever object it returns
/// is dispatched on, and its type must satisfy the `Unsize<V>`
/// judgement to be returnable at all — so `T` always matches the
/// object); it can never produce a `T`/object mismatch.
///
/// K's own obligation shrinks to: not get rid of the returned
/// pointer while the handle lives — and it cannot do so safely: the
/// `&'x mut T` was taken out of the library's own `&'x mut K`, and
/// the only writer of the slot while a handle lives is `replace`
/// (on the outer), which re-mints the pair.
pub struct InnerHandle<T: ?Sized> {
    /// the state raw — carrying the box's STORED provenance (minted
    /// by a raw borrow of the K's deref place, not from any materialized
    /// `&mut`), so the persistent line's path to the allocation root
    /// crosses no plain mutable node. Every deref mints a FRESH
    /// reference off it; nothing persistent is ever a mutable line.
    state: *mut T,
    status: Arc<Shared>,
    _marker: PhantomData<fn(*mut T)>,
}

// SAFETY: the state raw is exclusively reachable through this handle
// for its life (the mint's discipline), the status Arc is Send+Sync,
// and T: Send crosses with it.
unsafe impl<T: ?Sized + Send> Send for InnerHandle<T> {}

// SAFETY: the payload crosses threads: the box (an InnerHandle: its
// T: Send by the HandleAble blanket), and the K cell it exclusively
// owns — moving the outer moves the exclusive access to both, so K
// must be Send for the pointee's sake.
unsafe impl<'o, K: Send, V: ?Sized + Send> Send for OuterHandle<'o, K, V> {}

/// The root task's checkpointing companion: knows the root only
/// through a **Weak** (the mutator side owns the last strong `Arc` —
/// its drop IS the close) and points at the root slot — the stable
/// storage `independent_task_root` borrows for the whole task.
/// `request` runs the stop-the-world handshake and hands back a
/// guard with direct access to the slot while every mutator parks.
///
/// Requester serialization lives in `gate`, shared by every clone —
/// two concurrent requesters must not both observe `Parked` (each
/// would take a guard over the same stopped world). It was `Shared`'s
/// `request_lock`; the guard can no longer borrow a lock out of a
/// `Shared` the requester only holds weakly, so the gate moved here.
/// The mutator never takes it, as before.
///
/// The `'o` brand ties it to the same root-task closure as its
/// `OuterHandle` — it cannot leave the call either. It has no `V`:
/// nothing V-typed crosses it, the guard derefs straight to `K`.
pub struct Checkpointer<'o, K> {
    slot: *mut PersistentCell<K>,
    weak: Weak<Shared>,
    gate: Arc<Mutex<()>>,
    _brand: PhantomData<fn(&'o ()) -> &'o K>,
}

// SAFETY: the slot pointer is only dereferenced through a guard,
// which exists exclusively while every mutator is parked (see
// `request`); the data crosses threads, so K must be Send.
unsafe impl<'o, K: Send> Send for Checkpointer<'o, K> {}
// SAFETY: as Send — sharing the checkpointer shares no slot data.
unsafe impl<'o, K: Send> Sync for Checkpointer<'o, K> {}

impl<'o, K> Clone for Checkpointer<'o, K> {
    fn clone(&self) -> Self {
        Self {
            slot: self.slot,
            weak: self.weak.clone(),
            gate: self.gate.clone(),
            _brand: PhantomData,
        }
    }
}

impl<'o, K: Send> Checkpointer<'o, K> {
    /// [`request`](Self::request) with a deadline: additionally gives
    /// up (returning `None`) if the world has not stopped in time —
    /// safe, because a mutator re-checks the request flag before
    /// parking, so a late parker returns immediately.
    pub fn request_timeout(&self, timeout: Duration) -> Option<CheckpointGuard<'_, K>> {
        let request_lock = self.gate.lock().unwrap_or_else(|e| {
            let guard = e.into_inner();
            recover_poisoned_gate(&self.weak);
            guard
        });
        match request_until_parked_timeout(&self.weak, timeout)? {
            Handshake::Parked => Some(CheckpointGuard {
                slot: self.slot,
                shared: self.weak.upgrade(),
                request_lock,
                _marker: PhantomData,
            }),
            Handshake::Closed => Some(CheckpointGuard {
                slot: self.slot,
                shared: None,
                request_lock,
                _marker: PhantomData,
            }),
        }
    }

    /// Perform the stop-the-world handshake: request all mutators to
    /// park (the top handle parks its subtree first) and return a
    /// guard with direct access to the root slot. Blocks until
    /// parked. A gone tree (mutator side dropped, the Weak empty)
    /// hands out the guard at once — nothing will ever mutate again.
    /// Concurrent `request`s serialize on the gate: a second request
    /// blocks until the first guard is dropped and the world resumed.
    pub fn request(&self) -> CheckpointGuard<'_, K> {
        // held for the whole guard life — the serialization. A
        // POISONED gate means a requester panicked mid-protocol under
        // it: the recovery runs exactly there, where the poison is
        // the evidence — a clean gate promises a clean flag, because
        // the previous cycle's release ran under this same gate.
        let request_lock = self.gate.lock().unwrap_or_else(|e| {
            let guard = e.into_inner();
            recover_poisoned_gate(&self.weak);
            guard
        });
        match request_until_parked(&self.weak) {
            Handshake::Parked => CheckpointGuard {
                slot: self.slot,
                // the mutator is parked under OUR pending — its own
                // Arc keeps the Shared alive, so the lookup cannot
                // fail here; held for the guard's life so the drop's
                // release lands on a live Shared
                shared: self.weak.upgrade(),
                request_lock,
                _marker: PhantomData,
            },
            Handshake::Closed => CheckpointGuard {
                slot: self.slot,
                shared: None,
                request_lock,
                _marker: PhantomData,
            },
        }
    }
}

/// Read-only access to the root slot while every mutator is parked,
/// through `Deref` — the snapshot. There is deliberately NO
/// `DerefMut`: a write could install a state of a different type
/// under the parked mutator's handle (the slot is type-erased; the
/// chain would cast the new pointee back to its original `T` — type
/// confusion, Miri-verified). On main the guard's target is the node
/// type itself, so a write swaps same-typed values and is sound;
/// here the erasure makes the guard read-only. Dropping the guard
/// resumes the world.
pub struct CheckpointGuard<'a, K> {
    slot: *mut PersistentCell<K>,
    /// Some: the world is stopped under our request — the drop
    /// releases it. None: the mutator side was already gone — access
    /// is free and the drop has nothing to resume.
    shared: Option<Arc<Shared>>,
    /// Held for the guard's whole life: concurrent requesters block
    /// until this guard (and therefore the stop it caused) is done.
    /// Never read — its existence IS the lock.
    #[allow(dead_code)] // the guard's lifetime is the synchronization
    request_lock: MutexGuard<'a, ()>,
    _marker: PhantomData<fn(&'a ()) -> &'a K>,
}

impl<K> std::ops::Deref for CheckpointGuard<'_, K> {
    type Target = PersistentCell<K>;
    fn deref(&self) -> &PersistentCell<K> {
        // SAFETY: the guard exists only while the handshake proved
        // every mutator parked (or gone); the reborrow is tied to the
        // guard, so the world cannot resume underneath it.
        unsafe { &*self.slot }
    }
}

impl<K> Drop for CheckpointGuard<'_, K> {
    fn drop(&mut self) {
        if let Some(shared) = self.shared.take() {
            release_request(&shared);
        }
        // request_lock drops with the struct, releasing the requesters.
    }
}

/// The root task: mint the `OuterHandle` over `slot`, hand the
/// callback BOTH the outer handle and its `Checkpointer`, and keep
/// `slot` inaccessible (borrowed) for the whole call. The `'o` brand
/// arises only inside this higher-ranked closure, so neither handle
/// can leave the call — the discipline is structural.
///
/// `V` is the tree's vocabulary — what the callback drives through its
/// outer handle — and the slot's content must implement
/// [`HandleAble<V>`]: either a CONCRETE node (the blanket impl,
/// `InnerHandle<Node>: Unsize<V>`) or an already-erased one (`dyn Erased`
/// with `Erased: HandleAble<V>` — the sketch demos' shape, e.g.
/// `Box<PersistentCell<dyn HandleAble<V>>>`). `K` and its line are
/// reached through [`AsPtr`] alone: the mint projects through raws.
pub fn independent_task_root<K, V: ?Sized, R>(
    slot: &mut PersistentCell<K>,
    f: impl for<'o> FnOnce(OuterHandle<'o, K, V>, Checkpointer<'o, K>) -> R,
) -> R
where
    K: AsPtr + Sized + Send,
    K::Target: AsPtr,
    <K::Target as AsPtr>::Target: HandleAble<V>,
{
    // the raw is derived ONCE and shared by the outer's chain and the
    // checkpointer — a second `from_mut` on the same `&mut` would
    // invalidate the first raw's tag (the function-entry retag).
    // SAFETY: slot is exclusively borrowed for the whole call (the
    // fn signature itself); the outer and the checkpointer die
    // before it returns.
    let slot_ptr: *mut PersistentCell<K> = std::ptr::from_mut(slot);
    // SAFETY: slot is exclusively borrowed for the whole call (the
    // fn signature itself); the outer and the checkpointer die
    // before it returns.
    let outer = unsafe { OuterHandle::over_slot(slot_ptr) };
    // the Checkpointer knows the root's Shared only WEAKLY — the
    // outer's own Arc is the last strong one, so its drop IS the
    // close; the gate serializes the requester side (see Checkpointer)
    let checkpointer = Checkpointer {
        slot: slot_ptr,
        weak: Arc::downgrade(outer.shared()),
        gate: Arc::new(Mutex::new(())),
        _brand: PhantomData,
    };
    f(outer, checkpointer)
}

impl<'o, K, V: ?Sized> OuterHandle<'o, K, V> {
    fn shared(&self) -> &Arc<Shared> {
        &self.shared
    }
}

impl<'o, K, V: ?Sized> std::ops::Deref for OuterHandle<'o, K, V> {
    type Target = V;
    fn deref(&self) -> &V {
        // the box is the load-bearing invariant (always an
        // InnerHandle<_, V> over the K-shaped slot) — deref lands on
        // the V through it, the box itself is never handed out
        let current: *const V = self.current.as_ref() as *const V;
        // SAFETY: the box holds an InnerHandle<_, V> (the invariant);
        // viewing it as the V it erases is the Unsize correspondence.
        unsafe { &*current }
    }
}

impl<'o, K, V: ?Sized> std::ops::DerefMut for OuterHandle<'o, K, V> {
    fn deref_mut(&mut self) -> &mut V {
        let current: *mut V = self.current.as_mut() as *mut V;
        // SAFETY: as Deref — exclusive.
        unsafe { &mut *current }
    }
}

impl<'o, K, V: ?Sized> OuterHandle<'o, K, V>
where
    K: AsPtr + Send,
    K::Target: AsPtr,
    <K::Target as AsPtr>::Target: HandleAble<V>,
{
    /// Mint over a live slot the driver owns (crate-internal: the
    /// root task is the only minter through here).
    ///
    /// # Safety
    ///
    /// The caller's `&mut` slot (through `independent_task_root`)
    /// resolves validity and mint-time exclusivity; what remains:
    ///
    /// - The `K` must stay valid and live, at that address, for the
    ///   handle's whole life (the raw stored in the minted chain
    ///   derives from this `&mut`).
    /// - The `K` must not be reachable by shared reference from other
    ///   threads while the handle lives: its `Deref` would alias the
    ///   pointee the chain owns exclusively.
    /// - Every `&mut` into the slot that is active during the
    ///   handle's lifetime must be derived from this pointer.
    /// - Such a `&mut` must either come from `DerefMut` (the chain's
    ///   own derefs), or exist only while the handle's `Shared` is
    ///   parked (a guard's world-stopped access).
    ///
    /// The constructor establishes the `OuterHandle` invariant: the
    /// box is a valid `InnerHandle<T, V>` whose `state` is the raw
    /// of the pointee of the `K` as stored by the box (the loaded
    /// provenance), and the typed `slot` stored here is exactly the
    /// `K` cell the raw was taken from.
    unsafe fn over_slot(slot: *mut PersistentCell<K>) -> Self
    where
        K: Sized,
    {
        // the state raw: the mint projects through RAWS ONLY — the
        // slot's hatch, the container's stored line, the content's
        // line. No reference is materialized anywhere on the mint
        // path: a `&mut` here would go Unique on the first write and
        // freeze on the guard's read, forbidding every later write
        // through the chain (see `AsPtr`).
        // SAFETY: the caller's contract above — the root task holds the
        // slot exclusively for the whole call, so the projections are
        // the only live access (`as_ptr`'s caller obligation).
        let k_ptr: *mut K = unsafe { AsPtr::as_ptr(slot) };
        // SAFETY: as above.
        let target_ptr: *mut K::Target = unsafe { AsPtr::as_ptr(k_ptr) };
        // SAFETY: as above.
        let state_raw = unsafe { AsPtr::as_ptr(target_ptr) };
        // SAFETY: the state raw is the content's own line, live under
        // the caller's contract; the receiver is transient and dies at
        // return. It is the CONTENT, bounded only by `HandleAble<V>`:
        // the mint dispatches through that trait — never through
        // `dyn HandleAble<V>`.
        let erased: &mut <K::Target as AsPtr>::Target = unsafe { &mut *state_raw };
        // the chain's ORIGIN: a fresh root Shared moves in
        let shared = Arc::new(Shared::new());
        let state_erased: *mut () = state_raw as *mut ();
        let shared_ref = SharedRef(shared.clone());
        // SAFETY: the receiver came from this slot (the contract).
        let current = unsafe { erased.box_innerhandle(state_erased, shared_ref) };
        OuterHandle { current, slot, shared, _brand: PhantomData }
    }

    /// The successor install: claim the replacement's content, write the
    /// K cell THIS handle owns (typed — no casting), then re-mint
    /// through the replacement's `HandleAble` and keep the fresh handle
    /// in the box. The chain survives: the outer's `Shared` passes on to
    /// the successor, so parking identity is untouched.
    ///
    /// # Errors
    ///
    /// [`ContentClaimTimeout`] when the replacement's content is still
    /// driven by another chain after `CONTENT_CLAIM_WINDOW`. Nothing has
    /// been written in that case — the slot and this handle are exactly
    /// as they were.
    pub fn replace(&mut self, mut val: K) -> Result<(), ContentClaimTimeout>
    where
        <K::Target as AsPtr>::Target: HandleAble<V>,
    {
        // 1) THE CONTENT CLAIM, on the tree-wide registry, BEFORE
        // anything is written. An error gate has to leave the tree as it
        // was: with the install first, a lost claim would return from
        // here with this handle's inner handle pointing into the old K —
        // freed by the write.
        //    The key is the replacement's content address as read from
        // `val` where it stands. Content another chain can be driving is
        // reached through a pointer inside the container (it has to be,
        // for two containers to share it), so its address is the same
        // here and in the slot — the key the re-mint below mints under.
        // Content that lives inline in its container cannot be shared,
        // so no other chain can hold a claim on it either way.
        let content_key: usize = {
            // the raw hops on `val` where it stands — two, not three:
            // `val` IS the container, it is not behind a slot's hatch
            // yet. No reference is materialized (see `AsPtr`).
            // SAFETY: `val` is owned here, so the projections are the
            // only live access (`as_ptr`'s caller obligation).
            let target_ptr: *mut K::Target = unsafe { AsPtr::as_ptr(std::ptr::addr_of_mut!(val)) };
            // SAFETY: as above.
            let state_raw = unsafe { AsPtr::as_ptr(target_ptr) };
            state_raw as *const () as usize
        };
        let claims: Arc<DashMap<usize, ()>> = Arc::clone(&self.shared.claims);
        let claimed = claim_content(&claims, &self.shared, || content_key)?;
        let _content_claim = ClaimGuard { claims: Arc::clone(&claims), key: claimed };
        // 2) the swap — plain assignment through the typed slot
        let replacement = PersistentCell::new(val);
        // SAFETY: the slot is live and exclusively reachable through
        // this chain (the mint's discipline; the old inner handle is
        // replaced by the re-mint below, so no stale pointer
        // survives the write).
        unsafe { *self.slot = replacement };
        // 3) re-mint: use the HandleAble OF the replacement —
        // reachable through the slot it now lives in (the slot's own
        // AsPtr hatch, consulted once) — to mint a fresh InnerHandle
        // over that same slot, and assign it into the own box
        let fresh: Box<V> = {
            // the raw hops — the slot's hatch, the container's stored
            // line, the content's line; no reference is materialized
            // (see `AsPtr`).
            // SAFETY: this chain exclusively owns the slot (the mint's
            // discipline; the old inner handle is replaced by the
            // re-mint below), so the projections are the only live
            // access (`as_ptr`'s caller obligation).
            let k_ptr: *mut K = unsafe { AsPtr::as_ptr(self.slot) };
            // SAFETY: as above.
            let target_ptr: *mut K::Target = unsafe { AsPtr::as_ptr(k_ptr) };
            // SAFETY: as above.
            let state_raw = unsafe { AsPtr::as_ptr(target_ptr) };
            // SAFETY: the state raw is the content's own line, live under
            // the mint's discipline; the receiver is transient and dies
            // at return. As in `over_slot`: the content itself,
            // dispatched through `HandleAble<V>`.
            let erased: &mut <K::Target as AsPtr>::Target = unsafe { &mut *state_raw };
            let state_erased: *mut () = state_raw as *mut ();
            let shared_ref = SharedRef(self.shared.clone());
            // SAFETY: the mint's discipline over the fresh content.
            unsafe { erased.box_innerhandle(state_erased, shared_ref) }
        };
        self.current = fresh;
        Ok(())
    }
}

/// A container the mint can project through **on raws alone** — the
/// line the container's own `Deref`/`DerefMut` spring from. Two
/// impls, one per shape:
///
/// - **`PersistentCell<T>`**: the content through the hatch — the
///   interior-mutable line (the sound shape for erased states);
/// - **`Box<X>`**: the *stored* line (`&raw mut **this`) — for
///   containers whose target is itself a box.
///
/// The receiver is a RAW on purpose: no reference — least of all a
/// `&mut` — is materialized anywhere on the mint path. A materialized
/// `&mut` goes `Unique` on the first write and `Frozen` on the first
/// foreign read (the checkpoint guard's read through the shared
/// side), which forbids every later write through the chain's whole
/// subtree (Tree Borrows). Raws never freeze.
///
/// # Safety
///
/// For implementors: the returned pointer must be the line the
/// container's own `Deref`/`DerefMut` spring from — never a freshly
/// minted reference to foreign storage — and the backing store must
/// not be deleted until the next `&mut` method call on the container.
/// The impl must not rely on an exclusive reference existing: the
/// exclusivity is the CALLER's burden (the method's `unsafe` contract
/// and the chain's discipline), not the receiver's.
///
/// **Coherence is structural:** these are the only library impls,
/// and a client cannot write one for `PersistentCell` or `Box`
/// (foreign trait, foreign type — E0210). For a CONCRETE state `T`
/// as a slot target (`Box<T>`), the client writes their own one-line
/// identity impl (`unsafe impl AsPtr for MyState { type Target =
/// MyState; unsafe fn as_ptr(this: *mut Self) -> *mut Self::Target {
/// this } }`) — orphan-legal for their local type, and unable to
/// overlap ours.
pub unsafe trait AsPtr {
    /// The line this container hands out: the store its own
    /// `Deref`/`DerefMut` spring from.
    type Target: ?Sized;

    /// Project the container to its line, through the raw alone.
    ///
    /// # Safety
    ///
    /// The caller must hold the chain's exclusivity over the container
    /// (the mint's discipline): no other access to it may be live, and
    /// none may be created, until the mint is done. A `&mut self`
    /// receiver would enforce that — and would put a materialized
    /// reference on the mint path (see the trait's docs). The raw
    /// receiver keeps every reference off it; the obligation is the
    /// caller's.
    unsafe fn as_ptr(this: *mut Self) -> *mut Self::Target;
}

// the cell: the content through the hatch — the IM-covered line.
// NO vocabulary bounds: the impl is fully general over the content.
// SAFETY: the projection takes exactly the wrapper's own field — the
// store the Deref chain springs from — never foreign storage; the
// store lives until the next `&mut` method call on the cell. The
// `UnsafeCell<T>` → `*mut T` cast preserves the unsize metadata —
// `UnsafeCell<T>`'s layout is documented identical to `T` (the same
// cast `get` performs); `addr_of_mut` computes the field's true
// offset, and the sole field sits at offset 0.
unsafe impl<T: ?Sized> AsPtr for PersistentCell<T> {
    type Target = T;

    unsafe fn as_ptr(this: *mut Self) -> *mut T {
        // SAFETY: `this` is a live `PersistentCell` under the caller's
        // exclusivity contract (the mint's discipline); the raw borrow
        // of the field is the wrapper's own hatch — a place projection,
        // no reference materialized.
        let cell: *mut UnsafeCell<T> = unsafe { std::ptr::addr_of_mut!((*this).cell) };
        // `UnsafeCell<T>` is layout-identical to `T` — the same cast
        // `UnsafeCell::get` performs.
        cell as *mut T
    }
}

// the box: the STORED line, not the deref-fresh one. Fully general.
// SAFETY: the raw borrow of the builtin deref place loads the
// stored pointer itself — the box's own line, no fresh reference.
/// The sketch's `&raw mut **this`, with the box taken by reference: the
/// builtin box-deref place, borrowed raw. A safe function on purpose —
/// nothing in it is unsafe, and it needs no layout knowledge.
fn stored_line<T: ?Sized>(boxed: &mut Box<T>) -> *mut T {
    &raw mut **boxed
}

// SAFETY: the projection is the builtin box-deref place borrowed raw —
// the stored pointer itself, the box's own line — reached through a
// reference to the box (its one pointer), never to the content.
unsafe impl<T: ?Sized> AsPtr for Box<T> {
    type Target = T;

    unsafe fn as_ptr(this: *mut Self) -> *mut T {
        // SAFETY: `this` is a live `Box` under the caller's exclusivity
        // contract (the mint's discipline). The reference is to the box
        // itself — its one pointer — never to the content; the content's
        // line is the raw borrow of the box-deref place.
        let boxed: &mut Box<T> = unsafe { &mut *this };
        stored_line(boxed)
    }
}

/// A child node in the tree: the container `K` behind the library's
/// own `UnsafeCell`, so **both sides of a checkpoint reach it through
/// shared references**. The mutator's fanout only needs `&self` (the
/// split collects `&PersistentCell<K>`), and the reader's guard sees the
/// same wrapper — so references to the node may still exist whenever
/// the client looks, and a client deleting or moving the container
/// through its own interior mutability would do so while shared
/// references to this wrapper are live: excluded, because taking the
/// `K` out requires `&mut self` here, and no sound client code can
/// produce that while our shared refs exist.
///
/// Safe surface: [`Deref`] and [`DerefMut`] (the latter through
/// `&mut self` — exclusive on the wrapper, the ordinary interior-
/// mutability pattern). The mint path projects through [`AsPtr`] —
/// raws only, no reference materialized anywhere on it: a materialized
/// `&mut` on the mint path would go `Unique` on the first write and
/// freeze on the guard's first read (Tree Borrows), forbidding every
/// later write through the chain.
pub struct PersistentCell<K: ?Sized> {
    cell: UnsafeCell<K>,
}

impl<K: Sized> PersistentCell<K> {
    pub fn new(k: K) -> Self {
        PersistentCell { cell: UnsafeCell::new(k) }
    }
}

impl<K: Clone> Clone for PersistentCell<K> {
    fn clone(&self) -> Self {
        PersistentCell::new((**self).clone())
    }
}

impl<K: std::fmt::Debug> std::fmt::Debug for PersistentCell<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        (**self).fmt(f)
    }
}

#[cfg(feature = "serde")]
impl<K: serde::Serialize + ?Sized> serde::Serialize for PersistentCell<K> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        (**self).serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de, K: serde::Deserialize<'de>> serde::Deserialize<'de> for PersistentCell<K> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(PersistentCell::new(K::deserialize(deserializer)?))
    }
}


impl<K: ?Sized> PersistentCell<K> {
    /// Adopt a boxed value as a boxed cell — the cell's single field
    /// is `repr(transparent)`, so the layouts are identical and the
    /// box is re-typed in place. This is the constructor for cells
    /// over erased (`?Sized`) content: `Box<dyn ...>` in, boxed cell
    /// out, no reallocation.
    pub fn from_boxed(b: Box<K>) -> Box<PersistentCell<K>> {
        // SAFETY: UnsafeCell is repr(transparent); PersistentCell's
        // only field is the cell — layout-identical to K.
        unsafe { std::mem::transmute(b) }
    }
}

impl<K: ?Sized> Deref for PersistentCell<K> {
    type Target = K;
    fn deref(&self) -> &K {
        let cell_ptr: *const K = self.cell.get();
        // SAFETY: shared access to the cell's content — the wrapper's
        // own UnsafeCell grants exactly this.
        unsafe { &*cell_ptr }
    }
}

impl<K: ?Sized> DerefMut for PersistentCell<K> {
    fn deref_mut(&mut self) -> &mut K {
        let cell_ptr: *mut K = self.cell.get();
        // SAFETY: `&mut self` is exclusive on the wrapper — no other
        // access can exist.
        unsafe { &mut *cell_ptr }
    }
}

// SAFETY: the wrapper crosses threads exactly when its content does.
unsafe impl<K: ?Sized + Send> Send for PersistentCell<K> {}

// SAFETY: the cell's interior mutability is serialized by the
// library's own discipline — the global claims registry ensures one
// exclusive driver per slot at a time, and the checkpoints stop the
// world — so shared references may cross threads to waiting subtasks.
unsafe impl<K: ?Sized + Send> Sync for PersistentCell<K> {}

/// The erasure vocabulary: a state type `T` whose
/// `InnerHandle<T, V>` can be erased to `V`.
///
/// # Safety
///
/// For implementors — implementations are ONLY allowed inside this
/// crate; a foreign implementation would have to uphold all of the
/// following, which the blanket implementation here carries:
///
/// - `T` may implement this trait only if `InnerHandle<T>` can
///   be unsized to `V` (`InnerHandle<T>: Unsize<V>`).
/// - The `Box` returned by [`HandleAble::box_innerhandle`] must be
///   the `InnerHandle<T, V>` over the PASSED state raw (the blanket
///   impl stores it verbatim).
pub unsafe trait HandleAble<V: ?Sized>: Send {
    /// Mint the handle over `self`.
    ///
    /// # Safety
    ///
    /// For callers:
    ///
    /// - The `&mut self` receiver resolves validity and mint-time
    ///   exclusivity by construction; what remains: the K container
    ///   the receiver was taken from must stay valid and live at
    ///   that address for the returned handle's whole life.
    /// - The `K` must not be reachable by shared reference from other
    ///   threads while the handle lives: its `Deref` would alias the
    ///   pointee this handle owns exclusively.
    ///
    /// The `SharedRef` argument is the seal as well as the payload:
    /// it is an opaque shell (`Shared` private, the field private),
    /// so no client can construct one — the method is uncallable
    /// from outside the crate.
    unsafe fn box_innerhandle(&mut self, state: *mut (), shared: SharedRef) -> Box<V>;
}

// SAFETY: the blanket's where-clause (InnerHandle<T>: Unsize<V>)
// carries the whole contract — the client's vocabulary impl for
// the handle is the Unsize witness; the stored raw is the passed
// one, minted under the mint's discipline.
unsafe impl<T: Send, V: ?Sized> HandleAble<V> for T
where
    InnerHandle<T>: Unsize<V>,
{
    unsafe fn box_innerhandle(&mut self, state: *mut (), shared: SharedRef) -> Box<V> {
        // T is fixed by the dispatch that landed here; the persisted
        // raw is the PASSED one — the box's stored provenance. (Off
        // the receiver it would sit above a plain `&mut` line that
        // child writes turn Unique and the saves then Freeze —
        // measured, see the thread.)
        let raw: *mut T = state as *mut T;
        // bind at the sized type, then let the return coerce
        // Box<InnerHandle<T, V>> -> Box<V> (the Unsize where-clause).
        let handle: Box<InnerHandle<T>> = Box::new(InnerHandle {
            state: raw,
            status: shared.0,
            _marker: PhantomData,
        });
        handle
    }
}

impl<T: Sized> InnerHandle<T> {
    /// Park while a checkpoint has been requested, then return.
    /// Requires `&mut self`: no deref borrow can be live across it.
    pub fn safepoint(&mut self) {
        if self.status.is_pending() {
            self.status.park_self();
        }
    }

    pub fn is_stop_requested(&self) -> bool {
        self.status.is_pending()
    }

    /// The fan-out (the protocol per main: parent parks in the join
    /// loop when a checkpoint is requested). `split`
    /// runs on `&T` (a SHARED borrow of the node's own fields) and
    /// yields the children as `&PersistentCell<K>` — one slot per
    /// child. Each element
    /// becomes a `Box<V>` via `box_innerhandle`, and `f` runs on one
    /// scoped thread per box, whose results collect into the returned
    /// `Vec<R>` in DISPATCH order — `joins[i]` is child `i` (the
    /// spawn loop pushes in split order) and the joins only drain
    /// after every child finished, so no completion-order dependence.
    ///
    /// The callback is HIGHER-RANKED over the child brand: `f` must
    /// work for ANY `'n`, so the child `OuterHandle<'n>` can never
    /// appear in `R` (nor be stashed anywhere whose type mentions a
    /// fixed lifetime — the brand is invariant). The handle cannot
    /// leave the callback at all — strictly stronger than branding
    /// it with the root task's `'o`, which is why fanout no longer
    /// carries an `'o` of its own.
    ///
    /// # Errors
    ///
    /// [`ContentClaimTimeout`] when a child's content is still driven by
    /// another chain after `CONTENT_CLAIM_WINDOW` — the cycle breaker.
    /// That child was never minted and its callback never ran; every
    /// other child has finished by then, since the results are collected
    /// only once all of them have.
    ///
    /// # Panics
    ///
    /// Propagates a child that panicked.
    pub fn fanout<V: ?Sized + Send, K, F, R>(
        &mut self,
        split: &(dyn for<'x> Fn(&'x T) -> Vec<&'x PersistentCell<K>> + Send + Sync),
        f: F,
    ) -> Result<Vec<R>, ContentClaimTimeout>
    where
        T: Send,
        K: AsPtr + Sized + Send,
        K::Target: AsPtr,
        <K::Target as AsPtr>::Target: HandleAble<V>,
        F: for<'n> Fn(OuterHandle<'n, K, V>) -> R + Send + Sync,
        R: Send,
    {
        // Collect offsite through a SHARED node borrow: the split
        // hands out `&PersistentCell<K>` — the references may still be
        // alive from the reader's perspective, which is exactly what
        // excludes the client deleting the containers through sound
        // interior mutability. The library then unsafely turns them
        // into the mint's raws (the chain's exclusivity discipline).
        let child_nodes: Vec<&PersistentCell<K>> = {
            let node: &T = self;
            split(node)
        };
        // A shared split cannot promise uniqueness — the old `&mut`
        // split had it from borrowck — and a 'static PersistentCell
        // can be handed out across DIFFERENT levels and concurrent
        // fanouts, so the check must be GLOBAL. The tree-wide claims
        // registry (a DashMap created at the root, Arc-referenced
        // from every Shared) serializes identical subtasks instead of
        // dropping them: each child thread repeatedly tries to insert
        // its slot's address with a short backoff; only after the
        // insert succeeds does it mint and run, and the claim is
        // released (RAII, also on unwind) when the subtask ends.
        let claims: Arc<DashMap<usize, ()>> = Arc::clone(&self.status.claims);
        // each child's Shared is prepared in the parent (its parking
        // identity its own, the registry inherited); registration
        // under the parent's lock, per the convention
        let prepared: Vec<(&PersistentCell<K>, Arc<Shared>)> = child_nodes
            .into_iter()
            .map(|child| {
                let child_shared = Arc::new(self.status.new_child());
                self.status.register_child(&child_shared);
                (child, child_shared)
            })
            .collect();
        // THEN THREADS: one dedicated scoped thread per child. The
        // parent runs main's join loop: park (children first)
        // whenever a checkpoint is requested, poll the joins between
        // sleeps.
        let shared = self.status.clone();
        let f = &f;
        std::thread::scope(|scope| {
            let mut joins = Vec::with_capacity(prepared.len());
            for (child, child_shared) in prepared {
                let claims = Arc::clone(&claims);
                joins.push(
                    scope.spawn(move || -> Result<R, ContentClaimTimeout> {
                    // THE GLOBAL CLAIM: insert-retry with a short backoff.
                    // The Shared is preallocated and registered BEFORE
                    // this loop — so on each failed attempt the
                    // WAITING subtask must still answer checkpoint
                    // requests: park here when one is pending, or a
                    // save arriving during the backoff would deadlock
                    // the tree (the parent waits for this subtask to
                    // park; a bare sleep never would).
                    let key: usize = child as *const PersistentCell<K> as usize;
                    // the claim is taken through `try_claim`: the registry
                    // lock is gone before the backoff (which parks)
                    while !try_claim(&claims, key) {
                        child_shared.backoff_checkpoint();
                    }
                    // release on unwind or completion — a panicked
                    // subtask must not leave its slot claimed
                    let _claim = ClaimGuard { claims: Arc::clone(&claims), key };
                    // LEVEL 2 — THE CONTENT CLAIM, on the SAME map:
                    // de-duplicate on the `AsPtr` raw of the containee
                    // (two distinct cells holding container values that
                    // deref to the same content — a Copy/index
                    // container). A content duplicate WAITS, but only
                    // briefly: unlike cell duplicates (which resolve
                    // when the owning subtask ends), content waits can
                    // form CROSS-CLAIM CYCLES (A holds content x wants
                    // y; B holds y wants x) and INTRATHREAD SELF-
                    // DEADLOCKS (a nested fanout re-claiming content
                    // its own ancestor already holds) — so the wait is
                    // bounded, and when the window runs out this child
                    // gives up through the error gate: nothing is
                    // minted, and the error returns from the fan-out.
                    // The claim is released when this subtask returns.
                    let content_key = claim_content(&claims, &child_shared, || {
                        // the raw hops re-run per attempt; nothing is
                        // driven until the claim succeeds — no reference
                        // is materialized (see `AsPtr`).
                        // SAFETY: the CELL claim above makes this thread
                        // the sole driver of the slot (the mint's
                        // discipline — `as_ptr`'s caller obligation); the
                        // shared-to-raw cast is the split's own hand-off.
                        let child_ptr: *mut PersistentCell<K> =
                            child as *const PersistentCell<K> as *mut PersistentCell<K>;
                        // SAFETY: as above.
                        let k_ptr: *mut K = unsafe { AsPtr::as_ptr(child_ptr) };
                        // SAFETY: as above.
                        let target_ptr: *mut K::Target = unsafe { AsPtr::as_ptr(k_ptr) };
                        // SAFETY: as above.
                        let state_raw = unsafe { AsPtr::as_ptr(target_ptr) };
                        state_raw as *const () as usize
                    })?;
                    let _content_claim = ClaimGuard { claims: Arc::clone(&claims), key: content_key };
                    // THE MINT (only after BOTH uniqueness levels):
                    // the raw is claimed. The raw hops once more — no
                    // reference is materialized (see `AsPtr`).
                    let child_ptr: *mut PersistentCell<K> =
                        child as *const PersistentCell<K> as *mut PersistentCell<K>;
                    // SAFETY: the claims make this thread the sole
                    // driver of both the slot and its content; the
                    // mint's discipline holds (`as_ptr`'s caller
                    // obligation).
                    let k_ptr: *mut K = unsafe { AsPtr::as_ptr(child_ptr) };
                    // SAFETY: as above.
                    let target_ptr: *mut K::Target = unsafe { AsPtr::as_ptr(k_ptr) };
                    // SAFETY: as above.
                    let state_raw = unsafe { AsPtr::as_ptr(target_ptr) };
                    // SAFETY: as above — the receiver derives from the
                    // claimed raw; transient, dies at return. As in
                    // `over_slot`: the content itself, dispatched through
                    // `HandleAble<V>`.
                    let erased: &mut <K::Target as AsPtr>::Target = unsafe { &mut *state_raw };
                    let state_erased: *mut () = state_raw as *mut ();
                    let shared_ref = SharedRef(child_shared.clone());
                    // SAFETY: as above; the child's Shared carries the
                    // inherited registry and the registration.
                    let current = unsafe { erased.box_innerhandle(state_erased, shared_ref) };
                    let outer = OuterHandle {
                        current,
                        slot: child as *const PersistentCell<K> as *mut PersistentCell<K>,
                        shared: child_shared,
                        _brand: PhantomData,
                    };
                    Ok(f(outer))
                }));
            }
            loop {
                if shared.is_pending() {
                    shared.park_self();
                }
                if joins.iter().all(|j| j.is_finished()) {
                    break;
                }
                std::thread::sleep(Duration::from_millis(JOIN_POLL_INTERVAL_MS));
            }
            // all finished: join in order. A child's error gate is the
            // fan-out's error; a child that panicked is re-raised here.
            joins.into_iter().map(|j| j.join().expect("panicked child")).collect()
        })
    }
}

impl<T: Sized> Deref for InnerHandle<T> {
    type Target = T;
    fn deref(&self) -> &T {
        // a FRESH reference off the stored-provenance raw — nothing
        // persistent is ever a mutable line
        // SAFETY: the raw is live per the mint's discipline; T is
        // the pointed object's own type (the dispatch that minted
        // this handle fixed them together).
        unsafe { &*self.state }
    }
}

impl<T: Sized> DerefMut for InnerHandle<T> {
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: as Deref — exclusive, tied to `&mut self`.
        unsafe { &mut *self.state }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both registry tests register exactly two children.
    const REGISTERED_CHILDREN: usize = 2;
    /// A key whose registry shard the lock check looks at.
    const PROBE_KEY: usize = 1;
    /// How many keys the lock check sweeps (exclusive end) — enough that
    /// some hash to `PROBE_KEY`'s shard at any shard count.
    const PROBE_SWEEP_END: usize = 4096;
    /// Keys swept for a leaked registry shard lock.
    const PROBE_SWEEP: std::ops::Range<usize> = 0..PROBE_SWEEP_END;

    /// A FAILED claim must not leave its registry shard locked: the
    /// `Entry` — and with it the shard's write lock — is dropped before
    /// `try_claim` returns. Probed with `try_entry` from the same
    /// thread: with the shard still held, every key hashing to it would
    /// be unlockable.
    #[test]
    fn a_failed_claim_leaves_no_registry_shard_locked() {
        let claims = DashMap::new();
        assert!(try_claim(&claims, PROBE_KEY), "a free key is claimed");
        assert!(!try_claim(&claims, PROBE_KEY), "a taken key is not claimed twice");
        let lockable = PROBE_SWEEP.filter(|key| claims.try_entry(*key).is_some()).count();
        assert_eq!(lockable, PROBE_SWEEP.count(), "some key's shard is still locked");
    }

    /// A state the vocabulary below can read through its handle.
    struct Probe {
        n: u32,
    }

    /// The value a state starts with.
    const ORIGINAL: u32 = 1;
    /// The value a replacement starts with.
    const REPLACEMENT: u32 = 2;

    /// The probe's vocabulary.
    trait ProbeVocab: Send {
        fn read(&mut self) -> u32;
    }

    impl ProbeVocab for InnerHandle<Probe> {
        fn read(&mut self) -> u32 {
            self.n
        }
    }

    type ProbeSlot = Box<PersistentCell<Probe>>;

    /// The key the content claim on `slot`'s content is taken on —
    /// the two raw hops a claim goes through.
    fn content_address_of(slot: &mut ProbeSlot) -> usize {
        // SAFETY: `slot` is owned here, so the projections are the only
        // live access (`as_ptr`'s caller obligation).
        let target_ptr: *mut PersistentCell<Probe> =
            unsafe { AsPtr::as_ptr(std::ptr::addr_of_mut!(*slot)) };
        // SAFETY: as above.
        let state_raw = unsafe { AsPtr::as_ptr(target_ptr) };
        state_raw as *const () as usize
    }

    /// The error gate on `replace`: with the replacement's content
    /// claimed by another chain, `replace` gives up after
    /// `CONTENT_CLAIM_WINDOW` — and the slot and the handle are exactly
    /// as they were: the old state is still the one installed and
    /// driven. Once the claim is free, the replacement is installed.
    #[test]
    fn replace_gives_up_through_the_error_gate_and_changes_nothing() {
        let mut root = PersistentCell::new(Box::new(PersistentCell::new(Probe { n: ORIGINAL })));
        let (during, outcome, waited, after) = independent_task_root(
            &mut root,
            |mut outer: OuterHandle<'_, ProbeSlot, dyn ProbeVocab>, _checkpointer| {
                let mut replacement = Box::new(PersistentCell::new(Probe { n: REPLACEMENT }));
                // the other chain: whoever holds the claim on the
                // replacement's content — so TAKE the claim, and hold it
                let claims = Arc::clone(&outer.shared.claims);
                let key = content_address_of(&mut replacement);
                assert!(try_claim(&claims, key), "the stand-in takes the claim");
                let held = ClaimGuard { claims: Arc::clone(&claims), key };
                let started = std::time::Instant::now();
                let outcome = outer.replace(replacement);
                let waited = started.elapsed();
                let during = outer.read();
                drop(held);
                outer
                    .replace(Box::new(PersistentCell::new(Probe { n: REPLACEMENT })))
                    .expect("the content claim is free now");
                (during, outcome, waited, outer.read())
            },
        );
        assert_eq!(during, ORIGINAL, "a lost claim leaves the old state installed and driven");
        assert_eq!(outcome, Err(ContentClaimTimeout));
        assert!(waited >= CONTENT_CLAIM_WINDOW, "it waited the window first");
        assert_eq!(after, REPLACEMENT, "and the replacement installs once the claim is free");
    }

    /// A fan-out phase of probes, for the vocabulary below.
    struct Phase {
        children: Vec<PersistentCell<ProbeSlot>>,
    }

    /// How often a mutator waiting for a snapshot reaches its safepoint.
    const SNAPSHOT_POLL: std::time::Duration = std::time::Duration::from_millis(1);

    /// The phase's vocabulary: fan out over the children; park for a
    /// checkpoint.
    trait PhaseVocab: Send {
        fn fan(&mut self, ran: &std::sync::atomic::AtomicBool) -> Result<(), ContentClaimTimeout>;
        fn park(&mut self);
    }

    impl PhaseVocab for InnerHandle<Phase> {
        fn fan(&mut self, ran: &std::sync::atomic::AtomicBool) -> Result<(), ContentClaimTimeout> {
            let _joined: Vec<()> = self.fanout(
                &|phase: &Phase| phase.children.iter().collect(),
                |mut child: OuterHandle<'_, ProbeSlot, dyn ProbeVocab>| {
                    let saw_a_state = child.read() == ORIGINAL;
                    ran.store(saw_a_state, std::sync::atomic::Ordering::SeqCst);
                },
            )?;
            Ok(())
        }

        fn park(&mut self) {
            self.safepoint();
        }
    }

    /// The leaf's value, read through a checkpoint guard: every layer by
    /// its own `Deref`.
    fn leaf_through_guard(guard: &CheckpointGuard<'_, Box<PersistentCell<Phase>>>) -> u32 {
        // guard -> the root cell -> the box's cell -> the phase
        let inner: &PersistentCell<Phase> = guard;
        let phase: &Phase = inner;
        // the child's cell -> the box's cell -> the probe
        let child: &PersistentCell<ProbeSlot> = phase.children.first().unwrap();
        let child_inner: &PersistentCell<Probe> = child;
        child_inner.n
    }

    /// The error gate on `fanout`: a child whose content is claimed by
    /// another chain is never minted — its callback does not run — and
    /// the error returns from the fan-out. The point of the gate: the
    /// tree is whole when the caller gets the error — a checkpoint can
    /// still stop the world and read the leaf through the guard. Once the
    /// claim is free, the fan-out runs.
    #[test]
    fn fanout_gives_up_through_the_error_gate_and_the_tree_can_still_be_snapshotted() {
        let mut child: ProbeSlot = Box::new(PersistentCell::new(Probe { n: ORIGINAL }));
        let content_key = content_address_of(&mut child);
        let mut root =
            PersistentCell::new(Box::new(PersistentCell::new(Phase { children: vec![PersistentCell::new(child)] })));
        let ran = std::sync::atomic::AtomicBool::new(false);
        let observed = &ran;
        let (outcome, ran_while_claimed, snapshot, second) = independent_task_root(
            &mut root,
            |mut outer: OuterHandle<'_, Box<PersistentCell<Phase>>, dyn PhaseVocab>, checkpointer| {
                // the other chain: whoever holds the claim on the child's
                // content — so TAKE the claim, and hold it
                let claims = Arc::clone(&outer.shared.claims);
                assert!(try_claim(&claims, content_key), "the stand-in takes the claim");
                let held = ClaimGuard { claims: Arc::clone(&claims), key: content_key };
                let outcome = outer.fan(observed);
                let ran_while_claimed = observed.load(std::sync::atomic::Ordering::SeqCst);
                drop(held);
                // the snapshot: from a checker thread, as a snapshotter
                // would, while this thread keeps reaching its safepoint
                let seen: Arc<Mutex<Option<u32>>> = Arc::new(Mutex::new(None));
                std::thread::scope(|scope| {
                    let checkpointer = checkpointer.clone();
                    let reported = Arc::clone(&seen);
                    let checker = scope.spawn(move || {
                        let guard = checkpointer.request();
                        let leaf = leaf_through_guard(&guard);
                        drop(guard);
                        *reported.lock().unwrap_or_else(|e| e.into_inner()) = Some(leaf);
                    });
                    while seen.lock().unwrap_or_else(|e| e.into_inner()).is_none() {
                        outer.park();
                        std::thread::sleep(SNAPSHOT_POLL);
                    }
                    checker.join().expect("checker panicked");
                });
                let snapshot = *seen.lock().unwrap_or_else(|e| e.into_inner());
                let second = outer.fan(observed);
                (outcome, ran_while_claimed, snapshot, second)
            },
        );
        assert_eq!(outcome, Err(ContentClaimTimeout));
        assert!(!ran_while_claimed, "the child's callback never ran");
        assert_eq!(snapshot, Some(ORIGINAL), "and the checkpoint read the untouched leaf");
        assert_eq!(second, Ok(()));
        assert!(ran.load(std::sync::atomic::Ordering::SeqCst), "and it runs once the claim is free");
    }

    /// Two registered children, both listed in the Weak snapshot.
    #[test]
    fn registry_lists_children() {
        let parent = Arc::new(Shared::new());
        let a = Arc::new(Shared::new());
        let b = Arc::new(Shared::new());
        parent.register_child(&a);
        parent.register_child(&b);
        assert_eq!(parent.child_weaks().len(), REGISTERED_CHILDREN);
    }

    /// A WAITING subtask must still answer checkpoint requests: its
    /// backoff step parks on a pending request (or a save arriving
    /// during the wait would deadlock the tree).
    #[test]
    fn backoff_parks_on_pending() {
        let shared = Arc::new(Shared::new());
        let saw_parked = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let s2 = shared.clone();
        let flag = saw_parked.clone();
        let weak = Arc::downgrade(&shared);
        let requester = std::thread::spawn(move || {
            // blocks until the waiter parks
            let _ = request_until_parked(&weak);
            flag.store(true, std::sync::atomic::Ordering::SeqCst);
            release_request(&s2); // resume the world (and the waiter)
        });
        // the waiting subtask's role: spin until the request is
        // visible, then the backoff step — which must PARK (the
        // requester's handshake only completes when it does)
        while !shared.is_pending() {
            std::thread::sleep(PARK_SLEEP);
        }
        shared.backoff_checkpoint();
        requester.join().expect("requester");
        assert!(saw_parked.load(std::sync::atomic::Ordering::SeqCst));
        assert!(!shared.is_pending(), "the world resumed");
    }

    /// The handshake dance, thread-to-Weak: a mutator that parks at
    /// its safepoint releases the requester's `request_until_parked`,
    /// and the requester's release lets the mutator run again.
    #[test]
    fn handshake_parks_and_resumes() {
        let shared = Arc::new(Shared::new());
        let weak = Arc::downgrade(&shared);
        let mutator = shared.clone();
        let parked = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let saw_park = parked.clone();
        let side = std::thread::spawn(move || {
            // the mutator reaches its safepoint: the request must be
            // visible first (park_self would return immediately
            // otherwise and the handshake would miss)
            while !mutator.is_pending() {
                std::thread::sleep(PARK_SLEEP);
            }
            mutator.park_self();
            saw_park.store(true, std::sync::atomic::Ordering::SeqCst);
        });
        // the requester: waits until the mutator parks (through the
        // weak, like every requester-side look)
        match request_until_parked(&weak) {
            Handshake::Parked => {}
            Handshake::Closed => unreachable!("live mutator answered Closed"),
        }
        // release: the mutator runs again and finishes
        release_request(&shared);
        side.join().expect("mutator panicked");
        assert!(parked.load(std::sync::atomic::Ordering::SeqCst));
    }

    /// A gone mutator (its Arc dropped) answers a request with Closed
    /// instead of blocking — the real death-detection path.
    #[test]
    fn gone_mutator_never_blocks() {
        let weak = {
            let shared = Arc::new(Shared::new());
            Arc::downgrade(&shared)
            // the last strong Arc dies here: the mutator side is gone
        };
        match request_until_parked(&weak) {
            Handshake::Closed => {}
            Handshake::Parked => unreachable!("a gone mutator cannot park"),
        }
    }

    /// A dropped child is gone: its Weak no longer upgrades — still
    /// listed in the registry, but the parking request skips it (the
    /// lookup itself is the skip).
    #[test]
    fn dropped_child_is_gone() {
        let parent = Arc::new(Shared::new());
        let transient = Arc::new(Shared::new());
        let keeper = Arc::new(Shared::new());
        parent.register_child(&transient);
        parent.register_child(&keeper);
        let transient_weak = Arc::downgrade(&transient);
        drop(transient);
        assert!(transient_weak.upgrade().is_none());
        // the registry entry survives (a Weak can), the child doesn't
        assert_eq!(parent.child_weaks().len(), REGISTERED_CHILDREN);
        // and the request through it answers Closed, not a hang
        match request_until_parked(&transient_weak) {
            Handshake::Closed => {}
            Handshake::Parked => unreachable!("a dropped child cannot park"),
        }
    }

    /// A requester that panics mid-protocol (gate held, pending stuck
    /// true, no guard) poisons the gate — the recovery on the next
    /// lock releases the ownerless pending so parked mutators resume.
    #[test]
    fn poisoned_gate_recovers_stale_pending() {
        let shared = Arc::new(Shared::new());
        let weak = Arc::downgrade(&shared);
        let gate = Arc::new(Mutex::new(()));
        let gate2 = gate.clone();
        // the mid-protocol death: pending set, then the requester's own
        // next step FAILS — it re-enters the gate it is holding — and the
        // failed expectation unwinds through the held guard (poisoning
        // it), never releasing
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _gate_guard = gate2.lock().unwrap();
            shared.pending_request.store(true, std::sync::atomic::Ordering::SeqCst);
            let _never_reached = gate2.try_lock().expect("the requester dies here: its own gate is held");
        }));
        assert!(result.is_err());
        assert!(shared.is_pending());
        // the next requester: the poisoned closure recovers the world
        recover_poisoned_gate(&weak);
        assert!(!shared.is_pending());
    }
}
