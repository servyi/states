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
use std::sync::{Arc, Condvar, Mutex, MutexGuard, Weak};
use std::time::Duration;

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
pub struct Shared {
    children: Mutex<Vec<Weak<Shared>>>,
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

impl Shared {
    fn new() -> Self {
        Shared {
            children: Mutex::new(Vec::new()),
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
pub struct Checkpointer<K, V: ?Sized> {
    slot: *mut K,
    weak: Weak<Shared>,
    gate: Arc<Mutex<()>>,
    _marker: PhantomData<fn(*mut K, *mut V)>,
}

// SAFETY: the slot pointer is only dereferenced through a guard,
// which exists exclusively while every mutator is parked (see
// `request`); the data crosses threads, so K must be Send.
unsafe impl<K: Send, V: ?Sized> Send for Checkpointer<K, V> {}
// SAFETY: as Send — sharing the checkpointer shares no slot data.
unsafe impl<K: Send, V: ?Sized> Sync for Checkpointer<K, V> {}

impl<K, V: ?Sized> Clone for Checkpointer<K, V> {
    fn clone(&self) -> Self {
        Self {
            slot: self.slot,
            weak: self.weak.clone(),
            gate: self.gate.clone(),
            _marker: PhantomData,
        }
    }
}

impl<K: Send, V: ?Sized> Checkpointer<K, V> {
    /// Perform the stop-the-world handshake: request all mutators to
    /// park (the top handle parks its subtree first) and return a
    /// guard with direct access to the root slot. Blocks until
    /// parked. A gone tree (mutator side dropped, the Weak empty)
    /// hands out the guard at once — nothing will ever mutate again.
    /// Concurrent `request`s serialize on the gate: a second request
    /// blocks until the first guard is dropped and the world resumed.
    pub fn request(&self) -> CheckpointGuard<'_, K> {
        // held for the whole guard life — the serialization
        let request_lock = self.gate.lock().unwrap_or_else(|e| e.into_inner());
        // a stale pending from a requester that died mid-handshake:
        // reset the world before asking again (the mutator, parked
        // under that dead request, re-checks the flag and resumes)
        if let Some(shared) = self.weak.upgrade() {
            if shared.is_pending() {
                release_request(&shared);
            }
            drop(shared);
        }
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

/// Direct access to the root slot while every mutator is parked,
/// through `Deref`/`DerefMut` — the world is stopped for the guard's
/// whole life, so even mutable access is exclusive by construction.
/// Dropping the guard resumes the world.
pub struct CheckpointGuard<'a, K> {
    slot: *mut K,
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
    type Target = K;
    fn deref(&self) -> &K {
        // SAFETY: the guard exists only while the handshake proved
        // every mutator parked (or gone); the reborrow is tied to the
        // guard, so the world cannot resume underneath it.
        unsafe { &*self.slot }
    }
}

impl<K> std::ops::DerefMut for CheckpointGuard<'_, K> {
    fn deref_mut(&mut self) -> &mut K {
        // SAFETY: as Deref, but exclusive — the world is stopped for
        // the guard's whole life, so no mutator can coexist with this
        // borrow.
        unsafe { &mut *self.slot }
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
/// `slot` inaccessible (borrowed) for the whole call — the
/// discipline is structural, not by-convention.
pub fn independent_task_root<K, V: ?Sized + 'static, R>(
    slot: &mut K,
    f: impl FnOnce(OuterHandle<K, V>, Checkpointer<K, V>) -> R,
) -> R
where
    K: DerefMut<Target = dyn HandleAble<V>> + ReplaceAble<V> + Sized + 'static,
{
    // the raw is derived ONCE and shared by the outer's chain and the
    // checkpointer — a second `from_mut` on the same `&mut` would
    // invalidate the first raw's tag (the function-entry retag).
    // SAFETY: slot is exclusively borrowed for the whole call (the
    // fn signature itself); the outer and the checkpointer die
    // before it returns.
    let slot_ptr: *mut K = std::ptr::from_mut(slot);
    let outer = unsafe { OuterHandle::over_slot(slot_ptr) };
    // the Checkpointer knows the root's Shared only WEAKLY — the
    // outer's own Arc is the last strong one, so its drop IS the
    // close; the gate serializes the requester side (see Checkpointer)
    let checkpointer = Checkpointer {
        slot: slot_ptr,
        weak: Arc::downgrade(outer.shared()),
        gate: Arc::new(Mutex::new(())),
        _marker: PhantomData,
    };
    f(outer, checkpointer)
}

impl<K: ?Sized, V: ?Sized> OuterHandle<K, V> {
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
        V: Send + 'static,
        K: DerefMut<Target = dyn HandleAble<V>> + ReplaceAble<V> + Sized + Send + 'static,
        F: Fn(OuterHandle<K, V>) + Send + Sync,
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
        // wrap each box in the OUTER handle (marker + current) so the
        // callback can drive AND replace its subtree; then threads
        let outers: Vec<OuterHandle<K, V>> = boxes
            .into_iter()
            .map(|current| OuterHandle { current, _slot: PhantomData })
            .collect();
        // THEN THREADS: one dedicated scoped thread per outer (std, per
        // the 23:55 step — rayon left only noise in Miri). The parent
        // runs main's join loop: park (children first) whenever a
        // checkpoint is requested, poll the joins between sleeps.
        let shared = self.status.clone();
        std::thread::scope(|scope| {
            let mut joins = Vec::with_capacity(outers.len());
            for o in outers {
                joins.push(scope.spawn(|| f(o)));
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

    /// Two registered children, both listed in the Weak snapshot.
    #[test]
    fn registry_lists_children() {
        let parent = Arc::new(Shared::new());
        let a = Arc::new(Shared::new());
        let b = Arc::new(Shared::new());
        parent.register_child(&a);
        parent.register_child(&b);
        assert_eq!(parent.child_weaks().len(), 2);
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
                std::thread::sleep(Duration::from_millis(1));
            }
            mutator.park_self();
            saw_park.store(true, std::sync::atomic::Ordering::SeqCst);
        });
        // the requester: waits until the mutator parks (through the
        // weak, like every requester-side look)
        match request_until_parked(&weak) {
            Handshake::Parked => {}
            Handshake::Closed => panic!("live mutator answered Closed"),
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
            Handshake::Parked => panic!("a gone mutator cannot park"),
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
        assert_eq!(parent.child_weaks().len(), 2);
        // and the request through it answers Closed, not a hang
        match request_until_parked(&transient_weak) {
            Handshake::Closed => {}
            Handshake::Parked => panic!("a dropped child cannot park"),
        }
    }
}
