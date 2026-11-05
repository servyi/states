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
struct Shared {
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

/// The OUTER handle: purely a type marker over the current install
/// handle (`Box<V>`, which it knows is an `InnerHandle<T, V>`
/// pointing at a `K`-shaped slot) — it owns NO slot. `replace` is
/// the successor install: walk to the slot through the current
/// inner handle's own pointer, swap the slot via
/// the K-cell assignment, and keep the fresh handle in the box.
///
/// The `'o` brand is unforgeable: it only arises inside
/// [`independent_task_root`]'s higher-ranked closure, so the handle
/// (and everything minted under it, via `fanout`'s own `'o`)
/// structurally cannot leave that call.
pub struct OuterHandle<'o, K: ?Sized, V: ?Sized> {
    current: Box<V>,
    _slot: PhantomData<fn(*mut K, *mut V)>,
    /// Invariant in `'o`: the brand cannot shrink to a foreign
    /// lifetime — the handle stays inside the root task.
    _brand: PhantomData<fn(&'o ()) -> &'o ()>,
}

pub struct InnerHandle<T: ?Sized, V: ?Sized> {
    /// The K-shaped holder slot, as a fat pointer whose POINTee type
    /// carries `+ Send` — the holder's sendability is a static part
    /// of the pointer's own type (coercing a `*mut K` into it
    /// requires `K: Send`), so the Send impl below needs no proof
    /// about a K it cannot name.
    ptr: *mut (dyn DerefMut<Target = dyn HandleAble<V>> + Send + 'static),
    status: Arc<Shared>,
    _marker: PhantomData<fn(*mut T, *mut V)>,
}

// Sound now: the pointee's type is `dyn DerefMut<..> + Send` (Send by
// the auto trait IN the type), the status Arc is Send+Sync, and
// T: Send crosses with it. The K itself is unreachable as a type
// here — its sendability rode along in the pointer's type at mint.
unsafe impl<T: ?Sized + Send, V: ?Sized> Send for InnerHandle<T, V> {}

// The box is the payload (an InnerHandle: its ptr's pointee is
// `+ Send` by type, its T: Send by the HandleAble blanket); K: Send
// mirrors the pointer's own mint requirement — the fn-pointer marker
// would not need it, the pointee does.
unsafe impl<'o, K: ?Sized + Send, V: ?Sized + Send> Send for OuterHandle<'o, K, V> {}

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
    slot: *mut K,
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
pub fn independent_task_root<K, V: ?Sized + 'static, R>(
    slot: &mut K,
    f: impl for<'o> FnOnce(OuterHandle<'o, K, V>, Checkpointer<'o, K>) -> R,
) -> R
where
    K: DerefMut<Target = dyn HandleAble<V>> + Sized + Send + 'static,
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
        _brand: PhantomData,
    };
    f(outer, checkpointer)
}

impl<'o, K: ?Sized, V: ?Sized> OuterHandle<'o, K, V> {
    fn shared(&self) -> &Arc<Shared> {
        // SAFETY: the box is an InnerHandle<_, V> (the invariant) —
        // reading its status field through the erased view.
        unsafe {
            &(*(self.current.as_ref() as *const V as *const InnerHandle<(), V>)).status
        }
    }
}

impl<'o, K: ?Sized, V: ?Sized> std::ops::Deref for OuterHandle<'o, K, V> {
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

impl<'o, K: ?Sized, V: ?Sized> std::ops::DerefMut for OuterHandle<'o, K, V> {
    fn deref_mut(&mut self) -> &mut V {
        // SAFETY: as Deref — exclusive.
        unsafe { &mut *(self.current.as_mut() as *mut V) }
    }
}

impl<'o, K, V: ?Sized + 'static> OuterHandle<'o, K, V>
where
    K: DerefMut<Target = dyn HandleAble<V>> + Send + 'static,
{
    /// Mint over a live slot the driver owns (crate-internal: the
    /// root task is the only minter through here).
    ///
    /// # Safety
    ///
    /// The `&mut` resolves validity and mint-time exclusivity by
    /// construction; what remains:
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
    /// box is a valid `InnerHandle<T, V>` produced by
    /// `box_innerhandle` on the pointee of the `K`, and its `ptr`
    /// derives from exactly the holder `&mut` passed on.
    unsafe fn over_slot(slot: *mut K) -> Self
    where
        K: Sized,
    {
        // the receiver comes off the slot raw; the holder `&mut` is
        // derived at the call — its TYPE carries the validity and
        // mint-time exclusivity guarantees, the remaining contract
        // stays in the safety docs
        // SAFETY: the caller's contract above.
        let erased: &mut dyn HandleAble<V> = unsafe { (*slot).deref_mut() };
        // the chain's ORIGIN: a fresh root Shared moves in
        let shared = Arc::new(Shared::new());
        // SAFETY: the holder is the K-shaped slot the receiver came
        // from (the contract above); the pointee of the K is the
        // receiver (`self`); exclusive for this chain.
        let current =
            unsafe { erased.box_innerhandle(&mut *slot, SharedRef(shared)) };
        OuterHandle { current, _slot: PhantomData, _brand: PhantomData }
    }

    /// The successor install, per the sketch: walk to the slot through
    /// the CURRENT inner handle's own pointer, swap the slot (plain
    /// assignment — the slot IS a `K` cell), then re-mint through the
    /// replacement's `HandleAble` and keep the fresh inner handle in
    /// the box.
    pub fn replace(&mut self, val: K) {
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
        // 3) swap the slot — plain assignment: the slot IS a K cell
        // SAFETY: as above — the write lands in the K-shaped slot the
        // current chain exclusively owns.
        unsafe { *slot = val };
        // 4) re-mint: use the HandleAble OF val — reachable through the
        // slot it now lives in (K's own DerefMut) — to create a new
        // InnerHandle pointing at that same slot, and assign it into
        // the own box
        // SAFETY: the mint's discipline over the fresh slot content;
        // the receiver comes off the slot raw, the holder `&mut` is
        // derived at the call.
        let fresh: Box<V> = unsafe {
            let erased: &mut dyn HandleAble<V> = (*slot).deref_mut();
            erased.box_innerhandle(&mut *slot, SharedRef(shared))
        };
        self.current = fresh;
    }
}

/// The erasure vocabulary: a state type `T` whose
/// `InnerHandle<T, V>` can be erased to `V`.
///
/// # Safety
///
/// For implementors — implementations are ONLY allowed inside this
/// crate; a foreign implementation would have to uphold all of the
/// following, which the blanket implementation here carries:
///
/// - `T` may implement this trait only if `InnerHandle<T, V>` can
///   be unsized to `V` (`InnerHandle<T, V>: Unsize<V>`).
/// - The `Box` returned by [`HandleAble::box_innerhandle`] must be
///   the `InnerHandle<T, V>` minted over the passed K container
///   (its `ptr` is exactly the passed holder pointer).
pub unsafe trait HandleAble<V: ?Sized>: Send {
    /// Mint the handle over `holder`.
    ///
    /// # Safety
    ///
    /// For callers:
    ///
    /// - The `&mut` resolves validity and mint-time exclusivity by
    ///   construction; what remains: the K container must stay valid
    ///   and live at that address for the returned handle's whole
    ///   life (the raw stored in it derives from this `&mut`).
    /// - The K container's pointee must be `self`.
    /// - The `K` must not be reachable by shared reference from other
    ///   threads while the handle lives: its `Deref` would alias the
    ///   pointee this handle owns exclusively.
    ///
    /// The `SharedRef` argument is the seal as well as the payload:
    /// it is an opaque shell (`Shared` private, the field private),
    /// so no client can construct one — the method is uncallable
    /// from outside the crate.
    unsafe fn box_innerhandle(
        &mut self,
        holder: &mut (dyn DerefMut<Target = dyn HandleAble<V>> + Send + 'static),
        shared: SharedRef,
    ) -> Box<V>;
}

unsafe impl<T: Send, V: ?Sized> HandleAble<V> for T
where
    InnerHandle<T, V>: Unsize<V>,
{
    unsafe fn box_innerhandle(
        &mut self,
        holder: &mut (dyn DerefMut<Target = dyn HandleAble<V>> + Send + 'static),
        shared: SharedRef,
    ) -> Box<V> {
        // the raw that outlives this call — derived from the `&mut`,
        // whose type proved validity and exclusivity at mint time
        let ptr: *mut (dyn DerefMut<Target = dyn HandleAble<V>> + Send + 'static) =
            std::ptr::from_mut(holder);
        // bind at the sized type, then let the return coerce
        // Box<InnerHandle<T, V>> -> Box<V> (the Unsize where-clause).
        let handle: Box<InnerHandle<T, V>> = Box::new(InnerHandle {
            ptr,
            status: shared.0,
            _marker: PhantomData,
        });
        handle
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
    pub fn fanout<K, F, R>(
        &mut self,
        split: &(dyn for<'x> Fn(&'x mut T) -> Vec<&'x mut K> + Send + Sync),
        f: F,
    ) -> Vec<R>
    where
        T: Send,
        V: Send + 'static,
        K: DerefMut<Target = dyn HandleAble<V>> + Sized + Send + 'static,
        F: for<'n> Fn(OuterHandle<'n, K, V>) -> R + Send + Sync,
        R: Send,
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
                // the receiver comes off the split's raw; the holder
                // `&mut` is derived at the call — its type carries the
                // validity and mint-time exclusivity guarantees
                // SAFETY: the split yielded each child exactly
                // once, disjointly; exclusive for this mint.
                let erased: &mut dyn HandleAble<V> = unsafe { (*raw).deref_mut() };
                // each child gets its OWN Shared (its parking
                // identity), weakly registered into the parent's vec
                // — insert under the lock, release (the convention)
                let child_shared = Arc::new(Shared::new());
                self.status.register_child(&child_shared);
                // SAFETY: the holder is the child's K-shaped slot the
                // receiver derived from; the pointee of that K is the
                // receiver; yielded exactly once, disjointly, exclusive.
                unsafe { erased.box_innerhandle(&mut *raw, SharedRef(child_shared)) }
            })
            .collect();
        // wrap each box in the OUTER handle (marker + current) so the
        // callback can drive AND replace its subtree; then threads.
        // The brand lifetime is free here — `f`'s higher-ranked bound
        // instantiates it at the call, and its universality is what
        // keeps the handle inside the callback.
        let outers: Vec<OuterHandle<'_, K, V>> = boxes
            .into_iter()
            .map(|current| OuterHandle { current, _slot: PhantomData, _brand: PhantomData })
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
            // all finished: join in order, propagate a panicked child
            joins
                .into_iter()
                .map(|j| j.join().expect("panicked child"))
                .collect()
        })
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

    /// A requester that panics mid-protocol (gate held, pending stuck
    /// true, no guard) poisons the gate — the recovery on the next
    /// lock releases the ownerless pending so parked mutators resume.
    #[test]
    fn poisoned_gate_recovers_stale_pending() {
        let shared = Arc::new(Shared::new());
        let weak = Arc::downgrade(&shared);
        let gate = Arc::new(Mutex::new(()));
        let gate2 = gate.clone();
        // the mid-protocol death: pending set, then the panic unwinds
        // through the held gate (poisoning it), never releasing
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _gate_guard = gate2.lock().unwrap();
            shared.pending_request.store(true, std::sync::atomic::Ordering::SeqCst);
            panic!("requester died mid-protocol");
        }));
        assert!(result.is_err());
        assert!(shared.is_pending());
        // the next requester: the poisoned closure recovers the world
        recover_poisoned_gate(&weak);
        assert!(!shared.is_pending());
    }
}
