//! The two-piece handle library: erased state trees under
//! stop-the-world checkpointing, with in-place successor installs.
//!
//! # The safety architecture
//!
//! All `unsafe` below is justified against five invariants. I1-I3
//! are established at the mints and carried by ownership; I4 is
//! maintained entirely by safe code (the handshake protocol); I5 is
//! a type-level consequence of the mint's dispatch.
//!
//! - **I1 (one chain per slot):** every slot (`K` cell) and every
//!   state inside it is reachable through exactly ONE handle chain.
//!   Established: `independent_task_root` takes `&mut K` (borrowck
//!   forbids a second path), `fanout`'s split yields each child
//!   exactly once and disjointly (the `&mut` yields are disjoint by
//!   construction, and the HRTB closure cannot stash them), and
//!   `replace` writes the cell while holding `&mut self` (the only
//!   chain). Maintained by safe code: handles are `!Clone`, states'
//!   methods can only reach their own subtree, and the `'o`/`'n`
//!   brands keep handles inside their task/callback.
//! - **I2 (slot stability):** a slot cell stays live at a fixed
//!   address for its whole task: the root task borrows `&mut K` for
//!   the call; children cells live inside their parent's state,
//!   which the parent exclusively owns.
//! - **I3 (the stored line):** minted state raws carry the pointer
//!   line the container's `Deref`s spring from (the `AsPtr`
//!   contract), never a freshly minted reference — so the checker's
//!   reads (which spring from the same line) are ancestors in the
//!   borrow tree, not demoting siblings. This is what keeps
//!   mutate-after-checkpoint defined under Tree Borrows.
//! - **I4 (the world stops):** `Checkpointer::request` returns a
//!   guard only after every registered mutator parked; a parked
//!   mutator provably holds no deref borrow (safepoint takes
//!   `&mut self`, so borrowck forbids a live borrow across it).
//! - **I5 (T-matching):** a handle's `T` equals the type of the
//!   object its raw points at: the raw is minted through a dyn
//!   dispatch ON that object, and `box_innerhandle`'s receiver IS
//!   that object (so `from_mut`-of-receiver cannot disagree).

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
pub struct OuterHandle<'o, K: ?Sized, V: ?Sized> {
    current: Box<V>,
    /// the K cell this chain owns — `replace`'s write target, typed
    slot: *mut K,
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
pub struct InnerHandle<T: ?Sized, V: ?Sized> {
    /// the state raw — carrying the box's STORED provenance (minted
    /// by a raw borrow of the K's deref place, not from any materialized
    /// `&mut`), so the persistent line's path to the allocation root
    /// crosses no plain mutable node. Every deref mints a FRESH
    /// reference off it; nothing persistent is ever a mutable line.
    state: *mut T,
    status: Arc<Shared>,
    _marker: PhantomData<fn(*mut T, *mut V)>,
}

// SAFETY (preconditions of the construct): a raw `*mut T` may be
// moved to another thread only if (a) no access to the pointee
// remains on the sending thread, and (b) the pointee's data may
// cross threads (`T: Send`).
// Why they hold: (a) is I1 — the handle is the one chain to this
// state, it is `!Clone`, and moving it moves the exclusivity with
// it; the sending thread keeps no other path (the mint consumed the
// only `&mut` chain, and safe code cannot copy a raw out of a
// private field). (b) is supplied by the `T: Send` bound. The other
// fields (`Arc<Shared>`, fn-marker) are `Send` by construction.
unsafe impl<T: ?Sized + Send, V: ?Sized> Send for InnerHandle<T, V> {}

// SAFETY (preconditions of the construct): the payload — `Box<V>`
// (an `InnerHandle`: its `T: Send` via the `HandleAble` blanket),
// the slot raw `*mut K` (crossing moves the cell's exclusivity, so
// the cell's pointee must be `Send`: `K: Send`), and `Arc<Shared>`
// — may cross threads only with no residual access on the sender
// and all crossed data `Send`.
// Why they hold: I1 (the outer is the one chain over both the cell
// and the inner handle; `!Clone`; safe code cannot extract the
// fields), and every field is `Send` under the stated bounds.
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

// SAFETY (preconditions of the construct): `Send` — the raw slot
// pointer crosses threads; sound only if it is never dereferenced
// outside a guard and the pointee data may cross (`K: Send`).
// Why they hold: the field is private; the ONLY dereference site is
// `CheckpointGuard::deref`, and a guard exists only after the
// handshake parked every mutator (I4, maintained by the safe
// protocol code); `K: Send` is the bound. `Sync` — sharing `&K` of
// the CHECKPOINTER exposes only the gate (`Arc<Mutex>`: `Sync`) and
// the `Weak` (`Sync`); the slot raw is only read under the gate, so
// concurrent `request`s serialize and still never coexist with a
// running mutator.
unsafe impl<'o, K: Send> Send for Checkpointer<'o, K> {}
// SAFETY: as the `Send` justification above — no slot data is
// shared by sharing the checkpointer; the gate serializes guards.
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
        // SAFETY (preconditions of the construct): `&*self.slot`
        // requires the slot to be live and no conflicting access to
        // its content while the shared read is outstanding.
        // Why they hold: liveness is I2 (the root task's `&mut K`
        // outlives every guard — the checkpointer cannot leave the
        // task, so neither can the guard); no conflicting access is
        // I4: the handshake returned `Parked` (or `Closed`), every
        // mutator is parked holding no deref borrow, and the world
        // cannot resume until this guard drops (the guard holds the
        // gate, and only `Drop` releases the request) — all of it
        // maintained by the safe protocol code.
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
/// The root task: mint the `OuterHandle` over `slot` and hand the
/// callback BOTH the outer handle and its `Checkpointer`, keeping
/// `slot` borrowed for the whole call. Safe to call for any `K`
/// meeting the bounds: all of the callee's obligations are
/// discharged by this function's own signature and the bounds —
/// there are no caller-side preconditions beyond the type system.
pub fn independent_task_root<K, V: ?Sized + 'static, R>(
    slot: &mut K,
    f: impl for<'o> FnOnce(OuterHandle<'o, K, V>, Checkpointer<'o, K>) -> R,
) -> R
where
    K: DerefMut + AsPtr + Sized + Send + 'static,
    K::Target: Unsize<dyn HandleAble<V>> + 'static,
{
    // the raw is derived ONCE and shared by the outer's chain and the
    // checkpointer — a second `from_mut` on the same `&mut` would
    // invalidate the first raw's tag (the function-entry retag).
    let slot_ptr: *mut K = std::ptr::from_mut(slot);
    // SAFETY (preconditions of the construct, from `over_slot`'s
    // contract): the slot raw must point to a live, exclusively-
    // reachable `K` for the handle's whole life.
    // Why they hold: liveness and exclusivity are the present
    // function's own `&mut K` — borrowck guarantees no other path to
    // the cell exists for the duration of the call (I1/I2), and the
    // handles cannot outlive the call (the `'o` brand arises only
    // inside `f`'s HRTB, so neither handle can escape `f`'s type).
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
        &self.shared
    }
}

// The OUTER-VIEW INVARIANT, used by both derefs below: the private
// field `current` ALWAYS holds a `Box<InnerHandle<T, V>>` (erased
// to `Box<V>`) minted by `box_innerhandle` over this chain's slot.
// Maintained by safe code: the only constructors of `OuterHandle`
// are `over_slot`, `fanout`, and `replace`, and each stores exactly
// a `box_innerhandle` result (never anything else) into `current`.
impl<'o, K: ?Sized, V: ?Sized> std::ops::Deref for OuterHandle<'o, K, V> {
    type Target = V;
    fn deref(&self) -> &V {
        // SAFETY (preconditions of the construct): viewing the box's
        // contents as `V` requires them to BE an `InnerHandle<_, V>`
        // (a `Box<V>` could hold any V — the type does not know).
        // Why they hold: the outer-view invariant above; the Unsize
        // coercion at the mint made the erasure type-correct, so
        // `&V` dispatches to the inner handle's `V` impl, and the
        // reference is tied to `&self` (the box outlives it).
        unsafe { &*(self.current.as_ref() as *const V) }
    }
}

impl<'o, K: ?Sized, V: ?Sized> std::ops::DerefMut for OuterHandle<'o, K, V> {
    fn deref_mut(&mut self) -> &mut V {
        // SAFETY: as `Deref` (the outer-view invariant), plus
        // exclusivity: `&mut self` is the one chain (I1) and no guard
        // can coexist with it (I4).
        unsafe { &mut *(self.current.as_mut() as *mut V) }
    }
}

impl<'o, K, V: ?Sized + 'static> OuterHandle<'o, K, V>
where
    K: DerefMut + AsPtr + Send + 'static,
    K::Target: Unsize<dyn HandleAble<V>> + 'static,
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
    /// box is a valid `InnerHandle<T, V>` whose `state` is the raw
    /// of the pointee of the `K` as stored by the box (the loaded
    /// provenance), and the typed `slot` stored here is exactly the
    /// `K` cell the raw was taken from.
    /// # Safety (caller preconditions)
    ///
    /// `slot` must point to a live `K` that stays live, at that
    /// address, exclusively reachable through the minted handle's
    /// chain for the handle's whole life (I1/I2).
    ///
    /// Crate-internal; the only caller is `independent_task_root`,
    /// which discharges all of it via its `&mut K` signature.
    unsafe fn over_slot(slot: *mut K) -> Self
    where
        K: Sized,
    {
        // SAFETY (preconditions of the construct): `&mut *slot`
        // requires a live, unaliased pointee.
        // Why they hold: this function's own contract above (I1/I2,
        // discharged by the one caller's `&mut K`).
        let k: &mut K = unsafe { &mut *slot };
        // `as_ptr` is a safe call carrying an unsafe CONTRACT; its
        // two requirements hold per K's own `AsPtr` impl, and the
        // caller-side discipline (exclusive ownership for the
        // handle's life) is this function's contract above.
        let state_raw: *mut K::Target = K::as_ptr(k);
        // SAFETY (preconditions of the construct): dereferencing
        // `state_raw` as `&mut dyn HandleAble<V>` requires the fat
        // raw to be live, to point at an object whose type both
        // implements `HandleAble<V>` and coerces to it, and to be
        // exclusive for the borrow.
        // Why they hold: liveness and same-object-as-Deref are
        // `AsPtr`'s requirements 1-2 (K's impl); the type judgement
        // is the where-clause `K::Target: Unsize<dyn HandleAble<V>>`
        // (the upcast is type-correct by the bound); exclusivity is
        // I1 (this chain is the only one). The receiver `erased` is
        // transient: it dies at the end of this statement.
        let erased: &mut dyn HandleAble<V> = unsafe { &mut *(state_raw as *mut dyn HandleAble<V>) };
        // the chain's ORIGIN: a fresh root Shared moves in
        let shared = Arc::new(Shared::new());
        // SAFETY (preconditions of the construct, from
        // `box_innerhandle`'s contract): the receiver must BE the
        // object at `state`; the raw must be kept live and exclusive
        // for the returned handle's whole life; the `SharedRef` must
        // be one of ours.
        // Why they hold: the receiver was DERIVED from `state_raw`
        // one statement up (same object — I5 fixes `T` to it);
        // liveness/exclusivity are `AsPtr`'s requirement 2 plus I1
        // (the handle is stored in `self.current` and dies with the
        // chain); the `SharedRef` wraps the `shared` minted here.
        let current = unsafe { erased.box_innerhandle(state_raw as *mut (), SharedRef(shared.clone())) };
        OuterHandle { current, slot, shared, _brand: PhantomData }
    }

    /// The successor install: write the K cell THIS handle owns
    /// (typed — no casting), then re-mint through the replacement's
    /// `HandleAble` and keep the fresh handle in the box. The chain
    /// survives: the outer's `Shared` passes on to the successor,
    /// so parking identity is untouched.
    pub fn replace(&mut self, val: K) {
        // 1) the swap — plain assignment through the typed slot
        // SAFETY: the slot is live and exclusively reachable through
        // this chain (the mint's discipline; the old inner handle is
        // replaced by the re-mint below, so no stale pointer
        // survives the write).
        // SAFETY (preconditions of the construct): writing through
        // `*mut K` requires the cell to be live and no outstanding
        // access to the OLD content when its `Drop` runs.
        // Why they hold: liveness is I2 (the cell outlives the
        // task); no outstanding access is I1 — this method holds
        // `&mut self`, i.e. the only chain, so every state method
        // has returned (borrowck on the driver side: `replace`
        // cannot be called while a deref borrow is live), no guard
        // exists (a guard would require this very chain to be
        // parked, which it is not — it is executing here), and the
        // old inner handle's raw dies with `self.current` below.
        unsafe { *self.slot = val };
        // 2) re-mint: use the HandleAble OF the replacement —
        // reachable through the slot it now lives in (K's own
        // DerefMut, consulted once) — to mint a fresh InnerHandle
        // over that same slot, and assign it into the own box
        // SAFETY: the mint's discipline over the fresh slot content;
        // the receiver comes off the slot raw this handle owns.
        let fresh: Box<V> = {
            // SAFETY (preconditions of the construct): `&mut
            // *self.slot` requires a live, unaliased cell.
            // Why they hold: I2 for liveness; I1 for aliasing (the
            // `&mut self` receiver is the one chain; the write above
            // installed `val` as the sole content one statement ago).
            let k: &mut K = unsafe { &mut *self.slot };
            // contract-bearing safe call, as at every mint (see
            // `over_slot` for the full justification)
            let state_raw: *mut K::Target = K::as_ptr(k);
            // SAFETY: as the upcast deref in `over_slot`: live fat
            // raw (AsPtr 1-2), type-correct upcast (the `Unsize`
            // bound), exclusive (I1), transient receiver.
            let erased: &mut dyn HandleAble<V> = unsafe { &mut *(state_raw as *mut dyn HandleAble<V>) };
            // SAFETY: as the `box_innerhandle` call in `over_slot`:
            // the receiver is the object at `state_raw` (derived one
            // statement up; I5), the raw stays live/exclusive for
            // the new handle's life (stored into `self.current`,
            // replacing the old — the old raw is dead by I1), and
            // the `SharedRef` wraps this chain's own `shared`.
            unsafe { erased.box_innerhandle(state_raw as *mut (), SharedRef(self.shared.clone())) }
        };
        self.current = fresh;
    }
}

/// A container that can hand out the pointer its `Deref`s spring
/// from — the provenance vehicle the mint needs.
///
/// # Safety
///
/// For implementors:
///
/// 1. The returned pointer must NOT be a freshly minted reference:
///    it must originate from the same raw line the container's
///    `Deref`/`DerefMut` also spring from (the underlying stored
///    pointer). Under Tree Borrows a freshly minted reference is a
///    fresh tag — a sibling of the readers' tags — and every
///    checkpoint read would then permanently freeze the writer's
///    line (the 450 pattern, demonstrated on custom containers and
///    sized builtin targets).
/// 2. The backing store must not be deleted until the next `&mut`
///    method call on the container: disposal may only happen when
///    the container regains exclusive control, and by then every
///    handle minted over the old content must already be dropped
///    (the mint's existing contract).
pub unsafe trait AsPtr: DerefMut {
    /// The pointer to the container's current pointee, on the
    /// stored line (see the trait's safety requirements).
    ///
    /// For callers (the library's mints): the returned raw is kept
    /// for the minted handle's whole life; the caller must
    /// exclusively own the container for that period, and must
    /// re-mint (drop the old handle) before the container's next
    /// `&mut` method call — both are the mint's existing discipline
    /// (I1/I2), which is why the mints may call this safely.
    fn as_ptr(&mut self) -> *mut Self::Target;
}

// SAFETY (preconditions of the construct, from the trait):
// requirement 1 — the raw must originate from the line the Derefs
// spring from, not a freshly minted reference; requirement 2 — the
// backing store must not be deleted until the next `&mut` method
// call on the container.
// Why they hold: (1) `&raw mut **self` borrows the BUILTIN box-deref
// PLACE — the loaded stored pointer itself; no reference is
// materialized (a raw borrow does not retag), so the raw carries
// exactly the stored line that `Box::deref`/`deref_mut` hand out
// references to. (2) a `Box` cannot delete or move its allocation
// except by dropping or moving the box itself, which requires
// exclusive access — the `&mut self` receiver here IS that access,
// and the mint's discipline (I1/I2: the caller exclusively owns the
// container and re-mints before the next `&mut` call) bounds every
// previously returned raw's lifetime to end before any disposal.
unsafe impl<T: ?Sized> AsPtr for Box<T> {
    fn as_ptr(&mut self) -> *mut Self::Target {
        &raw mut **self
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
///   the `InnerHandle<T, V>` over the PASSED state raw (the blanket
///   impl stores it verbatim).
pub unsafe trait HandleAble<V: ?Sized>: Send {
    /// Mint the handle over `self`.
    ///
    /// # Safety
    ///
    /// Caller preconditions:
    ///
    /// 1. The receiver must be the object living at `state` (the
    ///    returned handle stores that raw and derefs through it).
    /// 2. The pointee of `state` must stay valid, live, and
    ///    exclusively reachable through the returned handle's chain
    ///    for that handle's whole life.
    /// 3. No other thread may hold shared access to the container
    ///    while the handle lives (its `Deref` would alias the
    ///    exclusive pointee).
    ///
    /// The library's three mint sites discharge 1 by deriving the
    /// receiver from the same `as_ptr` raw (I5: the dispatch fixes
    /// `T` to that object), 2 by `AsPtr`'s requirement 2 plus I1/I2,
    /// and 3 by I1 (the container is exclusively owned by the chain).
    ///
    /// The `SharedRef` argument is the seal as well as the payload:
    /// it is an opaque shell (`Shared` private, the field private),
    /// so no client can construct one — the method is uncallable
    /// from outside the crate.
    unsafe fn box_innerhandle(&mut self, state: *mut (), shared: SharedRef) -> Box<V>;
}

// SAFETY (preconditions of the construct, from the trait): the
// impl is only allowed in-crate (upheld: it is written here); `T`
// must unsized-coerce (`InnerHandle<T, V>: Unsize<V>` — the
// where-clause); the returned `Box` must be the `InnerHandle<T, V>`
// over the PASSED raw.
// Why they hold: the body stores the passed raw VERBATIM (never a
// receiver-derived one — the receiver is only the dispatch vehicle;
// I5 already fixed `T` to the object it points at, so raw and
// receiver agree by construction), and the returned box coerces by
// the where-clause. `T: Send` upholds the trait's supertrait.
unsafe impl<T: Send, V: ?Sized> HandleAble<V> for T
where
    InnerHandle<T, V>: Unsize<V>,
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
        let handle: Box<InnerHandle<T, V>> = Box::new(InnerHandle {
            state: raw,
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
    K: DerefMut + AsPtr + Sized + Send + 'static,
    K::Target: Unsize<dyn HandleAble<V>> + 'static,
        F: for<'n> Fn(OuterHandle<'n, K, V>) -> R + Send + Sync,
        R: Send,
    {
        // Collect offsite under the &mut node borrow (the ROOT's
        // pattern): the split's references die at the raw conversion,
        // the earliest point; everything after touches only raws.
        let child_raws: Vec<*mut K> = {
            let node: &mut T = self;
            split(node).into_iter().map(std::ptr::from_mut).collect()
        };
        // MINT THE BOXES: each slot becomes a Box<V> via the blanket
        // impl — the receiver derives from the raw (one borrow
        // stack). The typed raw and the child's Shared ride along
        // for the outer that owns them.
        let mints: Vec<(Box<V>, *mut K, Arc<Shared>)> = child_raws
            .into_iter()
            .map(|raw| {
                // the receiver: K's DerefMut consulted once, upcast
                // via the judgement
                // SAFETY: the split yielded each child exactly
                // once, disjointly; exclusive for this mint.
                // SAFETY (preconditions of the construct): `&mut
                // *raw` requires a live, unaliased child cell.
                // Why they hold: the split (safe code, HRTB) yielded
                // each child exactly once and disjointly — an
                // `iter_mut`-style collector cannot alias — and the
                // children live inside the parent's state, which
                // this chain exclusively owns (I1/I2).
                let k: &mut K = unsafe { &mut *raw };
                // contract-bearing safe call, as at every mint
                let state_raw: *mut K::Target = K::as_ptr(k);
                // SAFETY: as the upcast deref in `over_slot`: live
                // fat raw (AsPtr 1-2), type-correct upcast (the
                // `Unsize` bound), exclusive (I1), transient.
                let erased: &mut dyn HandleAble<V> = unsafe { &mut *(state_raw as *mut dyn HandleAble<V>) };
                // each child gets its OWN Shared (its parking
                // identity), weakly registered into the parent's vec
                // — insert under the lock, release (the convention)
                let child_shared = Arc::new(Shared::new());
                self.status.register_child(&child_shared);
                // SAFETY: as the `box_innerhandle` call in
                // `over_slot`: the receiver is the object at
                // `state_raw` (I5), the raw stays live/exclusive for
                // the child handle's life (the handle moves into the
                // child outer, whose brand confines it to the
                // callback), and the `SharedRef` wraps the child's
                // own freshly minted `shared`.
                let current =
                    unsafe { erased.box_innerhandle(state_raw as *mut (), SharedRef(child_shared.clone())) };
                (current, raw, child_shared)
            })
            .collect();
        // wrap each box in the OUTER handle — the child's typed slot
        // plus its Shared — so the callback can drive AND replace its
        // subtree; then threads. The brand lifetime is free here —
        // `f`'s higher-ranked bound instantiates it at the call, and
        // its universality is what keeps the handle inside the
        // callback.
        let outers: Vec<OuterHandle<'_, K, V>> = mints
            .into_iter()
            .map(|(current, slot, shared)| OuterHandle {
                current,
                slot,
                shared,
                _brand: PhantomData,
            })
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
        // SAFETY (preconditions of the construct): `&*self.state`
        // requires the raw to be live and `T` to be the pointee's
        // type; a shared read must not race another access.
        // Why they hold: liveness is the mint's contract (the raw is
        // kept live for the handle's life — `AsPtr` requirement 2
        // plus I2); type agreement is I5; and the only possible
        // cross-chain reader is a checkpoint guard, which cannot
        // exist while this handle's chain is executing (I4: a guard
        // exists only while every mutator — including this one — is
        // parked, and a parked chain holds no deref borrow because
        // `safepoint` takes `&mut self`).
        unsafe { &*self.state }
    }
}

impl<T: Sized, V: ?Sized> DerefMut for InnerHandle<T, V> {
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY (preconditions of the construct): `&mut
        // *self.state` requires live, type-matching, UNALIASED
        // pointee for the borrow.
        // Why they hold: as `Deref` (mint contract, I5), plus
        // exclusivity: `&mut self` is the one chain (I1), and no
        // guard can coexist with a running chain (I4) — so no other
        // access exists anywhere for the borrow's duration.
        unsafe { &mut *self.state }
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
