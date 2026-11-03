# The issue #18 assignable exploration (two-piece shape, 16:38 sketch)

Handle subcrate: `InnerHandle<T> { ptr, status: Arc<Shared> }` — no
lifetimes; `Shared` is Arc-OWNED with a `Mutex<Vec<Weak<Shared>>>`
children registry (register on new, merge-on-drop with closed
filtering) — the borrow-based registry is gone, and with it the
Shared-inside-the-node aliasing question.

Dynstate subcrate: the **`Assignable`** — the swappable storage IN the
tree (one heap'd value + its membership). `new(value, parent)` heaps
the value, registers membership; **`set(value)`** is the successor
install: allocate the new value, re-mint the inner handle over it with
the SAME membership, drop the old value by the OLD dropper. The
`HandleAble` blanket (the sketch's shortcut path) also lives here.

Client — variant A (the chain): vocabulary on the INNER handles
(`impl State for InnerHandle<Countdown>` — every state method runs on
its handle and CAN safepoint mid-run; Countup does). The chain = an
`Assignable` + the current `Box<dyn State>`; `Switch` carries the
successor VALUE (type-erased via a client `Next` seam); the install is
the ONE door — `assignable.set(value)` — value swapped, membership
KEPT, fresh inner handle minted over the new value. Runs: `result: 42`
(Countdown -> Countup with a mid-run safepoint), `save: countdown:3`.
**Miri clean** — after catching a real bug (set() dropped the old
value by the NEW dropper: type confusion; now old-data/old-dropper).

Client — variant B (the sketch-literal path): states call their own
`box_innerhandle()` and the outer `Box<dyn State>` is swapped
wholesale. Difference vs A: the minted handle's membership is a FRESH
root — nothing parks it from a parent; the whole old box (state +
membership) dies on swap. This is exactly the sketch's "missing
Assignable in the tree" gap, which A closes.

Challenges (details on the issue):
- `dyn HandleAble<V>` is not dyn-compatible (the `InnerHandle<Self>:
  Unsize<V>` where-clause cannot hold for the erased Self) — the
  erasure happens at the vocabulary trait instead.
- `set` must drop old by old dropper (Miri-caught).
- The successor payload needs its own type-erasure seam (`Next`), since
  `Switch` cannot name the state type.
