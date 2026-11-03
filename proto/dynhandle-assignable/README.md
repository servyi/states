# The issue #18 assignable exploration (two-piece shape, 16:38 sketch)

Rebuilt to follow the sketch literally after review: **the Assignable
is a Box a new value can be stored into** — a STABLE CELL at a fixed
heap address, whose content (a `Box<V>`) is swapped by `store`.
Inner handles point AT the cell (`*mut dyn Cell`), not at the value;
`set` follows the sketch's four steps verbatim: (1) store the new
value into the cell; (2) the old typed view is forgotten (the outer
box replaces it); (3) the membership Arc is cloned; (4) a fresh typed
handle is minted over the SAME cell, and the outer `Box<dyn State>`
swaps to it.

Pieces:
- handle: `Cell` trait (stable address, interior-mutable content:
  `data_ptr`/`data_ptr_mut`), `Shared` (Arc-owned, Weak children,
  register/merge-on-drop), `InnerHandle<T>` (points at the cell,
  carries the membership, derefs through the cell to the typed
  content — valid because handles are re-minted after every store).
- dynstate: `Assignable<V>` (RefCell<Option<Box<V>>> — shared
  borrows only on the cell itself), `mint`, `set`, `erase`.
- client: vocabulary on the inner handles (`impl State for
  InnerHandle<Countdown>` — mid-run safepoints work), `Switch`
  carries the successor through a typed `Install` seam (store +
  re-mint); the chain holds the cell as a raw pointer (stable
  address, sole ownership, manual drop).

Runs `result: 42` (Countdown -> Countup, mid-run safepoint through
the handle), `save: countdown:3`. **Miri clean**, clippy clean.

Miri taught three lessons on the way (all fixed):
1. the cell must be at a stable address — moving it (even boxing then
   moving the Box) dangles every handle minted over it;
2. the cell must only ever be borrowed SHAREDLY after the first mint
   — a `&mut` reborrow pops the handles' raw-pointer tags (Stacked
   Borrows); stores go through RefCell interior mutability;
3. the write path (DerefMut) must derive its pointer with write
   intent (borrow_mut), the read path without — one shared `data_ptr`
   cannot serve both.

Known gap vs a full port: fanout/`box_innerhandle` (variant B of the
earlier round) is not re-wired onto the cell yet; the sketch's
`new_child` registration exists but no state exercises it.
