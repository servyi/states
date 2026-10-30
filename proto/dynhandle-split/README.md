# The issue #18 prototype: the three-way split

Standalone workspace (`handle` / `dynstate` / `client`), green:
`cargo run` prints `result: 42`, `save: countdown:3`.

- `handle` — the existing checkpoint Handle; delta vs `src/checkpoint.rs`
  on main is `#[repr(C)]` (layout identity for the re-types).
- `dynstate` — the erased-chain MECHANISM only: `DynHandle<'o, T>`, the
  branded capability `Unwrap<'o>`/`Unwrapper<'o, S>` (in-crate witness),
  `ContextData<'o, S>` with the equality bound `Unwrap<'o, Type = S>`
  (ties the capability to the chain's state type without naming any
  state trait), and `install` (the owned erase + the pairing).
  Successor installs are REMOVED (the open design question).
- `client` — the chain vocabulary (`State`, `Outcome`, `Context` with
  its two inline entry pointers), the states, the run loop. ZERO
  unsafe; the entries use `U::unwrap_mut` + `Handle`'s Deref/DerefMut.

Unsafe census: four one-liners, all in the library crates
(`Deref`/`DerefMut`, `unwrap_mut`'s re-type, `install`'s erase).

Attack findings (verified against the compiler): cross-brand entry
application rejected (E0521/E0308 — a `DynHandle` cannot even escape
its `with_pair` closure); brand erasure (`Box<dyn for<'any> Fn..>`)
does not compile; `data` is private (E0616); `U::unwrap_mut` as a
bare fn pointer is nameable but its type bakes in the brand and the
state type, so it cannot be applied to another pair's handle.
