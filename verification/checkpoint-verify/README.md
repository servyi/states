# Formal verification of servyi/states PR #1 (checkpointer)

**Verdict:**
1. **SOUNDNESS — PROVEN.** The stop-the-world protocol's core is memory-safe:
   the mutator's `&mut` into the checkpointed node and the snapshotter's `&`
   can never alias, for any unbounded interleaving, any timing, any number of
   park/release cycles. Machine-checked by Verus (SMT-based deductive
   verification over all executions — **not** model checking): **25 verified,
   0 errors** in this crate (plus 2059 verified vstd obligations).
2. **LIVENESS — BROKEN.** The protocol as written in the PR can deadlock
   (stop-the-world handshake wedges forever). Reproduced deterministically
   under the PR's own test binary within ~5s; witnesses captured (see
   `evidence/`).

## What is verified (src/lib.rs)

| PR code (src/checkpoint.rs) | Verified port | Status |
|---|---|---|
| `Shared::{request_until_parked, release_request, close}` | `request_until_parked`, `release_request`, `close_with_lease` | verified |
| `Handle::park_self` (bank/park/wait/resume, late-parker) | `park_and_wait` | verified |
| `Handle::{safepoint, block_and_get, block_and_get_mut}` | same names | verified |
| `CheckpointRequester::{request, request_timeout}` | `request_timeout` | verified |
| `CheckpointGuard::{node, Drop}` | `node`, `release` | verified |
| `with_checkpoint_pair` | same (returns the pair) | verified |
| client discipline (borrows vs safepoints) | `demo_mutator_step`, `demo_snapshot` | verified |

### Theorems (T1–T3 in the source header)

- **T1 (no aliasing).** `block_and_get_mut`'s `&mut T` requires the node's
  unique `PointsTo` permission ("the baton") held mutably in the handle's
  slot for the borrow's whole lifetime; `CheckpointGuard::node`'s `&T`
  requires the baton in the guard. A safepoint needs the same slot
  (`&mut self`), so a borrow cannot straddle a park — the PR's
  "parking is borrow-checked" design, machine-checked. The baton is
  created once and moves only through verified atomic transitions, so the
  two borrows can never overlap.
- **T2 (raw-pointer safety).** Every node dereference site is verified to
  possess the baton with matching `pptr` and init state (`wf` contracts).
- **T3 (`unsafe impl Send`).** Justified by T1/T2 + `T: Send`: at most one
  accessor exists at any instant.

The mode machine (one SeqCst word fusing the PR's `pending`/`parked`/
`closed` flags, with the baton banked in the word's ghost account):

```
Idle --request--> Requested --park(bank)--> Parked0
Parked0 --grant(withdraw)--> Parked1 --release(re-bank)--> Draining
Draining --resume(withdraw)--> Idle
Idle|Requested --close(bank)--> Closed0 --grant--> Closed1
Requested --timeout--> Idle   (late parker: CAS fails, no park)
```

The `Parked0/Parked1` distinction makes a double grant unrepresentable
(the grant CAS cannot succeed twice).

## Trust base (everything else is verified)

- **B1** `pt_take`/`pt_give`: proof-mode `mem::take`/`mem::put` of the
  baton (same trust class as vstd's own `PCell::take`; moves a value,
  creates no aliasing).
- **B2** Condvars are **not modeled**: every condvar wait is abstracted to
  a spin loop that rechecks a flag — a behavioral *superset* of blocking.
  Soundness therefore holds for any wakeup discipline, including total
  wakeup loss (this is what makes the deadlock below a *liveness* bug
  only).
- **B3** The PR's unlocked fast-path reads (`safepoint`'s early return,
  `request_impl`'s closed fast path) are abstracted as nondeterministic
  choices; every real (possibly stale) outcome is admitted, and staleness
  in the PR is defer-only.
- **B4** Deadlines are nondeterministic events.
- **B5** The PR's `request_lock` (requester serialization) is modeled by a
  single requester; guard-vs-mutator exclusivity does not depend on the
  requester count.
- **B6** `fanout`'s `split: fn(&mut T) -> I` has no spec (plain fn
  pointer); the tree argument needs its documented pairwise-disjointness
  contract plus the PR's children-park-before-parent ordering. The
  parent/child handshake itself is the same verified protocol (one
  `Shared` per pair); the split axiom is the one item not machine-checked.
- **B7** Scoped-thread spawn/join semantics (std).
- **B8** One `assume(false)` in `close_with_lease`'s unreachable arm
  (Parked/Draining while holding the baton — excluded by baton
  linearity; see comments at the site).

## Memory-ordering refinement

All protocol transitions are SeqCst atomic RMWs (vstd-verified), matching
the PR's SeqCst accesses. The PR's flag accesses outside its parked-mutex
are abstracted per B3; the model admits every real protocol behavior
(strict superset), so soundness of the model implies soundness of the PR.

## The deadlock (liveness bug) — evidence

- `evidence/forensic.txt`: gdb backtrace of the wedged PR test binary:
  the top-level mutator blocked forever in `park_self` ->
  `children.retain` -> `request_until_parked(None)` (an **unbounded**
  wait on a child subtree to park), the timer/snapshotter spinning in
  `request_timeout` retries, workers running.
- `evidence/repro-log-tail.txt`: instrumented protocol log (the
  instrumentation is `evidence/checkpoint_instrumented.rs`) — the log
  freezes mid-handshake with a guard granted and never released.
- The PR's own test `full_run_with_stw_checkpoints_and_nested_fanout`
  hangs (killed at 120s timeout in a clean VM; `cargo test` does not
  finish).
- Additional test bug: `tests/checkpoint.rs` never constructs
  `Stage::Nested`, so the machine's `Done` stage is unreachable and the
  top-level fanout re-runs forever with an ever-growing `children` vec.

Mechanism sketch: `request_until_parked(None)` on children is an
unbounded wait whose enabling condition (the child eventually parking)
can be lost when the requesting side's own handshake times out or moves
on; combined with the guard-holding serialization this wedges the STW
handshake. Soundness is unaffected (T1–T3 hold regardless — verified
under B2, which assumes wakeups can be lost entirely).

## Reproducing

```
# verification (Verus 0.2026.09.20, toolchain 1.98.1)
cargo-verus verify          # in this directory

# deadlock reproduction
cd /workspace/states
cargo test full_run -- --test-threads=1   # hangs; kill after ~30s
```
