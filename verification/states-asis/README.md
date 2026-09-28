# states-asis: the PR's checkpointer verified as written

Verus verification of the PR's `src/checkpoint.rs` **protocol logic
verbatim** — same control flow, conditions, and step ordering — with
specs and ghost state added, answering "can we just add specs to
Mutex/Condvar and verify the code as-is?".

**Result: 13 verified, 0 errors** (plus 2059 vstd obligations).

## What is literally the PR's

- `request`: set pending → lock → loop { parked? grant : wait }
- `release` (the guard's Drop): return baton → clear pending →
  wait until the mutator has un-parked
- `park_self` (via `safepoint`): lock → bank baton → park →
  `while pending { wait }` → withdraw → un-park
- borrows: `unsafe { &*node }` / `&mut *node` with the `&mut self`
  discipline
- put-before-flag discipline everywhere (bank before publishing
  parked; return before clearing pending)

## What was added/swapped (the complete list)

| PR | here | why |
|---|---|---|
| `Mutex<bool>` + `park_cv` + `resume_cv` | trusted ghost-carrying `TrustedLock` with `lock`/`wait_park`/`wait_resume` mirroring the PR's helper seam (`lock_parked`/`wait_park`/`wait_resume`) | std types have no per-instance ghost; the PR's own helper layer is the natural spec attachment point |
| `pending_request: AtomicBool` | ghost flag inside the lock (mark/clear/still_pending) | the flag's protocol meaning is lock-mediated; the unlocked store/reads are linearized (B3') |
| `Shared` | `Shared<T>` (phantom) | the lock's ghost holds this pair's baton |
| `Arc<Shared>` sharing | `&'a Shared<T>` | same pair identity, lifetime-shared |
| `with_checkpoint_pair` body | trusted (signature verbatim) | Verus cannot call generic closures whose args carry tracked fields |
| — | ghost baton fields on `Handle`/`CheckpointGuard`, `wf` contracts | the ownership the proof tracks |

The trusted lock specs are the exact transition relations (bank /
grant 0→1 / return 1→0 / resume / mark / clear); mutual exclusion and
those transitions are the trust. Everything above them — every
critical section, the invariant `guarded_inv` (parked ⇔ banked-or-
guard, guards ≤ 1, guards ⇒ pending), the baton's node matching, the
`&mut`-vs-safepoint discipline — is **machine-checked**: `lock()`
ensures the invariant, `wait`/`put_back` require it, so each critical
section proves it at exit.

## Theorems

T1/T2/T3 as in `../checkpoint-verify`: the baton (the node's
`PointsTo`) is created once and moves only handle → banked → guard →
banked → handle; `block_and_get_mut` needs it mutably in the handle
slot, `guard.node()` needs it in the guard, so the two borrows can
never alias; every raw-pointer deref is justified (pptr + init).
Scope: core single-pair protocol; `fanout`'s split contract and
scoped threads remain the documented axioms (B6/B7); requester
serialization (B5) and deadlines (B4 superset) as before.

## Run

```
cargo-verus verify   # in this directory
```
