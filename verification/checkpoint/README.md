# checkpoint

The actual checkpointer protocol — the PR's handshake logic, function
for function — verified by Verus against a **two-primitive unverified
surface**, and runnable as-is:

```
cargo-verus verify            # 14 + 2059 verified, 0 errors
cargo test                    # 3/3 (stable across runs)
cargo +nightly miri test      # 16 seeds × aggressive preemption, clean
                              # (with -Zmiri-ignore-leaks: vstd's PPtr
                              #  heap-allocates the node and the verified
                              #  code never frees it — a documented
                              #  artifact of the pointer model; the PR's
                              #  original kept the node on the stack)
```

## The spec is implementable: `verified_lock` (fully verified)

`src/verified_lock.rs` (built under `cargo-verus verify`; gated by
`verus_only` for plain builds) implements the **same lock spec** with
**zero trusted bodies** — pure vstd atomics (`PAtomicBool` spin bit +
`PCell` + one `AtomicInvariant`, the vstd `basic_lock` pattern):
`32 verified, 0 errors` alongside the protocol's obligations. The
guard *owns* the protected state (the banked payload checks out at
acquire and back at release), so the spec's view is a real function of
owned state and `wait_*` is literally `put_back + lock`. This is the
machine-checked witness that the spec assumes nothing beyond a lock —
which is precisely the refinement the `std::sync` impl is trusted to
provide.

## Architecture (exactly two unverified primitives)

1. **Lock spec** — `TrustedLock`/`TrustedGuard`: mutual exclusion over
   `Guarded { parked, pending, baton }` + the exact ghost-transition
   contracts (bank / grant 0→1 / return 1→0 / resume / mark / clear /
   park reads+writes). Implemented by the unverified `std::sync::Mutex`
   (per-op lock/unlock bodies — real code).
2. **Condvar spec** — `wait_park`/`wait_resume` are exactly *unlock +
   [block] + lock*, with the state afterwards arbitrary but
   invariant-consistent (a superset of any real condvar, so soundness
   holds even for total wakeup loss). Implemented by the unverified
   `std::sync::Condvar` (`wait_timeout(1ms)` — the wakeup is a hint;
   the verified outer loops re-check their predicates, which also
   makes the implementation immune to lost wakeups and capture races,
   a real bug found and fixed during bring-up).

The refinement argument for both: **a blocked thread is a thread
taking no steps**, so blocking refines spinning; wakeup timeliness is
liveness, which is not verified.

## Verified (against those two specs)

Every protocol function: `with_checkpoint_pair`, `safepoint`,
`park_self`, `block_and_get`, `block_and_get_mut`,
`CheckpointRequester::request`, `CheckpointGuard::node`, the release
logic (`release`), and the shared invariant `guarded_inv` is
machine-checked at every critical-section exit (`put_back`/`wait`
require it; `lock` ensures it).

The theorems (T1/T2/T3): the node's `PointsTo` ("baton") moves only
handle → banked → guard → banked → handle, so the mutator's `&mut`
and the snapshotter's `&` can never alias, every raw-pointer deref is
justified, and the `unsafe impl Send`s are discharged.

## Remaining trust base

Verus + vstd + SMT; the two primitive impls above conformance to
their contracts; `nondet_bool` (trivial); `with_checkpoint_pair`'s
construction body (generic-closure limitation); the `Drop` twin of
`release` (Drop cannot carry the verified preconditions; it performs
the identical sequence).

## Runtime notes

- `is_pending` takes a fresh locked read — a valid instance of its
  any-value spec (the PR's unlocked read may be stale; staleness is
  defer-only).
- Liveness caveat inherited from the protocol: a `request()` landing
  after the mutator's last safepoint blocks; use the drain pattern
  (all tests do).
