# Verification artifacts for the checkpoint framework

Three self-contained Verus crates accompanying the formal verification
of PR #1's `src/checkpoint.rs`:

| crate | what it is | result |
|---|---|---|
| `checkpoint-verify` | port with a fused mode machine (`AtomicU8` ghost carries the `PointsTo` baton) | 25 verified, 0 errors — T1 (no aliasing), T2 (raw-pointer sites), T3 (Send impls) |
| `states-asis` | the PR's protocol logic verbatim, verified against a trusted lock/condvar spec | 13 verified, 0 errors — same theorems |
| `checkpoint` | runnable crate: the actual code verified above a two-primitive unverified surface (lock spec + condvar=unlock+lock), plus `verified_lock.rs`: the lock spec implemented by a **fully verified** spinlock (`basic_lock` pattern) — the spec assumes nothing beyond a lock | 32 verified, 0 errors; `cargo test` 3/3; Miri clean (16 seeds × aggressive preemption, `-Zmiri-ignore-leaks` for vstd PPtr's node allocation) |

Verdict: the stop-the-world protocol is **sound** (the mutator's `&mut`
and the snapshotter's `&` can never alias, for any unbounded
interleaving). Separately, the protocol has a **liveness bug**: the STW
handshake can deadlock — reproduced under the PR's own test binary
(`full_run_with_stw_checkpoints_and_nested_fanout` hangs; gdb
witnesses and an instrumented protocol log in
`checkpoint-verify/evidence/`); soundness does not depend on wakeups
(condvars are modeled as unlock+lock, a superset of any real condvar).

Reproduce: `cargo-verus verify` in each crate (Verus
0.2026.09.20, toolchain 1.98.1).
