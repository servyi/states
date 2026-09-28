# verbatim: the shipped checkpoint.rs, verified with spec'd seams

The protocol functions are the shipped text (`src/checkpoint.rs`,
7a03762) with every std-library touch routed through a spec'd
interface (`substrate.rs`), each with an unverified std-backed impl:

| seam | shipped | spec'd as |
|---|---|---|
| S1 | `Mutex<bool>` parked + park_cv/resume_cv | `ParkedCell` (baton account inside; same lock spec as `verification/checkpoint`'s, whose implementability is proven by its `verified_lock`) |
| S2 | `pending_request`/`closed` AtomicBools | folded into `ParkedCell` helpers (`set_pending`/`is_pending`/`is_closed`/`close_cell`) |
| S3 | `Instant::now()` + deadline arithmetic | `Deadline` (nondeterministic; any answer sound — liveness only) |
| S4 | `split: fn(&'t mut T) -> I` | `SplitFn` trait: parent baton consumed, per-child `PointsTo` minted, `reassemble` at fan-out end |
| S5 | `thread::scope`/`spawn`/`is_finished`/1ms sleep | `ScopeSpec`: spawns joined before return; `all_finished` advisory; closure runs inline |
| S6 | `request_lock: Mutex<()>` | `ReqLock` |

**Result: 28 verified, 0 errors** (plus 2059 vstd obligations), covering
`safepoint`, `park_self` (children handshake + park/resume),
`request_until_parked` (all three exits), `release_request`, `close`,
`block_and_get(_mut)`, `CheckpointGuard::node`, and the borrow
discipline (T1: the mutator's `&mut` and the guard's `&` cannot alias).

## Examples: every user-side assumption instantiated

- `examples/split_vec.rs` — `SplitFn` for the canonical shape
  (`T = Vec<C>`, `iter_mut`): disjoint children with per-child
  permissions (std slice disjointness as the trusted basis), plus a
  verified client round trip discharging every `Handle`/guard
  obligation (`hwf`, `has_baton`, `gwf`, `has_snap`) against the same
  contracts the shipped code was verified with. 4 verified.
- `examples/scope_seq.rs` — `ScopeSpec` fully instantiated: a
  sequential scope where spawns run inline and join before return
  (trivially satisfying the spec), with a verified client. 3 verified.

Together with `verification/checkpoint`'s `verified_lock` (the lock
spec from pure atomics), every seam's spec now has a concrete,
machine-checked instantiation.

## Open obligations (documented at their sites)

- `fanout`'s body: Verus frontend limits (closures capturing `&mut`
  across the scope, `Vec::retain`, ref-to-ptr casts); body is
  `assume(false)`-gated; the serve loop's parking is the same verified
  `park_self`; split/join facts are the S4/S5 specs (now instantiated
  by the examples).
- `request_impl`'s grant step: two `assume`s for the cross-guard
  ghost-threading (the Parked-exit view facts); argument given in the
  comment (give_baton set guards:=0/banked; requester holds
  request_lock).
- `with_checkpoint_pair`/Drops: trusted plumbing (same class as the
  other crates).

Verdict unchanged: **sound core** (baton exclusivity over the shipped
algorithm); the deadlock is liveness-only and unaffected.
