# The issue #18 installable exploration (on the two-brand)

Same workspace; the client explores the `Box<dyn InnerState>` shape:
the pairing is with the BOX (one Context for all inner states),
successors install by overwriting the box, safepoints sit between
transitions at the entry.

Attack finding (the probe in `client`'s `cross_child` module): a
child handle CANNOT escape its action call — `E0521: borrowed data
escapes outside of closure` on `*stashed = Some(h)`. So the
client-reachable cross-child attack is closed BY CONSTRUCTION (a
client never holds two children's handles in one scope), regardless
of what region inference does with the implementation's 'c
instantiations. The remaining trust anchor is fanout's own internals;
the one-pointer `Shared` check inside `unwrap_mut` is the cheap
backstop.

## The installable exploration (this branch)

Per the 15:50 comment: `Installed<S, T> { ctx: T, state: S }` moved
INTO the dynstate library, generalized over state and context — the
wrapper, its derefs, the projection (`Proj<S, T>`, a ZST so the
fn-pointer entries reach it without capturing), `install`, and the
safe `erased_mut` dig all live there. The old (unused, wonky)
DynHandle machinery is deleted from dynstate. The client carries only
its vocabulary (Env, Outcome, Context, InnerState/OuterState), the
entries, the states, and the driver — ZERO unsafe. Runs:
`result: 42` (Countdown inner -> Countup OUTER with a mid-run
safepoint), `save: countdown:3`.

The projection casts to the WRAPPER and then takes an ordinary field
borrow — no offset assumptions, any field order.

Walls found (W5/W6 new this round, in the client's notes):
1. brand-free Context required (with_pair fixes T before the brands);
2. the unwrapper's fat->thin offset projection is the one unsafe;
3. flavor mixing works only into boxes (both flavors install);
4. box swap replaces state+entries together;
5. fn-pointer entries cannot capture the unwrapper value (E0308) —
   it rides as a ZST projection type;
6. outer states cannot receive (handle, &mut Self) together (E0499) —
   they take the handle and project per use; the ??? of
   `Handle<??? Self>` resolves to the BOX handle.

Earlier round (the DynHandle-based box flavor, now superseded but kept
below in git history): the client ran the `Box<dyn InnerState>` shape
end to end:
`result: 42` (Countdown -> Countup through the box OVERWRITE),
`save: countdown:3`. ONE `Context` for every inner state (the pairing
is with the box, `Erased`); successors install as plain `*boxed =
next` writes through the same unwrap; the entry safepoints BETWEEN
transitions. Zero unsafe; neither subcrate touched.

Walls found (probe module `outer_probes`, E0119 verified live):
1. one Context cannot serve both flavors — the generic direct impl
   and the Erased impl conflict (E0119; `Erased` is 'static, and no
   negative bounds exist to exclude it);
2. two Context types split the chain — `DynHandle<'o,'c,Ctx>` fixes
   ONE data type, so the flavors cannot interleave;
3. a box-flavor Switch cannot express a direct successor — `install`
   needs a `Handle<S>` that only exists at with_pair/fanout mint
   points; an inner state holds no handle to re-install from;
4. inner states structurally cannot safepoint mid-run — the
   object-safe method sees no handle; a long-running inner execute
   delays the park until it returns.
