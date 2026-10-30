# The issue #18 two-brand prototype (on the split)

Same workspace, one change: the identity carries TWO brands —
`Handle<'o, 'c, T>`, invariant in both. `'o` is the PAIR (the whole
tree), `'c` is the CHILD within it: `with_pair` mints both as `'o`
for a top-level chain; `fanout`'s action is `for<'c> Fn(Handle<'_, 'c, T>)`,
so each child's brand is unnameable by client code. `DynHandle`,
`Unwrap`, `Unwrapper`, `ContextData`, `install` gain the parameter;
bodies unchanged. Green: `result: 42`, `save: countdown:3`.

Attack finding (the probe in `client`'s `cross_child` module): a
child handle CANNOT escape its action call — `E0521: borrowed data
escapes outside of closure` on `*stashed = Some(h)`. So the
client-reachable cross-child attack is closed BY CONSTRUCTION (a
client never holds two children's handles in one scope), regardless
of what region inference does with the implementation's 'c
instantiations. The remaining trust anchor is fanout's own internals;
the one-pointer `Shared` check inside `unwrap_mut` is the cheap
backstop.
