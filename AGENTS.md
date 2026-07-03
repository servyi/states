# Agents

## Commands

- Build: `cargo build`
- Test: `cargo test`
- Lint: `cargo clippy -- -W clippy::all`

## ContextId Rules

- **`ContextId` Clone is illegal.** Cloning resets `child_counter`, producing duplicate child IDs. There is no `Clone` impl and there must never be one.
- **`PhaseTracker` uses `u64` keys** (from `ctx.id()`), not `&ContextId`.

## Tracing Rules

- The context_id `Display` output is the dotted name (e.g., `1.2.3`), not the raw `u64` id.
- `MessageLogLayer` captures events containing fields named `ctx`, `context_id`, `func_ctx`, or `fctx`. The `self.` prefix is normalized away automatically.

## Code Quality

- Never bypass compiler warnings with `#[allow(...)]` or similar suppression attributes.
- No code duplication. Use good abstractions and put common code into logically self-contained submodules.
- Tests must never document or assert known buggy behavior. If a test reveals a bug, fix the code rather than encoding the buggy behavior as an expected result.
