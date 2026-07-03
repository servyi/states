# servyi-states

Generic state machine infrastructure: hierarchical context IDs, phase trackers, and structured message logging.

## Components

### ContextId

Hierarchical identifiers with dotted-path display (e.g., `1.2.3`). Each `ContextId` has:
- A unique `u64` hash (internal, used as HashMap key)
- A dotted name path (display/human-readable)
- Thread-safe child counter for creating children

**No `Clone`.** Cloning would reset the child counter and produce duplicate IDs. Pass by reference or move.

### PhaseTracker

Tracks phase state per `ContextId`:

```rust
pub trait PhaseTracker<P>: Send + Sync {
    fn advance(&self, ctx: &ContextId, from: Option<P>, to: P);
    fn expect_any_and_set(&self, ctx: &ContextId, valid_from: &[P], to: P);
    fn get_phase(&self, ctx: &ContextId) -> Option<P>;
}
```

`DefaultPhaseTracker<P>` — DashMap-backed implementation with assertion-checked transitions.

### State Machine

Generic async state machine traits and helpers:

- `State<K>` — single-step async state
- `IOState<K>` — perform IO then transition
- `ParallelState<K, I>` — split into parallel sub-states, collect results
- `run_to_completion<I>()` — drive a state chain to completion
- `execute_io()`, `execute_parallel()` — helpers

### MessageLog

A `tracing_subscriber::Layer` that captures events containing context ID fields (`ctx`, `context_id`, `func_ctx`, `fctx`) into a structured `DashMap` keyed by context ID string. Supports:

- Chronological ordering via sequence counter
- Status tracking per context ID
- Ancestor traversal for hierarchical display
- `self.` prefix normalization on field names

## Usage

```toml
[dependencies]
servyi-states = { git = "https://github.com/servyi/states.git" }
```

```rust
use servyi_states::{ContextId, DefaultPhaseTracker, PhaseTracker, MessageLog, MessageLogLayer};

let root = ContextId::root();
let child = root.new_child(); // displays as "1"

let tracker = DefaultPhaseTracker::<String>::new();
tracker.advance(&child, None, "open".to_string());
assert_eq!(tracker.get_phase(&child), Some("open".to_string()));

let log = std::sync::Arc::new(MessageLog::new());
let layer = MessageLogLayer::new(log.clone());
// Add layer to tracing subscriber to capture events
```
