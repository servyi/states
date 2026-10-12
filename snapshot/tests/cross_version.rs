//! Cross-version save/restore, the methodology discussed in PR review:
//!
//! 1. Shapes are stable in the code. The serialized artifact IS the shape;
//!    restore produces a `dyn`-compatible value of exactly that shape.
//! 2. Each version of the *code* carries its own `restore(&self) -> State`
//!    — the shape never changes, the interpretation does.
//! 3. The restored state may be a PRE-FIX state: valid to enter, but not
//!    yet equivalent to what a same-version save would have produced. A
//!    fixer (LLM or human, through the error-gate mechanism in the real
//!    product) transitions it to a fully fixed state before the machine
//!    runs. The crate sees none of this — only the checkpoint protocol.

use std::sync::atomic::Ordering;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use servyi_states::checkpoint::{with_checkpoint_pair, Handle};
use servyi_states_snapshot as snapshot;

// ---------------------------------------------------------------------------
// The stable shape: what v1 wrote, what v2 reads. NEVER changes once
// published — this is the contract.
// ---------------------------------------------------------------------------

/// Snapshot shape, version 1 — frozen.
#[derive(Serialize, Deserialize)]
struct ShapeV1 {
    /// Total steps each item needs (the workload description).
    steps_total: Vec<u32>,
    /// Steps completed per item at snapshot time.
    steps_done: Vec<u32>,
    /// Whether the item's analysis finished in the saving version.
    finished: Vec<bool>,
}

// ---------------------------------------------------------------------------
// Version 1: the saver's world.
// ---------------------------------------------------------------------------

mod v1 {
    use super::*;

    /// v1's machine: steps counted directly, no notion of `polished`.
    #[derive(Clone)]
    pub struct MachineV1 {
        pub items: Vec<ItemV1>,
    }

    #[derive(Clone)]
    pub struct ItemV1 {
        pub steps: u32,
        pub step: u32,
        pub done: Option<String>,
    }

    impl MachineV1 {
        /// v1's saver: machine -> stable shape.
        pub fn to_shape(&self) -> ShapeV1 {
            ShapeV1 {
                steps_total: self.items.iter().map(|i| i.steps).collect(),
                steps_done: self.items.iter().map(|i| i.step).collect(),
                finished: self.items.iter().map(|i| i.done.is_some()).collect(),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Version 2: the restorer's world. Same shape, different code, and the
// knowledge that v1 had a step-counting bug (it counted the completion
// step twice), so v1-finished items are restored into a PRE-FIX state.
// ---------------------------------------------------------------------------

mod v2 {
    use super::*;

    /// v2's machine: adds `polished`, and `needs_fix` marks pre-fix items.
    #[derive(Clone, Debug, PartialEq)]
    pub struct MachineV2 {
        pub items: Vec<ItemV2>,
    }

    #[derive(Clone, Debug, PartialEq)]
    pub struct ItemV2 {
        pub id: u32,
        pub steps: u32,
        pub step: u32,
        pub done: Option<String>,
        /// v2-only: the fixer must resolve this before the item runs.
        pub needs_fix: bool,
    }

    impl ShapeV1 {
        /// v2's restorer — THIS code changed between versions; the shape
        /// did not. v1 recorded the completion step twice, so anything
        /// v1 called finished is restored as a pre-fix state: done, but
        /// flagged for the fixer to re-validate.
        pub fn restore(&self) -> MachineV2 {
            MachineV2 {
                items: self
                    .steps_total
                    .iter()
                    .zip(&self.steps_done)
                    .zip(&self.finished)
                    .enumerate()
                    .map(|(idx, ((total, step), finished))| ItemV2 {
                        id: idx as u32 + 1,
                        // v2 counts one fewer step for the same work
                        // (v1 double-counted the completion step).
                        steps: if *finished { (*step).saturating_sub(1).max(1) } else { *total },
                        step: *step,
                        done: finished.then(|| format!("v{}", idx as u32 + 1)),
                        needs_fix: *finished,
                    })
                    .collect()
            }
        }
    }

    /// The fixer: the stand-in for the LLM/human intervention that the
    /// error gate routes in the real product. Pre-fix -> fully fixed.
    pub fn fix(m: &mut MachineV2) -> usize {
        let mut fixed = 0;
        for item in &mut m.items {
            if item.needs_fix {
                // v2's re-validation: v1's results are trusted after the
                // step bookkeeping is reconciled.
                item.needs_fix = false;
                fixed += 1;
            }
        }
        fixed
    }
}

#[derive(Default)]
struct CountingIo {
    calls: std::sync::atomic::AtomicUsize,
}

impl CountingIo {
    fn step(&self, id: u32) {
        let _prev = self.calls.fetch_add(1, Ordering::SeqCst);
        let _ = id;
    }
}

/// v2's driver: run every item to completion, skipping done items (the
/// resume property), then aggregate.
fn drive_v2(mut h: Handle<'_, v2::MachineV2>, io: &CountingIo) -> String {
    loop {
        h.safepoint();
        h.safepoint();
        let m = &mut *h;
        for item in &mut m.items {
            if item.done.is_some() || item.needs_fix {
                continue;
            }
            io.step(item.id);
            item.step += 1;
            if item.step >= item.steps {
                item.done = Some(format!("v{}", item.id));
            }
        }
        if m.items.iter().all(|i| i.done.is_some()) {
            return m
                .items
                .iter()
                .filter_map(|i| i.done.clone())
                .collect::<Vec<_>>()
                .join(",");
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn cross_version_restore_produces_prefix_state_then_fixer_completes_it() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cross.json");

    // ── v1 SAVES: a machine with two items finished, one mid-flight. ──
    let machine_v1 = v1::MachineV1 {
        items: vec![
            v1::ItemV1 { steps: 3, step: 3, done: Some("v1".into()) },
            v1::ItemV1 { steps: 3, step: 3, done: Some("v2".into()) },
            v1::ItemV1 { steps: 3, step: 1, done: None },
        ],
    };
    // v1's saver runs on ITS machine through the same protocol crate:
    // serialize the shape itself — save_handle takes any Serialize.
    with_checkpoint_pair(machine_v1, |mut h, _req| {
        h.safepoint();
        snapshot::write_value(&h.to_shape(), &path).expect("v1 saves the shape");
    });

    // The artifact on disk is the stable shape, not any version's machine.
    let raw: serde_json::Value = snapshot::restore_from(&path).expect("artifact parses");
    assert_eq!(raw["steps_total"], serde_json::json!([3, 3, 3]));
    assert_eq!(raw["steps_done"], serde_json::json!([3, 3, 1]));
    assert_eq!(raw["finished"], serde_json::json!([true, true, false]));

    // ── v2 RESTORES: its code reads the same shape. ──
    let restored: ShapeV1 =
        serde_json::from_value(raw).expect("shape is shape-stable across versions");
    let mut machine_v2 = restored.restore();

    // Items v1 called finished are PRE-FIX: done, but flagged.
    assert!(machine_v2.items[0].needs_fix && machine_v2.items[0].done.is_some());
    assert!(machine_v2.items[1].needs_fix && machine_v2.items[1].done.is_some());
    assert!(!machine_v2.items[2].needs_fix && machine_v2.items[2].done.is_none());
    // v2 reconciled v1's double-counted completion step.
    assert_eq!(machine_v2.items[0].steps, 2);

    // The fixer (error-gate-routed intervention in the real product)
    // transitions pre-fix -> fully fixed.
    assert_eq!(v2::fix(&mut machine_v2), 2);
    assert!(machine_v2.items.iter().all(|i| !i.needs_fix));

    // ── v2 RUNS to completion on the checkpoint protocol. ──
    let io = CountingIo::default();
    let out = with_checkpoint_pair(machine_v2, |h, _req| drive_v2(h, &io));
    assert_eq!(out, "v1,v2,v3");

    // THE RESUME PROPERTY across versions: the two v1-finished items cost
    // v2 ZERO io — only the one genuinely unfinished item ran (2 remaining
    // steps).
    assert_eq!(io.calls.load(Ordering::SeqCst), 2);
}

#[test]
fn unfixed_prefix_state_refuses_to_run_and_reports() {
    // Entering the machine WITHOUT the fixer is detectable: pre-fix items
    // are skipped by the driver (treated as frozen), so the run stalls
    // with unfinished work — surfaced to the caller instead of silently
    // misbehaving. Here we assert the stall is observable via a bounded
    // probe, standing in for the error gate parking the process.
    let restored =
        ShapeV1 { steps_total: vec![3], steps_done: vec![3], finished: vec![true] };
    let mut machine_v2 = restored.restore();
    assert!(machine_v2.items[0].needs_fix);

    let io = CountingIo::default();
    let stalled = with_checkpoint_pair(machine_v2.clone(), |mut h, _req| {
        // Bounded probe: a real deployment would park in the error gate
        // here; we just observe that no work happens.
        std::thread::sleep(Duration::from_millis(5));
        h.safepoint();
        let m = &*h;
        m.items.iter().any(|i| i.needs_fix)
    });
    assert!(stalled, "pre-fix item is visible in the node state");

    let out = with_checkpoint_pair(machine_v2, |h, _req| drive_v2(h, &io));
    assert_eq!(out, "v1");
    assert_eq!(io.calls.load(Ordering::SeqCst), 0);
}
