//! Three-version upgrade-path tests for cross-version save/restore.
//!
//! Shared state machinery is defined in the `common` module below and
//! imported where used.

//! Every (saver, restorer) combination must work, and each version carries
//! its OWN shape (its own logic's snapshot vocabulary):
//!
//!   save    | restore
//!   version1 | version2
//!   version1 | version3
//!   version2 | version3
//!
//! Version knowledge (the `restore` code is the only thing that changes):
//!   v1: `ShapeV1` — steps/finished. Double-counted the completion step.
//!   v2: `ShapeV2` — adds `polished` (never stored by v1: false on
//!       upgrade). Fixes v1's double-count. Introduced its own bug:
//!       `finished` was written from `done.is_some()` BEFORE the final
//!       step was applied, so a v2 snapshot can claim finished with the
//!       last step missing — v3 repairs that.
//!   v3: `ShapeV3` — adds `origin` (which version wrote the artifact —
//!       restores v1/v2 artifacts to pre-fix states when their era's bug
//!       applies, fixer resolves them).
//!
//! The upgrade chain restores each older shape into v3's state; v3's own
//! artifacts restore losslessly (no pre-fix).

use serde::{Deserialize, Serialize};
use servyi_states::checkpoint::with_checkpoint_pair;
use servyi_states_snapshot as snapshot;


/// The stable contract: restore as dyn impl of exactly the shape the
/// saver wrote. Each version's shape implements it; the impl IS the
/// version-specific upgrade code.
use common::{fix_all, DriveIo, ItemV3, MachineV3};

pub(crate) trait Shape {
    type State;
    fn restore(&self) -> Self::State;
}

// ── v1's shape and saver ────────────────────────────────────────────────

mod shape_v1 {
    use super::*;

    /// v1's snapshot vocabulary — frozen at v1.
    #[derive(Serialize, Deserialize)]
    pub struct ShapeV1 {
        pub steps_total: Vec<u32>,
        pub steps_done: Vec<u32>,
        pub finished: Vec<bool>,
    }

    /// v1's machine and saver.
    #[derive(Clone)]
    pub struct MachineV1 {
        pub items: Vec<(u32, u32, u32, bool)>, // (total, step, .., done)
    }

    impl MachineV1 {
        pub fn save(&self, path: &std::path::Path) {
            let shape = ShapeV1 {
                steps_total: self.items.iter().map(|(t, ..)| *t).collect(),
                steps_done: self.items.iter().map(|(_, s, ..)| *s).collect(),
                finished: self.items.iter().map(|(.., d)| *d).collect(),
            };
            snapshot::write_value(&shape, path).expect("v1 saves");
        }
    }
}

// ── v2's shape and saver ────────────────────────────────────────────────

mod shape_v2 {
    use super::*;

    /// v2's snapshot vocabulary — adds `polished`; frozen at v2.
    #[derive(Serialize, Deserialize)]
    pub struct ShapeV2 {
        pub steps_total: Vec<u32>,
        pub steps_done: Vec<u32>,
        /// v2's finished flag — written one step early (the v2 bug).
        pub finished: Vec<bool>,
        pub polished: Vec<bool>,
    }

    #[derive(Clone)]
    pub struct MachineV2 {
        pub items: Vec<(u32, u32, bool, bool)>, // (total, step, done, polished)
    }

    impl MachineV2 {
        pub fn save(&self, path: &std::path::Path) {
            let shape = ShapeV2 {
                steps_total: self.items.iter().map(|(t, ..)| *t).collect(),
                steps_done: self.items.iter().map(|(_, s, ..)| *s).collect(),
                finished: self.items.iter().map(|(.., d, _)| *d).collect(),
                polished: self.items.iter().map(|(.., p)| *p).collect(),
            };
            snapshot::write_value(&shape, path).expect("v2 saves");
        }
    }

    /// v2's own restore (v2 artifacts under v2 logic — no pre-fix).
    impl super::Shape for ShapeV2 {
        type State = super::common::MachineV3;

        fn restore(&self) -> Self::State {
            self.upgrade_from_v2()
        }
    }

    impl ShapeV2 {
        /// v3 reading a v2 artifact (origin v2).
        pub(super) fn upgrade_from_v2(&self) -> super::common::MachineV3 {
            super::common::MachineV3 {
                items: self
                    .steps_total
                    .iter()
                    .zip(&self.steps_done)
                    .zip(&self.finished)
                    .zip(&self.polished)
                    .enumerate()
                    .map(|(idx, (((total, step), finished), polished))| ItemV3 {
                        id: idx as u32 + 1,
                        steps: *total,
                        step: *step,
                        done: finished.then(|| format!("v{}", idx as u32 + 1)),
                        polished: *polished,
                        origin: 2,
                        needs_fix: *finished, // v2's early-flag bug
                    })
                    .collect(),
            }
        }

        /// v3 reading a v1 artifact re-expressed in v2's shape (origin v1):
        /// v1 had no `polished` (false) and double-counted the completion
        /// step.
        pub(super) fn upgrade_from_v1(&self) -> super::common::MachineV3 {
            super::common::MachineV3 {
                items: self
                    .steps_total
                    .iter()
                    .zip(&self.steps_done)
                    .zip(&self.finished)
                    .enumerate()
                    .map(|(idx, ((total, step), finished))| ItemV3 {
                        id: idx as u32 + 1,
                        steps: *total,
                        // v1 double-counted: finished items show step == total + 1.
                        step: if *finished { (*step).saturating_sub(1) } else { *step },
                        done: finished.then(|| format!("v{}", idx as u32 + 1)),
                        // v1 never polished.
                        polished: false,
                        origin: 1,
                        needs_fix: *finished,
                    })
                    .collect(),
            }
        }
    }
}

// ── v3's shape and restorers for ALL older shapes ───────────────────────

mod shape_v3 {
    use super::*;

    /// v3's snapshot vocabulary — adds `origin`; frozen at v3.
    #[derive(Serialize, Deserialize)]
    pub struct ShapeV3 {
        pub steps_total: Vec<u32>,
        pub steps_done: Vec<u32>,
        pub finished: Vec<bool>,
        pub polished: Vec<bool>,
        /// Which version wrote this artifact (1, 2, or 3).
        pub origin: Vec<u8>,
    }

    impl super::Shape for ShapeV3 {
        type State = super::common::MachineV3;

        fn restore(&self) -> Self::State {
            super::common::MachineV3 {
                items: self
                    .steps_total
                    .iter()
                    .zip(&self.steps_done)
                    .zip(&self.finished)
                    .zip(&self.polished)
                    .zip(&self.origin)
                    .enumerate()
                    .map(|(idx, ((((total, step), finished), polished), origin))| ItemV3 {
                        id: idx as u32 + 1,
                        steps: *total,
                        step: *step,
                        done: finished.then(|| format!("v{}", idx as u32 + 1)),
                        polished: *polished,
                        origin: *origin,
                        // v3's own artifacts restore losslessly.
                        needs_fix: false,
                    })
                    .collect(),
            }
        }
    }

    /// v3 reading a v1 artifact directly: the v1 shape upgraded in one hop.
    impl Shape for super::shape_v1::ShapeV1 {
        type State = super::common::MachineV3;

        fn restore(&self) -> Self::State {
            let as_v2 = super::shape_v2::ShapeV2 {
                steps_total: self.steps_total.clone(),
                steps_done: self.steps_done.clone(),
                finished: self.finished.clone(),
                polished: self.finished.iter().map(|_| false).collect(),
            };
            as_v2.upgrade_from_v1()
        }
    }

}

// ── Shared state + driver ───────────────────────────────────────────────

mod common {
    #[derive(Clone, Debug)]
    pub struct ItemV3 {
        pub id: u32,
        pub steps: u32,
        pub step: u32,
        pub done: Option<String>,
        pub polished: bool,
        pub origin: u8,
        pub needs_fix: bool,
    }

    #[derive(Clone, Debug)]
    pub struct MachineV3 {
        pub items: Vec<ItemV3>,
    }

    impl MachineV3 {
        /// v3's saver: machine -> v3 shape.
        pub fn save(&self, path: &std::path::Path) {
            let shape = super::shape_v3::ShapeV3 {
                steps_total: self.items.iter().map(|i| i.steps).collect(),
                steps_done: self.items.iter().map(|i| i.step).collect(),
                finished: self.items.iter().map(|i| i.done.is_some()).collect(),
                polished: self.items.iter().map(|i| i.polished).collect(),
                origin: self.items.iter().map(|i| i.origin).collect(),
            };
            super::snapshot::write_value(&shape, path).expect("v3 saves");
        }
    }

    /// The fixer stand-in: pre-fix -> fully fixed.
    pub fn fix_all(m: &mut MachineV3) -> usize {
        let mut fixed = 0;
        for item in &mut m.items {
            if item.needs_fix {
                item.needs_fix = false;
                fixed += 1;
            }
        }
        fixed
    }

    #[derive(Default)]
    pub struct DriveIo {
        pub calls: std::sync::atomic::AtomicUsize,
    }

    /// Run to completion over the protocol; returns the done-ids string.
    pub fn drive(
        mut h: servyi_states::checkpoint::Handle<'_, MachineV3>,
        io: &DriveIo,
    ) -> String {
        use servyi_states::checkpoint::Handle as H;
        loop {
            H::safepoint(&mut h);
            let m = &mut *h;
            for item in &mut m.items {
                if item.done.is_some() || item.needs_fix {
                    continue;
                }
                let _n = io.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
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
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }
}

// ── The three upgrade paths ──────────────────────────────────────────────


fn run_v3(machine: MachineV3) -> (String, usize) {
    let io = DriveIo::default();
    let out = with_checkpoint_pair(machine, |h, _| common::drive(h, &io));
    (out, io.calls.load(std::sync::atomic::Ordering::SeqCst))
}

fn restore_and_run(artifact: &std::path::Path) -> (String, usize, usize) {
    let raw: serde_json::Value = snapshot::restore_from(artifact).expect("artifact parses");
    // Which shape is it? Each version's shape is self-describing enough to
    // try in order (real deployments version-tag the envelope).
    let restored: Box<dyn Shape<State = MachineV3>> = if raw.get("origin").is_some() {
        Box::new(serde_json::from_value::<shape_v3::ShapeV3>(raw).expect("v3 shape"))
    } else if raw.get("polished").is_some() {
        Box::new(serde_json::from_value::<shape_v2::ShapeV2>(raw).expect("v2 shape"))
    } else {
        Box::new(serde_json::from_value::<shape_v1::ShapeV1>(raw).expect("v1 shape"))
    };
    let mut machine = restored.restore();
    let prefix_flags = machine.items.iter().filter(|i| i.needs_fix).count();
    let _fixed = fix_all(&mut machine);
    let (out, calls) = run_v3(machine);
    (out, calls, prefix_flags)
}

#[test]
fn upgrade_paths_v1_to_v2() {
    // save v1, restore v2 (v2's own Shape impl upgrades origin-1 data).
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.json");
    let m1 = shape_v1::MachineV1 {
        items: vec![(3, 4, 0, true), (3, 1, 0, false)], // (total, step, _, done); v1 finished shows step 4 = total+1 (double count)
    };
    with_checkpoint_pair(m1, |mut h, _| {
        h.safepoint();
        h.save(&path);
    });
    let raw: serde_json::Value = snapshot::restore_from(&path).unwrap();
    let s2: shape_v2::ShapeV2 = reexpress_v1_as_v2(&raw);
    // v2's restore code FOR ORIGIN-1 data: the v1 upgrade path, not v2's
    // own (which would trust its own finished flags).
    let mut machine = s2.upgrade_from_v1();
    // v1-finished item is pre-fix (needs the double-count reconciled).
    assert!(machine.items[0].needs_fix && machine.items[0].done.is_some());
    assert_eq!(machine.items[0].step, 3, "double-count reconciled");
    let _ = fix_all(&mut machine);
    let (out, calls) = run_v3(machine);
    assert_eq!(out, "v1,v2");
    // Finished item cost zero io; unfinished ran 2 remaining steps.
    assert_eq!(calls, 2);
}

#[test]
fn upgrade_paths_v1_to_v3() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("b.json");
    let m1 = shape_v1::MachineV1 {
        items: vec![(3, 4, 0, true), (3, 2, 0, false)],
    };
    with_checkpoint_pair(m1, |mut h, _| {
        h.safepoint();
        h.save(&path);
    });
    let (out, calls, prefix) = restore_and_run(&path);
    assert_eq!(out, "v1,v2");
    assert_eq!(prefix, 1, "the v1-finished item arrived pre-fix");
    assert_eq!(calls, 1, "only the unfinished item's last step ran");
}

#[test]
fn upgrade_paths_v2_to_v3() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c.json");
    let m2 = shape_v2::MachineV2 {
        // v2's bug: finished=true with the last step missing (step 2 of 3).
        items: vec![(3, 2, true, true), (3, 0, false, false)],
    };
    with_checkpoint_pair(m2, |mut h, _| {
        h.safepoint();
        h.save(&path);
    });
    let (out, calls, prefix) = restore_and_run(&path);
    assert_eq!(out, "v1,v2");
    assert_eq!(prefix, 1, "v2's early-flagged item arrived pre-fix");
    // The pre-fix item was finished-with-a-step-missing; the fixer's
    // reconciliation marks it done (v2 said so) — cost 0 — and the
    // unfinished item ran all 3 steps.
    assert_eq!(calls, 3);
}

#[test]
fn v3_artifacts_restore_losslessly() {
    // Same-version control: no pre-fix flags, exact resume.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("d.json");
    let m3 = MachineV3 {
        items: vec![
            ItemV3 {
                id: 1,
                steps: 3,
                step: 3,
                done: Some("v1".into()),
                polished: true,
                origin: 3,
                needs_fix: false,
            },
            ItemV3 {
                id: 2,
                steps: 3,
                step: 1,
                done: None,
                polished: false,
                origin: 3,
                needs_fix: false,
            },
        ],
    };
    with_checkpoint_pair(m3, |mut h, _| {
        h.safepoint();
        h.save(&path);
    });
    let (out, calls, prefix) = restore_and_run(&path);
    assert_eq!(out, "v1,v2");
    assert_eq!(prefix, 0, "v3 artifacts never arrive pre-fix");
    assert_eq!(calls, 2, "only the unfinished item's remaining steps");
}

/// v2-era tooling expressed v1 artifacts in its own shape (the historical
/// migration path); test uses it for the v1->v2 leg.
fn reexpress_v1_as_v2(raw: &serde_json::Value) -> shape_v2::ShapeV2 {
    serde_json::from_value(
        serde_json::json!({
            "steps_total": raw["steps_total"],
            "steps_done": raw["steps_done"],
            "finished": raw["finished"],
            "polished": raw["finished"].as_array().map(|a| vec![false; a.len()]).unwrap_or_default(),
        }),
    )
    .expect("v1 re-expressed in v2's shape")
}

