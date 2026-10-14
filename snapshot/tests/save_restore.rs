
// The scenario literals below are test data (see header); they are
// the one class the inline_literal policy exempts via allow.
#![allow(unknown_lints)] // plain clippy does not know the driver's lints
#![allow(inline_literal)]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use servyi_states::checkpoint::{with_checkpoint_pair, Handle};
use servyi_states_snapshot as snapshot;



#[derive(Debug, Clone)]
struct Io {
    calls: Arc<AtomicUsize>,
    step_delay: Duration,
}

impl Io {
    fn new() -> Self {
        Self { calls: Arc::new(AtomicUsize::new(0)), step_delay: Duration::from_millis(0) }
    }

    fn step(&self, _id: u32) {
        let _n = self.calls.fetch_add(1, Ordering::SeqCst);
        if !self.step_delay.is_zero() {
            std::thread::sleep(self.step_delay);
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WorkItem {
    id: u32,
    steps: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnalyzeState {
    item: WorkItem,
    step: u32,
    /// Result stored by the worker INSIDE the node — the fan-out returns
    /// nothing; ownership of results stays with the state machine.
    done: Option<String>,
}

impl AnalyzeState {
    fn result(&self) -> (u32, String) {
        (self.item.id, self.done.clone().unwrap_or_default())
    }
}

fn run_analyze(mut h: Handle<'_, AnalyzeState>, io: &Io) {
    loop {
        h.safepoint();
        let (id, steps, already_done) = {
            let st = &*h;
            (st.item.id, st.item.steps, st.step >= st.item.steps)
        };
        // A restored child that was already finished does NO work —
        // this is the resume-skips-finished property under test.
        if already_done {
            h.done = Some(format!("v{id}"));
            return;
        }
        io.step(id);
        h.safepoint();
        let finished = {
            let st = &mut *h;
            st.step += 1;
            st.step >= steps
        };
        if finished {
            h.done = Some(format!("v{id}"));
            return;
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Group {
    gid: u32,
    children: Vec<AnalyzeState>,
}

impl Group {
    fn result(&self) -> (u32, String) {
        let ids: Vec<String> =
            self.children.iter().map(|c| c.done.clone().unwrap_or_default()).collect();
        (self.gid, ids.join(","))
    }
}

fn run_group(mut h: Handle<'_, Group>, io: &Io) {
    h.fanout(&|g| g.children.iter_mut().collect(), |a| run_analyze(a, io));
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum Stage {
    Seq { step: u32 },
    Scatter { children: Vec<AnalyzeState> },
    Nested { groups: Vec<Group> },
    Done,
}

impl Default for Stage {
    fn default() -> Self {
        Stage::Seq { step: 0 }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct Machine {
    items: Vec<WorkItem>,
    done: Vec<(u32, String)>,
    stage: Stage,
    result: String,
}

/// One top-level transition. The node is touched exclusively through the
/// root handle: every access borrows the handle, so nothing can be live
/// across the safepoints inside the fan-outs.
fn step<'o>(mut h: Handle<'o, Machine>, io: &Io) -> (Handle<'o, Machine>, bool) {
    let mut next: Option<Stage> = None;
    let keep_going = {
        let m = &mut *h;
        match &mut m.stage {
            Stage::Seq { step } => {
                if (*step as usize) < m.items.len() {
                    io.step(m.items[*step as usize].id);
                    *step += 1;
                    true
                } else {
                    next = Some(Stage::Scatter {
                        children: m
                            .items
                            .iter()
                            .map(|i| AnalyzeState { item: i.clone(), step: 0, done: None })
                            .collect(),
                    });
                    true
                }
            }
            Stage::Scatter { .. } => true,
            Stage::Nested { .. } => true,
            Stage::Done => false,
        }
    };
    if let Some(n) = next {
        h.stage = n;
    }
    if !keep_going {
        return (h, false);
    }
    // Fan-out stages: run while serving checkpoints, then read results
    // from the node. Which fan-out to run, without borrowing `h` for the
    // match arm (NLL limitation; Polonius would allow the direct match).
    enum Fan {
        Scatter,
        Nested,
    }
    // Seq advances (or Done exits) inside the block above; only the
    // fan-out stages reach here.
    let which = match &h.stage {
        Stage::Scatter { .. } => Fan::Scatter,
        Stage::Nested { .. } => Fan::Nested,
        Stage::Seq { .. } | Stage::Done => return (h, true),
    };
    match which {
        Fan::Scatter => h.fanout(
            &|m: &mut Machine| match &mut m.stage {
                Stage::Scatter { children } => children.iter_mut().collect(),
                _ => unreachable!("checked above"),
            },
            |a| run_analyze(a, io),
        ),
        Fan::Nested => h.fanout(
            &|m: &mut Machine| match &mut m.stage {
                Stage::Nested { groups } => groups.iter_mut().collect(),
                _ => unreachable!("checked above"),
            },
            |g| run_group(g, io),
        ),
    }
    let m = &mut *h;
    let mut collected: Vec<(u32, String)> = Vec::new();
    match &m.stage {
        Stage::Scatter { children } => collected.extend(children.iter().map(AnalyzeState::result)),
        Stage::Nested { groups } => collected.extend(groups.iter().map(Group::result)),
        _ => {}
    }
    // Stage transitions live with the collection they follow: Scatter
    // results feed the Nested groups (take/skip split, gids 100/200),
    // Nested results finish the machine. Dropped in the 7a03762
    // redesign — restored from 0acc997's semantics.
    let next: Option<Stage> = match &m.stage {
        Stage::Scatter { .. } => Some(Stage::Nested {
            groups: vec![
                Group {
                    gid: 100,
                    children: m
                        .items
                        .iter()
                        .take(2)
                        .map(|i| AnalyzeState { item: i.clone(), step: 0, done: None })
                        .collect(),
                },
                Group {
                    gid: 200,
                    children: m
                        .items
                        .iter()
                        .skip(2)
                        .map(|i| AnalyzeState { item: i.clone(), step: 0, done: None })
                        .collect(),
                },
            ],
        }),
        Stage::Nested { .. } => {
            let ids: Vec<String> = m.done.iter().map(|(i, _)| i.to_string()).collect();
            m.result = format!("done: {}", ids.join(","));
            Some(Stage::Done)
        }
        _ => None,
    };
    m.done.extend(collected);
    if let Some(n) = next {
        m.stage = n;
    }
    (h, true)
}

fn drive(
    machine: Machine,
    io: &Io,
    checkpoint: Option<(Duration, std::path::PathBuf)>,
) -> Machine {
    with_checkpoint_pair(machine, move |mut h, req| {
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stop2 = stop.clone();

        std::thread::scope(|scope| {
            let timer = checkpoint.map(|(every, path)| {
                let t = scope.spawn(move || {
                    while !stop2.load(Ordering::SeqCst) {
                        std::thread::sleep(every);
                        if stop2.load(Ordering::SeqCst) {
                            break;
                        }
                        let Some(guard) = req.request_timeout(Duration::from_millis(50)) else {
                            continue;
                        };
                        let bytes = serde_json::to_string(&*guard).expect("serialize");
                        drop(guard);
                        let tmp = path.with_extension("tmp");
                        if std::fs::write(&tmp, &bytes).is_ok() {
                            let _prev = std::fs::rename(&tmp, &path);
                        }
                    }
                });
                (t, every)
            });

            loop {
                h.safepoint();
                let (h2, keep_going) = step(h, io);
                h = h2;
                if !keep_going {
                    break;
                }
            }

            stop.store(true, Ordering::SeqCst);
            while h.is_stop_requested() {
                h.safepoint();
            }
            if let Some((t, _)) = timer {
                if let Err(payload) = t.join() {
                    std::panic::resume_unwind(payload);
                }
            }
            // Snapshot the final state out of the node through the handle.
            let out: Machine = h.clone();
            out
        })
    })
}

fn tempdir() -> (tempfile::TempDir, std::path::PathBuf) {
    let d = tempfile::tempdir().expect("tempdir");
    let p = d.path().to_path_buf();
    (d, p)
}

#[test]
fn save_restore_roundtrip() {
    let (_g, dir) = tempdir();
    let path = dir.join("snap.json");
    let m = Machine {
        items: (1..=3).map(|id| WorkItem { id, steps: 2 }).collect(),
        ..Machine::default()
    };
    // Run to completion, snapshotting at the end of the pair closure.
    let out = with_checkpoint_pair(m, |mut h, _req| {
        loop {
            h.safepoint();
            let (h2, go) = step(h, &Io::new());
            h = h2;
            if !go {
                break;
            }
        }
        // Mutator-side save at a safepoint: no handshake, cannot block.
        snapshot::save_handle(&h, &path).expect("save at end");
        h.clone()
    });
    let restored: Machine = snapshot::restore_from(&path).expect("restore");
    assert_eq!(out.done, restored.done);
    assert_eq!(restored.result, out.result);
}

#[test]
fn snapshot_mid_fanout_resumes_without_rerunning_finished_children() {
    let (_g, dir) = tempdir();
    let path = dir.join("mid.json");

    // Deterministic mid-fan-out moment: children 1-2 run to completion and
    // then hold; children 3-4 hold before their first step. The snapshot
    // fires in between (2 done, 2 not started), gated by a barrier the
    // snapper releases after its first successful write.
    let snap_taken = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let snap_taken_ref = &snap_taken;
    // The snapper must not fire before BOTH early children finished.
    let early_done = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let early_done_ref = &early_done;
    let slow_io = Io { step_delay: Duration::from_millis(1), ..Io::new() };

    let m = Machine {
        items: (1..=4).map(|id| WorkItem { id, steps: 6 }).collect(),
        ..Machine::default()
    };
    with_checkpoint_pair(m, |mut h, req| {
        std::thread::scope(|scope| {
            let path2 = path.clone();
            let snapper = scope.spawn(move || {
                // ONE snapshot, taken while the barrier holds the children
                // mid-fan-out — AFTER both early children finished, and
                // before the late ones start. The file must keep exactly
                // that state, so the snapper exits after the write.
                while early_done_ref.load(Ordering::SeqCst) < 2 {
                    std::thread::sleep(Duration::from_millis(1));
                }
                while !snapshot::try_save_requester(&req, &path2, Duration::from_millis(50))? {
                    std::thread::sleep(Duration::from_millis(1));
                }
                snap_taken_ref.store(true, Ordering::SeqCst);
                Ok::<_, std::io::Error>(())
            });
            // Seq stage up to Scatter — hand-rolled: `step` would run the
            // fan-outs internally, bypassing this test's barrier action.
            loop {
                h.safepoint();
                let advanced = {
                    let m = &mut *h;
                    match &mut m.stage {
                        Stage::Seq { step } if (*step as usize) < m.items.len() => {
                            *step += 1;
                            true
                        }
                        Stage::Seq { .. } => {
                            m.stage = Stage::Scatter {
                                children: m
                                    .items
                                    .iter()
                                    .map(|i| AnalyzeState {
                                        item: i.clone(),
                                        step: 0,
                                        done: None,
                                    })
                                    .collect(),
                            };
                            true
                        }
                        _ => false,
                    }
                };
                if !advanced || matches!(h.stage, Stage::Scatter { .. }) {
                    break;
                }
            }
            // Scatter fan-out with the barrier action.
            if matches!(h.stage, Stage::Scatter { .. }) {
                h.fanout(
                    &|mm: &mut Machine| match &mut mm.stage {
                        Stage::Scatter { children } => children.iter_mut().collect(),
                        _ => unreachable!(),
                    },
                    |mut a| {
                        let early = a.item.id <= 2;
                        if early {
                            // Complete ALL steps first…
                            for _ in 0..a.item.steps {
                                a.safepoint();
                                slow_io.step(a.item.id);
                                a.safepoint();
                                a.step += 1;
                            }
                            a.done = Some(format!("v{}", a.item.id));
                            let _prev = early_done_ref.fetch_add(1, Ordering::SeqCst);
                        }
                        // …then everyone holds until the snapshot fired.
                        while !snap_taken_ref.load(Ordering::SeqCst) {
                            std::thread::sleep(Duration::from_millis(1));
                            a.safepoint();
                        }
                        if !early {
                            // Not-started children now run AFTER the snapshot.
                            run_analyze(a, &slow_io);
                        }
                    },
                );
            }
            let _ = snapper.join().expect("snapshotter");
        });
    });

    // The snapshot exists (the barrier guarantees at least one write) and
    // records exactly two children finished.
    let snap: Machine = snapshot::restore_from(&path).expect("snapshot parses");
    let unfinished_at_snap: usize = match &snap.stage {
        Stage::Scatter { children } => {
            children.iter().filter(|c| c.done.is_none()).count()
        }
        other => unreachable!("snapshot should catch the Scatter stage, got {other:?}"),
    };
    assert_eq!(unfinished_at_snap, 2, "2 done + 2 not started at snapshot");

    // RESUME from the mid-flight snapshot and run to completion.
    let io2 = Io::new();
    let resumed = drive(snap, &io2, None);
    let mut ids: Vec<u32> = resumed.done.iter().map(|(i, _)| *i).collect();
    ids.sort();
    assert_eq!(ids, vec![1, 2, 3, 4, 100, 200], "{}", resumed.result);

    // THE PROPERTY: children already done at snapshot time were NOT
    // re-run — zero IO for them. The unfinished children run ≤ 6 steps
    // each; the Nested stage (fresh machines, 4 children × 6 steps) is
    // the only other IO the resumed run performs.
    let calls = io2.calls.load(Ordering::SeqCst);
    let bound = unfinished_at_snap * 6 + 4 * 6;
    assert!(
        calls <= bound,
        "resume must skip finished children: {calls} calls, bound {bound}"
    );
}
