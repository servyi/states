use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use servyi_states::checkpoint::{with_checkpoint_pair, CheckpointRequester, Handle};

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
        let (id, steps) = {
            let st = h.block_and_get();
            (st.item.id, st.item.steps)
        };
        io.step(id);
        h.safepoint();
        let finished = {
            let st = h.block_and_get_mut();
            st.step += 1;
            st.step >= steps
        };
        if finished {
            h.block_and_get_mut().done = Some(format!("v{id}"));
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
    h.fanout(|g| g.children.iter_mut(), |a| run_analyze(a, io));
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
        let m = h.block_and_get_mut();
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
        h.block_and_get_mut().stage = n;
    }
    if !keep_going {
        return (h, false);
    }
    // Fan-out stages: run while serving checkpoints, then read results
    // from the node.
    let which = match &h.block_and_get().stage {
        Stage::Scatter { .. } => 0,
        Stage::Nested { .. } => 1,
        _ => -1,
    };
    match which {
        0 => h.fanout(
            |m| match &mut m.stage {
                Stage::Scatter { children } => children.iter_mut(),
                _ => unreachable!("checked above"),
            },
            |a| run_analyze(a, io),
        ),
        1 => h.fanout(
            |m| match &mut m.stage {
                Stage::Nested { groups } => groups.iter_mut(),
                _ => unreachable!("checked above"),
            },
            |g| run_group(g, io),
        ),
        _ => {}
    }
    let m = h.block_and_get_mut();
    let mut collected: Vec<(u32, String)> = Vec::new();
    match &m.stage {
        Stage::Scatter { children } => collected.extend(children.iter().map(AnalyzeState::result)),
        Stage::Nested { groups } => {
            collected.extend(groups.iter().map(Group::result));
            let ids: Vec<String> = m.done.iter().map(|(i, _)| i.to_string()).collect();
            m.result = format!("done: {}", ids.join(","));
            m.stage = Stage::Done;
        }
        _ => {}
    }
    m.done.extend(collected);
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
                        let bytes = serde_json::to_string(guard.node()).expect("serialize");
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
            let out: Machine = h.block_and_get().clone();
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
fn freeze_window_allows_access_then_safepoint() {
    let out = with_checkpoint_pair(vec![7u32], |mut h, _req| {
        h.fanout(|c| c.iter_mut(), |mut child| {
            assert_eq!(*child.block_and_get(), 7);
            child.safepoint();
            *child.block_and_get_mut() += 1;
        });
        h.block_and_get().clone()
    });
    assert_eq!(out, vec![8]);
}

#[test]
fn snapshotter_reads_live_tree_through_guard() {
    let (observed, final_v) = with_checkpoint_pair(41u32, |h, req| {
        let mut h = h;
        let (observed, final_v) = std::thread::scope(|scope| {
            let mutifier = scope.spawn(move || {
                let mut v = 41;
                for _ in 0..50 {
                    h.safepoint();
                    *h.block_and_get_mut() += 1;
                    v = *h.block_and_get();
                    std::thread::sleep(Duration::from_millis(1));
                }
                v
            });
            let mut observed = Vec::new();
            for _ in 0..5 {
                std::thread::sleep(Duration::from_millis(2));
                let Some(guard) = req.request() else {
                    continue;
                };
                observed.push(*guard.node());
                drop(guard);
            }
            (observed, mutifier.join().expect("mutifier"))
        });
        (observed, final_v)
    });
    assert_eq!(final_v, 91);
    for w in observed.windows(2) {
        assert!(w[0] <= w[1], "monotonic guard observations: {observed:?}");
    }
    assert!(observed.iter().all(|v| (41..=91).contains(v)));
}

#[test]
fn concurrent_requesters_serialize_instead_of_deadlocking() {
    with_checkpoint_pair(0u32, |h, req: CheckpointRequester<'_, u32>| {
        let mut h = h;
        let mutifier_done = std::sync::atomic::AtomicBool::new(false);
        let done_ref = &mutifier_done;
        std::thread::scope(|scope| {
            let _mutifier = scope.spawn(move || {
                for _ in 0..200 {
                    h.safepoint();
                    *h.block_and_get_mut() += 1;
                }
                done_ref.store(true, Ordering::SeqCst);
            });
            let readers: Vec<_> = (0..2)
                .map(|_| {
                    scope.spawn(|| {
                        let mut last = 0;
                        for _ in 0..20 {
                            if let Some(guard) = req.request_timeout(Duration::from_millis(500)) {
                                let v = *guard.node();
                                drop(guard);
                                assert!(v >= last, "observations must be monotonic per reader");
                                last = v;
                            }
                        }
                    })
                })
                .collect();
            for r in readers {
                r.join().expect("reader");
            }
            assert!(mutifier_done.load(Ordering::SeqCst), "mutator must not be starved");
        });
    });
}

#[test]
fn full_run_with_stw_checkpoints_and_nested_fanout() {
    let (_g, dir) = tempdir();
    let path = dir.join("cp.json");
    let io = Io { step_delay: Duration::from_millis(1), ..Io::new() };
    let m = Machine {
        items: (1..=4).map(|id| WorkItem { id, steps: 2 }).collect(),
        ..Machine::default()
    };
    let start = Instant::now();
    let m = drive(m, &io, Some((Duration::from_millis(2), path.clone())));
    let elapsed = start.elapsed();

    let mut ids: Vec<u32> = m.done.iter().map(|(i, _)| *i).collect();
    ids.sort();
    assert_eq!(ids, vec![1, 2, 3, 4, 100, 200], "{}", m.result);
    assert!(path.exists(), "checkpoints written");
    assert!(elapsed.as_millis() < 2000, "scoped threads run concurrently: {elapsed:?}");
}

#[test]
fn mid_run_snapshot_resumes() {
    let (_g, dir) = tempdir();
    let path = dir.join("cp.json");
    let io = Io { step_delay: Duration::from_millis(4), ..Io::new() };

    let m = Machine {
        items: (1..=4).map(|id| WorkItem { id, steps: 4 }).collect(),
        ..Machine::default()
    };
    std::thread::scope(|s| {
        s.spawn(|| drive(m, &io, Some((Duration::from_millis(3), path.clone()))));
        std::thread::sleep(Duration::from_millis(60));
    });

    let snap = std::fs::read_to_string(&path).expect("mid-run snapshot exists");
    let resumed: Machine = serde_json::from_str(&snap).expect("snapshot parses");
    let io2 = Io::new();
    let resumed = drive(resumed, &io2, None);
    let mut ids: Vec<u32> = resumed.done.iter().map(|(i, _)| *i).collect();
    ids.sort();
    assert_eq!(ids, vec![1, 2, 3, 4, 100, 200], "{}", resumed.result);
    assert!(io2.calls.load(Ordering::SeqCst) < 4 * 4 + 2 * 4);
}
