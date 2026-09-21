use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use servyi_states::checkpoint::{
    fanout, fanout_handle, CheckpointRequester, Checkpointer,
};

#[derive(Debug, Clone)]
struct Io {
    calls: Arc<AtomicUsize>,
    step_delay: Duration,
}

impl Io {
    fn new() -> Self {
        Self { calls: Arc::new(AtomicUsize::new(0)), step_delay: Duration::from_millis(0) }
    }

    fn step(&self, id: u32) {
        self.calls.fetch_add(1, Ordering::SeqCst);
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
}

fn run_analyze<'s>(
    mut h: servyi_states::checkpoint::Handle<'s, AnalyzeState>,
    mut cp: Checkpointer<'s>,
    io: &Io,
) -> (u32, String) {
    loop {
        cp.safepoint();
        let (id, steps) = {
            let st = cp.block_and_get(&h);
            (st.item.id, st.item.steps)
        };
        io.step(id);
        cp.safepoint();
        let finished = {
            let st = cp.block_and_get_mut(&mut h);
            st.step += 1;
            st.step >= steps
        };
        if finished {
            return (id, format!("v{id}"));
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Group {
    gid: u32,
    children: Vec<AnalyzeState>,
}

fn run_group<'s>(
    mut h: servyi_states::checkpoint::Handle<'s, Group>,
    mut cp: Checkpointer<'s>,
    io: &Io,
) -> (u32, String) {
    cp.safepoint();
    let gid = cp.block_and_get(&h).gid;
    let rs = fanout_handle(&mut h, &mut cp, |g: &mut Group| g.children.iter_mut(), |a, acp| {
        run_analyze(a, acp, io)
    });
    let ids: Vec<String> = rs.iter().map(|(i, _)| i.to_string()).collect();
    (gid, ids.join(","))
}

#[derive(Debug, Serialize, Deserialize)]
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

#[derive(Debug, Default, Serialize, Deserialize)]
struct Machine {
    items: Vec<WorkItem>,
    done: Vec<(u32, String)>,
    stage: Stage,
    result: String,
}

fn step(m: &mut Machine, cp: &mut Checkpointer<'_>, io: &Io) -> bool {
    let mut next: Option<Stage> = None;
    let keep_going = match &mut m.stage {
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
                        .map(|i| AnalyzeState { item: i.clone(), step: 0 })
                        .collect(),
                });
                true
            }
        }
        Stage::Scatter { children } => {
            let rs = fanout(children, cp, |c| c.iter_mut(), |h, c| run_analyze(h, c, io));
            m.done.extend(rs);
            next = Some(Stage::Nested {
                groups: vec![
                    Group {
                        gid: 100,
                        children: m
                            .items
                            .iter()
                            .take(2)
                            .map(|i| AnalyzeState { item: i.clone(), step: 0 })
                            .collect(),
                    },
                    Group {
                        gid: 200,
                        children: m
                            .items
                            .iter()
                            .skip(2)
                            .map(|i| AnalyzeState { item: i.clone(), step: 0 })
                            .collect(),
                    },
                ],
            });
            true
        }
        Stage::Nested { groups } => {
            let rs = fanout(groups, cp, |g| g.iter_mut(), |h, c| run_group(h, c, io));
            m.done.extend(rs);
            let ids: Vec<String> = m.done.iter().map(|(i, _)| i.to_string()).collect();
            m.result = format!("done: {}", ids.join(","));
            next = Some(Stage::Done);
            true
        }
        Stage::Done => false,
    };
    if let Some(n) = next {
        m.stage = n;
    }
    keep_going
}

fn drive(
    machine: &mut Machine,
    io: &Io,
    checkpoint: Option<(Duration, std::path::PathBuf)>,
) {
    let mut cp = Checkpointer::new();
    // SAFETY: `machine` is borrowed for the whole of `drive`, and the only
    // mutators are this thread (between `safepoint` calls) and, during
    // fan-outs, parked subtasks. The requester dies with this frame.
    let requester =
        unsafe { CheckpointRequester::from_node_ptr(machine as *const Machine, &cp) };
    let req = requester.clone();

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
                        let _ = std::fs::rename(&tmp, &path);
                    }
                }
            });
            (t, every)
        });

        loop {
            cp.safepoint();
            if !step(machine, &mut cp, io) {
                break;
            }
        }

        stop.store(true, Ordering::SeqCst);
        while cp.is_stop_requested() {
            cp.safepoint();
        }
        if let Some((t, _)) = timer {
            t.join().ok();
        }
    });
}

fn tempdir() -> (tempfile::TempDir, std::path::PathBuf) {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().to_path_buf();
    (d, p)
}

#[test]
fn freeze_window_allows_access_then_safepoint() {
    let mut cp = Checkpointer::new();
    let mut children = vec![7u32];
    let out = fanout(&mut children, &mut cp, |c| c.iter_mut(), |mut h, mut ccp| {
        {
            let a = ccp.block_and_get(&h);
            assert_eq!(*a, 7);
        }
        ccp.safepoint();
        *ccp.block_and_get_mut(&mut h)
    });
    assert_eq!(out, vec![7]);
}

#[test]
fn snapshotter_reads_live_tree_through_guard() {
    let mut machine = 41u32;
    let mut cp = Checkpointer::new();
    let req = unsafe { CheckpointRequester::from_node_ptr(&machine as *const u32, &cp) };
    let (observed, final_v) = std::thread::scope(|scope| {
        let mutifier = scope.spawn(|| {
            for _ in 0..50 {
                cp.safepoint();
                machine += 1;
                std::thread::sleep(Duration::from_millis(1));
            }
            machine
        });
        let mut observed = Vec::new();
        for _ in 0..5 {
            std::thread::sleep(Duration::from_millis(2));
            let guard = req.request();
            observed.push(*guard.node());
            drop(guard);
        }
        (observed, mutifier.join().unwrap())
    });
    assert_eq!(final_v, 91);
    for w in observed.windows(2) {
        assert!(w[0] <= w[1], "monotonic guard observations: {observed:?}");
    }
    assert!(observed.iter().all(|v| (41..=91).contains(v)));
}

#[test]
fn full_run_with_stw_checkpoints_and_nested_fanout() {
    let (_g, dir) = tempdir();
    let path = dir.join("cp.json");
    let io = Io { step_delay: Duration::from_millis(1), ..Io::new() };
    let mut m = Machine {
        items: (1..=4).map(|id| WorkItem { id, steps: 2 }).collect(),
        ..Machine::default()
    };
    let start = Instant::now();
    drive(&mut m, &io, Some((Duration::from_millis(2), path.clone())));
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

    let mut m = Machine {
        items: (1..=4).map(|id| WorkItem { id, steps: 4 }).collect(),
        ..Machine::default()
    };
    std::thread::scope(|s| {
        s.spawn(|| drive(&mut m, &io, Some((Duration::from_millis(3), path.clone()))));
        std::thread::sleep(Duration::from_millis(60));
    });

    let snap = std::fs::read_to_string(&path).expect("mid-run snapshot exists");
    let mut resumed: Machine = serde_json::from_str(&snap).expect("snapshot parses");
    let io2 = Io::new();
    drive(&mut resumed, &io2, None);
    let mut ids: Vec<u32> = resumed.done.iter().map(|(i, _)| *i).collect();
    ids.sort();
    assert_eq!(ids, vec![1, 2, 3, 4, 100, 200], "{}", resumed.result);
    assert!(io2.calls.load(Ordering::SeqCst) < 4 * 4 + 2 * 4);
}
