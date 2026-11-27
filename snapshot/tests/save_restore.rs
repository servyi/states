// The snapshot crate's integration tests: the roundtrip and the
// mid-fan-out snapshot+resume, rebuilt on the two-piece protocol with
// per-type vocab traits (the harness shape that compiles cleanly on
// this toolchain).
#![allow(unknown_lints)] // plain clippy does not know the driver's lints
#![allow(inline_literal)]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use servyi_states::checkpoint::{
    Checkpointer, InnerHandle, OuterHandle, PersistentCell, independent_task_root,
};
use servyi_states_snapshot as snapshot;

#[path = "../../tests/support/deadline.rs"]
mod deadline;

/// Steps per work item in the mid-run tests.
const WORK_STEPS_PER_ITEM: u32 = 6;

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
    done: Option<String>,
}

impl AnalyzeState {
    fn result(&self) -> (u32, String) {
        (self.item.id, self.done.clone().unwrap_or_default())
    }
}

type AnalyzeSlot = Box<PersistentCell<AnalyzeState>>;
type GroupSlot = Box<PersistentCell<Group>>;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Group {
    gid: u32,
    children: Vec<PersistentCell<AnalyzeSlot>>,
}

impl Group {
    fn result(&self) -> (u32, String) {
        let ids: Vec<String> =
            self.children.iter().map(|c| c.done.clone().unwrap_or_default()).collect();
        (self.gid, ids.join(","))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum Stage {
    Seq { step: u32 },
    Scatter { children: Vec<PersistentCell<AnalyzeSlot>> },
    Nested { groups: Vec<PersistentCell<GroupSlot>> },
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

/// The tree's vocab: analyze nodes run to completion; groups run their
/// children; the machine runs one stage transition.
trait VocabIo: Send {
    fn run(&mut self, io: &Io);
    /// The inner-machine reach (the mid-fan-out test's hand-rolled
    /// fan-out); None on non-machine nodes.
    fn machine(&mut self) -> Option<&mut InnerHandle<Machine>>;
    /// The mid-fan-out barrier: early children complete then hold;
    /// late children hold then run — everything until the snapshot.
    fn barrier_run(
        &mut self,
        io: &Io,
        snap_taken: &std::sync::atomic::AtomicBool,
        early_done: &AtomicUsize,
    ) {
        let _ = (io, snap_taken, early_done);
    }
}


impl VocabIo for InnerHandle<AnalyzeState> {
    fn machine(&mut self) -> Option<&mut InnerHandle<Machine>> {
        None
    }
    fn barrier_run(
        &mut self,
        io: &Io,
        snap_taken: &std::sync::atomic::AtomicBool,
        early_done: &AtomicUsize,
    ) {
        let early = self.item.id <= 2;
        if early {
            for _ in 0..self.item.steps {
                self.safepoint();
                io.step(self.item.id);
                self.safepoint();
                self.step += 1;
            }
            self.done = Some(format!("v{}", self.item.id));
            let _prev = early_done.fetch_add(1, Ordering::SeqCst);
        }
        while !snap_taken.load(Ordering::SeqCst) {
            std::thread::sleep(Duration::from_millis(1));
            self.safepoint();
        }
        if !early {
            self.run(io);
        }
    }
    fn run(&mut self, io: &Io) {
        loop {
            self.safepoint();
            let (id, steps, done_steps) = {
                let st = &**self;
                (st.item.id, st.item.steps, st.step)
            };
            if done_steps >= steps {
                self.done = Some(format!("v{id}"));
                return;
            }
            io.step(id);
            self.step += 1;
        }
    }
}

impl VocabIo for InnerHandle<Group> {
    fn machine(&mut self) -> Option<&mut InnerHandle<Machine>> {
        None
    }
    fn run(&mut self, io: &Io) {
        let io_ref: &Io = io;
        let _ = self.fanout(
            &|g: &Group| g.children.iter().collect(),
            |mut a: OuterHandle<'_, AnalyzeSlot, dyn VocabIo>| a.run(io_ref),
        )
        .expect("the cycle breaker does not fire in this scenario");    }
}

impl VocabIo for InnerHandle<Machine> {
    fn machine(&mut self) -> Option<&mut InnerHandle<Machine>> {
        Some(self)
    }
    fn run(&mut self, io: &Io) {
        self.safepoint();
        let seq_done = match &self.stage {
            Stage::Seq { step } => (*step as usize) >= self.items.len(),
            _ => false,
        };
        if seq_done {
            let items = self.items.clone();
            self.stage = Stage::Scatter {
                children: items
                    .iter()
                    .map(|i| {
                        PersistentCell::new(Box::new(PersistentCell::new(AnalyzeState {
                            item: i.clone(),
                            step: 0,
                            done: None,
                        })))
                    })
                    .collect(),
            };
            return;
        }
        let seq_next: Option<(usize, u32)> = match &self.stage {
            Stage::Seq { step } if (*step as usize) < self.items.len() => {
                Some((*step as usize, self.items[*step as usize].id))
            }
            _ => None,
        };
        if let Some((idx, id)) = seq_next {
            io.step(id);
            if let Stage::Seq { step } = &mut self.stage {
                *step = (idx + 1) as u32;
            }
            return;
        }
        match &mut self.stage {
            Stage::Seq { .. } => {}
            Stage::Scatter { .. } => {
                let io_ref: &Io = io;
                let _ = self.fanout(
                    &|m: &Machine| match &m.stage {
                        Stage::Scatter { children } => children.iter().collect(),
                        _ => unreachable!("checked above"),
                    },
                    |mut a: OuterHandle<'_, AnalyzeSlot, dyn VocabIo>| a.run(io_ref),
                )
                .expect("the cycle breaker does not fire in this scenario");                let collected: Vec<(u32, String)> = match &self.stage {
                    Stage::Scatter { children } => {
                        children.iter().map(|c| c.result()).collect()
                    }
                    _ => unreachable!("just ran Scatter"),
                };
                self.done.extend(collected);
                let items: Vec<WorkItem> = self.items.clone();
                let g100 = Group {
                    gid: 100,
                    children: items
                        .iter()
                        .take(2)
                        .map(|i| {
                            PersistentCell::new(Box::new(PersistentCell::new(AnalyzeState {
                                item: i.clone(),
                                step: 0,
                                done: None,
                            })))
                        })
                        .collect(),
                };
                let g200 = Group {
                    gid: 200,
                    children: items
                        .iter()
                        .skip(2)
                        .map(|i| {
                            PersistentCell::new(Box::new(PersistentCell::new(AnalyzeState {
                                item: i.clone(),
                                step: 0,
                                done: None,
                            })))
                        })
                        .collect(),
                };
                self.stage = Stage::Nested {
                    groups: vec![
                        PersistentCell::new(Box::new(PersistentCell::new(g100))),
                        PersistentCell::new(Box::new(PersistentCell::new(g200))),
                    ],
                };
            }
            Stage::Nested { .. } => {
                let io_ref: &Io = io;
                let _ = self.fanout(
                    &|m: &Machine| match &m.stage {
                        Stage::Nested { groups } => groups.iter().collect(),
                        _ => unreachable!("checked above"),
                    },
                    |mut g: OuterHandle<'_, GroupSlot, dyn VocabIo>| g.run(io_ref),
                )
                .expect("the cycle breaker does not fire in this scenario");                let collected: Vec<(u32, String)> = match &self.stage {
                    Stage::Nested { groups } => groups.iter().map(|c| c.result()).collect(),
                    _ => unreachable!("just ran Nested"),
                };
                // result from the done-list AS IT WAS at Nested entry
                // (the old semantics: the stage ids, not the gids)
                let done_snapshot: Vec<(u32, String)> = self.done.clone();
                let ids: Vec<String> =
                    done_snapshot.iter().map(|(i, _)| i.to_string()).collect();
                self.result = format!("done: {}", ids.join(","));
                self.done.extend(collected);
                self.stage = Stage::Done;
            }
            Stage::Done => {}
        }
    }
}

/// Drive a machine to completion under the root task; returns the
/// final node (read back from the caller's cell).
fn drive(machine: Machine, io: &Io) -> Machine {
    let mut root: PersistentCell<Box<PersistentCell<Machine>>> =
        PersistentCell::new(Box::new(PersistentCell::new(machine)));
    independent_task_root(
        &mut root,
        |mut outer: OuterHandle<'_, Box<PersistentCell<Machine>>, dyn VocabIo>, _ck| loop {
            outer.run(io);
            let done = matches!(
                outer.machine().map(|inner| &inner.stage),
                Some(Stage::Done)
            );
            if done {
                break;
            }
        },
    );
    (***root).clone()
}

fn tempdir() -> (tempfile::TempDir, std::path::PathBuf) {
    let d = tempfile::tempdir().expect("tempdir");
    let p = d.path().to_path_buf();
    (d, p)
}

#[test]
fn save_restore_roundtrip() {
    let _deadline = deadline::Deadline::start();

    let (_g, dir) = tempdir();
    let path = dir.join("snap.json");
    let m = Machine {
        items: (1..=3).map(|id| WorkItem { id, steps: 2 }).collect(),
        ..Default::default()
    };
    let out = drive(m.clone(), &Io::new());
    assert_eq!(out.result, "done: 1,2,3", "{}", out.result);
    // Mutator-side save of the completed node; restore must agree.
    let mut root: PersistentCell<Box<PersistentCell<Machine>>> =
        PersistentCell::new(Box::new(PersistentCell::new(out.clone())));
    independent_task_root(
        &mut root,
        |_outer: OuterHandle<'_, Box<PersistentCell<Machine>>, dyn VocabIo>, _ck| {},
    );
    snapshot::write_value(&(**root), &path).expect("write");
    let restored: Machine = snapshot::restore_from(&path).expect("restore");
    assert_eq!(out.done, restored.done);
    assert_eq!(restored.result, out.result);
}

/// The mid-fan-out snapshot + resume: two children complete and hold,
/// two hold before their first step; the snapshot fires between; the
/// resumed run must NOT rerun the finished children.
#[test]
fn snapshot_mid_fanout_resumes_without_rerunning_finished_children() {
    let _deadline = deadline::Deadline::start();

    let (_g, dir) = tempdir();
    let path = dir.join("mid.json");

    let snap_taken = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let snap_taken_ref = &snap_taken;
    let early_done = Arc::new(AtomicUsize::new(0));
    let early_done_ref = &early_done;
    let slow_io = Io { step_delay: Duration::from_millis(1), ..Io::new() };

    let m = Machine {
        items: (1..=4).map(|id| WorkItem { id, steps: WORK_STEPS_PER_ITEM }).collect(),
        ..Default::default()
    };
    let mut root: PersistentCell<Box<PersistentCell<Machine>>> =
        PersistentCell::new(Box::new(PersistentCell::new(m)));
    independent_task_root(
        &mut root,
        |mut h: OuterHandle<'_, Box<PersistentCell<Machine>>, dyn VocabIo>,
         req: Checkpointer<'_, Box<PersistentCell<Machine>>>| {
            // Seq to Scatter: run transitions until the stage flips
            loop {
                h.run(&slow_io);
                let at_scatter = matches!(
                    h.machine().map(|inner| &inner.stage),
                    Some(Stage::Scatter { .. })
                );
                if at_scatter {
                    break;
                }
            }
            std::thread::scope(|scope| {
                let path2 = path.clone();
                let req2 = req.clone();
                let snapper = scope.spawn(move || {
                    while early_done_ref.load(Ordering::SeqCst) < 2 {
                        std::thread::sleep(Duration::from_millis(1));
                    }
                    while !snapshot::try_save_requester(&req2, &path2, Duration::from_millis(50))? {
                        std::thread::sleep(Duration::from_millis(1));
                    }
                    snap_taken_ref.store(true, Ordering::SeqCst);
                    Ok::<_, std::io::Error>(())
                });
                // the barrier-aware Scatter: the vocab machine is in
                // Scatter now; run its fan-out by hand with the barrier
                let io_ref: &Io = &slow_io;
                let Some(inner) = h.machine() else { unreachable!("the root IS the machine") };
                let _ = inner.fanout(
                    &|mm: &Machine| match &mm.stage {
                        Stage::Scatter { children } => children.iter().collect(),
                        _ => unreachable!("checked above"),
                    },
                    |mut a: OuterHandle<'_, AnalyzeSlot, dyn VocabIo>| {
                        a.barrier_run(io_ref, snap_taken_ref, early_done_ref);
                    },
                )
                .expect("the cycle breaker does not fire in this scenario");                let _ = snapper.join().expect("snapshotter");
            });
        },
    );

    // The snapshot records exactly two children finished.
    let snap: Machine = snapshot::restore_from(&path).expect("snapshot parses");
    let unfinished_at_snap: usize = match &snap.stage {
        Stage::Scatter { children } => children.iter().filter(|c| c.done.is_none()).count(),
        other => unreachable!("snapshot should catch the Scatter stage, got {other:?}"),
    };
    assert_eq!(unfinished_at_snap, 2, "2 done + 2 not started at snapshot");

    // RESUME and run to completion.
    let io2 = Io::new();
    let resumed = drive(snap, &io2);
    let mut ids: Vec<u32> = resumed.done.iter().map(|(i, _)| *i).collect();
    ids.sort();
    assert_eq!(ids, vec![1, 2, 3, 4, 100, 200], "{}", resumed.result);

    // THE PROPERTY: finished children were NOT rerun.
    let calls = io2.calls.load(Ordering::SeqCst);
    let bound = unfinished_at_snap * WORK_STEPS_PER_ITEM as usize + 4 * WORK_STEPS_PER_ITEM as usize;
    assert!(
        calls <= bound,
        "resume must skip finished children: {calls} calls, bound {bound}"
    );
}


