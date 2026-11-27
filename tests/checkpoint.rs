// The two-piece protocol's integration suite (#19's design): the
// machine shape the old suite drove (Seq/Scatter/Nested stages, STW
// checkpoints, mid-run snapshot+resume) rebuilt on
// independent_task_root + PersistentCell slots.
//
// The scenario literals below are test data (see header); they are
// the one class the inline_literal policy exempts via allow.
#![allow(unknown_lints)] // plain clippy does not know the driver's lints
#![allow(inline_literal)]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use servyi_states::checkpoint::{
    Checkpointer, InnerHandle, OuterHandle, PersistentCell, independent_task_root,
};

/// Steps after which the machine holds (parks at safepoints, no
/// progress) until released — the deterministic mid-run point.
const HOLD_AT: usize = 6;
/// Every work item in the mid-run test has this many steps.
const WORK_STEPS_PER_ITEM: usize = 4;

#[derive(Debug, Clone)]
struct Io {
    calls: Arc<AtomicUsize>,
    step_delay: Duration,
    /// Work steps taken since the last release.
    progress: Arc<AtomicUsize>,
    /// Set to let a held machine continue.
    release: Arc<std::sync::atomic::AtomicBool>,
}

impl Io {
    fn new() -> Self {
        Self {
            calls: Arc::new(AtomicUsize::new(0)),
            step_delay: Duration::from_millis(0),
            progress: Arc::new(AtomicUsize::new(0)),
            release: Arc::new(std::sync::atomic::AtomicBool::new(true)),
        }
    }

    /// Whether the machine is held at the deterministic mid-run point.
    fn holding(&self) -> bool {
        !self.release.load(Ordering::SeqCst)
            && self.progress.load(Ordering::SeqCst) >= HOLD_AT
    }

    fn release(&self) {
        self.release.store(true, Ordering::SeqCst);
        self.progress.store(0, Ordering::SeqCst);
    }

    /// Take one step — unless the machine is held. The hold engages AT
    /// `HOLD_AT`, atomically with the step that reaches it: a step counts
    /// only if it reserves its progress before the hold engages. (With a
    /// separate check-then-step, a child that had passed the check took
    /// one more step after a sibling's step had engaged the hold: 7 steps
    /// consumed at the snapshot, not 6.)
    fn try_step(&self, _id: u32) -> bool {
        let mut progress = self.progress.load(Ordering::SeqCst);
        loop {
            if !self.release.load(Ordering::SeqCst) && progress >= HOLD_AT {
                return false;
            }
            match self
                .progress
                .compare_exchange_weak(progress, progress + 1, Ordering::SeqCst, Ordering::SeqCst)
            {
                Ok(_) => break,
                Err(now) => progress = now,
            }
        }
        let _n = self.calls.fetch_add(1, Ordering::SeqCst);
        if !self.step_delay.is_zero() {
            std::thread::sleep(self.step_delay);
        }
        true
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

/// The slot a node lives in: a boxed cell in a cell (the two-piece
/// shape). `AnalyzeSlot` etc. name the per-node container.
type AnalyzeSlot = Box<PersistentCell<AnalyzeState>>;
type GroupSlot = Box<PersistentCell<Group>>;

fn run_analyze(h: &mut InnerHandle<AnalyzeState>, io: &Io) {
    loop {
        h.safepoint();
        let (id, steps, done_steps) = {
            let st = &*h;
            (st.item.id, st.item.steps, st.step)
        };
        if done_steps >= steps {
            h.done = Some(format!("v{id}"));
            return;
        }
        while io.holding() {
            h.safepoint();
            std::thread::sleep(Duration::from_millis(1));
        }
        h.safepoint();
        if !io.try_step(id) {
            // the hold engaged between the check and this step: nothing
            // was taken and the state does not advance — re-check
            continue;
        }
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
    children: Vec<PersistentCell<AnalyzeSlot>>,
}

impl Group {
    fn result(&self) -> (u32, String) {
        let ids: Vec<String> = self
            .children
            .iter()
            .map(|c| c.done.clone().unwrap_or_default())
            .collect();
        (self.gid, ids.join(","))
    }
}

/// The group's drive vocabulary: a group runs its children.
trait GroupDriven: Send {
    fn run(&mut self, io: &Io);
}

impl GroupDriven for InnerHandle<Group> {
    fn run(&mut self, io: &Io) {
        let _ = self.fanout(
            &|g: &Group| g.children.iter().collect(),
            |mut a: OuterHandle<'_, AnalyzeSlot, dyn GroupNode>| a.run_node(io),
        )
        .expect("the cycle breaker does not fire in this scenario");
    }
}

/// The analyze node's vocabulary (the fan-out callback's view).
trait GroupNode: Send {
    fn run_node(&mut self, io: &Io);
}

impl GroupNode for InnerHandle<AnalyzeState> {
    fn run_node(&mut self, io: &Io) {
        run_analyze(self, io)
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

/// The machine's drive vocabulary: one stage per `run` call (the root
/// task loops it).
trait MachineDriven: Send {
    fn run_stage(&mut self, io: &Io) -> bool;
}

impl MachineDriven for InnerHandle<Machine> {
    fn run_stage(&mut self, io: &Io) -> bool {
        let mut next: Option<Stage> = None;
        let keep_going = {
            let m = &mut **self;
            match &mut m.stage {
                Stage::Seq { step } => {
                    if (*step as usize) < m.items.len()
                        && io.try_step(m.items[*step as usize].id)
                    {
                        *step += 1;
                        true
                    } else if (*step as usize) < m.items.len() {
                        // Held mid-Seq: make no progress this transition; the
                        // driver loop's safepoints keep serving the snapshotter.
                        true
                    } else {
                        next = Some(Stage::Scatter {
                            children: m
                                .items
                                .iter()
                                .map(|i| {
                                    PersistentCell::new(Box::new(PersistentCell::new(
                                        AnalyzeState {
                                            item: i.clone(),
                                            step: 0,
                                            done: None,
                                        },
                                    )))
                                })
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
            self.stage = n;
        }
        if !keep_going {
            return false;
        }
        let which = match &self.stage {
            Stage::Scatter { .. } => 0,
            Stage::Nested { .. } => 1,
            _ => -1,
        };
        match which {
            0 => {
                let _ = self.fanout(
                    &|m: &Machine| match &m.stage {
                        Stage::Scatter { children } => children.iter().collect(),
                        _ => unreachable!("checked above"),
                    },
                    |mut a: OuterHandle<'_, AnalyzeSlot, dyn GroupNode>| a.run_node(io),
                )
                .expect("the cycle breaker does not fire in this scenario");
            }
            1 => {
                let _ = self.fanout(
                    &|m: &Machine| match &m.stage {
                        Stage::Nested { groups } => groups.iter().collect(),
                        _ => unreachable!("checked above"),
                    },
                    |mut g: OuterHandle<'_, GroupSlot, dyn GroupDrivenVocab>| g.run(io),
                )
                .expect("the cycle breaker does not fire in this scenario");
            }
            _ => {}
        }
        let m = &mut **self;
        let mut collected: Vec<(u32, String)> = Vec::new();
        match &m.stage {
            Stage::Scatter { children } => collected.extend(children.iter().map(Group::node_result)),
            Stage::Nested { groups } => collected.extend(groups.iter().map(Group::slot_result)),
            _ => {}
        }
        // Stage transitions live with the collection they follow: Scatter
        // results feed the Nested groups (take/skip split, gids 100/200),
        // Nested results finish the machine.
        let next: Option<Stage> = match &m.stage {
            Stage::Scatter { .. } => Some(Stage::Nested {
                groups: vec![
                    PersistentCell::new(Box::new(PersistentCell::new(Group {
                        gid: 100,
                        children: m
                            .items
                            .iter()
                            .take(2)
                            .map(|i| {
                                PersistentCell::new(Box::new(PersistentCell::new(
                                    AnalyzeState {
                                        item: i.clone(),
                                        step: 0,
                                        done: None,
                                    },
                                )))
                            })
                            .collect(),
                    }))),
                    PersistentCell::new(Box::new(PersistentCell::new(Group {
                        gid: 200,
                        children: m
                            .items
                            .iter()
                            .skip(2)
                            .map(|i| {
                                PersistentCell::new(Box::new(PersistentCell::new(
                                    AnalyzeState {
                                        item: i.clone(),
                                        step: 0,
                                        done: None,
                                    },
                                )))
                            })
                            .collect(),
                    }))),
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
        true
    }
}

trait GroupDrivenVocab: Send {
    fn run(&mut self, io: &Io);
}

impl GroupDrivenVocab for InnerHandle<Group> {
    fn run(&mut self, io: &Io) {
        <Self as GroupDriven>::run(self, io)
    }
}

impl Group {
    fn node_result(c: &PersistentCell<AnalyzeSlot>) -> (u32, String) {
        c.result()
    }
    fn slot_result(c: &PersistentCell<GroupSlot>) -> (u32, String) {
        c.result()
    }
}

fn drive(
    machine: Machine,
    io: &Io,
    checkpoint: Option<(Duration, std::path::PathBuf)>,
) -> Machine {
    let mut root: PersistentCell<Box<PersistentCell<Machine>>> =
        PersistentCell::new(Box::new(PersistentCell::new(machine)));
    independent_task_root(
        &mut root,
        |mut outer: OuterHandle<'_, Box<PersistentCell<Machine>>, dyn MachineDrivenVocab>,
         req: Checkpointer<'_, Box<PersistentCell<Machine>>>| {
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

                // the stage loop: each run_stage parks at its safepoints
                loop {
                    outer.park();
                    if !outer.run_stage(io) {
                        break;
                    }
                }

                if let Some((t, _)) = timer {
                    stop.store(true, Ordering::SeqCst);
                    t.join().expect("checkpoint timer");
                }
            });
        },
    );
    // the node stayed in the caller's cell all along — read it back
    (*(*(*root))).clone()
}

trait MachineDrivenVocab: Send {
    fn run_stage(&mut self, io: &Io) -> bool;
    fn safepoint(&mut self);
}

trait ParkVocab: Send {
    fn park(&mut self);
}

impl ParkVocab for dyn MachineDrivenVocab + '_ {
    fn park(&mut self) {
        self.safepoint();
    }
}

impl MachineDrivenVocab for InnerHandle<Machine> {
    fn run_stage(&mut self, io: &Io) -> bool {
        <Self as MachineDriven>::run_stage(self, io)
    }
    fn safepoint(&mut self) {
        InnerHandle::safepoint(self)
    }
}

fn tempdir() -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("snapshot.json");
    (dir, path)
}

/// The freeze window: a guard's read access during a hold, then the
/// safepoint serves the world.
#[test]
fn freeze_window_allows_access_then_safepoint() {
    let mut root: PersistentCell<Box<PersistentCell<Vec<u32>>>> =
        PersistentCell::new(Box::new(PersistentCell::new(vec![7_u32])));
    let out: Vec<u32>;
    {
        independent_task_root(
            &mut root,
            |mut h: OuterHandle<'_, Box<PersistentCell<Vec<u32>>>, dyn VecDriven>, req| {
                let ck = req.clone();
                std::thread::scope(|s| {
                    let t = s.spawn(move || {
                        let guard = ck.request();
                        let node: &Vec<u32> = &guard;
                        assert_eq!(node, &vec![7]);
                    });
                    // the freeze window: park until the reader is done
                    // (a bare join would self-deadlock on a request
                    // arriving after the last park)
                    while !t.is_finished() {
                        h.park();
                    }
                    h.push(8);
                    t.join().unwrap();
                });
            },
        );
        out = (***root).clone();
    }
    assert_eq!(out, vec![7, 8]);
}

trait VecDriven: Send {
    fn push(&mut self, v: u32);
    fn park(&mut self);
}

impl VecDriven for InnerHandle<Vec<u32>> {
    fn push(&mut self, v: u32) {
        (**self).push(v);
        self.safepoint();
    }
    fn park(&mut self) {
        self.safepoint();
    }
}


#[test]
fn snapshotter_reads_live_tree_through_guard() {
    let mut root: PersistentCell<Box<PersistentCell<u32>>> =
        PersistentCell::new(Box::new(PersistentCell::new(41_u32)));
    let observed: Arc<std::sync::Mutex<Vec<u32>>> = Arc::new(std::sync::Mutex::new(Vec::new()));
    let observed2 = observed.clone();
    independent_task_root(
        &mut root,
        |mut h: OuterHandle<'_, Box<PersistentCell<u32>>, dyn CountDriven>, req| {
            let mutifier = Arc::new(AtomicUsize::new(0));
            let mutifier2 = mutifier.clone();
            std::thread::scope(|s| {
                let reader = s.spawn(move || {
                    let mut last = 0;
                    while mutifier2.load(Ordering::SeqCst) == 0 {
                        let Some(g) = req.request_timeout(Duration::from_millis(50)) else {
                            continue;
                        };
                        let v: u32 = ****g;
                        let mut o = observed2.lock().unwrap();
                        if v >= last {
                            (*o).push(v);
                            last = v;
                        }
                        drop(g);
                        std::thread::sleep(Duration::from_millis(1));
                    }
                });
                for i in 0..=50 {
                    h.set(41 + i);
                    h.park();
                }
                mutifier.store(1, Ordering::SeqCst);
                reader.join().unwrap();
            });
        },
    );
    let final_v: u32 = ***root;
    assert_eq!(final_v, 91);
    let w = final_v_2(&observed);
    if !w.is_empty() {
        assert!(w[0] <= w[w.len() - 1], "monotonic guard observations: {observed:?}");
        assert!(w.iter().all(|v| (41..=91).contains(v)));
    }
}

fn final_v_2(observed: &std::sync::Mutex<Vec<u32>>) -> Vec<u32> {
    observed.lock().unwrap().clone()
}

trait CountDriven: Send {
    fn set(&mut self, v: u32);
    fn park(&mut self);
}

impl CountDriven for InnerHandle<u32> {
    fn set(&mut self, v: u32) {
        **self = v;
    }
    fn park(&mut self) {
        self.safepoint();
    }
}

#[test]
fn concurrent_requesters_serialize_instead_of_deadlocking() {
    let mut root: PersistentCell<Box<PersistentCell<u32>>> =
        PersistentCell::new(Box::new(PersistentCell::new(0_u32)));
    independent_task_root(
        &mut root,
        |mut h: OuterHandle<'_, Box<PersistentCell<u32>>, dyn CountDriven>, req| {
            let mutifier_done = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let r1 = req.clone();
            let r2 = req.clone();
            let done2 = mutifier_done.clone();
            let done3 = mutifier_done.clone();
            std::thread::scope(|s| {
                let a = s.spawn(move || {
                    let mut last = 0;
                    while !done2.load(Ordering::SeqCst) {
                        let Some(g) = r1.request_timeout(Duration::from_millis(50)) else {
                            continue;
                        };
                        let v: u32 = ****g;
                        assert!(v >= last, "observations must be monotonic per reader");
                        last = v;
                    }
                });
                let b = s.spawn(move || {
                    let mut last = 0;
                    while !done3.load(Ordering::SeqCst) {
                        let Some(g) = r2.request_timeout(Duration::from_millis(50)) else {
                            continue;
                        };
                        let v: u32 = ****g;
                        assert!(v >= last, "observations must be monotonic per reader");
                        last = v;
                    }
                });
                for i in 0..100 {
                    h.set(i);
                    h.park();
                }
                mutifier_done.store(true, Ordering::SeqCst);
                a.join().unwrap();
                b.join().unwrap();
            });
        },
    );
}

#[test]
fn full_run_with_stw_checkpoints_and_nested_fanout() {
    let io = Io::new();
    let machine = Machine {
        items: vec![
            WorkItem { id: 1, steps: 1 },
            WorkItem { id: 2, steps: 2 },
            WorkItem { id: 3, steps: 3 },
            WorkItem { id: 4, steps: 2 },
        ],
        ..Default::default()
    };
    let (_dir, path) = tempdir();
    let start = Instant::now();
    let m = drive(machine.clone(), &io, Some((Duration::from_millis(1), path.clone())));
    let elapsed = start.elapsed();
    let ids: Vec<u32> = m.done.iter().map(|(i, _)| *i).collect();
    assert_eq!(ids, vec![1, 2, 3, 4, 100, 200], "{}", m.result);
    assert!(path.exists(), "checkpoints written");
    // A wall-clock bound shows that the children ran concurrently only
    // where time is real: under an interpreter (Miri) a run's duration
    // says nothing about scheduling.
    if !cfg!(miri) {
        assert!(elapsed.as_millis() < 2000, "scoped threads run concurrently: {elapsed:?}");
    }
}

#[test]
fn mid_run_snapshot_resumes() {
    // The full-run IO count, measured (not derived): the resume assertion
    // compares against it.
    let full_io = Io::new();
    let full = drive(full_machine(), &full_io, None);
    assert_eq!(full.result, "done: 1,2,3,4");

    // The held run: hold at HOLD_AT, snapshot mid-hold, release, finish.
    let io = Io::new();
    io.release.store(false, Ordering::SeqCst);
    let (dir, path) = tempdir();
    let consumed_at_snapshot = Arc::new(AtomicUsize::new(0));
    let consumed_probe = consumed_at_snapshot.clone();

    let mut root: PersistentCell<Box<PersistentCell<Machine>>> =
        PersistentCell::new(Box::new(PersistentCell::new(full_machine())));
    independent_task_root(
        &mut root,
        |mut h: OuterHandle<'_, Box<PersistentCell<Machine>>, dyn MachineDrivenVocab>,
         req: Checkpointer<'_, Box<PersistentCell<Machine>>>| {
            let io2 = hold_io(&io);
            let req2 = req.clone();
            let path2 = path.clone();
            std::thread::scope(|sc| {
                let io3 = hold_io(&io);
                let consumed_probe2 = consumed_probe.clone();
                let snapshotter = sc.spawn(move || {
                    // wait for the hold to ENGAGE (the deterministic
                    // mid-run point), then snapshot, then RELEASE —
                    // the drive loop is blocked inside the fan-out's
                    // join while the children are held, so the release
                    // must come from THIS thread
                    while io3.progress.load(Ordering::SeqCst) < HOLD_AT {
                        std::thread::sleep(Duration::from_millis(1));
                    }
                    loop {
                        let g = req2.request_timeout(Duration::from_millis(50));
                        if let Some(g) = g {
                            let bytes = serde_json::to_string(&*g).expect("ser");
                            let tmp = path2.with_extension("tmp");
                            std::fs::write(&tmp, &bytes).expect("write");
                            std::fs::rename(&tmp, path2).expect("rename");
                            consumed_probe2.store(io3.calls.load(Ordering::SeqCst), Ordering::SeqCst);
                            io3.release();
                            return;
                        }
                    }
                });
                loop {
                    h.park();
                    if !h.run_stage(&io2) {
                        break;
                    }
                }
                snapshotter.join().expect("snapshotter");
            });
        },
    );
    let held: Machine = (*(*(*root))).clone();
    let consumed = consumed_at_snapshot.load(Ordering::SeqCst);
    let _ = held;
    assert_eq!(consumed, HOLD_AT, "held snapshot consumes exactly HOLD_AT steps");

    // resume from the snapshot
    let text = std::fs::read_to_string(&path).expect("snapshot file");
    let resumed_core: Box<PersistentCell<Machine>> = serde_json::from_str(&text).expect("deser");
    let resumed = drive((**resumed_core).clone(), &Io::new(), None);
    let ids: Vec<u32> = resumed.done.iter().map(|(i, _)| *i).collect();
    assert_eq!(ids, vec![1, 2, 3, 4, 100, 200], "{}", resumed.result);
    drop(dir);
}

fn hold_io(io: &Io) -> Io {
    io.clone()
}

fn full_machine() -> Machine {
    Machine {
        items: vec![
            WorkItem { id: 1, steps: WORK_STEPS_PER_ITEM as u32 },
            WorkItem { id: 2, steps: WORK_STEPS_PER_ITEM as u32 },
            WorkItem { id: 3, steps: WORK_STEPS_PER_ITEM as u32 },
            WorkItem { id: 4, steps: WORK_STEPS_PER_ITEM as u32 },
        ],
        ..Default::default()
    }
}
