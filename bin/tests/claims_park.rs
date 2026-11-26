//! Three subtasks are handed the same cell, and a checkpoint arrives
//! while two of them wait for the claim. Every one of them — the winner
//! at its safepoint, the losers in their claim backoff — has to park, or
//! the handshake never completes.
//!
//! The scenario runs on its own thread and the TEST waits for it with a
//! deadline: if a waiting subtask cannot park, the scenario never
//! returns, and the regression fails the assertion instead of hanging
//! the suite (the stuck thread is simply left behind).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use hslib::{HandleAble, InnerHandle, PersistentCell, independent_task_root};

/// How long the subtask that WINS the contested cell keeps it, parking
/// at its safepoints the while.
const WINNER_RUNS_FOR: Duration = Duration::from_millis(300);
/// How long the winner sleeps between safepoints.
const SAFEPOINT_POLL: Duration = Duration::from_millis(1);
/// When the checker asks for a checkpoint: once the losers are waiting.
const CHECKPOINT_AFTER: Duration = Duration::from_millis(100);
/// How long the scenario may take before the test gives up on it. The
/// scenario itself needs about a second.
const SCENARIO_DEADLINE: Duration = Duration::from_secs(10);
/// How often the test checks whether the scenario has finished.
const FINISHED_POLL: Duration = Duration::from_millis(10);

trait Vocab: Send {
    fn step(&mut self);
}

type Slot = Box<PersistentCell<dyn HandleAble<dyn Vocab>>>;

/// Holds its claims for a while, parking whenever a checkpoint is
/// requested.
struct State;

impl Vocab for InnerHandle<State> {
    fn step(&mut self) {
        let until = std::time::Instant::now() + WINNER_RUNS_FOR;
        while std::time::Instant::now() < until {
            self.safepoint();
            std::thread::sleep(SAFEPOINT_POLL);
        }
    }
}

struct Root {
    children: Vec<PersistentCell<Slot>>,
}

impl Vocab for InnerHandle<Root> {
    fn step(&mut self) {
        self.fanout(
            // the ONE cell, handed to three subtasks: one wins the cell
            // claim, the other two wait for it
            &|root: &Root| {
                root.children.iter().flat_map(|child| [child; 3]).collect()
            },
            |mut outer| outer.step(),
        );
    }
}

#[test]
fn subtasks_waiting_for_a_claim_still_park_for_a_checkpoint() {
    let stopped = Arc::new(AtomicBool::new(false));
    let report = Arc::clone(&stopped);
    let scenario = std::thread::spawn(move || {
        let child: Slot = PersistentCell::from_boxed(Box::new(State));
        let root_node: Slot =
            PersistentCell::from_boxed(Box::new(Root { children: vec![PersistentCell::new(child)] }));
        let mut root = PersistentCell::new(root_node);
        independent_task_root(&mut root, |mut outer, checkpointer| {
            let checkpointer = checkpointer.clone();
            std::thread::scope(|scope| {
                let checker = scope.spawn(move || {
                    std::thread::sleep(CHECKPOINT_AFTER);
                    let guard = checkpointer.request();
                    drop(guard);
                    report.store(true, Ordering::SeqCst);
                });
                outer.step();
                checker.join().unwrap();
            })
        })
    });
    let give_up_at = std::time::Instant::now() + SCENARIO_DEADLINE;
    while !scenario.is_finished() && std::time::Instant::now() < give_up_at {
        std::thread::sleep(FINISHED_POLL);
    }
    assert!(
        scenario.is_finished(),
        "the checkpoint never completed within {SCENARIO_DEADLINE:?}: a subtask waiting for a claim could not park"
    );
    scenario.join().unwrap();
    assert!(stopped.load(Ordering::SeqCst), "the checkpoint completed");
}
