//! State machines on the checkpoint protocol.
//!
//! The machine's node is the current state, `Box<dyn State<..>>`; one
//! transition is [`Handle::execute`]: the state runs as the node's single
//! fan-out child and receives its [`Ctx`] — the fan-out and safepoint
//! facilities for the transition. Fan-outs live inside the state's
//! `execute`: [`Ctx::fanout`] splits the children (fields of the state)
//! directly off the node's tree, one dedicated thread per child, this
//! thread parked join-waiting. The successor is installed by the driver
//! with `*handle = Box::new(..)`.

use anyhow::Result;
use std::sync::Mutex;
use std::thread;

/// Interval between child-finish checks while the fan-out serve loop waits.
const JOIN_POLL_INTERVAL_MS: u64 = 1;

use crate::checkpoint::{Handle, Shared, park_self_fields, with_checkpoint_pair};

/// The machine node: the current state.
pub type StateNode<K, E> = Box<dyn State<K, Env = E> + Send + Sync>;

/// What a state decides: its successor, or the machine's result.
pub enum Step<K, E: ?Sized> {
    Continue(StateNode<K, E>),
    Done(K),
}

pub trait State<K>: Send + Sync + 'static
where
    K: Send + 'static,
    Self::Env: 'static,
{
    /// The machine's environment (providers and the like).
    type Env: ?Sized;

    /// One transition. `env` carries the machine's IO. Fan-outs run here:
    /// [`Ctx::fanout`] takes a slot vector — a field of `self` — and
    /// splits its children directly off the node's tree.
    fn execute(&mut self, ctx: &mut Ctx<'_, '_>, env: &Self::Env) -> Result<Step<K, Self::Env>>;
}

/// The fan-out and safepoint facilities of one transition, handed to the
/// running state. Derived from its node handle; borrows of the two never
/// overlap.
pub struct Ctx<'a, 'o> {
    shared: &'o Shared,
    children: &'a mut Vec<Shared>,
}

impl Ctx<'_, '_> {
    /// Park while a checkpoint has been requested, then return.
    pub fn safepoint(&mut self) {
        if self.shared.is_pending() {
            park_self_fields(self.shared, self.children);
        }
    }

    /// Fan out over `slots` — a field of the state that owns this ctx, so
    /// the children are part of the node's tree. One dedicated scoped
    /// thread per child runs `action` with the child's handle; this thread
    /// parks join-waiting while they run. Data lives in the slot for the
    /// child's whole life and is accessed through its handle.
    ///
    /// # Panics
    ///
    /// Panics if a panicked child propagates from the fan-out scope, or if
    /// a handshake lock is poisoned — see the invariant on `Shared::parked`.
    pub fn fanout<C, F>(&mut self, slots: &mut [C], action: F)
    where
        C: Send,
        F: Fn(Handle<'_, C>) + Send + Sync,
    {
        let child_ptrs: Vec<*mut C> = slots.iter_mut().map(std::ptr::from_mut).collect();
        let start = self.children.len();
        for _ in 0..child_ptrs.len() {
            self.children.push(Shared::new());
        }
        let handles: Vec<Handle<'_, C>> = child_ptrs
            .into_iter()
            .zip(&self.children[start..])
            .map(|(ptr, storage)| {
                // SAFETY (Handle constructor, per child): `slots` is a
                // field of the state that owns this ctx; each child is
                // yielded exactly once, disjointly; this handle is the only
                // reference to it for its whole life.
                Handle::<C>::from_raw_parts(ptr, storage)
            })
            .collect();
        let shared = self.shared;
        let children = &self.children[start..];
        thread::scope(|scope| {
            let action = &action;
            let mut joins = Vec::with_capacity(handles.len());
            for h in handles {
                joins.push(scope.spawn(move || action(h)));
            }
            loop {
                if shared.is_pending() {
                    park_self_fields(shared, children);
                }
                if joins.iter().all(|j| j.is_finished()) {
                    break;
                }
                thread::sleep(std::time::Duration::from_millis(JOIN_POLL_INTERVAL_MS));
            }
        });
        self.children.truncate(start);
    }
}

impl<'o, K, E> Handle<'o, StateNode<K, E>>
where
    K: Send + 'static,
    E: ?Sized + Sync + 'static,
{
    /// One transition of the machine: the current state executes through
    /// its own handle (as the node's single fan-out child, so its fan-outs
    /// split straight off the node), and its successor is installed with
    /// `*handle = Box::new(..)`. Returns the machine's result when done.
    ///
    /// # Panics
    ///
    /// Panics if a panicked child propagates from the fan-out scope — see
    /// [`Handle::fanout`].
    pub fn execute(&mut self, env: &E) -> Result<Option<K>> {
        self.safepoint();
        // The action runs exactly once (single child); the Mutex is only
        // here because fan-out actions must be Fn + Send + Sync.
        let outcome = Mutex::new(None::<Result<Step<K, E>>>);
        self.fanout(
            &|node: &mut StateNode<K, E>| vec![&mut *node],
            |mut child: Handle<'_, StateNode<K, E>>| {
                child.safepoint();
                *outcome.lock().unwrap_or_else(|e| e.into_inner()) = Some(child.step(env));
            },
        );
        match outcome
            .into_inner()
            .unwrap_or_else(|e| e.into_inner())
            .expect("the single child always runs")?
        {
            Step::Continue(next) => {
                (*self).set_state(next);
                Ok(None)
            }
            Step::Done(result) => Ok(Some(result)),
        }
    }

    /// Run the state behind this handle with its transition ctx: the state
    /// and the ctx come from disjoint fields of the handle.
    fn step(&mut self, env: &E) -> Result<Step<K, E>> {
        // SAFETY: the handle's type invariant guarantees `node` points at
        // the node's own storage whenever the handle is not parked; this
        // method runs on the mutator thread, not parked.
        let boxed = self.raw_node();
        // SAFETY: the handle's type invariant guarantees `boxed` points at
        // the node's own storage whenever the handle is not parked; this
        // method runs on the mutator thread, not parked.
        let node: &mut StateNode<K, E> = unsafe { &mut *boxed };
        let state: &mut (dyn State<K, Env = E> + Send + Sync) = node.as_mut();
        let mut ctx = Ctx { shared: self.shared(), children: self.raw_children() };
        state.execute(&mut ctx, env)
    }
}

/// Drive a machine to completion under one checkpoint pair: the node is
/// the current state; each transition is [`Handle::execute`].
///
/// # Panics
///
/// Panics if a state panics — the pair's fan-out scope propagates it.
pub fn run_to_completion<K, E>(
    machine: StateNode<K, E>,
    env: &E,
) -> Result<K>
where
    K: Send + 'static,
    E: ?Sized + Sync + 'static,
{
    with_checkpoint_pair(machine, |mut handle, _requester| {
        loop {
            handle.safepoint();
            if let Some(result) = handle.execute(env)? {
                return Ok(result);
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    /// Countdown start and expected result in the linear-machine test.
    const LINEAR_START: u32 = 3;
    const LINEAR_RESULT: u32 = 42;
    /// Steps observed by the linear test: LINEAR_START transitions plus Done.
    const LINEAR_CALLS: usize = 4;
    /// The fan-out test's work items: per-item steps.
    const WORK_STEPS: &[u32] = &[3, 2, 1];
    /// The fan-out test's second round index.
    const SECOND_ROUND: u32 = 1;
    /// A work item counts as served on its own dedicated thread.
    const MIN_DEDICATED_THREADS: usize = 2;

    struct Env {
        calls: Arc<AtomicUsize>,
    }

    struct CountDown(u32);

    impl State<u32> for CountDown {
        type Env = Env;

        fn execute(&mut self, ctx: &mut Ctx<'_, '_>, env: &Env) -> Result<Step<u32, Env>> {
            ctx.safepoint();
            let _n = env.calls.fetch_add(1, Ordering::SeqCst);
            if self.0 == 0 {
                Ok(Step::Done(LINEAR_RESULT))
            } else {
                Ok(Step::Continue(Box::new(CountDown(self.0 - 1))))
            }
        }
    }

    #[test]
    fn linear_machine_runs_to_completion() {
        let calls = Arc::new(AtomicUsize::new(0));
        let env = Env { calls: Arc::clone(&calls) };
        let result = run_to_completion(Box::new(CountDown(LINEAR_START)), &env).expect("run");
        assert_eq!(result, LINEAR_RESULT);
        assert_eq!(calls.load(Ordering::SeqCst), LINEAR_CALLS);
    }

    struct WorkItem {
        steps: u32,
        threads: Vec<thread::ThreadId>,
    }

    impl WorkItem {
        fn new(steps: u32) -> Self {
            Self { steps, threads: Vec::new() }
        }
    }

    struct Machine {
        round: u32,
        work: Vec<WorkItem>,
    }

    impl State<String> for Machine {
        type Env = Env;

        fn execute(&mut self, ctx: &mut Ctx<'_, '_>, env: &Env) -> Result<Step<String, Env>> {
            ctx.safepoint();
            let _n = env.calls.fetch_add(1, Ordering::SeqCst);
            if self.round == 0 {
                // The fan-out lives inside the transition: the children
                // are this state's own work slots.
                ctx.fanout(&mut self.work, |mut child| {
                    let _n = env.calls.fetch_add(1, Ordering::SeqCst);
                    child.threads.push(thread::current().id());
                    for _ in 0..child.steps {
                        child.safepoint();
                        std::thread::sleep(Duration::from_millis(JOIN_POLL_INTERVAL_MS));
                    }
                });
                self.round = SECOND_ROUND;
                Ok(Step::Continue(Box::new(Machine {
                    round: self.round,
                    work: std::mem::take(&mut self.work),
                })))
            } else {
                let distinct: std::collections::HashSet<thread::ThreadId> =
                    self.work.iter().flat_map(|w| w.threads.iter().copied()).collect();
                // Every work item ran on its own dedicated thread.
                assert!(distinct.len() >= self.work.len().min(MIN_DEDICATED_THREADS));
                Ok(Step::Done(format!("done:{}", distinct.len())))
            }
        }
    }

    #[test]
    fn fanout_machine_runs_children_on_dedicated_threads() {
        let calls = Arc::new(AtomicUsize::new(0));
        let env = Env { calls: Arc::clone(&calls) };
        let machine = Machine {
            round: 0,
            work: WORK_STEPS.iter().map(|&steps| WorkItem::new(steps)).collect(),
        };
        let result = run_to_completion(Box::new(machine), &env).expect("run");
        assert_eq!(result, format!("done:{}", WORK_STEPS.len()));
    }
}
