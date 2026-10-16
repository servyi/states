//! State machines on the checkpoint protocol.
//!
//! The machine's node is the current state ([`StateNode`]); one
//! transition is [`Handle::execute`]: the state runs directly through
//! the handle's `DerefMut` and receives its [`Ctx`] — the fan-out and
//! safepoint facilities of the transition. Fan-outs live inside the
//! state's `execute`: [`Ctx::fanout`] splits the children (fields of
//! the state) directly off the node's tree, one dedicated thread per
//! child, this thread parked join-waiting. A transition returns the
//! successor ([`Step::Next`]); the driver loop installs it with
//! `*handle = next`.

use anyhow::Result;
use std::thread;

use crate::checkpoint::{Handle, Shared, park_self_fields, with_checkpoint_pair};

/// Interval between child-finish checks while the fan-out serve loop waits.
const JOIN_POLL_INTERVAL_MS: u64 = 1;

/// The machine node: the current state. The elided object lifetime is
/// the machine's run — long enough for the env (borrowed providers and
/// the like), while the states themselves stay `'static`.
pub type StateNode<K, E> = Box<dyn State<K, E> + Send + Sync>;

/// What a state decides: its successor, or the machine's result. The
/// successor's object lifetime is the run (elided).
pub enum Step<K, E: ?Sized> {
    Next(StateNode<K, E>),
    Done(K),
}

pub trait State<K, E>: Send + Sync
where
    K: Send + 'static,
    E: ?Sized + Sync,
{
    /// One transition. `env` carries the machine's IO — providers and the
    /// like, supplied by the runner (an input, not an output, hence a
    /// generic parameter). It may borrow, and is shared with fan-out
    /// children; the machine's box types carry its lifetime. Fan-outs run
    /// here: [`Ctx::fanout`] takes a slot vector — a field of `self` — and
    /// splits its children directly off the node's tree.
    fn execute(&mut self, ctx: &mut Ctx<'_>, env: &E) -> Result<Step<K, E>>;
}

/// The fan-out and safepoint facilities of one transition, handed to the
/// running state: a copy of its node's shared parking state. A fan-out's
/// children are local to that fan-out — parking the running state parks
/// them through its own serve loop (parking propagates bottom-up), so no
/// registration anywhere is needed.
pub struct Ctx<'o> {
    shared: &'o Shared,
}

impl Ctx<'_> {
    /// Park while a checkpoint has been requested, then return. Any
    /// fan-out children this ctx owns are parked by the fan-out's own
    /// serve loop; outside a fan-out there are none.
    pub fn safepoint(&mut self) {
        if self.shared.is_pending() {
            park_self_fields(self.shared, &[]);
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
        let storages: Vec<Shared> = (0..child_ptrs.len()).map(|_| Shared::new()).collect();
        let handles: Vec<Handle<'_, C>> = child_ptrs
            .into_iter()
            .zip(&storages)
            .map(|(ptr, storage)| {
                // SAFETY (Handle constructor, per child): `slots` is a
                // field of the state that owns this ctx; each child is
                // yielded exactly once, disjointly; this handle is the only
                // reference to it for its whole life.
                Handle::<C>::from_raw_parts(ptr, storage)
            })
            .collect();
        let shared = self.shared;
        thread::scope(|scope| {
            let action = &action;
            let mut joins = Vec::with_capacity(handles.len());
            for h in handles {
                joins.push(scope.spawn(move || action(h)));
            }
            loop {
                if shared.is_pending() {
                    park_self_fields(shared, &storages);
                }
                if joins.iter().all(|j| j.is_finished()) {
                    break;
                }
                thread::sleep(std::time::Duration::from_millis(JOIN_POLL_INTERVAL_MS));
            }
        });
    }
}

impl<'o, K, E> Handle<'o, StateNode<K, E>>
where
    K: Send + 'static,
    E: ?Sized + Sync + 'o,
{
    /// One transition of the machine: the current state runs directly
    /// through the handle's `DerefMut`, with its [`Ctx`] for fan-outs and
    /// safepoints. Returns the state's decision — installing a `Next`
    /// successor is the driver loop's job (`*handle = next`).
    pub fn execute(&mut self, env: &E) -> Result<Step<K, E>> {
        self.safepoint();
        // The ctx is the machine's own shared parking ref (a copy — the
        // two borrows cannot overlap); the state runs directly through
        // the handle's DerefMut. Fan-outs inside the state park through
        // the ctx; a request on the machine parks this thread and,
        // bottom-up, whatever the ctx's fan-outs own.
        let mut ctx = Ctx { shared: self.shared() };
        let state: &mut (dyn State<K, E> + Send + Sync) = (*self).as_mut();
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
    E: ?Sized + Sync,
{
    with_checkpoint_pair(machine, |mut handle, _requester| {
        loop {
            handle.safepoint();
            match handle.execute(env)? {
                Step::Done(result) => return Ok(result),
                Step::Next(next) => *handle = next,
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
    /// The fan-out test's work items: per-item steps and the item count.
    const WORK_STEPS: &[u32] = &[3, 2, 1];
    /// The fan-out test's second round index.
    const SECOND_ROUND: u32 = 1;
    /// A work item counts as served on its own dedicated thread.
    const MIN_DEDICATED_THREADS: usize = 2;

    struct Env {
        calls: Arc<AtomicUsize>,
    }

    struct CountDown(u32);

    impl State<u32, Env> for CountDown {
        fn execute(&mut self, ctx: &mut Ctx<'_>, env: &Env) -> Result<Step<u32, Env>> {
            ctx.safepoint();
            let _n = env.calls.fetch_add(1, Ordering::SeqCst);
            if self.0 == 0 {
                Ok(Step::Done(LINEAR_RESULT))
            } else {
                Ok(Step::Next(Box::new(CountDown(self.0 - 1))))
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

    impl State<String, Env> for Machine {
        fn execute(&mut self, ctx: &mut Ctx<'_>, env: &Env) -> Result<Step<String, Env>> {
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
                Ok(Step::Next(Box::new(Machine {
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
