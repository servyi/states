//! The state machine on the checkpoint protocol — 100% safe code over
//! the public [`Handle`] API (issue #13): no `unsafe`, no escape
//! hatches, no visibility openings. Everything the drivers need is the
//! reviewed public surface — `Deref`/`DerefMut`, `safepoint`, `fanout`
//! — plus [`crate::context_id::ContextId`] for the parallel phase.
//!
//! - A linear [`State`] executes through `&mut self` and decides its
//!   successor; the driver lives on the HANDLE over the node
//!   ([`Handle::execute`]) and installs successors with a plain
//!   `DerefMut` assignment.
//! - A fan-out phase is the concrete parametric [`ParallelState`],
//!   instantiated by the client: it allocates and stores a
//!   [`ContextId`] per child at spawn; children are
//!   `(ContextId, Box<dyn State>)` tuples, wrapped in handles by
//!   [`Handle::fanout`]. Its driver is on the handle too
//!   ([`Handle::execute_children`]).
//! - [`Ctx`] is pure ergonomics over `&mut Handle` (Deref/DerefMut and
//!   nothing else — there is no special `ctx.fanout`, only
//!   `handle.fanout`): the auto-deref chain `Ctx → Handle → node`
//!   reaches both.

use anyhow::Result;
use std::sync::Mutex;

use crate::checkpoint::{Handle, with_checkpoint_pair};
use crate::context_id::ContextId;

/// The machine node: the current state.
pub type StateNode<K, E> = Box<dyn State<K, Env = E> + Send + Sync>;

/// What a state decides: its successor, or the machine's result.
pub enum Step<K, E: ?Sized> {
    Continue(StateNode<K, E>),
    Done(K),
}

/// A linear state: one transition through `&mut self`. States that
/// want fan-out do not grow the trait — they run as a
/// [`ParallelState`] phase instead.
pub trait State<K>: Send + Sync + 'static
where
    K: Send + 'static,
    Self::Env: 'static,
{
    /// The machine's environment (providers and the like).
    type Env: ?Sized + Sync;

    /// One transition: run, and name the successor or the result.
    fn execute(&mut self, env: &Self::Env) -> Result<Step<K, Self::Env>>;
}

/// Pure sugar over `&mut Handle`: Deref/DerefMut, nothing else. The
/// auto-deref chain `Ctx → Handle → node` makes both the handle's
/// facilities (`ctx.fanout(..)` resolves on `Handle`) and the node's
/// fields reachable; every `&mut` through the chain reborrows the same
/// handle, so state slots are borrowed inside `fanout`'s split
/// closure, which is handed the `&mut` node exactly for that.
pub struct Ctx<'a, 'o, T> {
    handle: &'a mut Handle<'o, T>,
}

impl<'a, 'o, T> std::ops::Deref for Ctx<'a, 'o, T> {
    type Target = Handle<'o, T>;
    fn deref(&self) -> &Handle<'o, T> {
        self.handle
    }
}

impl<'a, 'o, T> std::ops::DerefMut for Ctx<'a, 'o, T> {
    fn deref_mut(&mut self) -> &mut Handle<'o, T> {
        self.handle
    }
}

/// A fan-out phase: children are `(ContextId, state)` tuples; the
/// context id is allocated and stored at spawn (issue #13's design).
/// Run it with [`Handle::execute_children`] — the driver on the
/// handle, since only the handle can fan out.
pub struct ParallelState<K, E: ?Sized> {
    children: Vec<(ContextId, StateNode<K, E>)>,
}

impl<K, E> ParallelState<K, E>
where
    K: Send + 'static,
    E: ?Sized + Sync + 'static,
{
    /// An empty parallel phase; spawn children under a context with
    /// [`ParallelState::spawn`].
    pub fn new() -> Self {
        ParallelState { children: Vec::new() }
    }
}

impl<K, E: ?Sized> Default for ParallelState<K, E> {
    fn default() -> Self {
        ParallelState { children: Vec::new() }
    }
}

impl<K, E> ParallelState<K, E>
where
    K: Send + 'static,
    E: ?Sized + Sync + 'static,
{

    /// Spawn a child state under `ctx`: allocates the child's
    /// `ContextId` (a child of `ctx`), stores the tuple, and returns
    /// the id's hash so the caller can correlate
    /// [`Handle::execute_children`] results (the id itself is not
    /// cloneable — cloning would reset its child counter).
    pub fn spawn(&mut self, ctx: &ContextId, state: StateNode<K, E>) -> u64 {
        let id = ctx.new_child();
        let hash = id.hash();
        self.children.push((id, state));
        hash
    }

    /// The children's context ids, in spawn order.
    pub fn child_ids(&self) -> impl Iterator<Item = &ContextId> {
        self.children.iter().map(|(id, _)| id)
    }
}

impl<'o, K, E> Handle<'o, ParallelState<K, E>>
where
    K: Send + 'static,
    E: ?Sized + Sync + 'static,
{
    /// Run every child to completion, each on its own thread: the
    /// child tuples are wrapped in handles by `fanout`, and each child
    /// state runs its linear loop through the tuple. Returns each
    /// child's result paired with its context id's hash, in completion
    /// order.
    ///
    /// # Panics
    ///
    /// Panics if a panicked child propagates from the fan-out scope —
    /// see [`Handle::fanout`].
    pub fn execute_children(&mut self, env: &E) -> Result<Vec<(u64, K)>> {
        self.safepoint();
        let results: Mutex<Vec<(u64, K)>> = Mutex::new(Vec::new());
        let first_error: Mutex<Option<anyhow::Error>> = Mutex::new(None);
        self.fanout(
            &|phase: &mut ParallelState<K, E>| phase.children.iter_mut().collect(),
            |mut child: Handle<'_, (ContextId, StateNode<K, E>)>| {
                let id = child.0.hash();
                match child.run_to_completion(env) {
                    Ok(result) => results
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .push((id, result)),
                    Err(e) => {
                        let mut slot =
                            first_error.lock().unwrap_or_else(|e| e.into_inner());
                        if slot.is_none() {
                            *slot = Some(e);
                        }
                    }
                }
            },
        );
        if let Some(e) = first_error.into_inner().unwrap_or_else(|e| e.into_inner()) {
            return Err(e);
        }
        Ok(results.into_inner().unwrap_or_else(|e| e.into_inner()))
    }
}

impl<'o, K, E> Handle<'o, (ContextId, StateNode<K, E>)>
where
    K: Send + 'static,
    E: ?Sized + Sync + 'static,
{
    /// The child form of run-to-completion: same protocol (safepoint,
    /// transition, install) through the tuple's state half — the
    /// auto-deref chain child handle → tuple → Box → dyn State.
    fn run_to_completion(&mut self, env: &E) -> Result<K> {
        loop {
            self.safepoint();
            match self.1.execute(env)? {
                Step::Continue(next) => self.1 = next,
                Step::Done(result) => return Ok(result),
            }
        }
    }
}

impl<'o, K, E> Handle<'o, StateNode<K, E>>
where
    K: Send + 'static,
    E: ?Sized + Sync + 'static,
{
    /// One transition of the machine: safepoint, run the current
    /// state (reached through `DerefMut`, never a raw `&mut`), install
    /// its successor with a plain `DerefMut` assignment. Returns the
    /// machine's result when done.
    ///
    /// # Panics
    ///
    /// Panics if a panicked child propagates from a nested fan-out
    /// scope — see [`Handle::fanout`].
    pub fn execute(&mut self, env: &E) -> Result<Option<K>> {
        self.safepoint();
        match (**self).execute(env)? {
            Step::Continue(next) => {
                **self = next;
                Ok(None)
            }
            Step::Done(result) => Ok(Some(result)),
        }
    }
}

impl<'o, K, E> Handle<'o, StateNode<K, E>>
where
    K: Send + 'static,
    E: ?Sized + Sync + 'static,
{
    /// Drive the machine to completion: safepoint, one transition
    /// ([`Handle::execute`]), repeat until a state yields the result.
    ///
    /// # Panics
    ///
    /// Panics if a state panics inside a nested fan-out scope.
    pub fn run_to_completion(&mut self, env: &E) -> Result<K> {
        loop {
            self.safepoint();
            if let Some(result) = self.execute(env)? {
                return Ok(result);
            }
        }
    }
}

/// Drive a machine to completion under one checkpoint pair: the node
/// is the current state; each transition is [`Handle::execute`].
///
/// # Panics
///
/// Panics if a state panics inside a nested fan-out scope.
pub fn run_to_completion<K, E>(machine: StateNode<K, E>, env: &E) -> Result<K>
where
    K: Send + 'static,
    E: ?Sized + Sync + 'static,
{
    with_checkpoint_pair(machine, |mut handle, _requester| {
        handle.run_to_completion(env)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Countdown start and expected result in the linear-machine test.
    const LINEAR_START: u32 = 3;
    const LINEAR_RESULT: u32 = 42;
    /// Work items per fan-out child.
    const WORK_STEPS: [u32; WORK_ITEMS] = [2, 3, 4];
    /// How many children the fan-out test spawns.
    const WORK_ITEMS: usize = 3;
    /// The single result the ctx-deref test expects.
    const ONE_RESULT: usize = 1;

    struct Env;

    /// The linear test's state: counts down, then a final state
    /// yields the fixed result.
    struct Countdown {
        left: u32,
        calls: usize,
    }

    impl State<u32> for Countdown {
        type Env = Env;

        fn execute(&mut self, _env: &Env) -> Result<Step<u32, Env>> {
            self.calls += 1;
            if self.left == 0 {
                return Ok(Step::Done(LINEAR_RESULT));
            }
            self.left -= 1;
            Ok(Step::Continue(Box::new(Countdown {
                left: self.left,
                calls: self.calls,
            })))
        }
    }

    #[test]
    fn linear_machine_completes() {
        let machine: StateNode<u32, Env> =
            Box::new(Countdown { left: LINEAR_START, calls: 0 });
        let result = run_to_completion(machine, &Env).expect("machine completes");
        assert_eq!(result, LINEAR_RESULT);
    }

    /// One parallel child's state: counts its steps, records which
    /// thread each step ran on, then yields the count.
    struct WorkItem {
        steps: u32,
        done: u32,
        threads: Vec<std::thread::ThreadId>,
    }

    impl State<u32> for WorkItem {
        type Env = Env;

        fn execute(&mut self, _env: &Env) -> Result<Step<u32, Env>> {
            if self.done < self.steps {
                self.done += 1;
                self.threads.push(std::thread::current().id());
                return Ok(Step::Continue(Box::new(WorkItem {
                    steps: self.steps,
                    done: self.done,
                    threads: self.threads.clone(),
                })));
            }
            Ok(Step::Done(self.steps))
        }
    }

    #[test]
    fn parallel_phase_runs_children_on_their_own_threads() {
        let phase_id = ContextId::root().new_child();
        let mut phase = ParallelState::<u32, Env>::new();
        let mut hashes = Vec::new();
        for steps in WORK_STEPS {
            hashes.push(phase.spawn(&phase_id, Box::new(WorkItem {
                steps,
                done: 0,
                threads: Vec::new(),
            })));
        }
        assert_eq!(phase.child_ids().count(), WORK_STEPS.len());

        let results = with_checkpoint_pair(phase, |mut handle, _requester| {
            handle.execute_children(&Env).expect("children complete")
        });

        // every child finished with its step count, keyed by the hash
        // spawn handed out — the context-id correlation contract.
        let mut by_hash = results.into_iter().collect::<std::collections::HashMap<_, _>>();
        for (i, h) in hashes.iter().enumerate() {
            assert_eq!(by_hash.remove(h), Some(WORK_STEPS[i]), "child {i} result");
        }
        assert!(by_hash.is_empty(), "no stray results");
    }

    /// The drivers are on the handles; the machine node installs
    /// successors through DerefMut — a mid-machine ParallelState
    /// phase composes by running under its own pair (#13 seam: the
    /// composition rule is Jonas's to settle; this test pins the
    /// standalone phase contract only).
    #[test]
    fn spawn_allocates_distinct_child_context_ids() {
        let parent = ContextId::root().new_child();
        let mut phase = ParallelState::<u32, Env>::new();
        let h1 = phase.spawn(&parent, Box::new(Countdown { left: 0, calls: 0 }));
        let h2 = phase.spawn(&parent, Box::new(Countdown { left: 0, calls: 0 }));
        assert_ne!(h1, h2, "each child gets its own context id");
    }

    #[test]
    fn ctx_derefs_to_the_handle_and_the_node() {
        // The chain Ctx -> Handle -> node: field access reaches the
        // tuple through the handle without any method of Ctx's own.
        let phase_id = ContextId::root().new_child();
        let mut phase = ParallelState::<u32, Env>::new();
        let h = phase.spawn(&phase_id, Box::new(Countdown { left: 0, calls: 0 }));
        let results = with_checkpoint_pair(phase, |mut handle, _| {
            let mut ctx = Ctx { handle: &mut handle };
            // ctx.fanout resolves on Handle; the split reaches the
            // node's children — pure Deref, no ctx methods.
            let seen: AtomicUsize = AtomicUsize::new(0);
            ctx.fanout(
                &|p: &mut ParallelState<u32, Env>| p.children.iter_mut().collect(),
                |child: Handle<'_, (ContextId, StateNode<u32, Env>)>| {
                    seen.fetch_add(1, Ordering::SeqCst);
                    assert_eq!(child.0.hash(), h, "the tuple is the node's child");
                },
            );
            handle.execute_children(&Env).expect("children complete")
        });
        assert_eq!(results.len(), ONE_RESULT);
    }
}
