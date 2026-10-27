//! The state machine on the checkpoint protocol — 100% safe code over
//! the public [`Handle`] API (issue #13).
//!
//! The machine node is [`Context`]: the state's context id and the
//! state, side by side. The environment is an EXPLICIT parameter of
//! the trait (`State<K, E>`), and the node carries the object
//! lifetime (`Context<'e, K, E>`), so `E` may borrow — a machine can
//! hold `E = MachineEnv<'e>` over stack providers for the whole run.
//! Through the deref chain
//! `&mut Handle<Context>` → `&mut Context` → `&mut dyn State` the
//! driver calls the state's `execute` and installs successors with a
//! plain field assignment (`h.state = next`) — no `unsafe`, no
//! escape hatches, nothing added to [`crate::checkpoint`].
//!
//! Fan-outs live inside a state's `execute`: the driver splits its
//! handle into the node and a [`Ctx`] over the same tree, so a state
//! runs against `&mut self` while its `ctx` fans the state's own
//! child slots out — every child a member of the ONE enclosing tree,
//! checkpointable from the root. A fan-out phase is the concrete
//! [`ParallelState`], itself a [`State`]: it allocates each child's
//! [`ContextId`] at spawn, its `execute` contains the fan-out over
//! its children, results land in each child's own slot (read in spawn
//! order after the join), and `then` names the continuation built
//! from them.

use anyhow::Result;
use std::sync::Mutex;

use crate::checkpoint::{Ctx, Handle};
use crate::context_id::ContextId;

/// The machine node: the state's context id and the state. Derefs all
/// the way to the state (`&mut Context` → `&mut dyn State`), so a
/// handle over a Context drives the machine directly.
pub struct Context<'e, K, E: ?Sized + 'e> {
    /// This state's context id.
    pub id: ContextId,
    /// The current state.
    pub state: Box<dyn State<K, E> + Send + Sync + 'e>,
    /// The child's landing slot: a parallel worker writes its result
    /// here, through its own handle; the machine driver never touches
    /// it.
    pub result: Option<K>,
}

impl<'e, K, E: ?Sized + 'e> Context<'e, K, E> {
    /// A node for `state` under `ctx`: allocates the state's context
    /// id (a child of `ctx`).
    pub fn new(ctx: &ContextId, state: Box<dyn State<K, E> + Send + Sync + 'e>) -> Self {
        Context { id: ctx.new_child(), state, result: None }
    }
}

impl<'e, K, E: ?Sized + 'e> std::ops::Deref for Context<'e, K, E> {
    type Target = dyn State<K, E> + Send + Sync + 'e;

    fn deref(&self) -> &Self::Target {
        &*self.state
    }
}

impl<'e, K, E: ?Sized + 'e> std::ops::DerefMut for Context<'e, K, E> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut *self.state
    }
}

/// What a state decides: its successor, or the machine's result.
pub enum Step<'e, K, E: ?Sized + 'e> {
    Continue(Box<dyn State<K, E> + Send + Sync + 'e>),
    Done(K),
}

/// A state: one transition through `&mut self`. A state whose work
/// fans out passes a field of itself (its child slots) to
/// [`Ctx::fanout`] — the fan-out lives inside `execute`, in the
/// enclosing tree.
pub trait State<K, E: ?Sized>: Send + Sync
where
    K: Send + 'static,
    E: Sync,
{
    /// One transition: run, and name the successor or the result.
    /// `E` is an explicit parameter so it may carry a lifetime — the
    /// run's borrow of the providers. The successor's own object
    /// lifetime `'r` is caller-chosen (the run's `'e` at the driver);
    /// it must NOT elide to the `&mut self` borrow, or the successor
    /// would appear to borrow the state it replaces.
    fn execute<'r>(&mut self, ctx: &mut Ctx<'_, '_>, env: &E) -> Result<Step<'r, K, E>>;
}

/// The continuation of a fan-out phase: receives the children's
/// results in spawn order, builds the phase's successor state.
pub type PhaseThen<K, E> = Box<dyn FnOnce(Vec<K>) -> Box<dyn State<K, E> + Send + Sync> + Send + Sync>;

/// A fan-out phase: allocates each child's [`ContextId`] itself and
/// hands the id to the state's constructor; the children are
/// [`Context`]s of this same tree. Run it with
/// [`Handle::run_children`], or compose it into a machine as an
/// ordinary [`State`] (its `execute` contains the fan-out).
pub struct ParallelState<'e, K, E: ?Sized + Sync + 'static + 'e> {
    id: ContextId,
    children: Vec<Context<'e, K, E>>,
    /// The continuation: builds the phase's successor from the
    /// children's results, in spawn order. Named with
    /// [`ParallelState::then`] before the phase runs; the successor
    /// box is `'static`, so the phase requires `E: 'static` (a state
    /// with its own child slots fans out through [`Ctx::fanout`]
    /// under any env).
    then: Option<PhaseThen<K, E>>,
}

impl<'e, K, E> ParallelState<'e, K, E>
where
    K: Send + Sync + 'static,
    E: ?Sized + Sync + 'static,
{
    /// A phase under `ctx`: allocates the phase's own context id (a
    /// child of `ctx`); spawn children with [`ParallelState::spawn`].
    pub fn new(ctx: &ContextId) -> Self {
        ParallelState { id: ctx.new_child(), children: Vec::new(), then: None }
    }

    /// The continuation: `then` receives the children's results, in
    /// spawn order, and builds the phase's successor state.
    pub fn then(
        mut self,
        then: impl FnOnce(Vec<K>) -> Box<dyn State<K, E> + Send + Sync> + Send + Sync + 'static,
    ) -> Self {
        self.then = Some(Box::new(then));
        self
    }

    /// Spawn a child from a CONSTRUCTOR that receives the child's
    /// freshly allocated context id — the state embeds it at birth.
    pub fn spawn(
        &mut self,
        make: impl FnOnce(&ContextId) -> Box<dyn State<K, E> + Send + Sync + 'e>,
    ) {
        let child_id = self.id.new_child();
        self.children.push(Context::new(&self.id, make(&child_id)));
    }

    /// Drive every child to completion through `ctx` — the fan-out
    /// itself lives here: one dedicated thread per child, each child
    /// run to completion through its own handle, results landing in
    /// the children's slots, returned in spawn order. The first child
    /// error aborts the phase.
    fn drive(&mut self, ctx: &mut Ctx<'_, '_>, env: &E) -> Result<Vec<K>> {
        ctx.safepoint();
        let first_error: Mutex<Option<anyhow::Error>> = Mutex::new(None);
        ctx.fanout(
            &mut self.children,
            |mut child: Handle<'_, Context<'e, K, E>>| match child.run_to_completion(env) {
                Ok(result) => child.result = Some(result),
                Err(e) => {
                    let mut slot = first_error.lock().unwrap_or_else(|e| e.into_inner());
                    if slot.is_none() {
                        *slot = Some(e);
                    }
                }
            },
        );
        if let Some(e) = first_error.into_inner().unwrap_or_else(|e| e.into_inner()) {
            return Err(e);
        }
        Ok(self
            .children
            .iter_mut()
            .map(|c| c.result.take())
            .collect::<Option<Vec<_>>>()
            .unwrap_or_else(|| unreachable!("every child ran to completion and landed")))
    }
}

/// A phase is itself a state: its `execute` contains the fan-out over
/// its children (in the enclosing tree, checkpointable from the root),
/// and continues into the state its `then` builds from the landed
/// results. Composes under ONE checkpoint pair — no pair of its own.
impl<'e, K, E> State<K, E> for ParallelState<'e, K, E>
where
    K: Send + Sync + 'static,
    E: ?Sized + Sync + 'static,
{
    fn execute<'r>(&mut self, ctx: &mut Ctx<'_, '_>, env: &E) -> Result<Step<'r, K, E>> {
        let results = self.drive(ctx, env)?;
        let then = self
            .then
            .take()
            .expect("ParallelState::then must name the continuation before running");
        Ok(Step::Continue(then(results)))
    }
}

impl<'o, 'e, K, E> Handle<'o, ParallelState<'e, K, E>>
where
    K: Send + Sync + 'static,
    E: ?Sized + Sync + 'static,
    'e: 'o,
{
    /// Run every child to completion, each on its own thread: the
    /// children — [`Context`]s of this same tree — are wrapped in
    /// handles by `fanout` and run by the same
    /// [`Handle::run_to_completion`]. Each result lands in its
    /// child's own slot; the joined slots are returned in spawn
    /// order. The first child error aborts the phase.
    ///
    /// # Panics
    ///
    /// Panics if a panicked child propagates from the fan-out scope —
    /// see [`Handle::fanout`].
    pub fn run_children(&mut self, env: &E) -> Result<Vec<K>> {
        self.safepoint();
        let (phase, mut ctx) = self.step_parts();
        phase.drive(&mut ctx, env)
    }
}

impl<'o, 'e, K, E> Handle<'o, Context<'e, K, E>>
where
    K: Send + 'static,
    E: ?Sized + Sync,
{
    /// Drive the machine to completion: safepoint, run the current
    /// state through the deref chain, install its successor with a
    /// plain field assignment, repeat until a state yields the
    /// result.
    ///
    /// # Panics
    ///
    /// Panics if a panicked child propagates from a fan-out scope —
    /// see [`Handle::fanout`].
    pub fn run_to_completion(&mut self, env: &E) -> Result<K> {
        loop {
            self.safepoint();
            // The handle splits into the node and a Ctx over the same
            // tree: the state's own execute runs against the node
            // while its ctx carries the tree facilities.
            let step = {
                let (node, mut ctx) = self.step_parts();
                node.execute(&mut ctx, env)?
            };
            match step {
                Step::Continue(next) => self.state = next,
                Step::Done(result) => return Ok(result),
            }
        }
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::with_checkpoint_pair;

    /// Countdown start in the linear-machine test.
    const LINEAR_START: u32 = 3;
    /// The linear test's fixed result.
    const LINEAR_RESULT: u32 = 42;
    /// Work steps per parallel child.
    const WORK_STEPS: [u32; WORK_ITEMS] = [2, 3, 4];
    /// How many children the parallel phase spawns.
    const WORK_ITEMS: usize = 3;

    struct Env;

    /// The linear test's state: counts down, then yields the fixed
    /// result.
    struct Countdown {
        left: u32,
    }

    impl State<u32, Env> for Countdown {
        fn execute<'r>(&mut self, _ctx: &mut Ctx<'_, '_>, _env: &Env) -> Result<Step<'r, u32, Env>> {
            if self.left == 0 {
                return Ok(Step::Done(LINEAR_RESULT));
            }
            self.left -= 1;
            Ok(Step::Continue(Box::new(Countdown { left: self.left })))
        }
    }

    #[test]
    fn linear_machine_completes() {
        let machine = Context::new(ContextId::root(), Box::new(Countdown { left: LINEAR_START }));
        let result = with_checkpoint_pair(machine, |mut handle, _requester| {
            handle.run_to_completion(&Env)
        })
        .expect("machine completes");
        assert_eq!(result, LINEAR_RESULT);
    }

    /// One parallel child: its context id arrives at CONSTRUCTION and
    /// is embedded; it counts its steps and yields the id's hash, so
    /// results are correlated by what the state itself carries.
    struct WorkItem {
        id_hash: u64,
        steps: u32,
        done: u32,
    }

    impl State<u64, Env> for WorkItem {
        fn execute<'r>(&mut self, _ctx: &mut Ctx<'_, '_>, _env: &Env) -> Result<Step<'r, u64, Env>> {
            if self.done < self.steps {
                self.done += 1;
                return Ok(Step::Continue(Box::new(WorkItem {
                    id_hash: self.id_hash,
                    steps: self.steps,
                    done: self.done,
                })));
            }
            Ok(Step::Done(self.id_hash))
        }
    }

    /// The phase under its own pair: spawn allocates each child's
    /// context id and hands it to the constructor; children run on
    /// their own threads through the SAME tree; results land in the
    /// children's slots, in spawn order.
    #[test]
    fn parallel_children_run_and_land_in_spawn_order() {
        let hashes: Mutex<Vec<u64>> = Mutex::new(Vec::new());
        let mut phase = ParallelState::<u64, Env>::new(ContextId::root());
        for steps in WORK_STEPS {
            phase.spawn({
                let hashes = &hashes;
                move |id: &ContextId| {
                    hashes.lock().expect("hash vec").push(id.hash());
                    Box::new(WorkItem { id_hash: id.hash(), steps, done: 0 })
                        as Box<dyn State<u64, Env> + Send + Sync>
                }
            });
        }
        let expected: Vec<u64> = hashes.into_inner().expect("hash vec");
        assert_eq!(expected.len(), WORK_ITEMS);

        let results = with_checkpoint_pair(phase, |mut handle, _requester| {
            handle.run_children(&Env).expect("children complete")
        });
        assert_eq!(results, expected, "slots landed in spawn order");
    }

    /// The restored composition: a fan-out phase sits in the machine's
    /// successor chain as an ordinary state; its execute contains the
    /// fan-out (in the enclosing tree), the continuation is built from
    /// the landed results. One pair for the whole workload.
    #[test]
    fn phase_as_state_fans_out_inside_the_machine() {
        // Lead: counts down, then continues into the phase.
        struct Lead {
            left: u32,
        }

        impl State<u64, Env> for Lead {
            fn execute<'r>(
                &mut self,
                _ctx: &mut Ctx<'_, '_>,
                _env: &Env,
            ) -> Result<Step<'r, u64, Env>> {
                if self.left == 0 {
                    let mut phase = ParallelState::<u64, Env>::new(ContextId::root());
                    for steps in WORK_STEPS {
                        phase.spawn(|_id| {
                            Box::new(WorkItem { id_hash: steps as u64, steps, done: 0 })
                                as Box<dyn State<u64, Env> + Send + Sync>
                        });
                    }
                    return Ok(Step::Continue(Box::new(phase.then(|results| {
                        // The continuation: counts the sum down to zero,
                        // then yields it.
                        Box::new(Sums { left: results.iter().sum(), sum: results.iter().sum() })
                            as Box<dyn State<u64, Env> + Send + Sync>
                    }))));
                }
                self.left -= 1;
                Ok(Step::Continue(Box::new(Lead { left: self.left })))
            }
        }

        /// Counts the sum down to zero, then yields it.
        struct Sums {
            left: u64,
            sum: u64,
        }

        impl State<u64, Env> for Sums {
            fn execute<'r>(
                &mut self,
                _ctx: &mut Ctx<'_, '_>,
                _env: &Env,
            ) -> Result<Step<'r, u64, Env>> {
                if self.left == 0 {
                    return Ok(Step::Done(self.sum));
                }
                self.left -= 1;
                Ok(Step::Continue(Box::new(Sums { left: self.left, sum: self.sum })))
            }
        }

        let machine = Context::new(ContextId::root(), Box::new(Lead { left: LINEAR_START }));
        let result = with_checkpoint_pair(machine, |mut handle, _requester| {
            handle.run_to_completion(&Env)
        })
        .expect("machine with an inner phase completes");
        let expected: u64 = WORK_STEPS.iter().map(|s| *s as u64).sum();
        assert_eq!(result, expected, "the continuation ran to the summed result");
    }

    #[test]
    fn spawn_allocates_distinct_child_context_ids() {
        let parent = ContextId::root().new_child();
        let mut phase = ParallelState::<u32, Env>::new(&parent);
        let mut seen = Vec::new();
        for _ in 0..WORK_ITEMS {
            phase.spawn(|id: &ContextId| {
                seen.push(id.hash());
                Box::new(Countdown { left: 0 }) as Box<dyn State<u32, Env> + Send + Sync>
            });
        }
        let unique: std::collections::HashSet<_> = seen.iter().copied().collect();
        assert_eq!(unique.len(), WORK_ITEMS, "each child gets its own context id");
    }
    /// The label the borrowing-env test clones out of its provider.
    const BORROWED_LABEL: &str = "borrowed";

    /// The run-lifetime contract: a machine whose environment BORROWS
    /// stack providers — `E = MachineEnv<'e>` — with no `'static`
    /// anywhere. This is the shape refactor-check needs.
    struct Providers {
        label: String,
    }

    struct MachineEnv<'a> {
        providers: &'a Providers,
    }

    struct Borrowing {
        left: u32,
    }

    impl<'a> State<String, MachineEnv<'a>> for Borrowing {
        fn execute<'r>(
            &mut self,
            _ctx: &mut Ctx<'_, '_>,
            env: &MachineEnv<'a>,
        ) -> Result<Step<'r, String, MachineEnv<'a>>> {
            if self.left == 0 {
                return Ok(Step::Done(env.providers.label.clone()));
            }
            self.left -= 1;
            Ok(Step::Continue(Box::new(Borrowing { left: self.left })))
        }
    }

    #[test]
    fn machine_env_may_borrow_for_the_run() {
        let providers = Providers { label: String::from(BORROWED_LABEL) };
        let env = MachineEnv { providers: &providers };
        let machine = Context::new(ContextId::root(), Box::new(Borrowing { left: LINEAR_START }));
        let result = with_checkpoint_pair(machine, |mut handle, _requester| {
            handle.run_to_completion(&env)
        })
        .expect("borrowing machine completes");
        assert_eq!(result, providers.label);
    }
}
