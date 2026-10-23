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

/// A node that holds the current state and can replace it — the
/// "derefs into a state" capability the generalized drivers need
/// ([`Handle::execute`], [`Handle::run_to_completion`]): reading the
/// state through `&mut dyn` is not enough, installing a successor
/// needs the slot.
pub trait StateSlot {
    /// The machine's result type.
    type K: Send + 'static;
    /// The machine's environment type.
    type E: ?Sized + Sync + 'static;
    /// The current state.
    fn state(&mut self) -> &mut (dyn State<Self::K, Env = Self::E> + Send + Sync);
    /// Install the successor.
    fn install(&mut self, next: StateNode<Self::K, Self::E>);
}

impl<K, E> StateSlot for StateNode<K, E>
where
    K: Send + 'static,
    E: ?Sized + Sync + 'static,
{
    type K = K;
    type E = E;
    fn state(&mut self) -> &mut (dyn State<K, Env = E> + Send + Sync) {
        &mut **self
    }
    fn install(&mut self, next: StateNode<K, E>) {
        *self = next;
    }
}

/// One parallel child: the state and its LANDING SLOT — the result is
/// written in place, through the child's own handle, by the worker
/// that ran it (no lock, and the parent reads the slots in spawn
/// order after the join).
pub struct ParallelChild<K, E: ?Sized> {
    state: StateNode<K, E>,
    result: Option<K>,
}

impl<K, E> StateSlot for ParallelChild<K, E>
where
    K: Send + 'static,
    E: ?Sized + Sync + 'static,
{
    type K = K;
    type E = E;
    fn state(&mut self) -> &mut (dyn State<K, Env = E> + Send + Sync) {
        &mut *self.state
    }
    fn install(&mut self, next: StateNode<K, E>) {
        self.state = next;
    }
}

/// A fan-out phase: allocates each child's [`ContextId`] itself and
/// hands the id to the state's CONSTRUCTOR (the state owns its id
/// from birth); children are [`ParallelChild`] slots — state + result
/// — wrapped in handles by `fanout`, each child run to completion by
/// the generalized [`Handle::run_to_completion`]. When every child is
/// done, `then` builds the phase's successor from the results, in
/// spawn order.
///
/// [`State::execute`] for a `ParallelState` runs the phase under its
/// own checkpoint pair (one pair per phase — the composition trade
/// documented on refactor-check#10: a state inside a node cannot be
/// dispatched and handed the handle at once, so a fanning state is
/// the node of its own pair).
pub struct ParallelState<K, E: ?Sized> {
    id: ContextId,
    children: Vec<ParallelChild<K, E>>,
    then: Box<dyn FnOnce(Vec<K>) -> StateNode<K, E> + Send + Sync>,
}

impl<K, E> ParallelState<K, E>
where
    K: Send + 'static,
    E: ?Sized + Sync + 'static,
{
    /// A phase under `ctx`: allocates the phase's own context id (a
    /// child of `ctx`); spawn children with [`ParallelState::spawn`],
    /// name the continuation with [`ParallelState::then`].
    pub fn new(ctx: &ContextId) -> Self {
        ParallelState {
            id: ctx.new_child(),
            children: Vec::new(),
            then: Box::new(default_then),
        }
    }

    /// Spawn a child from a CONSTRUCTOR that receives the child's
    /// freshly allocated context id — the state embeds it at birth,
    /// so the id travels inside the state, not beside it.
    pub fn spawn(&mut self, make: impl FnOnce(&ContextId) -> StateNode<K, E>) {
        let child_id = self.id.new_child();
        self.children.push(ParallelChild {
            state: make(&child_id),
            result: None,
        });
    }

    /// The continuation: builds the phase's successor from the
    /// children's results, in spawn order.
    pub fn then(
        mut self,
        then: impl FnOnce(Vec<K>) -> StateNode<K, E> + Send + Sync + 'static,
    ) -> Self {
        self.then = Box::new(then);
        self
    }
}

fn default_then<K, E: ?Sized>(_results: Vec<K>) -> StateNode<K, E> {
    unreachable!("ParallelState::then must name the continuation before running")
}


impl<K, E> State<K> for ParallelState<K, E>
where
    K: Send + Sync + 'static,
    E: ?Sized + Sync + 'static,
{
    type Env = E;

    fn execute(&mut self, env: &E) -> Result<Step<K, Self::Env>> {
        // One pair per phase: the children move into the phase's own
        // tree for the run and land back in their slots.
        let children = std::mem::take(&mut self.children);
        let mut landed = with_checkpoint_pair(children, |mut phase, _requester| {
            phase.safepoint();
            phase.fanout(
                &|kids: &mut Vec<ParallelChild<K, E>>| kids.iter_mut().collect(),
                |mut child: Handle<'_, ParallelChild<K, E>>| {
                    let result = child.run_to_completion(env).expect("child machine completes");
                    child.result = Some(result);
                },
            );
            std::mem::take(&mut *phase)
        });
        let results = landed.iter_mut().map(|c| c.result.take()).collect::<Option<Vec<_>>>();
        let results = results.unwrap_or_else(|| {
            unreachable!("every child ran to completion and landed its result")
        });
        self.children = landed;
        let then = std::mem::replace(&mut self.then, Box::new(default_then));
        Ok(Step::Continue(then(results)))
    }
}

impl<'o, T> Handle<'o, T>
where
    T: StateSlot + 'o,
{
    /// One transition: safepoint, run the current state through the
    /// slot, install its successor. Returns the machine's result when
    /// done.
    ///
    /// # Panics
    ///
    /// Panics if a panicked child propagates from a nested fan-out
    /// scope — see [`Handle::fanout`].
    pub fn execute(&mut self, env: &T::E) -> Result<Option<T::K>> {
        self.safepoint();
        match self.state().execute(env)? {
            Step::Continue(next) => {
                self.install(next);
                Ok(None)
            }
            Step::Done(result) => Ok(Some(result)),
        }
    }

    /// Drive the machine to completion: safepoint, one transition
    /// ([`Handle::execute`]), repeat until a state yields the result.
    /// Generalized over every [`StateSlot`] — the machine node and
    /// the parallel children run the same protocol.
    ///
    /// # Panics
    ///
    /// Panics if a state panics inside a nested fan-out scope.
    pub fn run_to_completion(&mut self, env: &T::E) -> Result<T::K> {
        loop {
            self.safepoint();
            if let Some(result) = self.execute(env)? {
                return Ok(result);
            }
        }
    }
}

impl<'o, T> Handle<'o, T> {
    /// Sugar for call sites that carry both: the [`Ctx`] is pure
    /// `Deref`/`DerefMut` over this handle (there is no special
    /// `ctx.fanout`, only `handle.fanout`).
    pub fn as_ctx(&mut self) -> Ctx<'_, 'o, T> {
        Ctx { handle: self }
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
    use std::sync::Mutex;

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

    impl State<u32> for Countdown {
        type Env = Env;

        fn execute(&mut self, _env: &Env) -> Result<Step<u32, Env>> {
            if self.left == 0 {
                return Ok(Step::Done(LINEAR_RESULT));
            }
            self.left -= 1;
            Ok(Step::Continue(Box::new(Countdown { left: self.left })))
        }
    }

    #[test]
    fn linear_machine_completes() {
        let machine: StateNode<u32, Env> = Box::new(Countdown { left: LINEAR_START });
        let result = run_to_completion(machine, &Env).expect("machine completes");
        assert_eq!(result, LINEAR_RESULT);
    }

    /// One parallel child: its context id arrives at CONSTRUCTION and
    /// is embedded (the constructor design); it counts its steps and
    /// yields the id's hash, so results are correlated by what the
    /// state itself carries.
    struct WorkItem {
        id_hash: u64,
        steps: u32,
        done: u32,
    }

    impl State<u64> for WorkItem {
        type Env = Env;

        fn execute(&mut self, _env: &Env) -> Result<Step<u64, Env>> {
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

    /// The phase's continuation: receives the landed results (in
    /// spawn order) and yields their sum.
    struct Sum(Vec<u64>);

    impl State<u64> for Sum {
        type Env = Env;

        fn execute(&mut self, _env: &Env) -> Result<Step<u64, Env>> {
            Ok(Step::Done(self.0.iter().sum()))
        }
    }

    /// The whole composition: the PHASE is a state inside the dyn
    /// machine — spawn allocates each child's context id and hands it
    /// to the constructor; children land their results in their slots
    /// in spawn order; `then` builds the continuation from them; the
    /// machine's result is the sum.
    #[test]
    fn parallel_phase_composes_inside_a_machine() {
        let hashes: Mutex<Vec<u64>> = Mutex::new(Vec::new());
        let mut phase = ParallelState::<u64, Env>::new(ContextId::root());
        for steps in WORK_STEPS {
            phase.spawn({
                let hashes = &hashes;
                move |id: &ContextId| {
                    hashes.lock().expect("hash vec").push(id.hash());
                    Box::new(WorkItem { id_hash: id.hash(), steps, done: 0 })
                        as StateNode<u64, Env>
                }
            });
        }
        let expected: Vec<u64> = hashes.into_inner().expect("hash vec");
        assert_eq!(expected.len(), WORK_ITEMS);

        let machine: StateNode<u64, Env> =
            Box::new(phase.then(|results| Box::new(Sum(results)) as StateNode<u64, Env>));
        let result = run_to_completion(machine, &Env).expect("machine completes");
        let sum: u64 = expected.iter().sum();
        assert_eq!(result, sum, "the continuation saw the landed results");
    }

    #[test]
    fn spawn_allocates_distinct_child_context_ids() {
        let parent = ContextId::root().new_child();
        let mut phase = ParallelState::<u32, Env>::new(&parent);
        let mut seen = Vec::new();
        for _ in 0..WORK_ITEMS {
            phase.spawn(|id: &ContextId| {
                seen.push(id.hash());
                Box::new(Countdown { left: 0 }) as StateNode<u32, Env>
            });
        }
        let unique: std::collections::HashSet<_> = seen.iter().copied().collect();
        assert_eq!(unique.len(), WORK_ITEMS, "each child gets its own context id");
    }

    /// The Ctx chain: `ctx.fanout` resolves on `Handle` and the split
    /// reaches the node — pure Deref/DerefMut, no methods of Ctx's
    /// own (an empty fan-out; the deref shape is the point).
    #[test]
    fn ctx_derefs_to_the_handle_and_the_node() {
        with_checkpoint_pair(
            ParallelState::<u64, Env>::new(ContextId::root()),
            |mut handle, _| {
                let mut ctx = handle.as_ctx();
                ctx.fanout(
                    &|_p: &mut ParallelState<u64, Env>| Vec::new(),
                    |_child: Handle<'_, ParallelChild<u64, Env>>| unreachable!("no children"),
                );
            },
        );
    }
}
