//! A consumer machine whose env borrows: `MachineEnv<'a>` holds
//! `&'a Providers<'a>` — the pattern real machines (refactor-check) use.

use anyhow::Result;
use servyi_states::state::{Ctx, State, Step, run_to_completion};

struct Providers<'a> {
    calls: &'a std::sync::atomic::AtomicUsize,
}

struct MachineEnv<'a> {
    providers: &'a Providers<'a>,
}

struct BorrowingMachine {
    round: u32,
}

/// The borrowing machine's second round and its observed call count.
const SECOND_ROUND: u32 = 1;
const EXPECTED_CALLS: usize = 2;
/// A string literal is sanctioned; keep it named for symmetry anyway.
const DONE_MESSAGE: &str = "borrowed env machine done";

impl<'a> State<String, MachineEnv<'a>> for BorrowingMachine {
    fn execute(&mut self, ctx: &mut Ctx<'_>, env: &MachineEnv<'a>) -> Result<Step<String, MachineEnv<'a>>> {
        ctx.safepoint();
        let _ = env
            .providers
            .calls
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if self.round == 0 {
            self.round = SECOND_ROUND;
            Ok(Step::Next(Box::new(BorrowingMachine { round: self.round })))
        } else {
            Ok(Step::Done(DONE_MESSAGE.to_string()))
        }
    }
}

#[test]
fn machine_with_borrowed_env_runs_to_completion() {
    let calls = std::sync::atomic::AtomicUsize::new(0);
    let providers = Providers { calls: &calls };
    let env = MachineEnv { providers: &providers };
    let result = run_to_completion(Box::new(BorrowingMachine { round: 0 }), &env).expect("run");
    assert_eq!(result, DONE_MESSAGE);
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), EXPECTED_CALLS);
}
