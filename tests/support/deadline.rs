//! Every test runs under a deadline.
//!
//! The library's waits are blocking by design (a park until the world
//! resumes, a claim until the holder releases). A test that exercises
//! them cannot be made to "time out" from inside — the only honest way
//! to keep a regression from hanging the suite is from outside: a test
//! still running when its deadline passes has hit a wait that should have
//! completed. The watchdog then reports the test by name and aborts the
//! test binary, so the run FAILS within a bounded time instead of being
//! waited out.
//!
//! One definition for the whole workspace: the integration tests include
//! it by path, and so do the library's own unit tests and the snapshot
//! crate's tests.
//!
//! Usage — the guard declared FIRST in the test, so it is dropped LAST:
//!
//! ```ignore
//! #[test]
//! fn something() {
//!     let _deadline = deadline::Deadline::start();
//!     // ...
//! }
//! ```
//!
//! The guard records where it was started — the test's file and line —
//! so the report names the hung test without anyone having to label it.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// How long one test may take. The slowest healthy test takes well under
/// this — natively and under Miri, loaded or not — so the deadline only
/// ever fires on a hang.
pub const TEST_DEADLINE: Duration = Duration::from_secs(60);

/// How often the watchdog looks at the clock.
const WATCHDOG_POLL: Duration = Duration::from_millis(50);

/// Write straight to the process's stderr. libtest captures each test's
/// output, and a thread the test spawns inherits that capture — so a
/// plain `eprintln!` from the watchdog would land in a buffer that dies
/// with the aborted test, exactly when it is needed.
/// The process's standard error, by the POSIX convention.
#[cfg(unix)]
const STDERR_DESCRIPTOR: std::os::fd::RawFd = 2;

#[cfg(unix)]
fn report_to_stderr(message: &str) {
    use std::io::Write;
    use std::os::fd::{FromRawFd, IntoRawFd};
    // SAFETY: the descriptor is the process's stderr, open for the
    // process's whole life; it is taken back out right after the write,
    // so nothing is closed.
    let stderr = unsafe { std::fs::File::from_raw_fd(STDERR_DESCRIPTOR) };
    let mut stderr = stderr;
    let _ = stderr.write_all(message.as_bytes());
    let _descriptor_stays_open = stderr.into_raw_fd();
}

/// Without a descriptor to borrow, the best effort is the captured stream.
#[cfg(not(unix))]
fn report_to_stderr(message: &str) {
    eprintln!("{message}");
}

/// A test's deadline. Live from `start` until dropped.
pub struct Deadline {
    done: Arc<AtomicBool>,
    watchdog: Option<std::thread::JoinHandle<()>>,
}

impl Deadline {
    /// Arm the deadline for the test this is called from. Call it as the
    /// test's first statement: the file and line of the call are what the
    /// report names.
    #[track_caller]
    pub fn start() -> Self {
        let started_at = std::panic::Location::caller();
        let done = Arc::new(AtomicBool::new(false));
        let seen = Arc::clone(&done);
        let watchdog = std::thread::spawn(move || {
            let give_up_at = Instant::now() + TEST_DEADLINE;
            while !seen.load(Ordering::SeqCst) {
                if Instant::now() >= give_up_at {
                    let file = started_at.file();
                    let line = started_at.line();
                    report_to_stderr(&format!(
                        "\nHANG: the test at {file}:{line} did not finish within \
                         {TEST_DEADLINE:?}. A wait that should have completed did not: the code \
                         under test is hung. Aborting the test binary so this is reported as a \
                         failure.\n"
                    ));
                    std::process::abort();
                }
                std::thread::sleep(WATCHDOG_POLL);
            }
        });
        Deadline { done, watchdog: Some(watchdog) }
    }
}

impl Drop for Deadline {
    fn drop(&mut self) {
        self.done.store(true, Ordering::SeqCst);
        if let Some(watchdog) = self.watchdog.take() {
            let _finished = watchdog.join();
        }
    }
}
