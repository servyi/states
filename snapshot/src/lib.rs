//! serde-based snapshot save/restore, built ON TOP of the checkpoint
//! protocol — deliberately in its own crate so the checkpointer itself
//! stays an isolated, reusable protocol with no serialization baked in.
//!
//! Two save paths, both writing atomic JSON snapshots (temp file +
//! rename, so a crash mid-write never corrupts a previous one):
//!
//! - [`save_handle`] — mutator-side: at any point where dereferencing
//!   the handle is legal, the mutator thread owns the world, so no
//!   handshake is needed and the call can never block (self-save cannot
//!   deadlock).
//! - [`save_requester`] / [`try_save_requester`] — snapshotter-side:
//!   stop-the-world through the requester, then serialize through the
//!   guard. `try_save_requester` skips the round on timeout instead of
//!   wedging.
//!
//! [`restore_from`] deserializes a snapshot; feed the value straight back
//! into [`servyi_states::checkpoint::with_checkpoint_pair`] to resume.
//! Because fan-out actions store their results INSIDE the node, a
//! snapshot taken mid-fan-out records which children already completed,
//! and the resumed run skips them.

use std::path::Path;

use servyi_states::checkpoint::{CheckpointRequester, Handle};
use std::time::Duration;

/// Mutator-side save: serialize the handle's node atomically. Call at any
/// safepoint-legal point — this thread owns the world, so this never
/// blocks and can never deadlock against itself.
pub fn save_handle<T: serde::Serialize>(
    handle: &Handle<'_, T>,
    path: impl AsRef<Path>,
) -> std::io::Result<()> {
    write_snapshot(&**handle, path.as_ref())
}

/// Snapshotter-side save: stop the world (blocks until the mutator parks,
/// like [`CheckpointRequester::request`]) and serialize the node.
pub fn save_requester<T: serde::Serialize + Send>(
    requester: &CheckpointRequester<'_, T>,
    path: impl AsRef<Path>,
) -> std::io::Result<()> {
    let Some(guard) = requester.request() else {
        // A closed pair hands out the guard immediately; a live one
        // parks eventually. Not reached in practice.
        return Ok(());
    };
    write_snapshot(&*guard, path.as_ref())
}

/// [`save_requester`] with a deadline: `Ok(false)` means the world did
/// not stop in time and nothing was written.
pub fn try_save_requester<T: serde::Serialize + Send>(
    requester: &CheckpointRequester<'_, T>,
    path: impl AsRef<Path>,
    timeout: Duration,
) -> std::io::Result<bool> {
    match requester.request_timeout(timeout) {
        Some(guard) => write_snapshot(&*guard, path.as_ref()).map(|()| true),
        None => Ok(false),
    }
}

/// Deserialize a node from a snapshot written by the save functions.
pub fn restore_from<T: serde::de::DeserializeOwned>(
    path: impl AsRef<Path>,
) -> std::io::Result<T> {
    let text = std::fs::read_to_string(path.as_ref())?;
    serde_json::from_str(&text)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

/// Serialize the (quiesced or owned) node to `path`, atomically.
fn write_snapshot<T: serde::Serialize>(node: &T, path: &Path) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp");
    {
        let file = std::fs::File::create(&tmp)?;
        serde_json::to_writer(file, node)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    }
    std::fs::rename(&tmp, path)
}

/// Periodic snapshot taker for the snapshotter thread: saves the world to
/// one file every `every` (last complete write wins) until `stop` is set.
/// A world that will not stop within one period skips that round, so a
/// busy machine never wedges the snapshotter.
pub struct Snapshotter<'a, T> {
    requester: CheckpointRequester<'a, T>,
    path: std::path::PathBuf,
    every: Duration,
}

impl<'a, T: serde::Serialize + Send> Snapshotter<'a, T> {
    pub fn new(
        requester: CheckpointRequester<'a, T>,
        path: impl Into<std::path::PathBuf>,
        every: Duration,
    ) -> Self {
        Self { requester, path: path.into(), every }
    }

    /// Save every `every` until `stop` is set; returns the number of
    /// snapshots written.
    pub fn run(&self, stop: &std::sync::atomic::AtomicBool) -> std::io::Result<u64> {
        use std::sync::atomic::Ordering;
        let mut written: u64 = 0;
        while !stop.load(Ordering::SeqCst) {
            std::thread::sleep(self.every);
            if stop.load(Ordering::SeqCst) {
                break;
            }
            if try_save_requester(&self.requester, &self.path, self.every)? {
                written += 1;
            }
        }
        Ok(written)
    }
}
