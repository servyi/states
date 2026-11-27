//! serde-based snapshot save/restore, built ON TOP of the checkpoint
//! protocol — deliberately in its own crate so the checkpointer itself
//! stays vocabulary-free.
//!
//! Two sides, mirroring the protocol's:
//!
//! - mutator-side ([`save_handle`]): serialize the inner handle's node
//!   atomically; call at any safepoint-legal point.
//!
//! - snapshotter-side ([`save_requester`] / [`try_save_requester`]):
//!   stop-the-world through the checkpointer, then serialize through
//!   the guard. `try_save_requester` skips the round on timeout
//!   instead of blocking, so a busy machine never wedges the
//!   snapshotter.
//!
//! [`restore_from`] rebuilds a node from a snapshot; feed the node
//! into [`servyi_states::checkpoint::independent_task_root`] to resume.

use std::path::Path;
use std::time::Duration;

use servyi_states::checkpoint::{Checkpointer, InnerHandle, SnapshotSlot};

/// Mutator-side save: serialize the handle's node atomically. Call at any
/// safepoint-legal point — this thread owns the world, so this never
/// blocks and can never deadlock against itself.
pub fn save_handle<T: serde::Serialize>(
    handle: &InnerHandle<T>,
    path: impl AsRef<Path>,
) -> std::io::Result<()> {
    write_snapshot(&**handle, path.as_ref())
}

/// Snapshotter-side save: stop the world (blocks until the mutator
/// parks, like [`Checkpointer::request`]) and serialize the node the
/// slot holds.
pub fn save_requester<K>(
    requester: &Checkpointer<'_, K>,
    path: impl AsRef<Path>,
) -> std::io::Result<()>
where
    K: SnapshotSlot + Send,
    K::Node: serde::Serialize + Sized,
{
    let guard = requester.request();
    write_snapshot(guard.snapshot_node(), path.as_ref())
}

/// [`save_requester`] with a deadline: `Ok(false)` means the world did
/// not stop in time and nothing was written.
pub fn try_save_requester<K>(
    requester: &Checkpointer<'_, K>,
    path: impl AsRef<Path>,
    timeout: Duration,
) -> std::io::Result<bool>
where
    K: SnapshotSlot + Send,
    K::Node: serde::Serialize + Sized,
{
    match requester.request_timeout(timeout) {
        Some(guard) => write_snapshot(guard.snapshot_node(), path.as_ref()).map(|() | true),
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

/// Serialize any snapshot artifact (a stable shape, a versioned envelope,
/// or the node itself) to `path`, atomically — the primitive the save
/// functions are built on. Version-skew workflows save their SHAPE
/// through this, never their machine type.
pub fn write_value<T: serde::Serialize>(value: &T, path: impl AsRef<Path>) -> std::io::Result<()> {
    write_snapshot(value, path.as_ref())
}

/// Serialize the (quiesced or owned) node to `path`, atomically.
const TEMP_FILE_EXTENSION: &str = "tmp";
/// Where the written-snapshot counter starts.
const SNAPSHOTS_WRITTEN_START: u64 = 0;

fn write_snapshot<T: serde::Serialize>(node: &T, path: &Path) -> std::io::Result<()> {
    let tmp = path.with_extension(TEMP_FILE_EXTENSION);
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
pub struct Snapshotter<'a, K> {
    requester: Checkpointer<'a, K>,
    path: std::path::PathBuf,
    every: Duration,
}

impl<'a, K> Snapshotter<'a, K>
where
    K: SnapshotSlot + Send,
    K::Node: serde::Serialize + Sized,
{
    pub fn new(
        requester: Checkpointer<'a, K>,
        path: impl Into<std::path::PathBuf>,
        every: Duration,
    ) -> Self {
        Self { requester, path: path.into(), every }
    }

    /// Save every `every` until `stop` is set; returns the number of
    /// snapshots written.
    pub fn run(&self, stop: &std::sync::atomic::AtomicBool) -> std::io::Result<u64> {
        use std::sync::atomic::Ordering;
        let mut written: u64 = SNAPSHOTS_WRITTEN_START;
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
