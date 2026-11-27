//! The OLD `with_checkpoint_pair` shape over the two-piece root — for
//! code written against the one-piece API: the cross-version fixtures
//! and the cross-version tests.
//!
//! ONE definition, included by path (`#[path] mod legacy_pair;`). The
//! fixtures are deliberately separate codebases — not workspace
//! members — so this is shared as source rather than as a crate.

use servyi_states::checkpoint::{
    Checkpointer, InnerHandle, OuterHandle, PersistentCell, independent_task_root,
};

/// The pair shim's vocabulary: hand the closure the INNER handle —
/// exactly the old `Handle<'_, T>`.
pub trait PairInner<T>: Send {
    fn inner(&mut self) -> &mut InnerHandle<T>;
}

impl<T: Send + 'static> PairInner<T> for InnerHandle<T> {
    fn inner(&mut self) -> &mut Self {
        self
    }
}

/// The old `with_checkpoint_pair(node, |handle, requester| ...)`: the
/// node moves into the slot and the closure drives the inner handle.
pub fn with_pair<T, R>(
    node: T,
    f: impl FnOnce(&mut InnerHandle<T>, &Checkpointer<'_, Box<PersistentCell<T>>>) -> R,
) -> R
where
    T: Send + 'static,
{
    let mut slot: PersistentCell<Box<PersistentCell<T>>> =
        PersistentCell::new(Box::new(PersistentCell::new(node)));
    independent_task_root(
        &mut slot,
        |mut outer: OuterHandle<'_, Box<PersistentCell<T>>, dyn PairInner<T>>, ck| {
            f(outer.inner(), &ck)
        },
    )
}
