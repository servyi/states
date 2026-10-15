use std::fmt::Debug;

use dashmap::DashMap;

use crate::context_id::ContextId;

pub trait PhaseTracker<P>: Send + Sync
where
    P: PartialEq + Debug + Clone + Send + Sync,
{
    fn advance(&self, ctx: &ContextId, from: Option<P>, to: P);
    fn expect_any_and_set(&self, ctx: &ContextId, valid_from: &[P], to: P);
    fn get_phase(&self, ctx: &ContextId) -> Option<P>;
}

pub struct DefaultPhaseTracker<P> {
    phases: DashMap<u64, P>,
}

impl<P> DefaultPhaseTracker<P> {
    pub fn new() -> Self {
        Self {
            phases: DashMap::new(),
        }
    }
}

impl<P> PhaseTracker<P> for DefaultPhaseTracker<P>
where
    P: PartialEq + Debug + Clone + Send + Sync,
{
    fn advance(&self, ctx: &ContextId, from: Option<P>, to: P) {
        let id = ctx.hash();
        let _ = self
            .phases
            .entry(id)
            .and_modify(|phase| {
                if let Some(expected) = &from {
                    assert_eq!(
                        *phase,
                        *expected,
                        "{ctx} expected phase {expected:?} but was {phase:?}"
                    );
                }
                *phase = to.clone();
            })
            .or_insert_with(|| {
                assert!(
                    from.is_none(),
                    "{ctx} expected phase {from:?} but was absent",
                );
                to
            });
    }

    fn expect_any_and_set(&self, ctx: &ContextId, valid_from: &[P], to: P) {
        let id = ctx.hash();
        let _entry = self
            .phases
            .entry(id)
            .and_modify(|phase| {
                assert!(
                    valid_from.contains(phase),
                    "{ctx} expected one of {valid_from:?} but was {phase:?}",
                );
                *phase = to.clone();
            })
            .or_insert_with(|| {
                to
            });
    }

    fn get_phase(&self, ctx: &ContextId) -> Option<P> {
        self.phases.get(&ctx.hash()).map(|g| (*g.value()).clone())
    }
}

#[cfg(feature = "serde")]
mod serde_impls {
    use super::*;

    impl<P: serde::Serialize + Clone> serde::Serialize for DefaultPhaseTracker<P> {
        fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            // Cloned under each guard's lifetime, sorted by key for
            // deterministic artifacts.
            let mut entries: Vec<(u64, P)> =
                self.phases.iter().map(|e| (*e.key(), e.value().clone())).collect();
            entries.sort_by_key(|(hash, _)| *hash);
            entries.serialize(serializer)
        }
    }

    impl<'de, P: serde::Deserialize<'de>> serde::Deserialize<'de> for DefaultPhaseTracker<P> {
        fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            use serde::de::Error;
            let entries = Vec::<(u64, P)>::deserialize(deserializer)?;
            let tracker = Self::new();
            for (hash, phase) in entries {
                // Insert-without-expected is the sanctioned path (same as
                // first advance): phase discipline still holds for every
                // later transition.
                if tracker.phases.insert(hash, phase).is_some() {
                    return Err(D::Error::custom(format!(
                        "duplicate phase entry for hash {hash}"
                    )));
                }
            }
            Ok(tracker)
        }
    }
}

impl<P> Default for DefaultPhaseTracker<P>
where
    P: PartialEq + Debug + Clone + Send + Sync,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context_id::ContextId;

    /// Phase names of the little lifecycle the tests exercise.
    const PHASE_OPEN: &str = "open";
    const PHASE_CLOSED: &str = "closed";
    const PHASE_RETRY: &str = "retry";
    const PHASE_DONE: &str = "done";
    const PHASE_UNEXPECTED: &str = "wrong";
    const PHASE_OF_A: &str = "phase_a";
    const PHASE_OF_B: &str = "phase_b";

    #[test]
    fn test_advance_from_none() {
        let tracker = DefaultPhaseTracker::<&str>::new();
        let root = ContextId::root();
        let ctx = root.new_child();
        tracker.advance(&ctx, None, PHASE_OPEN);
        assert_eq!(tracker.get_phase(&ctx), Some(PHASE_OPEN));
    }

    #[test]
    fn test_advance_with_correct_from() {
        let tracker = DefaultPhaseTracker::<&str>::new();
        let root = ContextId::root();
        let ctx = root.new_child();
        tracker.advance(&ctx, None, PHASE_OPEN);
        tracker.advance(&ctx, Some(PHASE_OPEN), PHASE_CLOSED);
        assert_eq!(tracker.get_phase(&ctx), Some(PHASE_CLOSED));
    }

    #[test]
    #[should_panic(expected = "expected phase")]
    fn test_advance_wrong_from_panics() {
        let tracker = DefaultPhaseTracker::<&str>::new();
        let root = ContextId::root();
        let ctx = root.new_child();
        tracker.advance(&ctx, None, PHASE_OPEN);
        tracker.advance(&ctx, Some(PHASE_UNEXPECTED), PHASE_CLOSED);
    }

    #[test]
    fn test_expect_any_and_set() {
        let tracker = DefaultPhaseTracker::<&str>::new();
        let root = ContextId::root();
        let ctx = root.new_child();
        tracker.advance(&ctx, None, PHASE_OPEN);
        tracker.expect_any_and_set(&ctx, &[PHASE_OPEN, PHASE_RETRY], PHASE_DONE);
        assert_eq!(tracker.get_phase(&ctx), Some(PHASE_DONE));
    }

    #[test]
    fn test_get_phase_absent() {
        let tracker = DefaultPhaseTracker::<&str>::new();
        let root = ContextId::root();
        let ctx = root.new_child();
        assert_eq!(tracker.get_phase(&ctx), None);
    }

    #[test]
    fn test_multiple_contexts_independent() {
        let tracker = DefaultPhaseTracker::<&str>::new();
        let root = ContextId::root();
        let a = root.new_child();
        let b = root.new_child();
        tracker.advance(&a, None, PHASE_OF_A);
        tracker.advance(&b, None, PHASE_OF_B);
        assert_eq!(tracker.get_phase(&a), Some(PHASE_OF_A));
        assert_eq!(tracker.get_phase(&b), Some(PHASE_OF_B));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serde_roundtrip() {
        let root = ContextId::root();
        let a = root.new_child();
        let b = root.new_child();
        let tracker = DefaultPhaseTracker::<&str>::new();
        tracker.advance(&a, None, "open");
        tracker.advance(&b, None, "done");

        let json = serde_json::to_string(&tracker).expect("serialize");
        let entries: Vec<(u64, &str)> = serde_json::from_str(&json).expect("parse back");
        // Sorted by hash for deterministic artifacts.
        assert!(entries.windows(2).all(|w| w[0].0 < w[1].0));
        assert_eq!(entries.len(), 2);

        let restored: DefaultPhaseTracker<&str> =
            serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored.get_phase(&a), Some("open"));
        assert_eq!(restored.get_phase(&b), Some("done"));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serde_rejects_duplicate_entries() {
        let json = r#"[[1,"open"],[1,"done"]]"#;
        let result: Result<DefaultPhaseTracker<&str>, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }
}
