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
                    "{ctx} expected phase {:?} but was absent",
                    from.as_ref().unwrap(),
                );
                to
            });
    }

    fn expect_any_and_set(&self, ctx: &ContextId, valid_from: &[P], to: P) {
        let id = ctx.hash();
        self.phases
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

    #[test]
    fn test_advance_from_none() {
        let tracker = DefaultPhaseTracker::<&str>::new();
        let root = ContextId::root();
        let ctx = root.new_child();
        tracker.advance(&ctx, None, "open");
        assert_eq!(tracker.get_phase(&ctx), Some("open"));
    }

    #[test]
    fn test_advance_with_correct_from() {
        let tracker = DefaultPhaseTracker::<&str>::new();
        let root = ContextId::root();
        let ctx = root.new_child();
        tracker.advance(&ctx, None, "open");
        tracker.advance(&ctx, Some("open"), "closed");
        assert_eq!(tracker.get_phase(&ctx), Some("closed"));
    }

    #[test]
    #[should_panic(expected = "expected phase")]
    fn test_advance_wrong_from_panics() {
        let tracker = DefaultPhaseTracker::<&str>::new();
        let root = ContextId::root();
        let ctx = root.new_child();
        tracker.advance(&ctx, None, "open");
        tracker.advance(&ctx, Some("wrong"), "closed");
    }

    #[test]
    fn test_expect_any_and_set() {
        let tracker = DefaultPhaseTracker::<&str>::new();
        let root = ContextId::root();
        let ctx = root.new_child();
        tracker.advance(&ctx, None, "open");
        tracker.expect_any_and_set(&ctx, &["open", "retry"], "done");
        assert_eq!(tracker.get_phase(&ctx), Some("done"));
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
        tracker.advance(&a, None, "phase_a");
        tracker.advance(&b, None, "phase_b");
        assert_eq!(tracker.get_phase(&a), Some("phase_a"));
        assert_eq!(tracker.get_phase(&b), Some("phase_b"));
    }
}
