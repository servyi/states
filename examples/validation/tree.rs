// Example 1 — the library's user interface, exercised end to end:
// a mutator that fans out over a tree of workers (nested fanout)
// while a snapshotter repeatedly stops the world and reads the whole
// tree through the guard. Formally verified against the same
// contracts the implementation is verified with; tested by
// `tests/validation_examples.rs` (several scenarios).

use servyi_states::checkpoint::{with_checkpoint_pair, CheckpointRequester, Handle};
use vstd::prelude::*;

verus! {

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Leaf {
    pub steps_done: u32,
    pub result: Option<u32>,
}



pub fn run_leaf(mut h: Handle<'_, Leaf>)
{
    for _i in 0..3u64 {
        h.safepoint();
        let st = h.block_and_get_mut();
        st.steps_done += 1;
    }
    h.block_and_get_mut().result = Option::Some(7);
}

#[allow(clippy::ptr_arg)] // the split fn pointer's type IS the node's (&mut Vec<Group>)
fn root_children(r: &mut Vec<Group>) -> std::slice::IterMut<'_, Group> {
    // &[mut Vec]: the split fn pointer's parameter IS the node type
    // (Vec<Group>) — the fanout interface fixes it.
    r.iter_mut()
}

#[derive(Debug, Clone, PartialEq)]
pub struct Group {
    pub gid: u32,
    pub children: Vec<Leaf>,
}

impl Group {
    fn children_of(g: &mut Group) -> std::slice::IterMut<'_, Leaf> {
        g.children.iter_mut()
    }
}

pub fn run_group(mut h: Handle<'_, Group>)
{
    let split_group: for<'a> fn(&'a mut Group) -> std::slice::IterMut<'a, Leaf> =
        Group::children_of;
    h.fanout(split_group, run_leaf);
}

/// A full run over a two-group tree: nested fanout, then the results
/// collected through the handle (the mutator's own read-back path).
pub fn full_run(root: Vec<Group>) -> (collected: Vec<(u32, Option<u32>)>)
{
    with_checkpoint_pair(root, |mut h, _req| {
        let split_root: for<'a> fn(&'a mut Vec<Group>) -> std::slice::IterMut<'a, Group> =
            root_children;
        h.fanout(split_root, run_group);
        let m = h.block_and_get();
        m.iter().map(|g| (g.gid, g.children.iter().map(|c| c.result).sum::<Option<u32>>())).collect()
    })
}

/// The mutator/snapshotter interleaving: while the mutator works, a
/// snapshotter repeatedly stops the world and reads the root through
/// the guard. (The verification here discharges the client-side
/// obligations — hwf/has_baton at every borrow, the guard's gwf.)
pub fn with_snapshotter(node: u64, snapshots: u64) -> (final_v: u64)
{
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;
    with_checkpoint_pair(node, |mut h, req| {
        std::thread::scope(|scope| {
            let done = Arc::new(AtomicU64::new(0));
            let inflight = Arc::new(AtomicU64::new(0));
            let (done2, inflight2) = (done.clone(), inflight.clone());
            let _reader = scope.spawn(move || {
                let requester: &CheckpointRequester<'_, u64> = &req;
                for _k in 0..snapshots {
                    inflight2.store(1, Ordering::SeqCst);
                    if done2.load(Ordering::SeqCst) == 1 {
                        inflight2.store(0, Ordering::SeqCst);
                        break;
                    }
                    if let Some(g) = requester.request() {
                        let _observed = *g.node();
                    }
                    inflight2.store(0, Ordering::SeqCst);
                }
            });
            *h.block_and_get_mut() = *h.block_and_get() + 1;
            h.safepoint();
            let out = *h.block_and_get();
            // drain: stop issuing, serve the in-flight request, join
            done.store(1, Ordering::SeqCst);
            while inflight.load(Ordering::SeqCst) == 1 || h.is_stop_requested() {
                h.safepoint();
            }
            out
        })
    })
}

} // verus!

#[cfg(not(test))] // absent when included from the test crate
fn main() {
    let out = full_run(vec![
        Group { gid: 100, children: vec![Leaf::default(), Leaf::default()] },
        Group { gid: 200, children: vec![Leaf::default(), Leaf::default()] },
    ]);
    println!("full_run: {out:?}");
    println!("with_snapshotter: {}", with_snapshotter(41, 3));
}
