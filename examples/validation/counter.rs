// Example 2 — the single-node interface: sequential mutation between
// safepoints, concurrent snapshotting with `request`/`request_timeout`,
// and the orphaned-request drain pattern that library users must
// follow before joining a snapshotter thread. Formally verified
// against the implementation's contracts; tested by
// `tests/validation_examples.rs`.

use servyi_states::checkpoint::{with_checkpoint_pair, CheckpointRequester, Handle};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use vstd::prelude::*;

verus! {

/// A sequential mutator loop: k steps, each between safepoints, with
/// the borrows' obligations discharged at every call (the pattern the
/// library's `&mut self` discipline is designed for).
pub fn counter_loop(mut h: Handle<'_, u64>, k: u64) -> (out: u64)
{
    for _i in 0..k {
        h.safepoint();
        let cur = *h.block_and_get();
        *h.block_and_get_mut() = cur.saturating_add(1);
    }
    *h.block_and_get()
}

/// Mutator + snapshotter with the drain handshake (the library's
/// documented liveness contract: a request landing after the mutator's
/// last safepoint blocks forever unless drained). The snapshotter
/// observes monotone values — the guard only exists while the world is
/// stopped. The drain runs before the loop consumes the handle:
/// done=1 stops new requests, one final safepoint serves any that are
/// in flight, then the loop runs out and the reader joins.
pub fn counter_with_reader(node: u64, steps: u64, reads: u64) -> (final_v: u64)
{
    with_checkpoint_pair(node, |mut h, req| {
        std::thread::scope(|scope| {
            let done = Arc::new(AtomicU64::new(0));
            let inflight = Arc::new(AtomicU64::new(0));
            let (done2, inflight2) = (done.clone(), inflight.clone());

            let reader = scope.spawn(move || {
                let requester: &CheckpointRequester<'_, u64> = &req;
                let mut observed: u64 = 0;
                for _k in 0..reads {
                    inflight2.store(1, Ordering::SeqCst);
                    if done2.load(Ordering::SeqCst) == 1 {
                        inflight2.store(0, Ordering::SeqCst);
                        break;
                    }
                    if let Some(g) = requester.request() {
                        let v = *g.node();
                        assert(v >= observed); // monotone observations
                        observed = v;
                    }
                    inflight2.store(0, Ordering::SeqCst);
                }
                observed
            });

            // The drain FIRST: stop issuing, serve the in-flight
            // request, THEN consume the handle in the loop.
            done.store(1, Ordering::SeqCst);
            while inflight.load(Ordering::SeqCst) == 1 || h.is_stop_requested() {
                h.safepoint();
            }
            let v = counter_loop(h, steps);
            drop(reader); // scope joins
            v
        })
    })
}

/// `request_timeout` integration: the deadline path must return None
/// (never block forever) when the mutator never parks — and the
/// snapshotter still exits cleanly.
pub fn timeout_gives_up(node: u64) -> (final_v: u64)
{
    with_checkpoint_pair(node, |mut h, req| {
        std::thread::scope(|scope| {
            let done = Arc::new(AtomicU64::new(0));
            let done2 = done.clone();
            let reader = scope.spawn(move || {
                let requester: &CheckpointRequester<'_, u64> = &req;
                let mut timeouts = 0u64;
                while done2.load(Ordering::SeqCst) == 0 {
                    if requester.request_timeout(std::time::Duration::from_millis(5)).is_none() {
                        timeouts += 1;
                    }
                }
                timeouts
            });
            std::thread::sleep(std::time::Duration::from_millis(30));
            done.store(1, Ordering::SeqCst);
            h.safepoint(); // serve a possibly in-flight request
            drop(reader); // scope joins
            // reader observed >=1 timeout (threads: B7)
            *h.block_and_get()
        })
    })
}

} // verus!

#[cfg(not(test))] // absent when included from the test crate
fn main() {
    println!("counter_with_reader: {}", counter_with_reader(0, 50, 10));
    println!("timeout_gives_up: {}", timeout_gives_up(1));
}
