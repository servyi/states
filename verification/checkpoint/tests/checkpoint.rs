//! End-to-end tests for the `checkpoint` crate (mirrors of the
//! verified client demos in states-asis, plus the parallel-requester
//! pattern from the PR's own tests).

use checkpoint::{with_checkpoint_pair, CheckpointRequester, Handle};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

/// Mutator steps through the node between safepoints; a parallel
/// requester repeatedly quiesces and reads. Uses the request-drain
/// handshake before joining (see the liveness note in the lib docs).
#[test]
fn parallel_requester_reads_live_node() {
    let out = with_checkpoint_pair(41u32, |mut h, req: CheckpointRequester<'_, u32>| {
        thread::scope(|scope| {
            let done = Arc::new(AtomicBool::new(false));
            let inflight = Arc::new(AtomicBool::new(false));
            let done2 = done.clone();
            let inflight2 = inflight.clone();

            let snap = scope.spawn(move || {
                let mut observations = Vec::new();
                loop {
                    inflight2.store(true, Ordering::SeqCst);
                    if done2.load(Ordering::SeqCst) {
                        inflight2.store(false, Ordering::SeqCst);
                        break;
                    }
                    let guard = req.request();
                    observations.push(*guard.node());
                    drop(guard);
                    inflight2.store(false, Ordering::SeqCst);
                }
                observations
            });

            let mut v = 41;
            for _ in 0..50 {
                h.safepoint();
                *h.block_and_get_mut() += 1;
                v = *h.block_and_get();
                thread::sleep(Duration::from_millis(1));
            }

            done.store(true, Ordering::SeqCst);
            while inflight.load(Ordering::SeqCst) || h.is_stop_requested() {
                h.safepoint();
            }

            let obs = snap.join().expect("requester");
            (obs, v)
        })
    });
    let (obs, final_v) = out;
    assert_eq!(final_v, 91);
    assert!(!obs.is_empty());
    for w in obs.windows(2) {
        assert!(w[0] <= w[1], "monotone guard observations: {obs:?}");
    }
    assert!(obs.iter().all(|v| (41..=91).contains(v)));
}

/// The verified client demo (states-asis demo_mutator_step +
/// demo_snapshot), as one thread each: a read, a mutation that adds
/// the read value, then a final snapshot of the result.
#[test]
fn verified_demo_shape() {
    let final_v = with_checkpoint_pair(7u64, |mut h: Handle<'_, u64>, req| {
        thread::scope(|scope| {
            let done = Arc::new(AtomicBool::new(false));
            let inflight = Arc::new(AtomicBool::new(false));
            let (done2, inflight2) = (done.clone(), inflight.clone());

            let snap = scope.spawn(move || {
                inflight2.store(true, Ordering::SeqCst);
                let g = req.request();
                let r = *g.node();
                drop(g);
                inflight2.store(false, Ordering::SeqCst);
                done2.store(true, Ordering::SeqCst);
                r
            });
            // demo_mutator_step
            h.safepoint();
            let v = *h.block_and_get();
            h.safepoint();
            *h.block_and_get_mut() += v;
            h.safepoint();
            // drain the (single) request before joining: exit only
            // once the requester has finished (done) and is no longer
            // inflight and no request is pending
            while !done.load(Ordering::SeqCst)
                || inflight.load(Ordering::SeqCst)
                || h.is_stop_requested()
            {
                h.safepoint();
            }
            let before = snap.join().expect("snap");
            let after = *h.block_and_get();
            (before, after)
        })
    });
    let (before, after) = final_v;
    // the single snapshot lands before OR after the mutation
    assert!(before == 7 || before == 14, "snapshot sees a quiesced state: {before}");
    assert_eq!(after, 14);
}

/// The guard's read sees a fully quiesced node: a torn value is
/// unrepresentable (two-field struct updated between safepoints).
#[test]
fn guard_sees_quiesced_state() {
    #[derive(Debug, Clone, PartialEq)]
    struct Pair {
        a: u64,
        b: u64,
    }
    let out = with_checkpoint_pair(Pair { a: 0, b: 0 }, |mut h, req| {
        thread::scope(|scope| {
            let done = Arc::new(AtomicBool::new(false));
            let inflight = Arc::new(AtomicBool::new(false));
            let (done2, inflight2) = (done.clone(), inflight.clone());

            let snap = scope.spawn(move || {
                let mut seen = Vec::new();
                loop {
                    inflight2.store(true, Ordering::SeqCst);
                    if done2.load(Ordering::SeqCst) {
                        inflight2.store(false, Ordering::SeqCst);
                        break;
                    }
                    let g = req.request();
                    seen.push(g.node().clone());
                    drop(g);
                    inflight2.store(false, Ordering::SeqCst);
                }
                seen
            });

            for i in 0..100u64 {
                h.safepoint();
                let p = h.block_and_get_mut();
                p.a = i;
                p.b = i; // both fields set between safepoints
            }
            done.store(true, Ordering::SeqCst);
            while inflight.load(Ordering::SeqCst) || h.is_stop_requested() {
                h.safepoint();
            }
            let mut seen = snap.join().expect("snap");
            seen.push(h.block_and_get().clone());
            seen
        })
    });
    // Every observation must have a == b (a torn pair is unobservable)
    for p in &out {
        assert_eq!(p.a, p.b, "torn pair observed: {p:?}");
    }
    assert_eq!(out.last(), Some(&Pair { a: 99, b: 99 }));
}
