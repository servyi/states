//! Miri test: nested `fanout` with a parallel stop-the-world requester
//! reading the whole tree through the guard.
//!
//! Miri checks (Stacked Borrows + data race detection) exactly the
//! unsafe raw-pointer plumbing this API is built on:
//!   - `with_checkpoint_pair`'s `ptr::from_mut` pair,
//!   - `fanout`'s `&mut *this` reborrows in the serve loop and the
//!     split-item handles (`ptr::from_mut(child)`) moved to workers,
//!   - `block_and_get`/`block_and_get_mut` borrows vs the
//!     snapshotter's `guard.node()` over the whole tree.
//!
//! Run: cargo +nightly miri test --test checkpoint_miri
//! (real run:     cargo test --test checkpoint_miri)

use servyi_states::checkpoint::{with_checkpoint_pair, Handle};

#[derive(Debug)]
struct Leaf {
    #[allow(dead_code)]
    id: u32,
    step: u32,
    done: bool,
}

/// A leaf worker: mutates only through its own handle, parking at
/// safepoints between steps.
fn run_leaf(mut h: Handle<'_, Leaf>) {
    for _ in 0..3 {
        h.safepoint();
        let st = h.block_and_get_mut();
        st.step += 1;
    }
    h.block_and_get_mut().done = true;
}

#[derive(Debug)]
struct Group {
    #[allow(dead_code)]
    gid: u32,
    children: Vec<Leaf>,
}

/// A group worker: nested fan-out over its own leaves, serving
/// checkpoint requests while they run.
fn run_group(mut h: Handle<'_, Group>) {
    h.fanout(|g| g.children.iter_mut(), |a| run_leaf(a));
}

/// Whole-tree snapshot taken through the guard while every mutator is
/// parked (the stop-the-world contract).
fn census(root: &[Group]) -> (u32, u32) {
    let mut done = 0;
    let mut steps = 0;
    for g in root {
        for c in &g.children {
            done += c.done as u32;
            steps += c.step;
        }
    }
    (done, steps)
}

#[test]
fn miri_nested_fanout_with_parallel_requester() {
    let root: Vec<Group> = (0..2)
        .map(|gi| Group {
            gid: 100 + gi,
            children: (0..2)
                .map(|ci| Leaf { id: 10 * gi + ci, step: 0, done: false })
                .collect(),
        })
        .collect();

    let final_root = with_checkpoint_pair(root, |mut h, req| {
        std::thread::scope(|scope| {
            use std::sync::atomic::{AtomicBool, Ordering};

            let fanout_done = std::sync::Arc::new(AtomicBool::new(false));
            let request_inflight = std::sync::Arc::new(AtomicBool::new(false));
            let fd2 = fanout_done.clone();
            let ri2 = request_inflight.clone();

            // Parallel stop-the-world requester: repeatedly quiesces
            // the whole tree and reads it through the guard while the
            // nested fan-out runs; stops once the fan-out is done.
            let snap = scope.spawn(move || {
                let mut observations = Vec::new();
                loop {
                    ri2.store(true, Ordering::SeqCst);
                    if fd2.load(Ordering::SeqCst) {
                        ri2.store(false, Ordering::SeqCst);
                        break;
                    }
                    if let Some(guard) = req.request() {
                        observations.push(census(guard.node()));
                        drop(guard);
                    }
                    ri2.store(false, Ordering::SeqCst);
                }
                observations
            });

            // Root fan-out over groups; each group fans out over its
            // leaves (two levels of scoped threads + parking).
            // (explicit &mut reborrow of the handle for the call)
            let h_ref: &mut Handle<'_, Vec<Group>> = &mut h;
            h_ref.fanout(|r| r.iter_mut(), |g| run_group(g));

            // Drain any in-flight request before joining (mirrors the
            // PR's drive(): serve late requests so the requester can
            // not block on a world that will never park again).
            fanout_done.store(true, Ordering::SeqCst);
            while request_inflight.load(Ordering::SeqCst) || h.is_stop_requested() {
                h.safepoint();
            }

            let obs = snap.join().expect("requester");
            // Observations are monotone in `done` (the tree only
            // advances), and at least one snapshot was taken during
            // the fan-out.
            assert!(!obs.is_empty());
            for w in obs.windows(2) {
                assert!(w[0].0 <= w[1].0 && w[0].1 <= w[1].1, "monotone: {obs:?}");
            }
            // PROOF THE FAN-OUT EXECUTED: every leaf ran to completion
            // through its own handle (3 steps + done).
            let final_census = census(h.block_and_get());
            assert_eq!(final_census, (4, 12), "final tree: {final_census:?}");
            final_census
        })
    });
    assert_eq!(final_root, (4, 12));
}
