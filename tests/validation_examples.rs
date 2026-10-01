//! Tests for the `examples/validation/` examples — several scenarios
//! per example, exercising the library's user interface end to end.

mod example_tree {
    include!("../examples/validation/tree.rs");
}

mod example_counter {
    include!("../examples/validation/counter.rs");
}

use example_tree::{full_run, with_snapshotter, Group, Leaf};

fn tree() -> Vec<Group> {
    vec![
        Group { gid: 100, children: vec![Leaf::default(), Leaf::default()] },
        Group { gid: 200, children: vec![Leaf::default(), Leaf::default()] },
    ]
}

// ── example 1: nested fanout ─────────────────────────────────────

#[test]
fn full_run_completes_and_collects() {
    let out = full_run(tree());
    assert_eq!(
        out,
        vec![(100, Some(14)), (200, Some(14))],
        "each of 4 leaves contributes 7"
    );
}

#[test]
fn full_run_on_empty_tree() {
    let out = full_run(vec![]);
    assert_eq!(out, vec![]);
}

#[test]
fn full_run_single_group_single_leaf() {
    let out = full_run(vec![Group { gid: 1, children: vec![Leaf::default()] }]);
    assert_eq!(out, vec![(1, Some(7))]);
}

#[test]
fn full_run_is_deterministic() {
    assert_eq!(full_run(tree()), full_run(tree()));
}

#[test]
fn snapshotter_sees_monotone_final() {
    let v = with_snapshotter(41, 3);
    assert_eq!(v, 42);
}

#[test]
fn snapshotter_zero_snapshots() {
    let v = with_snapshotter(7, 0);
    assert_eq!(v, 8);
}

// ── example 2: counter + readers ─────────────────────────────────

use example_counter::{counter_loop, counter_with_reader, timeout_gives_up};

#[test]
fn counter_loop_increments() {
    with_checkpoint_pair(10u64, |h, _| {
        assert_eq!(counter_loop(h, 5), 15);
    });
}

#[test]
fn counter_loop_zero_steps() {
    with_checkpoint_pair(3u64, |h, _| {
        assert_eq!(counter_loop(h, 0), 3);
    });
}

#[test]
fn counter_with_reader_sees_result() {
    let v = counter_with_reader(0, 50, 10);
    assert_eq!(v, 50);
}

#[test]
fn counter_with_reader_no_reads() {
    let v = counter_with_reader(9, 20, 0);
    assert_eq!(v, 29);
}

#[test]
fn counter_with_reader_heavy_readers() {
    let v = counter_with_reader(0, 200, 100);
    assert_eq!(v, 200);
}

#[test]
fn timeout_gives_up_returns() {
    let v = timeout_gives_up(1);
    assert_eq!(v, 1);
}

use servyi_states::checkpoint::with_checkpoint_pair;
