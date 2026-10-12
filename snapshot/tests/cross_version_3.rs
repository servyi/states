//! Cross-version save/restore across THREE SEPARATE BINARIES — the
//! saver and restorer are different compiled versions, each with its own
//! shapes and restore code, communicating only through the artifact file
//! and stdout/stderr:
//!
//!   save (binary) | restore (binary)
//!   version1      | version2
//!   version1      | version3
//!   version2      | version3
//!   version3      | version3   (control: lossless)
//!
//! Verification is by OBSERVED TRACES, not just outcomes: each binary
//! emits tracing events on stderr (`[vN:save]`, `[vN:restore] origin=`,
//! `[vN:fix] item .. pre-fix resolved`, `[vN:run] item .. skipped`);
//! these tests capture the traces and assert the upgrade actually took
//! the claimed path.

use std::process::Command;

struct Run {
    out: String,
    err: String,
}

fn run(bin: &str, cmd: &str, path: &std::path::Path) -> Run {
    let output = Command::new(bin).arg(cmd).arg(path).output().expect("binary runs");
    assert!(
        output.status.success(),
        "{bin} {cmd} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Run {
        out: String::from_utf8_lossy(&output.stdout).into_owned(),
        err: String::from_utf8_lossy(&output.stderr).into_owned(),
    }
}

fn out_str(r: &Run) -> String {
    r.out
        .lines()
        .find_map(|l| l.strip_prefix("OUT "))
        .map(str::trim)
        .map(str::to_string)
        .unwrap_or_else(|| unreachable!("OUT missing from output: {}", r.out))
}

fn io_count(r: &Run) -> usize {
    r.out
        .lines()
        .find_map(|l| l.strip_prefix("IO "))
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or_else(|| unreachable!("IO missing from output: {}", r.out))
}

macro_rules! exe {
    ($name:literal) => {
        env!(concat!("CARGO_BIN_EXE_", $name))
    };
}

#[test]
fn path_v1_saved_by_version1_restored_by_version2() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.json");

    let save = run(exe!("version1"), "save", &path);
    assert!(
        save.err.contains("[v1:save] wrote ShapeV1"),
        "saver trace: {}",
        save.err
    );

    let r = run(exe!("version2"), "restore-run", &path);
    // OBSERVED: v2's restore code took the origin-1 upgrade path.
    assert!(
        r.err.contains("[v2:restore] origin=v1 (upgrading through v2's knowledge)"),
        "restore trace: {}",
        r.err
    );
    // OBSERVED: the v1-finished item was pre-fix and the fixer resolved it.
    assert!(r.err.contains("[v2:fix] item 1 pre-fix resolved"), "{}", r.err);
    // OBSERVED: the finished item was skipped by the runner (zero io).
    assert!(r.err.contains("[v2:run] item 1 skipped (done)"), "{}", r.err);
    assert!(r.err.contains("[v2:run] item 2 finished (step 3/3)"), "{}", r.err);

    assert_eq!(out_str(&r), "v1,v2", "two items done");
    // The unfinished item had 1 of 3 steps left (v1 recorded step 2).
    assert_eq!(io_count(&r), 1, "resume skips finished work");
}

#[test]
fn path_v1_saved_by_version1_restored_by_version3() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("b.json");

    let _saved = run(exe!("version1"), "save", &path);
    let r = run(exe!("version3"), "restore-run", &path);
    assert!(
        r.err.contains("[v3:restore] origin=v1 (one-hop upgrade)"),
        "restore trace: {}",
        r.err
    );
    assert!(r.err.contains("[v3:fix] item 1 pre-fix resolved"), "{}", r.err);
    assert!(r.err.contains("[v3:run] item 2 finished (step 3/3)"), "{}", r.err);
    assert_eq!(out_str(&r), "v1,v2");
    assert_eq!(io_count(&r), 1, "only the last step of the unfinished item");
}

#[test]
fn path_v2_saved_by_version2_restored_by_version3() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c.json");

    let save = run(exe!("version2"), "save", &path);
    assert!(
        save.err.contains("[v2:save] wrote ShapeV2"),
        "saver trace: {}",
        save.err
    );

    let r = run(exe!("version3"), "restore-run", &path);
    assert!(
        r.err
            .contains("[v3:restore] origin=v2 (repairing the early-flag bug)"),
        "restore trace: {}",
        r.err
    );
    // v2's early-flagged item arrived pre-fix and was resolved at zero cost.
    assert!(r.err.contains("[v3:fix] item 1 pre-fix resolved"), "{}", r.err);
    assert_eq!(out_str(&r), "v1,v2");
    assert!(r.err.contains("[v3:run] item 2 finished (step 3/3)"), "{}", r.err);
    assert_eq!(io_count(&r), 3, "the not-started item ran all 3 steps");
}

#[test]
fn control_v3_saved_and_restored_by_version3_is_lossless() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("d.json");

    let _saved = run(exe!("version3"), "save", &path);
    let r = run(exe!("version3"), "restore-run", &path);
    assert!(
        r.err.contains("[v3:restore] origin=v3 (own shape, lossless)"),
        "restore trace: {}",
        r.err
    );
    // OBSERVED: no fixer activity on same-version artifacts.
    assert!(!r.err.contains("[v3:fix]"), "no pre-fix expected: {}", r.err);
    assert!(r.err.contains("[v3:run] item 2 finished (step 3/3)"), "{}", r.err);
    assert_eq!(out_str(&r), "v1,v2");
    assert_eq!(io_count(&r), 2, "only the unfinished item remaining steps");
}
