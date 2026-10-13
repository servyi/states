//! Version 1's binary: contains ONLY v1's shape and saver logic.

use std::path::Path;

use serde::{Deserialize, Serialize};
use servyi_states::checkpoint::with_checkpoint_pair;
use servyi_states_snapshot as snapshot;

/// v1's snapshot vocabulary — frozen at v1. No other version's field
/// exists here.
#[derive(Serialize, Deserialize)]
struct ShapeV1 {
    steps_total: Vec<u32>,
    steps_done: Vec<u32>,
    finished: Vec<bool>,
}

/// v1's machine.
#[derive(Clone)]
struct MachineV1 {
    /// (total, step, done) — v1 double-counts the completion step, so a
    /// finished item shows step == total + 1.
    items: Vec<(u32, u32, bool)>,
}

fn main() {
    tracing_subscriber::fmt()
        .with_target(false)
        .without_time()
        .with_writer(std::io::stderr)
        .init();
    let mut args = std::env::args().skip(1);
    let cmd = args.next().unwrap_or_else(|| usage());
    let path = args.next().unwrap_or_else(|| usage());
    match cmd.as_str() {
        "save" => save(&path),
        _ => usage(),
    }
}

fn usage() -> ! {
    eprintln!("usage: version1 save <path>");
    std::process::exit(2);
}

fn save(path: &str) {
    // Demo workload matching the upgrade-path tests: two items finished
    // (step recorded as total+1 — the v1 bug), one mid-flight.
    let machine = MachineV1 { items: vec![(3, 4, true), (3, 2, false)] };
    with_checkpoint_pair(machine, |mut h, _req| {
        h.safepoint();
        let shape = ShapeV1 {
            steps_total: h.items.iter().map(|(t, ..)| *t).collect(),
            steps_done: h.items.iter().map(|(_, s, _)| *s).collect(),
            finished: h.items.iter().map(|(.., d)| *d).collect(),
        };
        snapshot::write_value(&shape, path).expect("v1 writes its shape");
        tracing::info!("[v1:save] wrote ShapeV1: {} items", shape.finished.len());
    });
    let _ = Path::new(path).exists();
}
