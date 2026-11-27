//! Version 1's binary: contains ONLY v1's shape and saver logic.

use std::path::Path;

use serde::{Deserialize, Serialize};
use servyi_states::checkpoint::{PersistentCell, independent_task_root};
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
    // v1's saver reads its OWN machine (the shape is version-skew
    // data, never the protocol's node) — the tree is exercised for
    // fidelity; the shape is built from the local machine directly.
    let machine = MachineV1 { items: vec![(3, 4, true), (3, 2, false)] };
    let mut root = PersistentCell::new(Box::new(PersistentCell::new(machine.clone())));
    independent_task_root(
        &mut root,
        |_outer: servyi_states::checkpoint::OuterHandle<
            '_,
            Box<PersistentCell<MachineV1>>,
            dyn FixtureDriven,
        >,
         _ck| {},
    );
    let shape = ShapeV1 {
        steps_total: machine.items.iter().map(|(t, ..)| *t).collect(),
        steps_done: machine.items.iter().map(|(_, s, _)| *s).collect(),
        finished: machine.items.iter().map(|(.., d)| *d).collect(),
    };
    snapshot::write_value(&shape, path).expect("v1 writes its shape");
    tracing::info!("[v1:save] wrote ShapeV1: {} items", shape.finished.len());
    let _ = Path::new(path).exists();
}

/// The fixture's (trivial) drive vocabulary: nothing to drive — the
/// saver only reads.
trait FixtureDriven: Send {}

impl FixtureDriven for servyi_states::checkpoint::InnerHandle<MachineV1> {}
