//! Version 2's binary: v2's own shape for its own logic, PLUS its
//! knowledge of v1 artifacts (the v1->v2 upgrade path). Restores through
//! a dyn impl of exactly the shape it reads.

use serde::{Deserialize, Serialize};
use servyi_states::checkpoint::with_checkpoint_pair;

/// v2's snapshot vocabulary — frozen at v2 (adds `polished`).
#[derive(Serialize, Deserialize)]
struct ShapeV2 {
    steps_total: Vec<u32>,
    steps_done: Vec<u32>,
    /// v2 wrote this one step early (the v2 bug) — v3 repairs it.
    finished: Vec<bool>,
    polished: Vec<bool>,
}

/// v1's shape, as parsed by v2 for the upgrade path (frozen at v1).
#[derive(Deserialize)]
struct ShapeV1 {
    steps_total: Vec<u32>,
    steps_done: Vec<u32>,
    finished: Vec<bool>,
}

/// The stable contract this binary restores through.
trait Shape {
    type State;
    fn restore(&self) -> Self::State;
}

#[derive(Clone, Debug)]
struct Item {
    id: u32,
    steps: u32,
    step: u32,
    done: Option<String>,
    polished: bool,
    needs_fix: bool,
}

#[derive(Clone, Debug)]
struct Machine {
    items: Vec<Item>,
}

impl Shape for ShapeV2 {
    type State = Machine;

    fn restore(&self) -> Machine {
        tracing::info!("[v2:restore] origin=v2 (own shape)");
        Machine {
            items: self
                .steps_total
                .iter()
                .zip(&self.steps_done)
                .zip(&self.finished)
                .zip(&self.polished)
                .enumerate()
                .map(|(idx, (((total, step), finished), polished))| Item {
                    id: idx as u32 + 1,
                    steps: *total,
                    step: *step,
                    done: finished.then(|| format!("v{}", idx as u32 + 1)),
                    polished: *polished,
                    needs_fix: false,
                })
                .collect(),
        }
    }
}

impl ShapeV1 {
    /// v2's upgrade code for origin-1 artifacts: reconcile the
    /// double-counted completion step; v1 never polished.
    fn upgrade(self) -> Machine {
        tracing::info!("[v2:restore] origin=v1 (upgrading through v2's knowledge)");
        Machine {
            items: self
                .steps_total
                .iter()
                .zip(&self.steps_done)
                .zip(&self.finished)
                .enumerate()
                .map(|(idx, ((total, step), finished))| Item {
                    id: idx as u32 + 1,
                    steps: *total,
                    step: if *finished { (*step).saturating_sub(1) } else { *step },
                    done: finished.then(|| format!("v{}", idx as u32 + 1)),
                    polished: false,
                    needs_fix: *finished,
                })
                .collect(),
        }
    }
}

fn fix(m: &mut Machine) {
    for item in &mut m.items {
        if item.needs_fix {
            tracing::info!("[v2:fix] item {} pre-fix resolved", item.id);
            item.needs_fix = false;
        }
    }
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
        "restore-run" => restore_run(&path),
        _ => usage(),
    }
}

fn usage() -> ! {
    eprintln!("usage: version2 save <path> | restore-run <path>");
    std::process::exit(2);
}

fn save(path: &str) {
    // v2's demo workload: one item finished one step early (the v2 bug),
    // one not started, one polished.
    let machine = Machine {
        items: vec![
            Item { id: 1, steps: 3, step: 2, done: Some("v1".into()), polished: true, needs_fix: false },
            Item { id: 2, steps: 3, step: 0, done: None, polished: false, needs_fix: false },
        ],
    };
    with_checkpoint_pair(machine, |mut h, _req| {
        h.safepoint();
        let shape = ShapeV2 {
            steps_total: h.items.iter().map(|i| i.steps).collect(),
            steps_done: h.items.iter().map(|i| i.step).collect(),
            finished: h.items.iter().map(|i| i.done.is_some()).collect(),
            polished: h.items.iter().map(|i| i.polished).collect(),
        };
        servyi_states_snapshot::write_value(&shape, path).expect("v2 writes its shape");
        tracing::info!("[v2:save] wrote ShapeV2: {} items", shape.finished.len());
    });
}

fn restore_run(path: &str) {
    let raw: serde_json::Value =
        servyi_states_snapshot::restore_from(path).expect("artifact parses");
    let mut machine = if raw.get("polished").is_some() {
        let shape: ShapeV2 = serde_json::from_value(raw).expect("v2 shape");
        <ShapeV2 as Shape>::restore(&shape)
    } else {
        let shape: ShapeV1 = serde_json::from_value(raw).expect("v1 shape");
        shape.upgrade()
    };
    fix(&mut machine);
    let (out, calls) = run(machine);
    println!("OUT {out}");
    println!("IO {calls}");
}

fn run(machine: Machine) -> (String, usize) {
    let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let c = std::sync::Arc::clone(&calls);
    let mut skip_logged = std::collections::HashSet::new();
    let out = with_checkpoint_pair(machine, move |mut h, _req| loop {
        h.safepoint();
        let m = &mut *h;
        for item in &mut m.items {
            if item.done.is_some() || item.needs_fix {
                // One skip event per item, not one per loop iteration.
                if skip_logged.insert(item.id) {
                    tracing::info!("[v2:run] item {} skipped (done)", item.id);
                }
                continue;
            }
            let _n = c.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            item.step += 1;
            if item.step >= item.steps {
                item.done = Some(format!("v{}", item.id));
                tracing::info!("[v2:run] item {} finished (step {}/{})", item.id, item.step, item.steps);
            }
        }
        if m.items.iter().all(|i| i.done.is_some()) {
            return m
                .items
                .iter()
                .filter_map(|i| i.done.clone())
                .collect::<Vec<_>>()
                .join(",");
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    });
    (out, calls.load(std::sync::atomic::Ordering::SeqCst))
}
