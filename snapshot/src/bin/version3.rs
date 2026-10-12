//! Version 3's binary: v3's own shape for its own logic, PLUS upgrade
//! knowledge for v1 and v2 artifacts (one-hop upgrades — the v1->v3 and
//! v2->v3 paths). Restores through a dyn impl of exactly the shape read.

use serde::{Deserialize, Serialize};
use servyi_states::checkpoint::with_checkpoint_pair;

/// v3's snapshot vocabulary — frozen at v3 (adds `origin`).
#[derive(Serialize, Deserialize)]
struct ShapeV3 {
    steps_total: Vec<u32>,
    steps_done: Vec<u32>,
    finished: Vec<bool>,
    polished: Vec<bool>,
    origin: Vec<u8>,
}

/// v1's shape, as parsed by v3.
#[derive(Deserialize)]
struct ShapeV1 {
    steps_total: Vec<u32>,
    steps_done: Vec<u32>,
    finished: Vec<bool>,
}

/// v2's shape, as parsed by v3.
#[derive(Deserialize)]
struct ShapeV2 {
    steps_total: Vec<u32>,
    steps_done: Vec<u32>,
    finished: Vec<bool>,
    polished: Vec<bool>,
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

impl Shape for ShapeV3 {
    type State = Machine;

    fn restore(&self) -> Machine {
        tracing::info!("[v3:restore] origin=v3 (own shape, lossless)");
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
    /// v3's one-hop upgrade for origin-1 artifacts: reconcile the
    /// double-count, no polish in v1's era, pre-fix the finished ones.
    fn upgrade(self) -> Machine {
        tracing::info!("[v3:restore] origin=v1 (one-hop upgrade)");
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

impl ShapeV2 {
    /// v3's upgrade for origin-2 artifacts: v2 flagged finished one step
    /// early — such items arrive pre-fix for the fixer to re-validate.
    fn upgrade(self) -> Machine {
        tracing::info!("[v3:restore] origin=v2 (repairing the early-flag bug)");
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
                    needs_fix: *finished,
                })
                .collect(),
        }
    }
}

fn fix(m: &mut Machine) {
    for item in &mut m.items {
        if item.needs_fix {
            tracing::info!("[v3:fix] item {} pre-fix resolved", item.id);
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
    eprintln!("usage: version3 save <path> | restore-run <path>");
    std::process::exit(2);
}

fn save(path: &str) {
    let machine = Machine {
        items: vec![
            Item { id: 1, steps: 3, step: 3, done: Some("v1".into()), polished: true, needs_fix: false },
            Item { id: 2, steps: 3, step: 1, done: None, polished: false, needs_fix: false },
        ],
    };
    with_checkpoint_pair(machine, |mut h, _req| {
        h.safepoint();
        let shape = ShapeV3 {
            steps_total: h.items.iter().map(|i| i.steps).collect(),
            steps_done: h.items.iter().map(|i| i.step).collect(),
            finished: h.items.iter().map(|i| i.done.is_some()).collect(),
            polished: h.items.iter().map(|i| i.polished).collect(),
            origin: h.items.iter().map(|_| 3u8).collect(),
        };
        servyi_states_snapshot::write_value(&shape, path).expect("v3 writes its shape");
        tracing::info!("[v3:save] wrote ShapeV3: {} items", shape.finished.len());
    });
}

fn restore_run(path: &str) {
    let raw: serde_json::Value =
        servyi_states_snapshot::restore_from(path).expect("artifact parses");
    let mut machine: Machine = if raw.get("origin").is_some() {
        let shape: ShapeV3 = serde_json::from_value(raw).expect("v3 shape");
        <ShapeV3 as Shape>::restore(&shape)
    } else if raw.get("polished").is_some() {
        let shape: ShapeV2 = serde_json::from_value(raw).expect("v2 shape");
        shape.upgrade()
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
    let out = with_checkpoint_pair(machine, move |mut h, _req| loop {
        h.safepoint();
        let m = &mut *h;
        for item in &mut m.items {
            if item.done.is_some() || item.needs_fix {
                tracing::info!("[v3:run] item {} skipped (done)", item.id);
                continue;
            }
            let _n = c.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            item.step += 1;
            if item.step >= item.steps {
                item.done = Some(format!("v{}", item.id));
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
