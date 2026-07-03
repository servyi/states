use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

static GLOBAL_COUNTER: AtomicU64 = AtomicU64::new(1);

static ROOT: OnceLock<ContextId> = OnceLock::new();

pub struct ContextId {
    id: u64,
    name: Vec<u64>,
    child_counter: Mutex<u64>,
}

impl ContextId {
    pub fn root() -> &'static ContextId {
        ROOT.get_or_init(|| ContextId {
            id: 0,
            name: Vec::new(),
            child_counter: Mutex::new(1),
        })
    }

    pub fn new_child(&self) -> ContextId {
        let child_num = {
            let mut counter = self.child_counter.lock().unwrap();
            let val = *counter;
            *counter += 1;
            val
        };
        let mut name = Vec::with_capacity(self.name.len() + 1);
        name.extend_from_slice(&self.name);
        name.push(child_num);
        ContextId {
            id: GLOBAL_COUNTER.fetch_add(1, Ordering::Relaxed),
            name,
            child_counter: Mutex::new(1),
        }
    }

    pub fn id(&self) -> u64 {
        self.id
    }
}

impl fmt::Display for ContextId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.name.is_empty() {
            return write!(f, "0");
        }
        for (i, seg) in self.name.iter().enumerate() {
            if i > 0 {
                write!(f, ".")?;
            }
            write!(f, "{}", seg)?;
        }
        Ok(())
    }
}

impl fmt::Debug for ContextId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl PartialEq for ContextId {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for ContextId {}

impl std::hash::Hash for ContextId {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn root_displays_as_0() {
        assert_eq!(ContextId::root().to_string(), "0");
    }

    #[test]
    fn child_increments_counter() {
        let root = ContextId::root();
        let c1 = root.new_child();
        let c2 = root.new_child();
        assert_ne!(c1, c2);
    }

    #[test]
    fn grandchild_appends_segment() {
        let root = ContextId::root();
        let child = root.new_child();
        let gc1 = child.new_child();
        let gc2 = child.new_child();
        let prefix = child.to_string();
        assert_eq!(gc1.to_string(), format!("{prefix}.1"));
        assert_eq!(gc2.to_string(), format!("{prefix}.2"));
    }

    #[test]
    fn siblings_have_independent_counters() {
        let root = ContextId::root();
        let a = root.new_child();
        let b = root.new_child();
        let a1 = a.new_child();
        let b1 = b.new_child();
        let a2 = a.new_child();
        let a_str = a.to_string();
        let b_str = b.to_string();
        assert_ne!(a_str, b_str);
        assert_eq!(a1.to_string(), format!("{a_str}.1"));
        assert_eq!(b1.to_string(), format!("{b_str}.1"));
        assert_eq!(a2.to_string(), format!("{a_str}.2"));
    }

    #[test]
    fn equality_based_on_global_id() {
        let root = ContextId::root();
        let a = root.new_child();
        let b = root.new_child();
        assert_eq!(a, a);
        assert_ne!(a, b);
    }

    #[test]
    fn deep_nesting() {
        let root = ContextId::root();
        let c1 = root.new_child();
        let c2 = c1.new_child();
        let c3 = c2.new_child();
        let p1 = c1.to_string();
        assert_eq!(c2.to_string(), format!("{p1}.1"));
        assert_eq!(c3.to_string(), format!("{p1}.1.1"));
    }

    #[test]
    fn global_ids_are_unique() {
        let root = ContextId::root();
        let a = root.new_child();
        let b = a.new_child();
        let c = b.new_child();
        assert_ne!(a.id(), b.id());
        assert_ne!(b.id(), c.id());
        assert_ne!(a.id(), c.id());
    }
}
