use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use dashmap::DashMap;
use tracing::field::Visit;
use tracing_subscriber::layer::Layer;

pub struct LogEntry {
    pub seq: u64,
    pub level: tracing::Level,
    pub target: String,
    pub message: String,
    pub fields: Vec<(String, String)>,
}

impl Clone for LogEntry {
    fn clone(&self) -> Self {
        Self {
            seq: self.seq,
            level: self.level,
            target: self.target.clone(),
            message: self.message.clone(),
            fields: self.fields.clone(),
        }
    }
}

pub struct MessageLog {
    entries: DashMap<String, Vec<LogEntry>>,
    statuses: DashMap<String, String>,
    counter: AtomicU64,
}

impl Default for MessageLog {
    fn default() -> Self {
        Self::new()
    }
}

impl MessageLog {
    pub fn new() -> Self {
        Self {
            entries: DashMap::new(),
            statuses: DashMap::new(),
            counter: AtomicU64::new(0),
        }
    }

    pub fn push(&self, context_id: String, entry: LogEntry) {
        self.entries
            .entry(context_id)
            .or_default()
            .push(entry);
    }

    pub fn next_seq(&self) -> u64 {
        self.counter.fetch_add(1, Ordering::Relaxed)
    }

    pub fn get(&self, context_id: &str) -> Vec<LogEntry> {
        self.entries
            .get(context_id)
            .map(|r| r.value().clone())
            .unwrap_or_default()
    }

    pub fn keys(&self) -> Vec<String> {
        self.entries.iter().map(|r| r.key().clone()).collect()
    }

    pub fn has(&self, context_id: &str) -> bool {
        self.entries.contains_key(context_id)
    }

    pub fn set_status(&self, context_id: String, status: String) {
        self.statuses.insert(context_id, status);
    }

    pub fn get_status(&self, context_id: &str) -> Option<String> {
        self.statuses.get(context_id).map(|r| r.value().clone())
    }
}

struct EventVisitor {
    context_ids: Vec<String>,
    message: String,
    fields: Vec<(String, String)>,
}

const CTX_FIELD_NAMES: &[&str] = &["ctx", "context_id", "func_ctx", "fctx"];

fn normalize_field_name(name: &str) -> &str {
    name.strip_prefix("self.").unwrap_or(name)
}

impl Visit for EventVisitor {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn fmt::Debug) {
        let formatted = format!("{:?}", value);
        let name = normalize_field_name(field.name());
        if name == "message" {
            self.message = formatted;
        } else if CTX_FIELD_NAMES.contains(&name) {
            self.context_ids.push(formatted);
        } else {
            self.fields.push((name.to_string(), formatted));
        }
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        let name = normalize_field_name(field.name());
        if name == "message" {
            self.message = value.to_string();
        } else if CTX_FIELD_NAMES.contains(&name) {
            self.context_ids.push(value.to_string());
        } else {
            self.fields.push((name.to_string(), value.to_string()));
        }
    }

    fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
        self.fields.push((field.name().to_string(), value.to_string()));
    }

    fn record_i64(&mut self, field: &tracing::field::Field, value: i64) {
        self.fields.push((field.name().to_string(), value.to_string()));
    }

    fn record_bool(&mut self, field: &tracing::field::Field, value: bool) {
        self.fields.push((field.name().to_string(), value.to_string()));
    }

    fn record_f64(&mut self, field: &tracing::field::Field, value: f64) {
        self.fields.push((field.name().to_string(), value.to_string()));
    }

    fn record_bytes(&mut self, field: &tracing::field::Field, value: &[u8]) {
        self.fields.push((field.name().to_string(), format!("{:?}", value)));
    }
}

pub struct MessageLogLayer {
    log: Arc<MessageLog>,
}

impl MessageLogLayer {
    pub fn new(log: Arc<MessageLog>) -> Self {
        Self { log }
    }
}

impl<S: tracing::Subscriber> Layer<S> for MessageLogLayer {
    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let mut visitor = EventVisitor {
            context_ids: Vec::new(),
            message: String::new(),
            fields: Vec::new(),
        };
        event.record(&mut visitor);

        if visitor.context_ids.is_empty() {
            return;
        }

        let level = *event.metadata().level();
        let target = event.metadata().target().to_string();
        let entry = LogEntry {
            seq: self.log.next_seq(),
            level,
            target,
            message: visitor.message,
            fields: visitor.fields,
        };

        for ctx_id in visitor.context_ids {
            self.log.push(ctx_id, entry.clone());
        }
    }
}

pub fn ancestors(ctx_id: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut current = ctx_id;
    loop {
        result.push(current.to_string());
        match current.rfind('.') {
            Some(pos) => current = &current[..pos],
            None => break,
        }
    }
    result
}

pub fn is_valid_ctx_id(s: &str) -> bool {
    !s.is_empty()
        && s.split('.').all(|seg| !seg.is_empty() && seg.chars().all(|c| c.is_ascii_digit()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ancestors_leaf() {
        assert_eq!(ancestors("1.2.3"), vec!["1.2.3", "1.2", "1"]);
    }

    #[test]
    fn test_ancestors_root_child() {
        assert_eq!(ancestors("5"), vec!["5"]);
    }

    #[test]
    fn test_ancestors_deep() {
        assert_eq!(ancestors("1.2.3.4"), vec!["1.2.3.4", "1.2.3", "1.2", "1"]);
    }

    #[test]
    fn test_is_valid_ctx_id() {
        assert!(is_valid_ctx_id("1"));
        assert!(is_valid_ctx_id("1.2.3"));
        assert!(!is_valid_ctx_id(""));
        assert!(!is_valid_ctx_id("abc"));
        assert!(!is_valid_ctx_id("1.2."));
        assert!(!is_valid_ctx_id(".1"));
        assert!(!is_valid_ctx_id("1.a.2"));
    }

    fn test_entry(log: &MessageLog, message: &str) -> LogEntry {
        LogEntry {
            seq: log.next_seq(),
            level: tracing::Level::INFO,
            target: "test".to_string(),
            message: message.to_string(),
            fields: vec![],
        }
    }

    #[test]
    fn test_message_log_push_and_get() {
        let log = MessageLog::new();
        log.push("1.2".to_string(), test_entry(&log, "hello"));
        let entries = log.get("1.2");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].message, "hello");
        assert!(log.get("9.9").is_empty());
    }
}
