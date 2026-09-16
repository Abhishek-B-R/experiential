//! Bounded durable experience records written outside the request executor.

use std::fs::OpenOptions;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use rusqlite::{params, Connection};
use serde_json::Value;

use super::{CaptureConfiguration, Policy};

struct Pending {
    policy: Policy,
    payload: String,
    experience_id: String,
    response_id: String,
    captured_at: u64,
}

pub(crate) struct CaptureStore {
    config: CaptureConfiguration,
    sender: Mutex<Option<mpsc::SyncSender<Pending>>>,
    worker: Mutex<Option<JoinHandle<()>>>,
    skipped: Arc<AtomicU64>,
}

impl CaptureStore {
    pub(crate) fn open(config: CaptureConfiguration) -> Result<Option<Arc<Self>>, String> {
        validate(&config)?;
        if !config.bindings.iter().any(|binding| binding.policy.enabled) {
            return Ok(None);
        }
        let mut connection = open_database(Path::new(&config.database_path))?;
        let policies: Vec<Policy> = config
            .bindings
            .iter()
            .filter(|binding| binding.policy.enabled)
            .map(|binding| binding.policy.clone())
            .collect();
        for policy in &policies {
            prune(&connection, policy, now()).map_err(safe_error)?;
        }
        let (sender, receiver) = mpsc::sync_channel::<Pending>(config.queue_capacity);
        let skipped = Arc::new(AtomicU64::new(0));
        let failures = skipped.clone();
        let worker = std::thread::Builder::new()
            .name("claas-capture".into())
            .spawn(move || {
                let mut last_cleanup = Instant::now();
                loop {
                    if last_cleanup.elapsed() >= Duration::from_secs(1) {
                        for policy in &policies {
                            if prune(&connection, policy, now()).is_err() {
                                failures.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                        last_cleanup = Instant::now();
                    }
                    match receiver.recv_timeout(Duration::from_secs(1)) {
                        Ok(item) => {
                            if persist(&mut connection, item).is_err() {
                                failures.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                        Err(mpsc::RecvTimeoutError::Timeout) => {}
                        Err(mpsc::RecvTimeoutError::Disconnected) => break,
                    }
                }
            })
            .map_err(|_| "cannot start CLaaS capture writer".to_string())?;
        Ok(Some(Arc::new(Self {
            config,
            sender: Mutex::new(Some(sender)),
            worker: Mutex::new(Some(worker)),
            skipped,
        })))
    }

    pub(crate) fn policy(&self, user: &str, alias: &str) -> Option<&Policy> {
        self.config
            .bindings
            .iter()
            .find(|binding| {
                binding.policy.enabled
                    && binding.alias == alias
                    && binding.policy.scope.user_id == user
            })
            .map(|binding| &binding.policy)
    }

    #[cfg(test)]
    pub(crate) fn database_path(&self) -> &str {
        &self.config.database_path
    }

    pub(crate) fn skipped_count(&self) -> u64 {
        self.skipped.load(Ordering::Relaxed)
    }

    pub(crate) fn skip(&self) {
        self.skipped.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn submit(&self, policy: Policy, value: Value, captured_at: u64) {
        let Some(experience_id) = value["experience_id"].as_str().map(str::to_owned) else {
            self.skip();
            return;
        };
        let Some(response_id) = value["response_id"].as_str().map(str::to_owned) else {
            self.skip();
            return;
        };
        let Ok(payload) = serde_json::to_string(&value) else {
            self.skip();
            return;
        };
        if payload.len() > policy.maximum_experience_bytes {
            self.skip();
            return;
        }
        let item = Pending {
            policy,
            payload,
            experience_id,
            response_id,
            captured_at,
        };
        let Ok(sender) = self.sender.lock() else {
            self.skip();
            return;
        };
        if sender
            .as_ref()
            .is_none_or(|sender| sender.try_send(item).is_err())
        {
            self.skip();
        }
    }

    /// Drain queued writes during graceful shutdown before the local process exits.
    pub(crate) fn close(&self) {
        if let Ok(mut sender) = self.sender.lock() {
            sender.take();
        }
        if let Ok(mut worker) = self.worker.lock() {
            if let Some(worker) = worker.take() {
                let _ = worker.join();
            }
        }
    }
}

fn safe_error(_: rusqlite::Error) -> String {
    "CLaaS capture database operation failed".into()
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|t| t.as_secs())
        .unwrap_or(0)
}

fn validate(config: &CaptureConfiguration) -> Result<(), String> {
    let invalid = || "invalid bounded CLaaS capture configuration".to_string();
    if !Path::new(&config.database_path).is_absolute()
        || !(1..=4096).contains(&config.queue_capacity)
    {
        return Err(invalid());
    }
    let mut keys = std::collections::HashSet::new();
    for binding in &config.bindings {
        let policy = &binding.policy;
        if [
            &binding.alias,
            &policy.scope.user_id,
            &policy.scope.application_id,
        ]
        .iter()
        .any(|value| value.trim().is_empty() || value.len() > 512)
            || !(1..=1_000_000).contains(&policy.maximum_experiences)
            || policy.maximum_experience_bytes == 0
            || policy.maximum_experience_bytes > policy.maximum_storage_bytes
            || policy.maximum_storage_bytes > i64::MAX as usize
            || policy.retention_seconds == 0
            || policy.retention_seconds > i64::MAX as u64 / 2
            || !keys.insert((&policy.scope.user_id, &binding.alias))
        {
            return Err(invalid());
        }
    }
    Ok(())
}

fn open_database(path: &Path) -> Result<Connection, String> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    match options.open(path) {
        Ok(file) => drop(file),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(_) => return Err("cannot create CLaaS capture database".into()),
    }
    let connection = Connection::open(path).map_err(safe_error)?;
    connection
        .busy_timeout(Duration::from_millis(100))
        .map_err(safe_error)?;
    connection
        .execute_batch(
            "PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; PRAGMA secure_delete=ON;
         CREATE TABLE IF NOT EXISTS claas_experiences (
           sequence INTEGER PRIMARY KEY AUTOINCREMENT,
           experience_id TEXT NOT NULL UNIQUE,
           user_id TEXT NOT NULL, application_id TEXT NOT NULL,
           response_id TEXT NOT NULL, captured_at INTEGER NOT NULL,
           expires_at INTEGER NOT NULL, payload_bytes INTEGER NOT NULL, payload TEXT NOT NULL,
           UNIQUE(user_id, application_id, response_id));
         CREATE INDEX IF NOT EXISTS claas_scope_sequence
           ON claas_experiences(user_id, application_id, sequence);",
        )
        .map_err(safe_error)?;
    Ok(connection)
}

fn persist(connection: &mut Connection, item: Pending) -> rusqlite::Result<()> {
    let transaction = connection.transaction()?;
    transaction.execute(
        "INSERT INTO claas_experiences
         (experience_id,user_id,application_id,response_id,captured_at,expires_at,payload_bytes,payload)
         VALUES (?1,?2,?3,?4,?5,?6,?7,?8) ON CONFLICT DO NOTHING",
        params![item.experience_id, item.policy.scope.user_id, item.policy.scope.application_id,
            item.response_id, item.captured_at as i64, (item.captured_at + item.policy.retention_seconds) as i64,
            item.payload.len() as i64, item.payload],
    )?;
    prune(&transaction, &item.policy, now())?;
    transaction.commit()
}

fn prune(connection: &Connection, policy: &Policy, timestamp: u64) -> rusqlite::Result<()> {
    connection.execute(
        "DELETE FROM claas_experiences WHERE user_id=?1 AND application_id=?2 AND expires_at<=?3",
        params![
            policy.scope.user_id,
            policy.scope.application_id,
            timestamp as i64
        ],
    )?;
    connection.execute(
        "DELETE FROM claas_experiences WHERE sequence IN (
           SELECT sequence FROM (
             SELECT sequence, ROW_NUMBER() OVER (ORDER BY sequence DESC) AS rank,
               SUM(payload_bytes) OVER (ORDER BY sequence DESC) AS bytes
             FROM claas_experiences WHERE user_id=?1 AND application_id=?2
           ) WHERE rank>?3 OR bytes>?4)",
        params![
            policy.scope.user_id,
            policy.scope.application_id,
            policy.maximum_experiences as i64,
            policy.maximum_storage_bytes as i64
        ],
    )?;
    Ok(())
}

#[cfg(test)]
#[path = "store_test.rs"]
mod tests;
