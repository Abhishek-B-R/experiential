//! SQLite transaction and retention mechanics, called only by native delivery.
use super::local::{CaptureConfiguration, Policy};
use rusqlite::{params, Connection, OpenFlags};
use std::fs::OpenOptions;
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub(crate) struct Pending {
    pub policy: Policy,
    pub payload: String,
    pub experience_id: String,
    pub response_id: String,
    pub captured_at: u64,
}

/// A committed write remains successful even when journal cleanup must retry.
pub(super) struct Persisted {
    pub maintenance_failed: bool,
}

fn safe_error(_: rusqlite::Error) -> String {
    "local gateway capture database operation failed".into()
}

pub(super) fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|t| t.as_secs())
        .unwrap_or(0)
}

pub(super) fn validate(config: &CaptureConfiguration) -> Result<(), String> {
    let invalid = || "invalid bounded local gateway capture configuration".to_string();
    if !Path::new(&config.database_path).is_absolute()
        || !(1..=4096).contains(&config.queue_capacity)
    {
        return Err(invalid());
    }
    let mut keys = std::collections::HashSet::new();
    let mut policies = std::collections::HashMap::new();
    for binding in &config.bindings {
        let policy = &binding.policy;
        if policies
            .insert(
                (&policy.scope.user_id, &policy.scope.application_id),
                policy,
            )
            .is_some_and(|previous| previous != policy)
        {
            return Err(invalid());
        }
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

pub(super) fn open_database(path: &Path) -> Result<Connection, String> {
    // The operator controls the parent directory. Resolve directory aliases
    // (including macOS /var) without resolving away the final file's symlink.
    let parent = path
        .parent()
        .and_then(|parent| parent.canonicalize().ok())
        .ok_or("cannot resolve local capture storage directory")?;
    let filename = path
        .file_name()
        .ok_or("invalid local capture storage filename")?;
    let canonical = parent.join(filename);
    let path = canonical.as_path();
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
        Err(_) => return Err("cannot create local gateway capture database".into()),
    }
    require_private_file(path, false)?;
    for suffix in ["-wal", "-shm", "-journal"] {
        let mut sidecar = path.as_os_str().to_owned();
        sidecar.push(suffix);
        require_private_file(Path::new(&sidecar), true)?;
    }
    // Creation above establishes private permissions. Never let SQLite recreate
    // a disappeared file with default permissions or follow a final symlink.
    let connection = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_WRITE
            | OpenFlags::SQLITE_OPEN_NO_MUTEX
            | OpenFlags::SQLITE_OPEN_NOFOLLOW,
    )
    .map_err(safe_error)?;
    let foreign_tables: bool = connection
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table'
             AND name NOT LIKE 'sqlite_%' AND name != 'gateway_captures')",
            [],
            |row| row.get(0),
        )
        .map_err(safe_error)?;
    if foreign_tables {
        return Err(
            "unsupported local capture database schema; preserve this file and use a fresh traffic database"
                .into(),
        );
    }
    connection
        .busy_timeout(Duration::from_millis(100))
        .map_err(safe_error)?;
    connection
        .execute_batch(
            "PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; PRAGMA secure_delete=ON;
         CREATE TABLE IF NOT EXISTS gateway_captures (
           sequence INTEGER PRIMARY KEY AUTOINCREMENT,
           experience_id TEXT NOT NULL UNIQUE,
           user_id TEXT NOT NULL, application_id TEXT NOT NULL,
           response_id TEXT NOT NULL, captured_at INTEGER NOT NULL,
           expires_at INTEGER NOT NULL, payload_bytes INTEGER NOT NULL, payload TEXT NOT NULL,
           UNIQUE(user_id, application_id, response_id));
         CREATE INDEX IF NOT EXISTS gateway_capture_scope_sequence
           ON gateway_captures(user_id, application_id, sequence);",
        )
        .map_err(safe_error)?;
    Ok(connection)
}

fn require_private_file(path: &Path, optional: bool) -> Result<(), String> {
    let metadata = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if optional && error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(_) => return Err("cannot inspect local capture storage permissions".into()),
    };
    if !metadata.file_type().is_file() {
        return Err("local capture storage must be a regular file, not a symlink".into());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if metadata.permissions().mode() & 0o077 != 0 {
            return Err("local capture storage requires owner-only file permissions".into());
        }
    }
    Ok(())
}

pub(super) fn persist(connection: &mut Connection, item: &Pending) -> rusqlite::Result<Persisted> {
    let transaction = connection.transaction()?;
    transaction.execute(
        "INSERT INTO gateway_captures
         (experience_id,user_id,application_id,response_id,captured_at,expires_at,payload_bytes,payload)
         VALUES (?1,?2,?3,?4,?5,?6,?7,?8) ON CONFLICT DO NOTHING",
        params![item.experience_id, item.policy.scope.user_id, item.policy.scope.application_id,
            item.response_id, item.captured_at as i64, (item.captured_at + item.policy.retention_seconds) as i64,
            item.payload.len() as i64, item.payload],
    )?;
    let removed = prune_rows(&transaction, &item.policy, now())?;
    transaction.commit()?;
    Ok(Persisted {
        maintenance_failed: removed > 0 && truncate_wal(connection).is_err(),
    })
}

pub(super) fn prune(
    connection: &Connection,
    policy: &Policy,
    timestamp: u64,
) -> rusqlite::Result<()> {
    prune_rows(connection, policy, timestamp)?;
    // Retry even when no rows were deleted: a prior checkpoint may have been
    // blocked by a reader. Idle destination maintenance calls this again.
    truncate_wal(connection)
}

fn truncate_wal(connection: &Connection) -> rusqlite::Result<()> {
    let busy: i64 =
        connection.query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |row| row.get(0))?;
    if busy != 0 {
        return Err(rusqlite::Error::SqliteFailure(
            rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_BUSY),
            None,
        ));
    }
    Ok(())
}

fn prune_rows(connection: &Connection, policy: &Policy, timestamp: u64) -> rusqlite::Result<usize> {
    let expired = connection.execute(
        "DELETE FROM gateway_captures WHERE user_id=?1 AND application_id=?2 AND expires_at<=?3",
        params![
            policy.scope.user_id,
            policy.scope.application_id,
            timestamp as i64
        ],
    )?;
    let evicted = connection.execute(
        "DELETE FROM gateway_captures WHERE sequence IN (
           SELECT sequence FROM (
             SELECT sequence, ROW_NUMBER() OVER (ORDER BY sequence DESC) AS rank,
               SUM(payload_bytes) OVER (ORDER BY sequence DESC) AS bytes
             FROM gateway_captures WHERE user_id=?1 AND application_id=?2
           ) WHERE rank>?3 OR bytes>?4)",
        params![
            policy.scope.user_id,
            policy.scope.application_id,
            policy.maximum_experiences as i64,
            policy.maximum_storage_bytes as i64
        ],
    )?;
    Ok(expired + evicted)
}

#[cfg(test)]
#[path = "local_store_test.rs"]
mod tests;
