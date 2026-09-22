//! Count- and byte-bounded delivery, isolated from request and bridge executors.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use serde::Deserialize;

use super::record::Record;
use super::response::WireResponse;

/// Local SQLite and hosted persistence implement the same off-path destination.
pub(crate) trait Sink: Send + 'static {
    type Prepared;

    /// Maximum retained preparation allocation, reserved before queue admission.
    fn preparation_bytes(maximum_record_bytes: usize) -> usize;

    /// Prepare once, off serving; retrying storage must not re-encode the record.
    fn prepare(record: &Record, maximum_bytes: usize) -> Result<Self::Prepared, ()>;

    /// Acknowledge an idempotent write or intentional policy exclusion. An error
    /// retains the payload for retry; error details must never include content.
    fn write(&mut self, prepared: &Self::Prepared) -> Result<(), ()>;

    /// Run retention maintenance without adding storage work to serving.
    fn maintain(&mut self) -> Result<(), ()> {
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Limits {
    pub maximum_records: usize,
    pub maximum_bytes: usize,
    pub maximum_record_bytes: usize,
}

impl Limits {
    pub(crate) fn validate(&self) -> Result<(), &'static str> {
        if !(1..=4096).contains(&self.maximum_records)
            || !(1..=8 * 1024 * 1024).contains(&self.maximum_record_bytes)
            || self.maximum_bytes < self.maximum_record_bytes
            || self.maximum_bytes > 256 * 1024 * 1024
        {
            return Err("invalid capture delivery bounds");
        }
        Ok(())
    }
}

#[derive(Default)]
struct Counters {
    bytes: AtomicUsize,
    preparation_bytes: AtomicUsize,
    pending: AtomicUsize,
    dropped: AtomicU64,
    persisted: AtomicU64,
    failed: AtomicU64,
    capacity: Mutex<()>,
    available: Condvar,
}

/// One worker prepares at a time. Its workspace cannot compete with a full queue.
struct PreparationBudget(Arc<Counters>);

impl Drop for PreparationBudget {
    fn drop(&mut self) {
        self.0.preparation_bytes.store(0, Ordering::Release);
    }
}

struct Pending {
    value: Record,
    wire: Option<WireResponse>,
    bytes: usize,
    counters: Arc<Counters>,
    completed: Option<mpsc::SyncSender<bool>>,
}

impl Drop for Pending {
    fn drop(&mut self) {
        let _capacity = self
            .counters
            .capacity
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        self.counters.bytes.fetch_sub(self.bytes, Ordering::AcqRel);
        self.counters.pending.fetch_sub(1, Ordering::AcqRel);
        self.counters.available.notify_all();
    }
}

pub(crate) struct Delivery {
    limits: Limits,
    maximum_queued_bytes: usize,
    sender: Mutex<Option<mpsc::SyncSender<Pending>>>,
    worker: Mutex<Option<JoinHandle<()>>>,
    counters: Arc<Counters>,
}

impl Delivery {
    pub(crate) fn new<S: Sink>(limits: Limits, mut sink: S) -> Result<Self, &'static str> {
        limits.validate()?;
        let preparation_bytes = S::preparation_bytes(limits.maximum_record_bytes);
        let maximum_queued_bytes = limits
            .maximum_bytes
            .checked_sub(preparation_bytes)
            .filter(|available| *available > 0)
            .ok_or("capture byte budget must fit destination preparation and queued content")?;
        let (sender, receiver) = mpsc::sync_channel::<Pending>(limits.maximum_records);
        let counters = Arc::new(Counters::default());
        let worker_counters = counters.clone();
        let maximum_record_bytes = limits.maximum_record_bytes;
        let worker = std::thread::Builder::new()
            .name("exp-capture".into())
            .spawn(move || {
                let mut maintained = Instant::now();
                loop {
                    if maintained.elapsed() >= Duration::from_secs(1) {
                        if sink.maintain().is_err() {
                            worker_counters.failed.fetch_add(1, Ordering::Relaxed);
                        }
                        maintained = Instant::now();
                    }
                    match receiver.recv_timeout(Duration::from_millis(100)) {
                        Ok(mut item) => {
                            worker_counters
                                .preparation_bytes
                                .store(preparation_bytes, Ordering::Release);
                            let _preparation = PreparationBudget(worker_counters.clone());
                            if let Some(wire) = item.wire.take() {
                                item.value.response = wire.decode();
                                if item.value.response.is_none() {
                                    item.value.provider_reasoning = None;
                                    item.value.provider_tool_calls_json = None;
                                }
                            }
                            let persisted = match S::prepare(&item.value, maximum_record_bytes) {
                                Ok(prepared) => {
                                    let mut delay = Duration::from_millis(25);
                                    while sink.write(&prepared).is_err() {
                                        worker_counters.failed.fetch_add(1, Ordering::Relaxed);
                                        // Hold the same queue slot and byte charge until
                                        // acknowledged, including across close timeouts.
                                        // Bound retry frequency, never expire accepted data.
                                        std::thread::sleep(delay);
                                        delay = (delay * 2).min(Duration::from_secs(1));
                                        if maintained.elapsed() >= Duration::from_secs(1) {
                                            if sink.maintain().is_err() {
                                                worker_counters
                                                    .failed
                                                    .fetch_add(1, Ordering::Relaxed);
                                            }
                                            maintained = Instant::now();
                                        }
                                    }
                                    true
                                }
                                Err(()) => false,
                            };
                            let counter = if persisted {
                                &worker_counters.persisted
                            } else {
                                &worker_counters.failed
                            };
                            counter.fetch_add(1, Ordering::Relaxed);
                            if let Some(completed) = &item.completed {
                                let _ = completed.send(persisted);
                            }
                        }
                        Err(mpsc::RecvTimeoutError::Timeout) => {}
                        Err(mpsc::RecvTimeoutError::Disconnected) => break,
                    }
                }
            })
            .map_err(|_| "cannot start capture delivery worker")?;
        Ok(Self {
            limits,
            maximum_queued_bytes,
            sender: Mutex::new(Some(sender)),
            worker: Mutex::new(Some(worker)),
            counters,
        })
    }

    /// Wait for capacity; accepted records are never discarded to make room.
    #[cfg(test)]
    pub(crate) fn submit(&self, value: Record) -> bool {
        self.enqueue(value, None, None)
    }

    /// A successful completion means the destination has persisted this update.
    pub(super) fn submit_wait(&self, value: Record, wire: Option<WireResponse>) -> bool {
        let (completed, outcome) = mpsc::sync_channel(1);
        self.enqueue(value, wire, Some(completed)) && outcome.recv().unwrap_or(false)
    }

    fn enqueue(
        &self,
        value: Record,
        wire: Option<WireResponse>,
        completed: Option<mpsc::SyncSender<bool>>,
    ) -> bool {
        let bytes = value.heap_bytes() + wire.as_ref().map_or(0, WireResponse::heap_bytes);
        if bytes > self.maximum_queued_bytes {
            return self.dropped();
        }
        // Clone before waiting. Shutdown closes new admissions, while producers
        // already waiting retain their right to deliver and keep the worker alive.
        let Some(sender) = self.sender.lock().ok().and_then(|sender| sender.clone()) else {
            return self.dropped();
        };
        let mut capacity = self
            .counters
            .capacity
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        while self.counters.pending.load(Ordering::Acquire) >= self.limits.maximum_records
            || self
                .counters
                .bytes
                .load(Ordering::Acquire)
                .saturating_add(bytes)
                > self.maximum_queued_bytes
        {
            capacity = self
                .counters
                .available
                .wait(capacity)
                .unwrap_or_else(|e| e.into_inner());
        }
        self.counters.bytes.fetch_add(bytes, Ordering::AcqRel);
        self.counters.pending.fetch_add(1, Ordering::AcqRel);
        drop(capacity);
        let item = Pending {
            value,
            wire,
            bytes,
            counters: self.counters.clone(),
            completed,
        };
        if sender.send(item).is_err() {
            return self.dropped();
        }
        true
    }

    fn dropped(&self) -> bool {
        self.counters.dropped.fetch_add(1, Ordering::Relaxed);
        false
    }

    /// Stop new submissions; a timeout reports an unfinished drain without purging it.
    pub(crate) fn close_until(&self, until: Instant) -> bool {
        if let Ok(mut sender) = self.sender.lock() {
            sender.take();
        }
        let Ok(mut worker) = self.worker.lock() else {
            return false;
        };
        let Some(handle) = worker.as_ref() else {
            return self.counters.pending.load(Ordering::Acquire) == 0;
        };
        while !handle.is_finished() && Instant::now() < until {
            std::thread::sleep(Duration::from_millis(1));
        }
        if !handle.is_finished() {
            return false;
        }
        worker.take().is_some_and(|handle| handle.join().is_ok())
    }

    pub(crate) fn counts(&self) -> [u64; 5] {
        [
            self.counters.pending.load(Ordering::Acquire) as u64,
            self.counters.bytes.load(Ordering::Acquire) as u64
                + self.counters.preparation_bytes.load(Ordering::Acquire) as u64,
            self.counters.persisted.load(Ordering::Relaxed),
            self.counters.failed.load(Ordering::Relaxed),
            self.counters.dropped.load(Ordering::Relaxed),
        ]
    }
}

#[cfg(test)]
#[path = "delivery_test.rs"]
mod tests;
