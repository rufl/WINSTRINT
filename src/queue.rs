use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::diagnostics;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QueueTicket {
    pub id: String,
    pub pid: u32,
    pub purpose: String,
    pub acquired_unix_secs: u64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct QueueState {
    pub active: Option<QueueTicket>,
    pub waiting: Vec<QueueTicket>,
}

#[derive(Clone, Debug)]
pub struct GpuQueue {
    path: PathBuf,
    lock_path: PathBuf,
}

pub struct QueueLease {
    queue: GpuQueue,
    ticket: Option<QueueTicket>,
}

impl GpuQueue {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        let lock_path = PathBuf::from(format!("{}.lock", path.display()));
        Self { path, lock_path }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn status(&self) -> Result<QueueState> {
        self.with_lock(|state| Ok(state.clone()))
    }

    pub fn acquire(&self, purpose: impl Into<String>, wait: bool) -> Result<QueueLease> {
        let ticket = QueueTicket {
            id: format!("{}-{}", std::process::id(), monotonic_id()),
            pid: std::process::id(),
            purpose: purpose.into(),
            acquired_unix_secs: now_unix_secs(),
        };
        let mut announced_waiting = false;
        loop {
            let acquired = self.with_lock(|state| {
                if state.active.is_none() {
                    state.waiting.retain(|item| item.id != ticket.id);
                    state.active = Some(ticket.clone());
                    Ok(true)
                } else if wait {
                    if !announced_waiting {
                        state.waiting.push(ticket.clone());
                    }
                    Ok(false)
                } else {
                    Ok(false)
                }
            })?;
            if acquired {
                return Ok(QueueLease {
                    queue: self.clone(),
                    ticket: Some(ticket),
                });
            }
            if !wait {
                let state = self.status()?;
                let owner = state
                    .active
                    .map(|item| format!("{} (pid {}, {})", item.id, item.pid, item.purpose))
                    .unwrap_or_else(|| "unknown owner".into());
                anyhow::bail!("GPU queue is busy: {owner}");
            }
            announced_waiting = true;
            thread::sleep(Duration::from_millis(500));
        }
    }

    pub fn release(&self, ticket_id: Option<&str>) -> Result<bool> {
        self.with_lock(|state| {
            let Some(active) = state.active.as_ref() else {
                return Ok(false);
            };
            if ticket_id.is_some_and(|id| id != active.id) {
                anyhow::bail!(
                    "active queue ticket is {}, not {}",
                    active.id,
                    ticket_id.unwrap_or_default()
                );
            }
            state.active = None;
            Ok(true)
        })
    }

    pub fn run_ticketed(
        &self,
        purpose: &str,
        command: &[String],
        force: bool,
    ) -> Result<std::process::ExitStatus> {
        if command.is_empty() {
            anyhow::bail!("ticketed command cannot be empty");
        }
        let _lease = self.acquire(purpose.to_string(), true)?;
        let current = diagnostics::collect()?;
        diagnostics::require_idle(&current, force, true)?;
        println!("GPU queue ticket granted: {purpose}");
        println!("Running: {}", command.join(" "));
        Command::new(&command[0])
            .args(&command[1..])
            .status()
            .with_context(|| format!("starting ticketed command {}", command[0]))
    }

    fn with_lock<T>(&self, operation: impl FnOnce(&mut QueueState) -> Result<T>) -> Result<T> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }
        let _lock = FileLock::acquire(&self.lock_path)?;
        let mut state = if self.path.exists() {
            let text = fs::read_to_string(&self.path)
                .with_context(|| format!("reading queue state {}", self.path.display()))?;
            serde_json::from_str(&text)
                .with_context(|| format!("parsing queue state {}", self.path.display()))?
        } else {
            QueueState::default()
        };
        prune_dead_tickets(&mut state);
        let result = operation(&mut state)?;
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&self.path)
            .with_context(|| format!("writing queue state {}", self.path.display()))?;
        serde_json::to_writer_pretty(&mut file, &state)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        Ok(result)
    }
}

impl QueueLease {
    pub fn ticket(&self) -> Option<&QueueTicket> {
        self.ticket.as_ref()
    }
    pub fn release(mut self) -> Result<bool> {
        let Some(ticket) = self.ticket.take() else {
            return Ok(false);
        };
        self.queue.release(Some(&ticket.id))
    }
}

impl Drop for QueueLease {
    fn drop(&mut self) {
        if let Some(ticket) = self.ticket.take() {
            let _ = self.queue.release(Some(&ticket.id));
        }
    }
}

struct FileLock {
    path: PathBuf,
}

impl FileLock {
    fn acquire(path: &Path) -> Result<Self> {
        for _ in 0..600 {
            match OpenOptions::new().write(true).create_new(true).open(path) {
                Ok(mut file) => {
                    let _ = writeln!(file, "{}", std::process::id());
                    return Ok(Self {
                        path: path.to_path_buf(),
                    });
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    if stale_lock(path) {
                        let _ = fs::remove_file(path);
                    } else {
                        thread::sleep(Duration::from_millis(50));
                    }
                }
                Err(error) => {
                    return Err(error)
                        .with_context(|| format!("creating queue lock {}", path.display()));
                }
            }
        }
        anyhow::bail!("timed out waiting for queue lock {}", path.display())
    }
}

impl Drop for FileLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn stale_lock(path: &Path) -> bool {
    fs::metadata(path)
        .and_then(|metadata| metadata.modified())
        .ok()
        .and_then(|modified| modified.elapsed().ok())
        .is_some_and(|age| age > Duration::from_secs(300))
}
fn prune_dead_tickets(state: &mut QueueState) {
    if state
        .active
        .as_ref()
        .is_some_and(|ticket| !process_alive(ticket.pid))
    {
        state.active = None;
    }
    state.waiting.retain(|ticket| process_alive(ticket.pid));
}

#[cfg(windows)]
fn process_alive(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }
    let filter = format!("PID eq {pid}");
    match Command::new("tasklist")
        .args(["/FI", &filter, "/FO", "CSV", "/NH"])
        .output()
    {
        Ok(output) if output.status.success() => {
            let needle = format!("\"{pid}\"");
            String::from_utf8_lossy(&output.stdout).contains(&needle)
        }
        Ok(_) | Err(_) => true,
    }
}

#[cfg(not(windows))]
fn process_alive(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }
    Command::new("kill")
        .args(["-0", &pid.to_string()])
        .status()
        .map(|status| status.success())
        .unwrap_or(true)
}

fn monotonic_id() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
}
fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dead_active_ticket_does_not_block_acquire() {
        let root = std::env::temp_dir().join(format!("winstrint-queue-test-{}", monotonic_id()));
        fs::create_dir_all(&root).expect("test directory");
        let path = root.join("queue.json");
        fs::write(
            &path,
            serde_json::to_vec(&QueueState {
                active: Some(QueueTicket {
                    id: "dead".into(),
                    pid: 0,
                    purpose: "dead".into(),
                    acquired_unix_secs: 0,
                }),
                waiting: Vec::new(),
            })
            .expect("queue JSON"),
        )
        .expect("queue state");

        let queue = GpuQueue::new(&path);
        let lease = queue
            .acquire("test", false)
            .expect("dead owner is reclaimed");
        assert_eq!(lease.ticket().expect("ticket").purpose, "test");
        lease.release().expect("release");
        let _ = fs::remove_dir_all(root);
    }
}
