use std::fs::{self, File, OpenOptions};
use std::io;
use std::path::{Component, Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use reqwest::blocking::Client;
use serde_json::json;
use zip::ZipArchive;

use crate::config::{AppConfig, BackendPreference};
use crate::diagnostics::{self, Diagnostics, GpuKind};
use crate::model::{self, ModelArtifact};
use crate::strata;

#[cfg(windows)]
use std::os::windows::io::AsRawHandle;

#[cfg(windows)]
use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
#[cfg(windows)]
use windows_sys::Win32::System::JobObjects::{
    AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation,
    SetInformationJobObject,
};

#[cfg(windows)]
struct ProcessTree {
    job: HANDLE,
}

#[cfg(not(windows))]
struct ProcessTree;

impl ProcessTree {
    fn new() -> Result<Self> {
        #[cfg(windows)]
        {
            let job = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
            if job.is_null() {
                anyhow::bail!(
                    "creating runtime process job object: {}",
                    io::Error::last_os_error()
                );
            }

            let mut limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
            limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            let configured = unsafe {
                SetInformationJobObject(
                    job,
                    JobObjectExtendedLimitInformation,
                    (&limits as *const JOBOBJECT_EXTENDED_LIMIT_INFORMATION).cast(),
                    std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                )
            };
            if configured == 0 {
                unsafe {
                    CloseHandle(job);
                }
                anyhow::bail!(
                    "configuring runtime process job object: {}",
                    io::Error::last_os_error()
                );
            }
            Ok(Self { job })
        }
        #[cfg(not(windows))]
        {
            Ok(Self)
        }
    }

    fn assign(&self, child: &Child) -> Result<()> {
        #[cfg(windows)]
        {
            let process = child.as_raw_handle();
            if process.is_null() {
                anyhow::bail!("runtime child returned a null process handle");
            }
            if unsafe { AssignProcessToJobObject(self.job, process) } == 0 {
                anyhow::bail!(
                    "assigning runtime child to its process job: {}",
                    io::Error::last_os_error()
                );
            }
        }
        #[cfg(not(windows))]
        {
            let _ = child;
        }
        Ok(())
    }
}

#[cfg(windows)]
impl Drop for ProcessTree {
    fn drop(&mut self) {
        unsafe {
            CloseHandle(self.job);
        }
    }
}

fn terminate_child(mut child: Child) {
    let _ = child.kill();
    let _ = child.wait();
}

fn spawn_runtime(mut command: Command, description: &str) -> Result<(Child, ProcessTree)> {
    let process_tree = ProcessTree::new()?;
    let child = command
        .spawn()
        .with_context(|| format!("starting {description}"))?;
    if let Err(error) = process_tree.assign(&child) {
        terminate_child(child);
        return Err(error);
    }
    Ok((child, process_tree))
}

const LLAMA_TAG: &str = "b11669";
const LLAMA_SYCL_URL: &str = "https://github.com/ggml-org/llama.cpp/releases/download/b11669/llama-b11669-bin-win-sycl-x64.zip";
const LLAMA_SYCL_SIZE: u64 = 149_038_704;
const LLAMA_SYCL_SHA256: &str = "a6fa46b20c5161ad71e5b8ec4a52f2dcf713573ce25cefde86aeff4ff0e56cf1";
const LLAMA_VULKAN_URL: &str = "https://github.com/ggml-org/llama.cpp/releases/download/b11669/llama-b11669-bin-win-vulkan-x64.zip";
const LLAMA_VULKAN_SIZE: u64 = 32_182_350;
const LLAMA_VULKAN_SHA256: &str =
    "2c07574e7bde71321cb785282b20cc821e951132149eeceeea32ddebdbf0598e";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResolvedBackend {
    Cpu,
    LlamaSycl,
    LlamaVulkan,
    Strata,
}
impl ResolvedBackend {
    pub fn label(self) -> &'static str {
        match self {
            Self::Cpu => "llama.cpp CPU",
            Self::LlamaSycl => "llama.cpp SYCL",
            Self::LlamaVulkan => "llama.cpp Vulkan",
            Self::Strata => "Strata native SYCL",
        }
    }
}

pub struct RuntimeProcess {
    child: Child,
    _process_tree: ProcessTree,
    pub endpoint: String,
}

#[derive(Clone, Copy)]
enum LlamaLaunch {
    Gpu(ResolvedBackend),
    Cpu,
}

impl LlamaLaunch {
    fn label(self) -> &'static str {
        match self {
            Self::Gpu(backend) => backend.label(),
            Self::Cpu => "llama.cpp CPU fallback",
        }
    }

    fn health_timeout(self) -> Duration {
        match self {
            Self::Gpu(_) => Duration::from_secs(300),
            Self::Cpu => Duration::from_secs(1_800),
        }
    }
}

pub fn select_backend(config: &AppConfig, diagnostics: &Diagnostics) -> Result<ResolvedBackend> {
    let target = diagnostics::target_gpu(diagnostics);
    match config.backend {
        BackendPreference::Cpu => Ok(ResolvedBackend::Cpu),
        BackendPreference::Sycl => {
            if !diagnostics.has_sycl {
                anyhow::bail!(
                    "SYCL prerequisites were not detected; install Intel oneAPI/Level Zero or choose --backend vulkan"
                );
            }
            Ok(ResolvedBackend::LlamaSycl)
        }
        BackendPreference::Vulkan => {
            if !diagnostics.has_vulkan {
                anyhow::bail!(
                    "Vulkan loader/driver was not detected; install an Intel Arc driver before starting"
                );
            }
            Ok(ResolvedBackend::LlamaVulkan)
        }
        BackendPreference::Strata => strata_backend(config, target),
        BackendPreference::Auto => {
            let candidate = match target.map(|gpu| gpu.kind) {
                Some(GpuKind::A770) if diagnostics.has_sycl => ResolvedBackend::LlamaSycl,
                Some(GpuKind::A770) if diagnostics.has_vulkan => ResolvedBackend::LlamaVulkan,
                Some(GpuKind::Arc140T) if diagnostics.has_vulkan => ResolvedBackend::LlamaVulkan,
                Some(GpuKind::Arc140T) if diagnostics.has_sycl => ResolvedBackend::LlamaSycl,
                Some(GpuKind::OtherIntel) if diagnostics.has_vulkan => ResolvedBackend::LlamaVulkan,
                Some(GpuKind::OtherIntel) if diagnostics.has_sycl => ResolvedBackend::LlamaSycl,
                _ if diagnostics.has_vulkan => ResolvedBackend::LlamaVulkan,
                _ if diagnostics.has_sycl => ResolvedBackend::LlamaSycl,
                _ => anyhow::bail!("no usable SYCL or Vulkan backend was detected"),
            };
            // This guard is only for llama.cpp, which cannot stream this model's experts.
            if matches!(
                candidate,
                ResolvedBackend::LlamaSycl | ResolvedBackend::LlamaVulkan
            ) {
                let model = model::spec(config.model);
                if let Some(gpu) = target
                    && gpu.adapter_ram_bytes > 0
                    && model.resident_bytes > gpu.adapter_ram_bytes
                {
                    println!(
                        "Auto backend: {:.1} GiB model shard exceeds {:.1} GiB GPU memory; using CPU.",
                        model.resident_bytes as f64 / 1e9,
                        gpu.adapter_ram_bytes as f64 / 1e9
                    );
                    return Ok(ResolvedBackend::Cpu);
                }
            }
            Ok(candidate)
        }
    }
}

pub fn start(
    config: &AppConfig,
    diagnostics: &Diagnostics,
    backend: ResolvedBackend,
    artifact: &ModelArtifact,
) -> Result<RuntimeProcess> {
    match backend {
        ResolvedBackend::Cpu | ResolvedBackend::LlamaSycl | ResolvedBackend::LlamaVulkan => {
            start_llama(config, diagnostics, backend, artifact)
        }
        ResolvedBackend::Strata => start_strata(config, diagnostics, artifact),
    }
}

pub fn ensure_llama_bundle(config: &AppConfig, backend: ResolvedBackend) -> Result<PathBuf> {
    let (suffix, url, size, sha) = match backend {
        ResolvedBackend::Cpu | ResolvedBackend::LlamaVulkan => (
            "vulkan",
            LLAMA_VULKAN_URL,
            LLAMA_VULKAN_SIZE,
            LLAMA_VULKAN_SHA256,
        ),
        ResolvedBackend::LlamaSycl => ("sycl", LLAMA_SYCL_URL, LLAMA_SYCL_SIZE, LLAMA_SYCL_SHA256),
        ResolvedBackend::Strata => anyhow::bail!("Strata does not use a llama.cpp runtime bundle"),
    };
    let directory = config
        .runtime_dir
        .join(format!("llama-{LLAMA_TAG}-{suffix}"));
    let executable = find_named_file(&directory, "llama-server.exe");
    let ready = executable.is_some()
        && fs::read_to_string(directory.join("bundle.json"))
            .ok()
            .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
            .and_then(|value| {
                value
                    .get("sha256")
                    .and_then(|value| value.as_str())
                    .map(str::to_owned)
            })
            .is_some_and(|value| value.eq_ignore_ascii_case(sha));
    if ready {
        return Ok(executable.expect("ready implies executable"));
    }
    fs::create_dir_all(&directory)?;
    let archive = config
        .runtime_dir
        .join(format!("llama-{LLAMA_TAG}-{suffix}.zip"));
    let client = Client::builder().user_agent("WINSTRINT/0.1").build()?;
    model::download_verified(&client, url, &archive, size, Some(sha))?;
    extract_zip(&archive, &directory)?;
    let executable = find_named_file(&directory, "llama-server.exe")
        .context("llama.cpp archive did not contain llama-server.exe")?;
    fs::write(
        directory.join("bundle.json"),
        serde_json::to_vec_pretty(
            &json!({"tag": LLAMA_TAG, "url": url, "sha256": sha, "executable": executable}),
        )?,
    )?;
    Ok(executable)
}

fn start_llama(
    config: &AppConfig,
    diagnostics: &Diagnostics,
    backend: ResolvedBackend,
    artifact: &ModelArtifact,
) -> Result<RuntimeProcess> {
    let executable = ensure_llama_bundle(config, backend)?;
    let log_dir = config.data_dir.join("logs");
    fs::create_dir_all(&log_dir)?;
    let log_path = log_dir.join("llama-server.log");
    let launch = match backend {
        ResolvedBackend::Cpu => LlamaLaunch::Cpu,
        ResolvedBackend::LlamaSycl | ResolvedBackend::LlamaVulkan => LlamaLaunch::Gpu(backend),
        ResolvedBackend::Strata => unreachable!(),
    };

    match launch {
        LlamaLaunch::Cpu => start_llama_process(
            config,
            diagnostics,
            artifact,
            &executable,
            &log_path,
            launch,
        ),
        LlamaLaunch::Gpu(gpu_backend) => match start_llama_process(
            config,
            diagnostics,
            artifact,
            &executable,
            &log_path,
            launch,
        ) {
            Ok(process) => Ok(process),
            Err(error) => {
                eprintln!("{} could not load the model: {error}", gpu_backend.label());
                eprintln!(
                    "Falling back to CPU execution; startup can take several minutes for this model."
                );
                start_llama_process(
                    config,
                    diagnostics,
                    artifact,
                    &executable,
                    &log_path,
                    LlamaLaunch::Cpu,
                )
                .with_context(|| {
                    format!("starting CPU fallback after {} failed", gpu_backend.label())
                })
            }
        },
    }
}

fn start_llama_process(
    config: &AppConfig,
    diagnostics: &Diagnostics,
    artifact: &ModelArtifact,
    executable: &Path,
    log_path: &Path,
    launch: LlamaLaunch,
) -> Result<RuntimeProcess> {
    let log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)?;
    let mut command = Command::new(executable);
    command
        .current_dir(executable.parent().unwrap_or_else(|| Path::new(".")))
        .arg("--model")
        .arg(&artifact.first_shard)
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(config.port.to_string())
        .arg("--ctx-size")
        .arg(config.context.to_string())
        .arg("--parallel")
        .arg("1")
        .stdout(Stdio::from(log.try_clone()?))
        .stderr(Stdio::from(log));
    match launch {
        LlamaLaunch::Gpu(backend) => {
            command
                .arg("--n-gpu-layers")
                .arg("auto")
                .arg("--fit")
                .arg("on")
                .arg("--moe-cache-mib")
                .arg(recommended_cache_mib(diagnostics).to_string());
            match backend {
                ResolvedBackend::LlamaSycl => {
                    command
                        .arg("--device")
                        .arg("SYCL0")
                        .env("ONEAPI_DEVICE_SELECTOR", "level_zero:0")
                        .env("SYCL_CACHE_PERSISTENT", "0");
                }
                ResolvedBackend::LlamaVulkan => {
                    command.arg("--device").arg("Vulkan0");
                }
                ResolvedBackend::Cpu | ResolvedBackend::Strata => unreachable!(),
            }
        }
        LlamaLaunch::Cpu => {
            command
                .arg("--n-gpu-layers")
                .arg("0")
                .arg("--device")
                .arg("none")
                .arg("--no-kv-offload")
                .arg("--no-op-offload")
                .arg("--moe-cache-mib")
                .arg("0");
        }
    }
    println!("Starting {}", launch.label());
    println!("Model: {}", artifact.first_shard.display());
    println!("Runtime log: {}", log_path.display());
    let executable_label = executable.display().to_string();
    let (mut child, process_tree) = spawn_runtime(command, &executable_label)?;
    let endpoint = format!("http://127.0.0.1:{}", config.port);
    if let Err(error) = wait_for_health(&endpoint, &mut child, launch.health_timeout(), log_path) {
        terminate_child(child);
        drop(process_tree);
        return Err(error);
    }
    Ok(RuntimeProcess {
        child,
        _process_tree: process_tree,
        endpoint,
    })
}

fn start_strata(
    config: &AppConfig,
    diagnostics: &Diagnostics,
    artifact: &ModelArtifact,
) -> Result<RuntimeProcess> {
    let launch = strata::prepare_launch(config, diagnostics::target_gpu(diagnostics), artifact)?;
    let log_path = &launch.log_path;
    if let Some(parent) = log_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)
        .with_context(|| format!("opening Strata runtime log {}", log_path.display()))?;
    let log_copy = log
        .try_clone()
        .with_context(|| format!("duplicating Strata runtime log {}", log_path.display()))?;
    let mut command = Command::new(&launch.program);
    command
        .args(&launch.args)
        .current_dir(&launch.cwd)
        .stdout(Stdio::from(log_copy))
        .stderr(Stdio::from(log));
    println!("Starting Strata native SYCL");
    println!("Model: {}", artifact.first_shard.display());
    println!("Runtime log: {}", log_path.display());
    let executable_label = launch.program.display().to_string();
    let (mut child, process_tree) = spawn_runtime(command, &executable_label)?;
    let endpoint = format!("http://127.0.0.1:{}", config.port);
    if let Err(error) = wait_for_health(&endpoint, &mut child, launch.health_timeout, log_path) {
        terminate_child(child);
        drop(process_tree);
        return Err(error);
    }
    Ok(RuntimeProcess {
        child,
        _process_tree: process_tree,
        endpoint,
    })
}

impl Drop for RuntimeProcess {
    fn drop(&mut self) {
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

fn wait_for_health(
    endpoint: &str,
    child: &mut Child,
    timeout: Duration,
    log_path: &Path,
) -> Result<()> {
    let client = Client::builder().timeout(Duration::from_secs(2)).build()?;
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(status) = child.try_wait().context("checking runtime process")? {
            anyhow::bail!(
                "runtime pid {} exited with status {status}; inspect {}",
                child.id(),
                log_path.display()
            );
        }
        if Instant::now() > deadline {
            anyhow::bail!(
                "runtime pid {} did not become healthy within {} seconds; inspect {}",
                child.id(),
                timeout.as_secs(),
                log_path.display()
            );
        }
        if let Ok(response) = client.get(format!("{endpoint}/health")).send()
            && response.status().is_success()
        {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(500));
    }
}

fn strata_backend(
    config: &AppConfig,
    target: Option<&crate::diagnostics::GpuInfo>,
) -> Result<ResolvedBackend> {
    strata::validate_for_runtime(config, target)?;
    Ok(ResolvedBackend::Strata)
}

fn recommended_cache_mib(diagnostics: &Diagnostics) -> u64 {
    match diagnostics::target_gpu(diagnostics).map(|gpu| gpu.kind) {
        Some(GpuKind::A770) => 8192,
        Some(GpuKind::Arc140T) => 4096,
        _ => 4096,
    }
}

fn extract_zip(archive: &Path, destination: &Path) -> Result<()> {
    let file = File::open(archive).with_context(|| format!("opening {}", archive.display()))?;
    let mut zip = ZipArchive::new(file).context("reading llama.cpp zip")?;
    for index in 0..zip.len() {
        let mut entry = zip.by_index(index)?;
        let Some(relative) = entry.enclosed_name() else {
            anyhow::bail!("zip contains an unsafe path at index {index}");
        };
        if relative.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        }) {
            anyhow::bail!("zip contains an unsafe path {}", relative.display());
        }
        let output = destination.join(&relative);
        if entry.is_dir() {
            fs::create_dir_all(&output)?;
            continue;
        }
        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut target = File::create(&output)?;
        io::copy(&mut entry, &mut target)?;
    }
    Ok(())
}

fn find_named_file(root: &Path, name: &str) -> Option<PathBuf> {
    if !root.exists() {
        return None;
    }
    let mut stack = vec![root.to_path_buf()];
    while let Some(directory) = stack.pop() {
        let entries = fs::read_dir(directory).ok()?;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path
                .file_name()
                .is_some_and(|file| file.eq_ignore_ascii_case(name))
            {
                return Some(path);
            }
        }
    }
    None
}
