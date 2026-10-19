use std::fs::{self, File, OpenOptions};
use std::io;
use std::path::{Component, Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use reqwest::blocking::Client;
use serde::Serialize;
use serde_json::json;
use zip::ZipArchive;

use crate::config::{AppConfig, BackendPreference};
use crate::diagnostics::{self, Diagnostics, GpuKind};
use crate::model::{self, ModelArtifact};

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
    LlamaSycl,
    LlamaVulkan,
    Strata,
}
impl ResolvedBackend {
    pub fn label(self) -> &'static str {
        match self {
            Self::LlamaSycl => "llama.cpp SYCL",
            Self::LlamaVulkan => "llama.cpp Vulkan",
            Self::Strata => "Strata native SYCL",
        }
    }
}

pub struct RuntimeProcess {
    child: Child,
    pub endpoint: String,
}

#[derive(Serialize)]
struct StrataConfig {
    backend: &'static str,
    args: Vec<String>,
    cwd: String,
    env: serde_json::Value,
}

pub fn select_backend(config: &AppConfig, diagnostics: &Diagnostics) -> Result<ResolvedBackend> {
    let target = diagnostics::target_gpu(diagnostics);
    match config.backend {
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
        BackendPreference::Auto => match target.map(|gpu| gpu.kind) {
            Some(GpuKind::A770) if diagnostics.has_sycl => Ok(ResolvedBackend::LlamaSycl),
            Some(GpuKind::A770) if diagnostics.has_vulkan => Ok(ResolvedBackend::LlamaVulkan),
            Some(GpuKind::Arc140T) if diagnostics.has_vulkan => Ok(ResolvedBackend::LlamaVulkan),
            Some(GpuKind::Arc140T) if diagnostics.has_sycl => Ok(ResolvedBackend::LlamaSycl),
            Some(GpuKind::OtherIntel) if diagnostics.has_vulkan => Ok(ResolvedBackend::LlamaVulkan),
            Some(GpuKind::OtherIntel) if diagnostics.has_sycl => Ok(ResolvedBackend::LlamaSycl),
            _ if diagnostics.has_vulkan => Ok(ResolvedBackend::LlamaVulkan),
            _ if diagnostics.has_sycl => Ok(ResolvedBackend::LlamaSycl),
            _ => anyhow::bail!("no usable SYCL or Vulkan backend was detected"),
        },
    }
}

pub fn start(
    config: &AppConfig,
    diagnostics: &Diagnostics,
    backend: ResolvedBackend,
    artifact: &ModelArtifact,
) -> Result<RuntimeProcess> {
    match backend {
        ResolvedBackend::LlamaSycl | ResolvedBackend::LlamaVulkan => {
            start_llama(config, diagnostics, backend, artifact)
        }
        ResolvedBackend::Strata => start_strata(config, diagnostics, artifact),
    }
}

pub fn ensure_llama_bundle(config: &AppConfig, backend: ResolvedBackend) -> Result<PathBuf> {
    let (suffix, url, size, sha) = match backend {
        ResolvedBackend::LlamaSycl => ("sycl", LLAMA_SYCL_URL, LLAMA_SYCL_SIZE, LLAMA_SYCL_SHA256),
        ResolvedBackend::LlamaVulkan => (
            "vulkan",
            LLAMA_VULKAN_URL,
            LLAMA_VULKAN_SIZE,
            LLAMA_VULKAN_SHA256,
        ),
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
    let log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)?;
    let mut command = Command::new(&executable);
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
        .arg("--n-gpu-layers")
        .arg("999")
        .arg("--parallel")
        .arg("1")
        .arg("--moe-cache-mib")
        .arg(recommended_cache_mib(diagnostics).to_string())
        .stdout(Stdio::from(log.try_clone()?))
        .stderr(Stdio::from(log));
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
        ResolvedBackend::Strata => unreachable!(),
    }
    println!("Starting {}", backend.label());
    println!("Model: {}", artifact.first_shard.display());
    println!("Runtime log: {}", log_path.display());
    let child = command
        .spawn()
        .with_context(|| format!("starting {}", executable.display()))?;
    let endpoint = format!("http://127.0.0.1:{}", config.port);
    wait_for_health(&endpoint, child.id(), &log_path)?;
    Ok(RuntimeProcess { child, endpoint })
}

fn start_strata(
    config: &AppConfig,
    diagnostics: &Diagnostics,
    artifact: &ModelArtifact,
) -> Result<RuntimeProcess> {
    if diagnostics
        .gpus
        .iter()
        .any(|gpu| matches!(gpu.kind, GpuKind::A770 | GpuKind::Arc140T))
    {
        anyhow::bail!(
            "Strata's current Windows SYCL branch is tested for Arc Pro B70, not A770/Arc 140T; use llama.cpp SYCL/Vulkan"
        );
    }
    let root = config.data_dir.join("strata");
    let script = root.join("sycl").join("serve").join("server_intel.py");
    let engine = root.join("engine").join("strata.exe");
    if !script.exists() || !engine.exists() {
        anyhow::bail!(
            "Strata is not provisioned; expected {} and {}",
            script.display(),
            engine.display()
        );
    }
    let python = std::env::var_os("WINSTRINT_PYTHON")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("python"));
    if !diagnostics::command_exists("python") && std::env::var_os("WINSTRINT_PYTHON").is_none() {
        anyhow::bail!("Python is required for the Strata server wrapper; set WINSTRINT_PYTHON");
    }
    let config_path = root.join("winstrint-strata.json");
    let strata_config = StrataConfig {
        backend: "sycl",
        args: vec![
            "--native".into(),
            artifact.first_shard.to_string_lossy().into_owned(),
            "--max-context".into(),
            config.context.to_string(),
            "--stream-experts".into(),
        ],
        cwd: root.to_string_lossy().into_owned(),
        env: json!({"SYCL_UR_USE_LEVEL_ZERO_V2":"1","SYCL_CACHE_PERSISTENT":"0","UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS":"1","ZES_ENABLE_SYSMAN":"1","STRATA_VERIFY_DEVICE_PLAN":"1","STRATA_VERIFY_NO_HOST":"1","STRATA_SYCL_HEARTBEAT_S":"10"}),
    };
    fs::write(&config_path, serde_json::to_vec_pretty(&strata_config)?)?;
    let log_path = config.data_dir.join("logs").join("strata-server.log");
    fs::create_dir_all(log_path.parent().expect("log path has parent"))?;
    let log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)?;
    let child = Command::new(python)
        .current_dir(&root)
        .arg(&script)
        .arg("--engine")
        .arg("strata")
        .arg("--config")
        .arg(&config_path)
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(config.port.to_string())
        .stdout(Stdio::from(log.try_clone()?))
        .stderr(Stdio::from(log))
        .spawn()
        .context("starting Strata server wrapper")?;
    let endpoint = format!("http://127.0.0.1:{}", config.port);
    wait_for_health(&endpoint, child.id(), &log_path)?;
    Ok(RuntimeProcess { child, endpoint })
}

impl Drop for RuntimeProcess {
    fn drop(&mut self) {
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

fn wait_for_health(endpoint: &str, pid: u32, log_path: &Path) -> Result<()> {
    let client = Client::builder().timeout(Duration::from_secs(2)).build()?;
    let deadline = Instant::now() + Duration::from_secs(180);
    loop {
        if Instant::now() > deadline {
            anyhow::bail!(
                "runtime pid {} did not become healthy within 180 seconds; inspect {}",
                pid,
                log_path.display()
            );
        }
        if let Ok(response) = client.get(format!("{endpoint}/health")).send() {
            if response.status().is_success() {
                return Ok(());
            }
        }
        thread::sleep(Duration::from_millis(500));
    }
}

fn strata_backend(
    config: &AppConfig,
    target: Option<&crate::diagnostics::GpuInfo>,
) -> Result<ResolvedBackend> {
    if target.is_some_and(|gpu| matches!(gpu.kind, GpuKind::A770 | GpuKind::Arc140T)) {
        anyhow::bail!(
            "Strata Windows support is not validated for A770/Arc 140T; use --backend sycl or --backend vulkan"
        );
    }
    let engine = config
        .data_dir
        .join("strata")
        .join("engine")
        .join("strata.exe");
    if !engine.exists() {
        anyhow::bail!(
            "Strata engine is not provisioned at {}; use llama.cpp or provision Strata explicitly",
            engine.display()
        );
    }
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
        let Some(relative) = entry.enclosed_name().map(PathBuf::from) else {
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
