use std::collections::BTreeSet;
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};
#[cfg(windows)]
use serde::Deserialize;
use serde::Serialize;
#[cfg(windows)]
use serde_json::Value;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum GpuKind {
    A770,
    Arc140T,
    OtherIntel,
    Unknown,
}

#[derive(Clone, Debug, Serialize)]
pub struct GpuInfo {
    pub name: String,
    pub pnp_device_id: String,
    pub driver_version: String,
    pub adapter_ram_bytes: u64,
    pub kind: GpuKind,
}

#[derive(Clone, Debug, Serialize)]
pub struct Activity {
    pub utilization_percent: Option<f64>,
    pub known_processes: Vec<String>,
    pub observable: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct Diagnostics {
    pub os_version: String,
    pub gpus: Vec<GpuInfo>,
    pub has_vulkan: bool,
    pub has_sycl: bool,
    pub vulkaninfo_available: bool,
    pub sycl_ls_available: bool,
    pub activity: Activity,
}

#[cfg(windows)]
#[derive(Deserialize)]
struct VideoController {
    #[serde(rename = "Name")]
    name: Option<String>,
    #[serde(rename = "PNPDeviceID")]
    pnp_device_id: Option<String>,
    #[serde(rename = "DriverVersion")]
    driver_version: Option<String>,
    #[serde(rename = "AdapterRAM")]
    adapter_ram: Option<Value>,
}

pub fn collect() -> Result<Diagnostics> {
    let os_version = os_version();
    let gpus = video_controllers();
    let vulkaninfo_available = command_exists("vulkaninfo");
    let sycl_ls_available = command_exists("sycl-ls");
    let has_vulkan = vulkaninfo_available || env::var_os("VK_ICD_FILENAMES").is_some();
    let has_sycl = sycl_ls_available
        || env::var_os("ONEAPI_ROOT").is_some_and(|path| Path::new(&path).exists())
        || env::var_os("SETVARS_COMPLETED").is_some()
        || oneapi_setvars().is_some();
    Ok(Diagnostics {
        os_version,
        gpus,
        has_vulkan,
        has_sycl,
        vulkaninfo_available,
        sycl_ls_available,
        activity: gpu_activity(),
    })
}

fn os_version() -> String {
    if cfg!(windows) {
        command_text("cmd.exe", &["/C", "ver"]).unwrap_or_else(|_| "unknown".into())
    } else {
        command_text("uname", &["-sr"]).unwrap_or_else(|_| "unknown".into())
    }
}

pub(crate) fn oneapi_setvars() -> Option<PathBuf> {
    let mut candidates = Vec::new();
    let script = if cfg!(windows) {
        "setvars.bat"
    } else {
        "setvars.sh"
    };
    if let Some(root) = env::var_os("ONEAPI_ROOT") {
        candidates.push(PathBuf::from(root).join(script));
    }
    #[cfg(windows)]
    for variable in ["ProgramFiles(x86)", "ProgramFiles"] {
        if let Some(root) = env::var_os(variable) {
            candidates.push(
                PathBuf::from(root)
                    .join("Intel")
                    .join("oneAPI")
                    .join(script),
            );
        }
    }
    #[cfg(windows)]
    if let Some(root) = env::var_os("USERPROFILE") {
        candidates.push(
            PathBuf::from(root)
                .join("Intel")
                .join("oneAPI")
                .join(script),
        );
    }
    #[cfg(not(windows))]
    {
        candidates.push(PathBuf::from("/opt/intel/oneapi").join(script));
        if let Some(root) = env::var_os("HOME") {
            candidates.push(
                PathBuf::from(root)
                    .join("intel")
                    .join("oneapi")
                    .join(script),
            );
        }
    }
    candidates.into_iter().find(|path| path.is_file())
}

#[cfg(not(windows))]
fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).into_owned())
}
pub fn command_exists(program: &str) -> bool {
    let path = Path::new(program);
    if path.is_absolute() || path.components().count() > 1 {
        return path.is_file();
    }
    let Some(search_path) = env::var_os("PATH") else {
        return false;
    };
    for directory in env::split_paths(&search_path) {
        if directory.join(program).is_file() {
            return true;
        }
        #[cfg(windows)]
        if let Some(extensions) = env::var_os("PATHEXT") {
            for extension in extensions.to_string_lossy().split(';') {
                if directory.join(format!("{program}{extension}")).is_file() {
                    return true;
                }
            }
        }
    }
    false
}

#[cfg(windows)]
fn video_controllers() -> Vec<GpuInfo> {
    let script = r#"$ErrorActionPreference='SilentlyContinue'; Get-CimInstance Win32_VideoController | Select-Object Name,PNPDeviceID,DriverVersion,AdapterRAM | ConvertTo-Json -Compress"#;
    let Some(value) = powershell_json(script) else {
        return Vec::new();
    };
    json_array(value)
        .into_iter()
        .filter_map(|item| serde_json::from_value::<VideoController>(item).ok())
        .filter_map(|controller| {
            let name = controller.name?.trim().to_string();
            if name.is_empty() {
                return None;
            }
            let pnp = controller.pnp_device_id.unwrap_or_default();
            let kind = classify_gpu(&name, &pnp);
            Some(GpuInfo {
                name: name.clone(),
                pnp_device_id: pnp.clone(),
                driver_version: controller
                    .driver_version
                    .unwrap_or_else(|| "unknown".into()),
                adapter_ram_bytes: normalized_memory(
                    kind,
                    number_value(controller.adapter_ram.as_ref()),
                ),
                kind,
            })
        })
        .collect()
}

#[cfg(not(windows))]
fn video_controllers() -> Vec<GpuInfo> {
    let mut gpus = command_output("vulkaninfo", &["--summary"])
        .map(|text| parse_vulkan_devices(&text))
        .unwrap_or_default();
    if gpus.is_empty() {
        if let Some(text) = command_output("sycl-ls", &[]) {
            for line in text.lines().map(str::trim).filter(|line| !line.is_empty()) {
                let lower = line.to_ascii_lowercase();
                if lower.contains("intel") && (lower.contains("gpu") || lower.contains("arc")) {
                    append_gpu(&mut gpus, line, line, "unknown", 0);
                }
            }
        }
    }
    gpus
}

#[cfg(not(windows))]
fn parse_vulkan_devices(text: &str) -> Vec<GpuInfo> {
    let mut gpus = Vec::new();
    let mut name: Option<String> = None;
    let mut device_id = String::new();
    let mut driver = String::from("unknown");
    let mut memory = 0;
    for line in text.lines() {
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim();
        let value = value.trim();
        if key == "deviceName" {
            if let Some(previous) = name.take() {
                let pnp = if device_id.is_empty() {
                    String::new()
                } else {
                    format!("VEN_8086&DEV_{device_id}")
                };
                append_gpu(&mut gpus, &previous, &pnp, &driver, memory);
            }
            name = Some(value.to_string());
            device_id.clear();
            driver.clear();
            driver.push_str("unknown");
            memory = 0;
        } else if name.is_some() {
            match key {
                "deviceID" => {
                    device_id = value.trim_start_matches("0x").to_ascii_uppercase();
                }
                "driverName" | "driverInfo" => {
                    if driver == "unknown" {
                        driver = value.to_string();
                    }
                }
                "deviceLocalMemory" => {
                    memory = value.parse().unwrap_or_default();
                }
                _ => {}
            }
        }
    }
    if let Some(last) = name {
        let pnp = if device_id.is_empty() {
            String::new()
        } else {
            format!("VEN_8086&DEV_{device_id}")
        };
        append_gpu(&mut gpus, &last, &pnp, &driver, memory);
    }
    gpus
}

#[cfg(not(windows))]
fn append_gpu(
    gpus: &mut Vec<GpuInfo>,
    name: &str,
    pnp_device_id: &str,
    driver_version: &str,
    adapter_ram_bytes: u64,
) {
    let name = name.trim();
    if name.is_empty() || gpus.iter().any(|gpu| gpu.name.eq_ignore_ascii_case(name)) {
        return;
    }
    let kind = classify_gpu(name, pnp_device_id);
    gpus.push(GpuInfo {
        name: name.to_string(),
        pnp_device_id: pnp_device_id.to_string(),
        driver_version: driver_version.trim().to_string(),
        adapter_ram_bytes: normalized_memory(kind, adapter_ram_bytes),
        kind,
    });
}

#[cfg(windows)]
fn gpu_activity() -> Activity {
    let utilization_script = r#"$ErrorActionPreference='SilentlyContinue'; try { Get-Counter '\GPU Engine(*)\Utilization Percentage' -ErrorAction Stop | Select-Object -ExpandProperty CounterSamples | ForEach-Object { [PSCustomObject]@{Value=$_.CookedValue} } | ConvertTo-Json -Compress } catch { '{}' }"#;
    let utilization_percent = powershell_json(utilization_script).and_then(|value| {
        json_array(value)
            .into_iter()
            .filter_map(|item| item.get("Value").and_then(Value::as_f64))
            .filter(|value| value.is_finite())
            .reduce(f64::max)
            .map(|value| value.clamp(0.0, 100.0))
    });
    let process_script = r#"$ErrorActionPreference='SilentlyContinue'; Get-Process | Where-Object { $_.ProcessName -match '^(llama-server|llama|strata|ollama|lmstudio|comfy|qwen)' } | Select-Object ProcessName,Id | ConvertTo-Json -Compress"#;
    let known_processes = powershell_json(process_script)
        .map(json_array)
        .unwrap_or_default()
        .into_iter()
        .filter_map(|item| {
            let name = item.get("ProcessName")?.as_str()?.to_string();
            let id = item.get("Id").and_then(Value::as_u64).unwrap_or_default();
            Some(format!("{name} ({id})"))
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    Activity {
        utilization_percent,
        observable: utilization_percent.is_some(),
        known_processes,
    }
}

#[cfg(not(windows))]
fn gpu_activity() -> Activity {
    let known_processes = command_output(
        "pgrep",
        &[
            "-af",
            "llama-server|llama|strata|ollama|lmstudio|comfy|qwen",
        ],
    )
    .map(|text| {
        text.lines()
            .filter_map(|line| {
                let mut fields = line.splitn(2, char::is_whitespace);
                let pid = fields.next()?.trim();
                let command = fields.next()?.trim();
                let executable = command.split_whitespace().next()?;
                let name = Path::new(executable).file_name()?.to_string_lossy();
                Some(format!("{name} ({pid})"))
            })
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    })
    .unwrap_or_default();
    Activity {
        utilization_percent: None,
        observable: false,
        known_processes,
    }
}

pub fn print_report(diagnostics: &Diagnostics) {
    println!("OS: {}", diagnostics.os_version.trim());
    if diagnostics.gpus.is_empty() {
        if cfg!(windows) {
            println!("GPU: no display adapter returned by WMI");
        } else {
            println!("GPU: no Intel adapter found in Vulkan/SYCL enumeration");
        }
    } else {
        for (index, gpu) in diagnostics.gpus.iter().enumerate() {
            let memory = if gpu.adapter_ram_bytes == 0 {
                "unknown".to_string()
            } else {
                format!("{:.1} GiB", gpu.adapter_ram_bytes as f64 / 1_073_741_824.0)
            };
            println!(
                "GPU {index}: {} | {:?} | {} | driver {}",
                gpu.name, gpu.kind, memory, gpu.driver_version
            );
            if !gpu.pnp_device_id.is_empty() {
                println!("  PNP: {}", gpu.pnp_device_id);
            }
        }
    }
    println!(
        "Backends: Vulkan={} (vulkaninfo={}), SYCL={} (sycl-ls={})",
        diagnostics.has_vulkan,
        diagnostics.vulkaninfo_available,
        diagnostics.has_sycl,
        diagnostics.sycl_ls_available
    );
    if cfg!(windows) {
        println!(
            "Strata: managed native Windows SYCL port requires Intel oneAPI DPC++/C++, Visual Studio Build Tools, and Level Zero; run `winstrint strata-status --verify`."
        );
    } else if diagnostics.has_sycl {
        println!(
            "Strata: native integration is Windows-only; llama.cpp SYCL prerequisites detected."
        );
    } else if diagnostics.has_vulkan {
        println!("Strata: native integration is Windows-only; Vulkan is available.");
    } else {
        println!("Strata: native integration is Windows-only; Intel Arc backends are unavailable.");
    }
    match diagnostics.activity.utilization_percent {
        Some(value) => println!("GPU activity: {:.1}%", value),
        None => println!("GPU activity: unavailable (no reliable counter)"),
    }
    if diagnostics.activity.known_processes.is_empty() {
        println!("Known GPU processes: none");
    } else {
        println!(
            "Known GPU processes: {}",
            diagnostics.activity.known_processes.join(", ")
        );
    }
}

pub fn require_idle(diagnostics: &Diagnostics, force: bool, benchmark: bool) -> Result<()> {
    if force {
        eprintln!("GPU safety override active (--force); queue ownership is still required.");
        return Ok(());
    }
    if !diagnostics.activity.known_processes.is_empty() {
        anyhow::bail!(
            "GPU safety gate: known GPU process active: {}",
            diagnostics.activity.known_processes.join(", ")
        );
    }
    if diagnostics
        .activity
        .utilization_percent
        .is_some_and(|value| value > 5.0)
    {
        anyhow::bail!("GPU safety gate: current utilization is above 5%");
    }
    if diagnostics.activity.utilization_percent.is_none() {
        if benchmark {
            anyhow::bail!(
                "GPU safety gate: utilization is unavailable; rerun with --force only after checking the GPU manually"
            );
        }
        eprintln!(
            "Warning: GPU utilization is unavailable; no known GPU process was found, so startup continues."
        );
    }
    Ok(())
}

pub fn target_gpu(diagnostics: &Diagnostics) -> Option<&GpuInfo> {
    diagnostics
        .gpus
        .iter()
        .find(|gpu| matches!(gpu.kind, GpuKind::A770 | GpuKind::Arc140T))
        .or_else(|| {
            diagnostics
                .gpus
                .iter()
                .find(|gpu| gpu.kind != GpuKind::Unknown)
        })
}

fn classify_gpu(name: &str, pnp: &str) -> GpuKind {
    let upper_name = name.to_ascii_uppercase();
    let upper_pnp = pnp.to_ascii_uppercase();
    if upper_pnp.contains("VEN_8086&DEV_56A0") || upper_name.contains("A770") {
        GpuKind::A770
    } else if upper_pnp.contains("VEN_8086&DEV_7D51") || upper_name.contains("140T") {
        GpuKind::Arc140T
    } else if upper_name.contains("INTEL") || upper_pnp.contains("VEN_8086") {
        GpuKind::OtherIntel
    } else {
        GpuKind::Unknown
    }
}

fn normalized_memory(kind: GpuKind, reported_bytes: u64) -> u64 {
    if matches!(kind, GpuKind::A770 | GpuKind::Arc140T) && reported_bytes < 8 * 1_073_741_824 {
        16 * 1_073_741_824
    } else {
        reported_bytes
    }
}

#[cfg(windows)]
fn powershell_json(script: &str) -> Option<Value> {
    let output = Command::new(powershell_program())
        .args([
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            script,
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    serde_json::from_str(stdout.trim()).ok()
}

#[cfg(windows)]
fn powershell_program() -> PathBuf {
    if cfg!(windows) {
        let windir = env::var_os("WINDIR").unwrap_or_else(|| "C:\\Windows".into());
        PathBuf::from(windir)
            .join("System32")
            .join("WindowsPowerShell")
            .join("v1.0")
            .join("powershell.exe")
    } else {
        PathBuf::from("powershell")
    }
}

fn command_text(program: &str, args: &[&str]) -> Result<String> {
    let output = Command::new(program)
        .args(args)
        .output()
        .with_context(|| format!("running {program}"))?;
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

#[cfg(windows)]
fn json_array(value: Value) -> Vec<Value> {
    match value {
        Value::Array(values) => values,
        Value::Object(_) => vec![value],
        _ => Vec::new(),
    }
}

#[cfg(windows)]
fn number_value(value: Option<&Value>) -> u64 {
    match value {
        Some(Value::Number(number)) => number
            .as_u64()
            .unwrap_or_else(|| number.as_f64().unwrap_or(0.0) as u64),
        Some(Value::String(text)) => text.parse().unwrap_or_default(),
        _ => 0,
    }
}
