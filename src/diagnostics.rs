use std::collections::BTreeSet;
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
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
    let os_version = command_text("cmd.exe", &["/C", "ver"]).unwrap_or_else(|_| "unknown".into());
    let gpus = video_controllers();
    let vulkaninfo_available = command_exists("vulkaninfo");
    let sycl_ls_available = command_exists("sycl-ls");
    let has_vulkan =
        vulkaninfo_available || Path::new(r"C:\Windows\System32\vulkan-1.dll").exists();
    let has_sycl = sycl_ls_available
        || env::var_os("ONEAPI_ROOT").is_some_and(|path| Path::new(&path).exists())
        || env::var_os("SETVARS_COMPLETED").is_some();
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

pub fn print_report(diagnostics: &Diagnostics) {
    println!("OS: {}", diagnostics.os_version.trim());
    if diagnostics.gpus.is_empty() {
        println!("GPU: no display adapter returned by WMI");
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

pub fn command_exists(program: &str) -> bool {
    let command = if cfg!(windows) { "where.exe" } else { "which" };
    Command::new(command)
        .arg(program)
        .output()
        .is_ok_and(|output| output.status.success())
}

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

fn json_array(value: Value) -> Vec<Value> {
    match value {
        Value::Array(values) => values,
        Value::Object(_) => vec![value],
        _ => Vec::new(),
    }
}

fn number_value(value: Option<&Value>) -> u64 {
    match value {
        Some(Value::Number(number)) => number
            .as_u64()
            .unwrap_or_else(|| number.as_f64().unwrap_or(0.0) as u64),
        Some(Value::String(text)) => text.parse().unwrap_or_default(),
        _ => 0,
    }
}
