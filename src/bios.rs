use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};
use serde::Serialize;

use crate::config::{AppConfig, BiosDowngradeArgs};
use crate::model;

const MODEL_NAME: &str = "21TL";
const BOARD_NAME: &str = "LNVNB161216";
const TARGET_VERSION: &str = "R5CN53WW";
const TARGET_PLATFORM: &str = "JLVI6";
const PACKAGE_URL: &str = "https://download.lenovo.com/consumer/mobiles/r5cn53ww.exe";
const PACKAGE_SIZE: u64 = 9_923_520;
const PACKAGE_SHA256: &str = "86c8b925188f0f95900fa170bc73aed050e5eb80869810bb8cf943dfbbfa9493";

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct BiosVersion {
    number: u16,
}

#[derive(Debug)]
struct Validation {
    package: PathBuf,
    package_size: u64,
    package_sha256: String,
    current_model: String,
    current_board: String,
    current_bios: String,
    target_bios: String,
}

#[derive(Debug, Serialize)]
struct ValidationReport {
    package: String,
    package_size: u64,
    package_sha256: String,
    source_url: &'static str,
    current_model: String,
    current_board: String,
    current_bios: String,
    target_bios: String,
    target_platform: &'static str,
    downgrade_required: bool,
    ready_to_stage: bool,
    staged_package: Option<String>,
    notes: Vec<&'static str>,
}

pub fn run(config: &AppConfig, args: &BiosDowngradeArgs) -> Result<()> {
    let validation = validate_package(&args.package)?;
    let mut staged_package = None;

    if args.execute && !config.yes {
        anyhow::bail!(
            "--execute requires the global --yes flag; validation is read-only without it"
        );
    }
    if args.execute && !args.confirm_downgrade {
        anyhow::bail!("--execute requires --confirm-downgrade");
    }
    if args.execute && !args.confirm_bitlocker_recovery {
        anyhow::bail!("--execute requires --confirm-bitlocker-recovery");
    }
    if args.execute && !args.confirm_ac_power {
        anyhow::bail!("--execute requires --confirm-ac-power");
    }
    if args.execute && !args.confirm_recovery_path {
        anyhow::bail!("--execute requires --confirm-recovery-path");
    }

    if args.execute {
        platform::require_ac_power()?;
    }

    if args.prepare || args.execute {
        let workspace = args
            .workspace
            .clone()
            .unwrap_or_else(|| config.data_dir.join("bios-downgrade"));
        let destination = stage_package(&validation, &workspace)?;
        staged_package = Some(destination.display().to_string());
    }

    let report = ValidationReport {
        package: validation.package.display().to_string(),
        package_size: validation.package_size,
        package_sha256: validation.package_sha256.clone(),
        source_url: PACKAGE_URL,
        current_model: validation.current_model.clone(),
        current_board: validation.current_board.clone(),
        current_bios: validation.current_bios.clone(),
        target_bios: validation.target_bios.clone(),
        target_platform: TARGET_PLATFORM,
        downgrade_required: true,
        ready_to_stage: true,
        staged_package: staged_package.clone(),
        notes: vec![
            "This target is the latest pre-R5CN54 Lenovo package verified here with PCIE Resizable BAR Support and Above 4GB MMIO BIOS assignment in SetupUtility.",
            "The package and firmware own version checks remain authoritative; this command does not bypass anti-rollback protections.",
            "A BIOS downgrade can reset settings and can require BitLocker recovery after reboot.",
            "Validation and staging never write BIOS, NVRAM, or PCI configuration.",
        ],
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_report(&report);
    }

    if args.execute {
        let executable = staged_package
            .map(PathBuf::from)
            .unwrap_or_else(|| validation.package.clone());
        println!();
        println!(
            "Launching the verified Lenovo BIOS updater: {}",
            executable.display()
        );
        println!("The updater may flash firmware and reboot the machine; do not interrupt it.");
        let child = Command::new(&executable)
            .current_dir(executable.parent().unwrap_or_else(|| Path::new(".")))
            .spawn()
            .with_context(|| format!("launching Lenovo BIOS updater {}", executable.display()))?;
        println!("Lenovo updater launched with process id {}.", child.id());
    }

    Ok(())
}

pub(crate) fn current_identity() -> Result<(String, String, String)> {
    let identity = platform::read_identity()?;
    Ok((identity.model, identity.board, identity.bios))
}

fn validate_package(path: &Path) -> Result<Validation> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("checking BIOS package {}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        anyhow::bail!(
            "BIOS package {} is not a normal file; refusing to use it",
            path.display()
        );
    }
    if metadata.len() != PACKAGE_SIZE {
        anyhow::bail!(
            "BIOS package {} is {} bytes; expected the pinned R5CN53WW package size {}",
            path.display(),
            metadata.len(),
            PACKAGE_SIZE
        );
    }

    let sha256 = model::sha256_file(path)?;
    if !sha256.eq_ignore_ascii_case(PACKAGE_SHA256) {
        anyhow::bail!(
            "BIOS package SHA-256 mismatch: got {}, expected {}",
            sha256,
            PACKAGE_SHA256
        );
    }

    let bytes =
        fs::read(path).with_context(|| format!("reading BIOS package {}", path.display()))?;
    if !bytes.starts_with(b"MZ") {
        anyhow::bail!(
            "BIOS package {} is not a Windows PE executable",
            path.display()
        );
    }

    let identity = platform::read_identity()?;
    let current = parse_version(&identity.bios).with_context(|| {
        format!(
            "current BIOS version {:?} is not a supported R5CN##WW value",
            identity.bios
        )
    })?;
    let target = parse_version(TARGET_VERSION).expect("pinned target BIOS version is valid");
    if identity.model != MODEL_NAME {
        anyhow::bail!(
            "refusing BIOS package: detected system model {:?}, expected Lenovo ThinkBook 16 G7+ IAH model {}",
            identity.model,
            MODEL_NAME
        );
    }
    if identity.board != BOARD_NAME {
        anyhow::bail!(
            "refusing BIOS package: detected baseboard {:?}, expected {}",
            identity.board,
            BOARD_NAME
        );
    }
    if current <= target {
        anyhow::bail!(
            "current BIOS {} is not newer than downgrade target {}; refusing a non-downgrade",
            identity.bios,
            TARGET_VERSION
        );
    }

    Ok(Validation {
        package: path.to_path_buf(),
        package_size: metadata.len(),
        package_sha256: sha256,
        current_model: identity.model,
        current_board: identity.board,
        current_bios: identity.bios,
        target_bios: TARGET_VERSION.to_owned(),
    })
}

fn stage_package(validation: &Validation, workspace: &Path) -> Result<PathBuf> {
    fs::create_dir_all(workspace)
        .with_context(|| format!("creating BIOS staging directory {}", workspace.display()))?;
    let destination = workspace.join("R5CN53WW.exe");
    let same_file =
        fs::canonicalize(&validation.package).ok() == fs::canonicalize(&destination).ok();
    if !same_file {
        fs::copy(&validation.package, &destination).with_context(|| {
            format!("copying verified BIOS package to {}", destination.display())
        })?;
    }
    let manifest = serde_json::json!({
        "model": MODEL_NAME,
        "baseboard": BOARD_NAME,
        "target_bios": TARGET_VERSION,
        "platform_id": TARGET_PLATFORM,
        "source_url": PACKAGE_URL,
        "size": validation.package_size,
        "sha256": validation.package_sha256,
        "package": destination.file_name().and_then(|name| name.to_str()).unwrap_or("R5CN53WW.exe"),
        "automatic_flash": false,
    });
    fs::write(
        workspace.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )
    .with_context(|| format!("writing BIOS staging manifest in {}", workspace.display()))?;
    Ok(destination)
}

fn print_report(report: &ValidationReport) {
    println!("Lenovo BIOS downgrade validation");
    println!("Model: {}", report.current_model);
    println!("Baseboard: {}", report.current_board);
    println!("Current BIOS: {}", report.current_bios);
    println!("Target BIOS: {}", report.target_bios);
    println!("Target platform ID: {}", report.target_platform);
    println!("Package: {}", report.package);
    println!("SHA-256: {}", report.package_sha256);
    if let Some(path) = &report.staged_package {
        println!("Staged package: {path}");
    }
    println!();
    for note in &report.notes {
        println!("Note: {note}");
    }
    println!();
    println!("Read-only validation passed; no firmware was flashed.");
}

fn parse_version(value: &str) -> Option<BiosVersion> {
    let value = value.trim();
    let number = value.strip_prefix("R5CN")?.strip_suffix("WW")?;
    if number.len() != 2 || !number.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    Some(BiosVersion {
        number: number.parse().ok()?,
    })
}

#[cfg(windows)]
mod platform {
    use super::Result;
    use anyhow::{Context, anyhow};
    use std::ptr::null_mut;
    use windows_sys::Win32::Foundation::ERROR_SUCCESS;
    use windows_sys::Win32::System::Power::{GetSystemPowerStatus, SYSTEM_POWER_STATUS};

    use windows_sys::Win32::System::Registry::{
        HKEY_LOCAL_MACHINE, KEY_READ, RegCloseKey, RegOpenKeyExW, RegQueryValueExW,
    };

    const BIOS_KEY: &str = r"HARDWARE\DESCRIPTION\System\BIOS";

    pub(super) struct Identity {
        pub model: String,
        pub board: String,
        pub bios: String,
    }

    pub(super) fn read_identity() -> Result<Identity> {
        let model = read_value("SystemProductName")?
            .trim_matches(char::from(0))
            .trim()
            .to_owned();
        let board = read_value("BaseBoardProduct")?
            .trim_matches(char::from(0))
            .trim()
            .to_owned();
        let bios = read_value("BIOSVersion")?;
        let bios = bios
            .split(|character: char| !character.is_ascii_alphanumeric())
            .find(|token| super::parse_version(token).is_some())
            .ok_or_else(|| anyhow!("registry BIOSVersion does not contain an R5CN##WW value"))?
            .to_owned();
        Ok(Identity { model, board, bios })
    }

    pub(super) fn require_ac_power() -> Result<()> {
        unsafe {
            let mut status: SYSTEM_POWER_STATUS = std::mem::zeroed();
            if GetSystemPowerStatus(&mut status) == 0 {
                return Err(std::io::Error::last_os_error())
                    .context("checking Windows AC power status");
            }
            if status.ACLineStatus != 1 {
                return Err(anyhow!(
                    "AC power is not reported as connected; connect the Lenovo AC adapter before --execute"
                ));
            }
        }
        Ok(())
    }

    fn read_value(name: &str) -> Result<String> {
        let key_path = wide(BIOS_KEY);
        let value_name = wide(name);
        unsafe {
            let mut key = std::ptr::null_mut();
            let status =
                RegOpenKeyExW(HKEY_LOCAL_MACHINE, key_path.as_ptr(), 0, KEY_READ, &mut key);
            if status != ERROR_SUCCESS {
                return Err(std::io::Error::from_raw_os_error(status as i32))
                    .with_context(|| format!("opening registry key {BIOS_KEY}"));
            }

            let mut value_type = 0u32;
            let mut byte_count = 0u32;
            let status = RegQueryValueExW(
                key,
                value_name.as_ptr(),
                null_mut(),
                &mut value_type,
                std::ptr::null_mut(),
                &mut byte_count,
            );
            if status != ERROR_SUCCESS {
                RegCloseKey(key);
                return Err(std::io::Error::from_raw_os_error(status as i32))
                    .with_context(|| format!("querying registry value {name}"));
            }
            let mut data = vec![0u8; byte_count as usize];
            let status = RegQueryValueExW(
                key,
                value_name.as_ptr(),
                null_mut(),
                &mut value_type,
                data.as_mut_ptr(),
                &mut byte_count,
            );
            RegCloseKey(key);
            if status != ERROR_SUCCESS {
                return Err(std::io::Error::from_raw_os_error(status as i32))
                    .with_context(|| format!("reading registry value {name}"));
            }
            if data.len() % 2 != 0 {
                return Err(anyhow!("registry value {name} has an odd byte length"));
            }
            let wide = data
                .chunks_exact(2)
                .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
                .collect::<Vec<_>>();
            Ok(String::from_utf16_lossy(&wide))
        }
    }

    fn wide(value: &str) -> Vec<u16> {
        value.encode_utf16().chain(std::iter::once(0)).collect()
    }
}

#[cfg(not(windows))]
mod platform {
    use super::Result;

    pub(super) struct Identity {
        pub model: String,
        pub board: String,
        pub bios: String,
    }

    pub(super) fn read_identity() -> Result<Identity> {
        anyhow::bail!("Lenovo BIOS downgrade validation requires Windows firmware identity")
    }

    pub(super) fn require_ac_power() -> Result<()> {
        anyhow::bail!("Lenovo BIOS downgrade execution requires Windows AC power status")
    }
}

#[cfg(test)]
mod tests {
    use super::parse_version;

    #[test]
    fn parses_supported_bios_versions() {
        assert!(parse_version("R5CN53WW").is_some());
        assert!(parse_version("R5CN54WW").is_some());
        assert!(parse_version("R5CN5WW").is_none());
        assert!(parse_version("R5CN51").is_none());
    }

    #[test]
    fn compares_bios_versions_numerically() {
        assert!(parse_version("R5CN54WW") > parse_version("R5CN53WW"));
    }
}
