use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use reqwest::blocking::Client;
use serde_json::json;
use sha2::{Digest, Sha256};

use crate::bios;
use crate::config::{AppConfig, RebarUefiArgs};
use crate::model;

const REBAR_PROJECT_URL: &str = "https://github.com/xCuri0/ReBarUEFI";
const REBAR_RELEASE: &str = "0.3";
const REBAR_GUID: &str = "{A3C5B77A-C88F-4A93-BF1C-4A92A32C65CE}";
const REBAR_DXE_URL: &str =
    "https://github.com/xCuri0/ReBarUEFI/releases/download/0.3/ReBarDxe.ffs";
const REBAR_DXE_SIZE: u64 = 2_578;
const REBAR_DXE_SHA256: &str = "ce45bd4beebbc61a1b4f4ce89e79c2e25a0deda8530783e3a87c991eb2033a0d";
const REBAR_STATE_URL: &str =
    "https://github.com/xCuri0/ReBarUEFI/releases/download/0.3/ReBarState.exe";
const REBAR_STATE_SIZE: u64 = 19_968;
const REBAR_STATE_SHA256: &str = "83a66a36b2f1d7448706473ab6bea89e6b9f1833add539562e88cd0484944fb1";

const MODEL_NAME: &str = "21TL";
const BOARD_NAME: &str = "LNVNB161216";
const SUPPORTED_BIOS: [&str; 4] = ["R5CN51WW", "R5CN53WW", "R5CN54WW", "R5CN55WW"];
const PCH_SETUP_GUID: &str = "{4570B7F1-ADE8-4943-8DC3-406472842384}";
const PCH_SETUP_SIZE: usize = 0x17C2;
const PCH_REBAR_OFFSET: usize = 0x3B6;
const SA_SETUP_GUID: &str = "{72C5E28C-7783-43A1-8767-FAD73FCCAFA4}";
const SA_SETUP_SIZE: usize = 0x300;
const SA_ABOVE_4G_OFFSET: usize = 0xA9;
const SA_MEMORY_REMAP_OFFSET: usize = 0x272;

#[derive(Debug)]
struct FirmwareVariable {
    bytes: Vec<u8>,
    attributes: u32,
}

#[derive(Debug)]
struct NativeState {
    pch: FirmwareVariable,
    sa: FirmwareVariable,
}

pub fn run(config: &AppConfig, args: &RebarUefiArgs) -> Result<()> {
    if args.download {
        let workspace = args
            .workspace
            .clone()
            .unwrap_or_else(|| config.data_dir.join("rebar-uefi"));
        download_assets(&workspace)?;
        return Ok(());
    }
    if args.native_status || args.native_enable {
        run_native(config, args)?;
        return Ok(());
    }

    println!("ReBarUEFI helper");
    println!("Project: {REBAR_PROJECT_URL}");
    println!("This command never patches or flashes BIOS firmware.");
    println!(
        "It only reads or writes the ReBarState UEFI variable; the ReBarDxe firmware module must already be installed."
    );
    println!(
        "UEFI boot is machine-checked; 4G Decoding and CSM are firmware settings that Windows does not expose reliably."
    );
    println!();
    if let Some(_) = args.set {
        if !config.yes {
            anyhow::bail!(
                "refusing to write the ReBarState UEFI variable without --yes; this affects the next boot"
            );
        }
        if !args.confirm_4g_decoding {
            anyhow::bail!(
                "refusing to write: pass --confirm-4g-decoding after manually verifying Above 4G Decoding"
            );
        }
        if !args.confirm_csm_disabled {
            anyhow::bail!(
                "refusing to write: pass --confirm-csm-disabled after manually verifying CSM is disabled"
            );
        }
    }

    let current = platform::read_state()?;
    print_state("Current", current);
    if let Some(value) = args.set {
        println!();
        println!("WARNING: writing ReBarState changes firmware NVRAM for the next boot.");
        println!("WARNING: it does not install ReBarDxe and has no effect on stock firmware.");
        println!(
            "WARNING: the 4G/CSM flags are explicit user confirmations, not automatic firmware reads."
        );
        println!("WARNING: keep a recovery/CMOS path available before rebooting.");
        platform::write_state(value)?;
        let after = platform::read_state()?.context("ReBarState disappeared after writing it")?;
        println!();
        print_state("Written", Some(after));
        println!("Reboot for the UEFI module to apply the new BAR size.");
    } else {
        println!();
        println!(
            "No NVRAM write requested. Use --set <0..32> --yes --confirm-4g-decoding --confirm-csm-disabled only after installing ReBarDxe."
        );
    }
    Ok(())
}

fn run_native(config: &AppConfig, args: &RebarUefiArgs) -> Result<()> {
    if args.native_enable {
        if !config.yes {
            anyhow::bail!(
                "refusing to write native PchSetup without --yes; this changes firmware NVRAM"
            );
        }
        if !args.confirm_native_rebar {
            anyhow::bail!("refusing to write native PchSetup without --confirm-native-rebar");
        }
    }

    let (model, board, bios_version) = bios::current_identity()?;
    validate_native_identity(&model, &board, &bios_version)?;
    println!("Native Lenovo ReBAR path");
    println!("Model: {model}");
    println!("Baseboard: {board}");
    println!("Current BIOS: {bios_version}");
    println!("PchSetup[0x{PCH_REBAR_OFFSET:04X}] is the native PCIe Resizable BAR setting.");
    println!(
        "Only that one PchSetup byte can be changed; the BIOS image is never patched or flashed."
    );
    println!();

    let state = platform::read_native()?;
    print_native_state(&state)?;
    if !args.native_enable {
        println!();
        println!("No native NVRAM write requested.");
        return Ok(());
    }

    let above_4g = native_byte(&state.sa.bytes, SA_ABOVE_4G_OFFSET, "SaSetup Above 4G MMIO")?;
    let memory_remap = native_byte(
        &state.sa.bytes,
        SA_MEMORY_REMAP_OFFSET,
        "SaSetup Memory Remap",
    )?;
    if above_4g != 1 {
        anyhow::bail!(
            "refusing native ReBAR write: SaSetup[0x{SA_ABOVE_4G_OFFSET:04X}] Above 4G MMIO is 0x{above_4g:02X}, not enabled"
        );
    }
    if memory_remap != 1 {
        anyhow::bail!(
            "refusing native ReBAR write: SaSetup[0x{SA_MEMORY_REMAP_OFFSET:04X}] Memory Remap is 0x{memory_remap:02X}, not enabled"
        );
    }

    let current = native_byte(
        &state.pch.bytes,
        PCH_REBAR_OFFSET,
        "PchSetup PCIe Resizable BAR Support",
    )?;
    if !matches!(current, 0..=2) {
        anyhow::bail!(
            "refusing native ReBAR write: PchSetup[0x{PCH_REBAR_OFFSET:04X}] has unsupported value 0x{current:02X}"
        );
    }
    if current == 1 {
        println!("Native PCIe Resizable BAR is already enabled; no NVRAM write was needed.");
        return Ok(());
    }

    let workspace = args
        .workspace
        .clone()
        .unwrap_or_else(|| config.data_dir.join("rebar-native"));
    write_native_backup(&workspace, &model, &board, &bios_version, &state, current)?;

    let mut updated = state.pch.bytes.clone();
    updated[PCH_REBAR_OFFSET] = 1;
    platform::write_native_rebar(&updated, state.pch.attributes)?;
    let after = platform::read_native()?;
    let written = native_byte(
        &after.pch.bytes,
        PCH_REBAR_OFFSET,
        "PchSetup PCIe Resizable BAR Support after write",
    )?;
    if written != 1 {
        anyhow::bail!(
            "native ReBAR write did not verify: PchSetup[0x{PCH_REBAR_OFFSET:04X}] is 0x{written:02X}"
        );
    }
    println!();
    print_native_state(&after)?;
    println!("Native ReBAR NVRAM write verified.");
    println!("Reboot before checking BAR allocation; firmware applies the setting during boot.");
    println!("Backup and manifest: {}", workspace.display());
    Ok(())
}

fn validate_native_identity(model: &str, board: &str, bios_version: &str) -> Result<()> {
    if model != MODEL_NAME {
        anyhow::bail!(
            "refusing native ReBAR path on unsupported model {model:?}; expected {MODEL_NAME}"
        );
    }
    if board != BOARD_NAME {
        anyhow::bail!(
            "refusing native ReBAR path on unsupported baseboard {board:?}; expected {BOARD_NAME}"
        );
    }
    if !SUPPORTED_BIOS.contains(&bios_version) {
        anyhow::bail!(
            "refusing native ReBAR path on unsupported BIOS {bios_version:?}; supported versions: {}",
            SUPPORTED_BIOS.join(", ")
        );
    }
    Ok(())
}

fn print_native_state(state: &NativeState) -> Result<()> {
    let pch_rebar = native_byte(
        &state.pch.bytes,
        PCH_REBAR_OFFSET,
        "PchSetup PCIe Resizable BAR Support",
    )?;
    let above_4g = native_byte(&state.sa.bytes, SA_ABOVE_4G_OFFSET, "SaSetup Above 4G MMIO")?;
    let memory_remap = native_byte(
        &state.sa.bytes,
        SA_MEMORY_REMAP_OFFSET,
        "SaSetup Memory Remap",
    )?;

    println!(
        "PchSetup: {} bytes, attributes 0x{:08X}",
        state.pch.bytes.len(),
        state.pch.attributes
    );
    println!(
        "  [0x{PCH_REBAR_OFFSET:04X}] PCIe Resizable BAR Support: 0x{pch_rebar:02X} ({})",
        describe_native_rebar(pch_rebar)
    );
    println!(
        "SaSetup: {} bytes, attributes 0x{:08X}",
        state.sa.bytes.len(),
        state.sa.attributes
    );
    println!(
        "  [0x{SA_ABOVE_4G_OFFSET:04X}] Above 4G MMIO: 0x{above_4g:02X} ({})",
        describe_toggle(above_4g)
    );
    println!(
        "  [0x{SA_MEMORY_REMAP_OFFSET:04X}] Memory Remap: 0x{memory_remap:02X} ({})",
        describe_toggle(memory_remap)
    );
    Ok(())
}

fn native_byte(bytes: &[u8], offset: usize, name: &str) -> Result<u8> {
    bytes
        .get(offset)
        .copied()
        .with_context(|| format!("{name} offset 0x{offset:04X} is outside the firmware variable"))
}

fn describe_native_rebar(value: u8) -> String {
    match value {
        0 => "disabled".to_owned(),
        1 => "enabled".to_owned(),
        2 => "auto".to_owned(),
        _ => "unknown".to_owned(),
    }
}

fn describe_toggle(value: u8) -> &'static str {
    match value {
        0 => "disabled",
        1 => "enabled",
        _ => "unknown",
    }
}

fn write_native_backup(
    workspace: &Path,
    model: &str,
    board: &str,
    bios_version: &str,
    state: &NativeState,
    current_rebar: u8,
) -> Result<()> {
    fs::create_dir_all(workspace).with_context(|| {
        format!(
            "creating native ReBAR backup directory {}",
            workspace.display()
        )
    })?;
    write_verified_backup(&workspace.join("PchSetup.before.bin"), &state.pch.bytes)?;
    write_verified_backup(&workspace.join("SaSetup.before.bin"), &state.sa.bytes)?;

    let manifest = json!({
        "operation": "native-pch-setup-rebar",
        "automatic_flash": false,
        "model": model,
        "baseboard": board,
        "bios": bios_version,
        "variables": {
            "PchSetup": {
                "guid": PCH_SETUP_GUID,
                "size": state.pch.bytes.len(),
                "attributes": state.pch.attributes,
                "rebar_offset": PCH_REBAR_OFFSET,
                "before": current_rebar,
                "after": 1,
                "sha256_before": sha256_bytes(&state.pch.bytes),
            },
            "SaSetup": {
                "guid": SA_SETUP_GUID,
                "size": state.sa.bytes.len(),
                "attributes": state.sa.attributes,
                "above_4g_offset": SA_ABOVE_4G_OFFSET,
                "memory_remap_offset": SA_MEMORY_REMAP_OFFSET,
                "sha256_before": sha256_bytes(&state.sa.bytes),
            },
        },
    });
    let manifest_bytes = serde_json::to_vec_pretty(&manifest)?;
    write_verified_backup(
        &workspace.join("native-rebar-manifest.json"),
        &manifest_bytes,
    )?;
    println!("Native NVRAM backup: {}", workspace.display());
    Ok(())
}

fn write_verified_backup(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Ok(metadata) = fs::symlink_metadata(path) {
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            anyhow::bail!("refusing to use non-regular backup path {}", path.display());
        }
        let existing = fs::read(path)
            .with_context(|| format!("reading existing backup {}", path.display()))?;
        if existing != bytes {
            anyhow::bail!(
                "refusing to overwrite changed backup {}; remove it only after reviewing the saved firmware state",
                path.display()
            );
        }
        return Ok(());
    }
    fs::write(path, bytes).with_context(|| format!("writing backup {}", path.display()))?;
    let written = fs::read(path).with_context(|| format!("verifying backup {}", path.display()))?;
    if written != bytes {
        anyhow::bail!("backup verification failed for {}", path.display());
    }
    Ok(())
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    hex::encode(digest.finalize())
}

fn download_assets(workspace: &Path) -> Result<()> {
    fs::create_dir_all(workspace)
        .with_context(|| format!("creating ReBarUEFI workspace {}", workspace.display()))?;
    let client = Client::builder()
        .user_agent("WINSTRINT/0.1 ReBarUEFI helper")
        .build()
        .context("creating ReBarUEFI download client")?;
    let assets = [
        (
            "ReBarDxe.ffs",
            REBAR_DXE_URL,
            REBAR_DXE_SIZE,
            REBAR_DXE_SHA256,
        ),
        (
            "ReBarState.exe",
            REBAR_STATE_URL,
            REBAR_STATE_SIZE,
            REBAR_STATE_SHA256,
        ),
    ];
    for (name, url, size, sha256) in assets {
        model::download_verified(&client, url, &workspace.join(name), size, Some(sha256))?;
    }
    let manifest = json!({
        "project": REBAR_PROJECT_URL,
        "release": REBAR_RELEASE,
        "assets": [
            {"name": "ReBarDxe.ffs", "size": REBAR_DXE_SIZE, "sha256": REBAR_DXE_SHA256},
            {"name": "ReBarState.exe", "size": REBAR_STATE_SIZE, "sha256": REBAR_STATE_SHA256},
        ],
    });
    fs::write(
        workspace.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )
    .with_context(|| format!("writing ReBarUEFI manifest in {}", workspace.display()))?;
    println!(
        "Verified ReBarUEFI {REBAR_RELEASE} assets in {}",
        workspace.display()
    );
    println!("Next: inspect a firmware backup and use UEFITool/UEFIPatch to add ReBarDxe.ffs.");
    println!("This launcher will not patch or flash the Lenovo BIOS.");
    Ok(())
}

fn print_state(label: &str, state: Option<u8>) {
    match state {
        Some(value) => println!("{label} ReBarState: {value} ({})", describe_state(value)),
        None => {
            println!("{label} ReBarState: absent (the ReBarUEFI variable has not been created)")
        }
    }
}

fn describe_state(value: u8) -> String {
    match value {
        0 => "disabled".into(),
        32 => "unlimited".into(),
        exponent => format!("{} MiB maximum BAR", 1u64 << exponent),
    }
}

#[cfg(windows)]
mod platform {
    use std::io;
    use std::ptr;

    use anyhow::{Context, Result};
    use windows_sys::Win32::Foundation::{
        CloseHandle, GetLastError, ERROR_ENVVAR_NOT_FOUND, ERROR_NOT_ALL_ASSIGNED, ERROR_SUCCESS,
        HANDLE,
    };
    use windows_sys::Win32::Security::{
        AdjustTokenPrivileges, LookupPrivilegeValueW, LUID_AND_ATTRIBUTES, SE_PRIVILEGE_ENABLED,
        SE_SYSTEM_ENVIRONMENT_NAME, TOKEN_ADJUST_PRIVILEGES, TOKEN_PRIVILEGES, TOKEN_QUERY,
    };
    use windows_sys::Win32::System::SystemInformation::{FirmwareTypeUefi, GetFirmwareType};
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};
    use windows_sys::Win32::System::WindowsProgramming::{
        GetFirmwareEnvironmentVariableExW, SetFirmwareEnvironmentVariableExW,
    };

    use super::{
        FirmwareVariable, NativeState, PCH_SETUP_GUID, PCH_SETUP_SIZE, REBAR_GUID, SA_SETUP_GUID,
        SA_SETUP_SIZE,
    };

    const VARIABLE_NAME: &str = "ReBarState";
    const VARIABLE_ATTRIBUTES: u32 = 0x0000_0007;

    pub fn read_state() -> Result<Option<u8>> {
        require_uefi()?;
        let token = enable_system_environment_privilege()?;
        let name = wide(VARIABLE_NAME);
        let guid = wide(REBAR_GUID);
        let mut value = 0u8;
        let mut attributes = 0u32;
        let size = unsafe {
            GetFirmwareEnvironmentVariableExW(
                name.as_ptr(),
                guid.as_ptr(),
                (&mut value as *mut u8).cast(),
                1,
                &mut attributes,
            )
        };
        let result = if size == 1 {
            Ok(Some(value))
        } else {
            let error = io::Error::last_os_error();
            if error.raw_os_error() == Some(ERROR_ENVVAR_NOT_FOUND as i32) {
                Ok(None)
            } else {
                Err(error).with_context(|| {
                    format!("reading {VARIABLE_NAME} UEFI variable with GUID {REBAR_GUID}")
                })
            }
        };
        unsafe {
            CloseHandle(token);
        }
        result
    }

    pub fn write_state(value: u8) -> Result<()> {
        require_uefi()?;
        let token = enable_system_environment_privilege()?;
        let name = wide(VARIABLE_NAME);
        let guid = wide(REBAR_GUID);
        let ok = unsafe {
            SetFirmwareEnvironmentVariableExW(
                name.as_ptr(),
                guid.as_ptr(),
                (&value as *const u8).cast(),
                1,
                VARIABLE_ATTRIBUTES,
            )
        };
        let result = if ok == 0 {
            Err(io::Error::last_os_error()).with_context(|| {
                format!("writing {VARIABLE_NAME} UEFI variable with GUID {REBAR_GUID}")
            })
        } else {
            Ok(())
        };
        unsafe {
            CloseHandle(token);
        }
        result
    }

    pub fn read_native() -> Result<NativeState> {
        require_uefi()?;
        let token = enable_system_environment_privilege()?;
        let result = (|| {
            let pch = read_variable("PchSetup", PCH_SETUP_GUID, PCH_SETUP_SIZE)?;
            let sa = read_variable("SaSetup", SA_SETUP_GUID, SA_SETUP_SIZE)?;
            Ok(NativeState { pch, sa })
        })();
        unsafe {
            CloseHandle(token);
        }
        result
    }

    pub fn write_native_rebar(bytes: &[u8], attributes: u32) -> Result<()> {
        if bytes.len() != PCH_SETUP_SIZE {
            anyhow::bail!(
                "refusing native PchSetup write with {} bytes; expected {}",
                bytes.len(),
                PCH_SETUP_SIZE
            );
        }
        require_uefi()?;
        let token = enable_system_environment_privilege()?;
        let result = write_variable("PchSetup", PCH_SETUP_GUID, bytes, attributes);
        unsafe {
            CloseHandle(token);
        }
        result
    }

    fn read_variable(name: &str, guid: &str, expected_size: usize) -> Result<FirmwareVariable> {
        let name_wide = wide(name);
        let guid_wide = wide(guid);
        let mut bytes = vec![0u8; expected_size];
        let mut attributes = 0u32;
        let size = unsafe {
            GetFirmwareEnvironmentVariableExW(
                name_wide.as_ptr(),
                guid_wide.as_ptr(),
                bytes.as_mut_ptr().cast(),
                expected_size as u32,
                &mut attributes,
            )
        };
        if size != expected_size as u32 {
            let error = io::Error::last_os_error();
            return Err(error).with_context(|| {
                format!(
                    "reading {name} UEFI variable with GUID {guid}; expected {expected_size} bytes, Windows returned {size}"
                )
            });
        }
        Ok(FirmwareVariable { bytes, attributes })
    }

    fn write_variable(name: &str, guid: &str, bytes: &[u8], attributes: u32) -> Result<()> {
        let name_wide = wide(name);
        let guid_wide = wide(guid);
        let ok = unsafe {
            SetFirmwareEnvironmentVariableExW(
                name_wide.as_ptr(),
                guid_wide.as_ptr(),
                bytes.as_ptr().cast(),
                bytes.len() as u32,
                attributes,
            )
        };
        if ok == 0 {
            return Err(io::Error::last_os_error())
                .with_context(|| format!("writing {name} UEFI variable with GUID {guid}"));
        }
        Ok(())
    }

    fn require_uefi() -> Result<()> {
        let mut firmware_type = 0;
        if unsafe { GetFirmwareType(&mut firmware_type) } == 0 {
            return Err(io::Error::last_os_error()).context("checking Windows firmware type");
        }
        if firmware_type != FirmwareTypeUefi {
            if firmware_type == 1 {
                anyhow::bail!(
                    "Windows is booted in legacy BIOS mode; EFI variable access requires a UEFI boot"
                );
            }
            anyhow::bail!("Windows is not booted with UEFI firmware");
        }
        Ok(())
    }

    fn enable_system_environment_privilege() -> Result<HANDLE> {
        let mut token: HANDLE = ptr::null_mut();
        let desired = TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY;
        if unsafe { OpenProcessToken(GetCurrentProcess(), desired, &mut token) } == 0 {
            return Err(io::Error::last_os_error()).context("opening the process token");
        }

        let mut luid = Default::default();
        if unsafe { LookupPrivilegeValueW(ptr::null(), SE_SYSTEM_ENVIRONMENT_NAME, &mut luid) } == 0
        {
            unsafe {
                CloseHandle(token);
            }
            return Err(io::Error::last_os_error())
                .context("looking up SeSystemEnvironmentPrivilege");
        }

        let privileges = TOKEN_PRIVILEGES {
            PrivilegeCount: 1,
            Privileges: [LUID_AND_ATTRIBUTES {
                Luid: luid,
                Attributes: SE_PRIVILEGE_ENABLED,
            }],
        };
        if unsafe {
            AdjustTokenPrivileges(
                token,
                0,
                &privileges,
                std::mem::size_of::<TOKEN_PRIVILEGES>() as u32,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        } == 0
        {
            unsafe {
                CloseHandle(token);
            }
            return Err(io::Error::last_os_error())
                .context("enabling SeSystemEnvironmentPrivilege");
        }
        let last_error = unsafe { GetLastError() };
        if last_error == ERROR_NOT_ALL_ASSIGNED {
            unsafe {
                CloseHandle(token);
            }
            anyhow::bail!(
                "SeSystemEnvironmentPrivilege is not assigned; run this command from an elevated administrator shell"
            );
        }
        if last_error != ERROR_SUCCESS {
            unsafe {
                CloseHandle(token);
            }
            anyhow::bail!(
                "enabling SeSystemEnvironmentPrivilege failed with Windows error {last_error}"
            );
        }
        Ok(token)
    }

    fn wide(value: &str) -> Vec<u16> {
        value.encode_utf16().chain(std::iter::once(0)).collect()
    }
}

#[cfg(not(windows))]
mod platform {
    use anyhow::{bail, Result};

    pub fn read_state() -> Result<Option<u8>> {
        bail!("the ReBarUEFI helper requires Windows UEFI firmware access")
    }

    pub fn write_state(_value: u8) -> Result<()> {
        bail!("the ReBarUEFI helper requires Windows UEFI firmware access")
    }

    pub fn read_native() -> Result<super::NativeState> {
        bail!("the native ReBAR path requires Windows UEFI firmware access")
    }

    pub fn write_native_rebar(_bytes: &[u8], _attributes: u32) -> Result<()> {
        bail!("the native ReBAR path requires Windows UEFI firmware access")
    }
}
