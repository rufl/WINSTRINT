use std::env;
use std::ffi::{OsStr, OsString};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use serde_json::{Map, Value, json};

use crate::config::{AppConfig, ModelChoice, StrataTarget};
use crate::diagnostics::{self, GpuInfo, GpuKind};
use crate::model::ModelArtifact;

/// The upstream Strata project this launcher tracks.
pub const STRATA_REPOSITORY: &str = "https://github.com/Niko1221/Strata.git";
/// Pinned upstream revision read during this integration pass.
pub const STRATA_COMMIT: &str = "e88ee39d1d6be6e3364183af3b39b62a7ed6f211";
const LLAMA_CPP_REPOSITORY: &str = "https://github.com/ggml-org/llama.cpp.git";
const LLAMA_CPP_COMMIT: &str = "3cf03257f219afbe7334045ff7c6a06ac68c627d";
const WINDOWS_PATCH_VERSION: &str = "windows-sycl-v8";
const PREVIOUS_WINDOWS_PATCH_VERSION: &str = "windows-sycl-v7";
const OLDER_WINDOWS_PATCH_VERSION: &str = "windows-sycl-v6";
const LEGACY_WINDOWS_PATCH_VERSION: &str = "windows-sycl-v5";
const WINDOWS_ENGINE_BUILD_VERSION: &str = "windows-sycl-engine-v2";
const ENGINE_HEALTH_TIMEOUT: Duration = Duration::from_secs(1_800);
const WINDOWS_CMAKE_RULES: &str = r#"
set(CMAKE_CXX_COMPILE_OBJECT "<CMAKE_CXX_COMPILER> -fsycl /EHsc /Qopenmp- <DEFINES> <INCLUDES> <FLAGS> -c <SOURCE> -o <OBJECT>")
set(CMAKE_CXX_LINK_EXECUTABLE "<CMAKE_CXX_COMPILER> -fsycl /Qopenmp- <OBJECTS> <LINK_FLAGS> <LINK_LIBRARIES> advapi32.lib kernel32.lib vcruntime.lib ucrt.lib msvcrt.lib libircmt.lib svml_dispmd.lib libmmd.lib oldnames.lib msvcprt.lib -o <TARGET>")
set(CMAKE_CXX_CREATE_STATIC_LIBRARY "lib.exe /OUT:<TARGET> <OBJECTS>")
set(CMAKE_C_CREATE_STATIC_LIBRARY "lib.exe /OUT:<TARGET> <OBJECTS>")
set(CMAKE_CXX_ARCHIVE_CREATE "lib.exe /OUT:<TARGET> <OBJECTS>")
set(CMAKE_C_ARCHIVE_CREATE "lib.exe /OUT:<TARGET> <OBJECTS>")
set(CMAKE_CXX_ARCHIVE_FINISH "")
set(CMAKE_C_ARCHIVE_FINISH "")
set(CMAKE_CXX_FLAGS_INIT "")
set(CMAKE_CXX_FLAGS_DEBUG_INIT "")
set(CMAKE_CXX_FLAGS_RELEASE_INIT "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO_INIT "")
set(CMAKE_CXX_FLAGS_MINSIZEREL_INIT "")
set(CMAKE_EXE_LINKER_FLAGS_INIT "")
set(CMAKE_EXE_LINKER_FLAGS_DEBUG_INIT "")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE_INIT "")
set(CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO_INIT "")
set(CMAKE_EXE_LINKER_FLAGS_MINSIZEREL_INIT "")
set(CMAKE_CXX_STANDARD_LIBRARIES "")
set(CMAKE_CXX_IMPLICIT_LINK_LIBRARIES "")
set(CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES "")
set(CMAKE_CXX_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreaded "")
set(CMAKE_CXX_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreadedDLL "")
set(CMAKE_CXX_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreadedDebug "")
set(CMAKE_CXX_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreadedDebugDLL "")
set(CMAKE_CXX_COMPILE_OPTIONS_MSVC_DEBUG_INFORMATION_FORMAT_Embedded "")
set(CMAKE_CXX_COMPILE_OPTIONS_MSVC_DEBUG_INFORMATION_FORMAT_ProgramDatabase "")
set(CMAKE_CXX_COMPILE_OPTIONS_MSVC_DEBUG_INFORMATION_FORMAT_EditAndContinue "")
set(CMAKE_CXX_COMPILE_OPTIONS_MSVC_RUNTIME_CHECKS_PossibleDataLoss "")
set(CMAKE_CXX_COMPILE_OPTIONS_MSVC_RUNTIME_CHECKS_StackFrameErrorCheck "")
set(CMAKE_CXX_COMPILE_OPTIONS_MSVC_RUNTIME_CHECKS_UninitializedVariable "")
set(CMAKE_CXX_COMPILE_OPTIONS_MSVC_RUNTIME_CHECKS_RTCsu "")
set(CMAKE_DEPFILE_FLAGS_CXX "")
"#;
const WINDOWS_PINNED_LIMIT_OLD: &str = r#"#ifdef _WIN32
// #243: how much of the arena the sliced registration may pin on Windows when the whole arena was refused.
// Page-locked memory the GPU maps is charged to its shared (non-local) WDDM segment; pinned slice by slice until
// the driver refused one (28 GiB of a 63 GB PC), that segment was left full and every later cudaMalloc failed
// "out of memory".  4 GiB of the budget stay free for what the engine allocates after the arena; without the DXGI
// numbers, RAM/2 - 8 GiB (the budget is about half the RAM).  False: no limit could be worked out.
bool sliced_pin_limit(uint64_t& limit, std::string& why) {
    constexpr uint64_t GiB = 1ull << 30;
    char buf[256];
    int dev = 0;
    cudaDeviceProp p{};
    uint64_t budget = 0, usage = 0;
    std::string err = "no CUDA device properties";
    if (cudaGetDevice(&dev) == cudaSuccess && cudaGetDeviceProperties(&p, dev) == cudaSuccess &&
        strata::platform::gpu_shared_memory_budget(p.luid, budget, usage, err)) {
        limit = budget > usage + 4 * GiB ? budget - usage - 4 * GiB : 0;
        std::snprintf(buf, sizeof buf, "the GPU's shared-memory budget %.1f GiB - %.1f GiB in use - 4 GiB",
                      (double) budget / GiB, (double) usage / GiB);
        why = buf;
        return true;
    }
    (void) cudaGetLastError();
    const uint64_t ram = strata::platform::total_physical_memory();
    if (ram == 0) return false;
    limit = ram / 2 > 8 * GiB ? ram / 2 - 8 * GiB : 0;
    std::snprintf(buf, sizeof buf, "%s: RAM/2 - 8 GiB of %.1f GiB", err.c_str(), (double) ram / GiB);
    why = buf;
    return true;
}
#endif
"#;
const WINDOWS_PINNED_LIMIT_NEW: &str = r#"#ifdef _WIN32
// SYCL has no CUDA device-properties object from which to obtain the adapter LUID.  Keep the conservative
// Windows fallback used when the DXGI query is unavailable instead of compiling CUDA-only symbols into the SYCL port.
bool sliced_pin_limit(uint64_t& limit, std::string& why) {
    constexpr uint64_t GiB = 1ull << 30;
    char buf[256];
    const uint64_t ram = strata::platform::total_physical_memory();
    if (ram == 0) return false;
    limit = ram / 2 > 8 * GiB ? ram / 2 - 8 * GiB : 0;
    std::snprintf(buf, sizeof buf, "SYCL Windows fallback: RAM/2 - 8 GiB of %.1f GiB", (double) ram / GiB);
    why = buf;
    return true;
}
#endif
"#;
const WINDOWS_GROUP_ENTANGLE_OLD: &str = "sycl::ext::oneapi::experimental::entangle(sg)";
const WINDOWS_GROUP_ENTANGLE_NEW: &str = "sycl::ext::oneapi::experimental::get_tangle_group(sg)";
const WINDOWS_GROUP_PARTITION_OLD: &str =
    "sycl::ext::oneapi::experimental::chunked_partition<LogicSubGroupSize>";
const WINDOWS_GROUP_PARTITION_NEW: &str =
    "sycl::ext::oneapi::experimental::get_fixed_size_group<LogicSubGroupSize>";
const WINDOWS_VERIFY_TEMPLATE_OLD: &str = "template <bool OUT>";
const WINDOWS_VERIFY_TEMPLATE_NEW: &str = "template <bool WRITE_OUT>";
const WINDOWS_VERIFY_CONDITION_OLD: &str = "if (!OUT) continue;";
const WINDOWS_VERIFY_CONDITION_NEW: &str = "if (!WRITE_OUT) continue;";
const WINDOWS_GENERATE_MEMMEM_OLD: &str = "const void* hit = memmem(other.data(), nb2, probe, 64);\n                                    if (hit != nullptr)";
const WINDOWS_GENERATE_MEMMEM_NEW: &str = "const uint8_t* hit = std::search(other.data(), other.data() + nb2, probe, probe + 64);\n                                    if (hit != other.data() + nb2)";
const WINDOWS_GENERATE_SMALL_OLD: &[(&str, &str)] = &[
    (
        "const int64_t small = old_rule();",
        "const int64_t small_chunk = old_rule();",
    ),
    ("ring_cap_for(small)", "ring_cap_for(small_chunk)"),
    ("if (small >= c)", "if (small_chunk >= c)"),
    (
        "if (small > 0) { chunk = small; return slots_for(small); }",
        "if (small_chunk > 0) { chunk = small_chunk; return slots_for(small_chunk); }",
    ),
    (
        "set_ring_budget((int) room_of(c), small)",
        "set_ring_budget((int) room_of(c), small_chunk)",
    ),
    (
        "const int64_t small = old_pick(only);",
        "const int64_t small_chunk = old_pick(only);",
    ),
    ("ring_cap_for(small)", "ring_cap_for(small_chunk)"),
    (
        "return std::max(c, small)",
        "return std::max(c, small_chunk)",
    ),
    (
        "const int64_t small = old_pick(nullptr);",
        "const int64_t small_chunk = old_pick(nullptr);",
    ),
    (
        "small >= chunk ? 0 : (int) ring_room(chunk, nullptr, strata::prefill::Prefill::ring_cap_for(small)),",
        "small_chunk >= chunk ? 0 : (int) ring_room(chunk, nullptr, strata::prefill::Prefill::ring_cap_for(small_chunk)),",
    ),
    (
        "                    small);",
        "                    small_chunk);",
    ),
];
const WINDOWS_ESIMD_SOURCE_OLD: &str = "            ${CMAKE_CURRENT_SOURCE_DIR}/src/prefill/esimd_moe.dp.cpp)       # 151-b70: fused int8 DPAS expert products (ESIMD, Battlemage)\n";
const WINDOWS_ESIMD_SOURCE_V7_OLD: &str = r#"            ${CMAKE_CURRENT_SOURCE_DIR}/src/prefill/esimd_moe.dp.cpp)       # 151-b70: fused int8 DPAS expert products (ESIMD, Battlemage)
if(WIN32)
  # Intel oneAPI sycl-post-link currently crashes in SROA on SwigluTag; keep the tested fallback on Windows.
  option(STRATA_WINDOWS_ESIMD "Build the fused ESIMD int8 kernels on Windows" OFF)
  if(NOT STRATA_WINDOWS_ESIMD)
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/src/prefill/esimd_moe.dp.cpp PROPERTIES COMPILE_DEFINITIONS STRATA_I8_NO_ESIMD=1)
  endif()
endif()
"#;
const WINDOWS_ESIMD_SOURCE_NEW: &str = r#"            ${CMAKE_CURRENT_SOURCE_DIR}/src/prefill/esimd_moe.dp.cpp)       # 151-b70: fused int8 DPAS expert products (ESIMD, Battlemage)
if(WIN32)
  # oneAPI 2025.3's sycl-post-link SROA pass crashes on explicit-SIMD kernels when this translation unit is optimized before post-link.
  option(STRATA_WINDOWS_ESIMD "Build the fused ESIMD int8 kernels on Windows" ON)
  if(STRATA_WINDOWS_ESIMD)
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/src/prefill/esimd_moe.dp.cpp PROPERTIES COMPILE_OPTIONS "/Od")
  else()
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/src/prefill/esimd_moe.dp.cpp PROPERTIES COMPILE_DEFINITIONS STRATA_I8_NO_ESIMD=1)
  endif()
endif()
"#;
const WINDOWS_RUNTIME_TARGET_OLD: &str =
    "target_link_libraries(strata PRIVATE strata_engine strata_prefill strata_spec)\n";
const WINDOWS_RUNTIME_TARGET_NEW: &str = "target_link_libraries(strata PRIVATE strata_engine strata_prefill strata_spec)\nif(WIN32)\n  target_sources(strata PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/platform/winstrint_windows_runtime.c)\nendif()\n";
const WINDOWS_RUNTIME_SOURCE: &str = r#"#include <math.h>

static float winstrint_ldexpf_impl(float value, int exponent) {
    return (float) ldexp((double) value, exponent);
}

/* AVX-512 ggml objects from icx reference the UCRT import spelling __imp_ldexpf,
   while the Windows UCRT import library exports __imp_ldexp. */
__declspec(selectany) float (*__imp_ldexpf)(float, int) = winstrint_ldexpf_impl;
"#;

pub struct StrataLaunch {
    pub program: PathBuf,
    pub args: Vec<OsString>,
    pub cwd: PathBuf,
    pub log_path: PathBuf,
    pub health_timeout: Duration,
}

pub fn strata_root(config: &AppConfig) -> PathBuf {
    absolute_path(&config.data_dir).join("strata")
}

fn absolute_path(path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    }
}

fn source_root(config: &AppConfig) -> PathBuf {
    strata_root(config).join("source")
}

fn build_root(config: &AppConfig) -> PathBuf {
    strata_root(config).join("build-sycl")
}

fn virtualenv_root(config: &AppConfig) -> PathBuf {
    strata_root(config).join("venv")
}

fn log_root(config: &AppConfig) -> PathBuf {
    absolute_path(&config.data_dir).join("logs")
}
fn engine_path(config: &AppConfig) -> PathBuf {
    build_root(config).join("strata.exe")
}
fn engine_marker(config: &AppConfig) -> PathBuf {
    build_root(config).join(".winstrint-engine")
}

fn python_path(config: &AppConfig) -> PathBuf {
    if cfg!(windows) {
        virtualenv_root(config).join("Scripts").join("python.exe")
    } else {
        virtualenv_root(config).join("bin").join("python")
    }
}

pub fn print_status(config: &AppConfig, verify: bool) -> Result<()> {
    let root = strata_root(config);
    println!("Strata source: {STRATA_REPOSITORY}");
    println!("Strata revision: {STRATA_COMMIT}");
    println!("Windows native SYCL port: {WINDOWS_PATCH_VERSION}");
    println!("  CMake/icx driver, direct Windows GGUF streaming I/O, and managed server runner");
    println!(
        "  Windows default: fused ESIMD int8 kernels are enabled with the oneAPI 2025.3 /Od post-link workaround; oneMKL remains the prefill fallback for unsupported devices"
    );
    println!("Managed Strata root: {}", root.display());
    println!("  source: {}", source_root(config).display());
    println!("  build:  {}", build_root(config).display());
    println!("  Python: {}", python_path(config).display());
    println!("  engine: {}", engine_path(config).display());
    match find_setvars() {
        Some(path) => println!("oneAPI setvars: {}", path.display()),
        None => println!("oneAPI setvars: missing"),
    }
    if verify {
        if let Err(error) = preflight_native_toolchain() {
            println!("Native toolchain: unavailable ({error:#})");
            return Err(error).context("native toolchain verification failed");
        }
        println!("Native toolchain: ready");
    }
    Ok(())
}

pub fn ensure_python_environment(config: &AppConfig) -> Result<PathBuf> {
    let source = ensure_source(config)?;
    ensure_gguf_py(&source)?;
    let python = python_path(config);
    if !python.exists() {
        if virtualenv_root(config).exists() {
            fs::remove_dir_all(virtualenv_root(config)).with_context(|| {
                format!(
                    "removing incomplete managed Strata Python environment {}",
                    virtualenv_root(config).display()
                )
            })?;
        }
        fs::create_dir_all(strata_root(config))?;
        let (program, prefix) = find_bootstrap_python()?;
        create_virtualenv(config, &program, &prefix)?;
    }
    let marker = virtualenv_root(config).join(".winstrint-requirements");
    let expected = format!("{STRATA_COMMIT}\n{WINDOWS_PATCH_VERSION}\n");
    if fs::read_to_string(&marker).ok().as_deref() != Some(expected.as_str()) {
        let requirements = source.join("requirements.txt");
        if !requirements.is_file() {
            bail!(
                "Strata source has no requirements.txt: {}",
                requirements.display()
            );
        }
        install_requirements(&python, &requirements, &source)?;
        fs::write(marker, expected)?;
    }
    Ok(python)
}

pub fn provision(config: &AppConfig, requested: StrataTarget, rebuild: bool) -> Result<()> {
    println!("Requested Strata target: {requested:?}");
    preflight_native_toolchain()?;
    let source = ensure_source(config)?;
    let _python = ensure_python_environment(config)?;
    let engine = ensure_engine(config, &source, rebuild)?;
    println!("Strata native SYCL engine ready: {}", engine.display());
    println!(
        "Model preparation is lazy; `--backend strata chat` creates the native pack and streamed model state."
    );
    Ok(())
}

pub fn probe(config: &AppConfig, _force: bool) -> Result<()> {
    preflight_native_toolchain()?;
    let source = source_root(config);
    if !source.join("sycl").join("CMakeLists.txt").is_file() {
        bail!(
            "Strata source is not provisioned at {}; run `winstrint strata-provision`",
            source.display()
        );
    }
    let engine = engine_path(config);
    if !engine.is_file() {
        bail!(
            "Strata engine is not built at {}; run `winstrint strata-provision`",
            engine.display()
        );
    }
    let setvars = find_setvars().context("locating Intel oneAPI setvars.bat")?;
    let command = format!(
        "call {} intel64 vs2022 && {} --help",
        batch_quote(&setvars),
        batch_quote(&engine)
    );
    let output = run_cmd_output(&command, &source, "probing the Strata native engine")?;
    println!("Strata probe succeeded: {}", engine.display());
    let stdout = String::from_utf8_lossy(&output.stdout);
    if let Some(line) = stdout.lines().find(|line| !line.trim().is_empty()) {
        println!("Engine: {}", line.trim());
    }
    Ok(())
}

pub fn validate_for_runtime(config: &AppConfig, target: Option<&GpuInfo>) -> Result<()> {
    if !cfg!(windows) {
        bail!("the managed Strata port currently requires native Windows oneAPI");
    }
    let gpu = target.context("no Intel Arc GPU was detected for Strata")?;
    if !matches!(gpu.kind, GpuKind::A770 | GpuKind::Arc140T) {
        bail!(
            "Strata requires Intel Arc A770 or Arc 140T; detected {:?} ({})",
            gpu.kind,
            gpu.name
        );
    }
    let model = crate::model::spec(config.model);
    if gpu.adapter_ram_bytes > 0 && model.resident_bytes > gpu.adapter_ram_bytes {
        let strategy = if matches!(gpu.kind, GpuKind::Arc140T) {
            "using a mapped expert tier and partial VRAM cache"
        } else {
            "using streamed experts and a host mirror"
        };
        println!(
            "Strata: {} needs {:.1} GiB resident and this GPU reports {:.1} GiB; {}",
            config.model.key(),
            model.resident_bytes as f64 / 1e9,
            gpu.adapter_ram_bytes as f64 / 1e9,
            strategy
        );
    }
    preflight_native_toolchain()
}

pub fn prepare_launch(
    config: &AppConfig,
    target: Option<&GpuInfo>,
    artifact: &ModelArtifact,
) -> Result<StrataLaunch> {
    validate_for_runtime(config, target)?;
    let source = ensure_source(config)?;
    let python = ensure_python_environment(config)?;
    let engine = ensure_engine(config, &source, false)?;
    let model = prepare_model(config, &source, &python, artifact)?;
    let alchemist = target.is_some_and(|gpu| matches!(gpu.kind, GpuKind::A770));
    let arc140t = target.is_some_and(|gpu| matches!(gpu.kind, GpuKind::Arc140T));
    let config_path = write_server_config(&config, &source, &engine, &model, alchemist, arc140t)?;
    let setvars = find_setvars().context("locating Intel oneAPI setvars.bat")?;
    let script = write_runtime_script(config, &source, &python, &setvars, &config_path, &engine)?;
    Ok(StrataLaunch {
        program: PathBuf::from("cmd.exe"),
        args: vec![
            os("/D"),
            os("/S"),
            os("/C"),
            os("call"),
            script.into_os_string(),
        ],
        cwd: source,
        log_path: log_root(config).join("strata-runtime.log"),
        health_timeout: ENGINE_HEALTH_TIMEOUT,
    })
}

fn preflight_native_toolchain() -> Result<()> {
    if !cfg!(windows) {
        bail!("native Strata SYCL provisioning is implemented for Windows only");
    }
    let missing = ["git", "cmake", "ninja"]
        .into_iter()
        .filter(|tool| !diagnostics::command_exists(tool))
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        bail!(
            "missing native build tools: {}; install Git, CMake, and Ninja and reopen the terminal",
            missing.join(", ")
        );
    }
    let setvars = find_setvars().context(
        "Intel oneAPI setvars.bat is missing; install Intel oneAPI DPC++/C++ with Visual Studio integration",
    )?;
    let command = format!(
        "call {} intel64 vs2022 >nul && where icx >nul && where icpx >nul",
        batch_quote(&setvars)
    );
    let output = run_cmd_allow_failure(
        &command,
        Path::new("."),
        "checking the Intel oneAPI Windows toolchain",
    )?;
    if !output.status.success() {
        bail!(
            "Intel oneAPI DPC++/C++ compiler was not found after setvars.bat; install the oneAPI compiler and Visual Studio 2022 Build Tools"
        );
    }
    Ok(())
}

fn find_setvars() -> Option<PathBuf> {
    crate::diagnostics::oneapi_setvars()
}

fn ensure_source(config: &AppConfig) -> Result<PathBuf> {
    let source = source_root(config);
    if source.join("sycl").join("CMakeLists.txt").is_file() {
        let revision = run_checked(
            OsStr::new("git"),
            &[os("rev-parse"), os("HEAD")],
            Some(&source),
            "reading the managed Strata revision",
        )?;
        if revision.trim() != STRATA_COMMIT {
            bail!(
                "managed Strata source is at {}, expected {}; remove {} and reprovision",
                revision.trim(),
                STRATA_COMMIT,
                source.display()
            );
        }
        apply_windows_patch(&source)?;
        return Ok(source);
    }
    if source.exists() {
        bail!(
            "managed Strata source directory is incomplete: {}; remove it and reprovision",
            source.display()
        );
    }
    if let Some(parent) = source.parent() {
        fs::create_dir_all(parent)?;
    }
    run_checked(
        OsStr::new("git"),
        &[
            os("clone"),
            os("--filter=blob:none"),
            os("--no-checkout"),
            os(STRATA_REPOSITORY),
            source.clone().into_os_string(),
        ],
        None,
        "cloning the pinned Strata source",
    )?;
    run_checked(
        OsStr::new("git"),
        &[
            os("fetch"),
            os("--depth=1"),
            os("origin"),
            os(STRATA_COMMIT),
        ],
        Some(&source),
        "fetching the pinned Strata revision",
    )?;
    run_checked(
        OsStr::new("git"),
        &[os("checkout"), os("--detach"), os(STRATA_COMMIT)],
        Some(&source),
        "checking out the pinned Strata revision",
    )?;
    apply_windows_patch(&source)?;
    Ok(source)
}

fn apply_windows_esimd_patch(source: &Path) -> Result<()> {
    let cmake = source.join("sycl").join("CMakeLists.txt");
    let text = fs::read_to_string(&cmake)
        .with_context(|| format!("reading Windows Strata CMake file {}", cmake.display()))?
        .replace("\r\n", "\n");
    if !text.contains(WINDOWS_ESIMD_SOURCE_NEW) {
        let old = if text.contains(WINDOWS_ESIMD_SOURCE_V7_OLD) {
            WINDOWS_ESIMD_SOURCE_V7_OLD
        } else {
            WINDOWS_ESIMD_SOURCE_OLD
        };
        patch_text(&cmake, &[(old, WINDOWS_ESIMD_SOURCE_NEW)])?;
    }
    let text = fs::read_to_string(&cmake)
        .with_context(|| format!("reading Windows Strata CMake file {}", cmake.display()))?
        .replace("\r\n", "\n");
    if !text.contains(WINDOWS_RUNTIME_TARGET_NEW) {
        patch_text(
            &cmake,
            &[(WINDOWS_RUNTIME_TARGET_OLD, WINDOWS_RUNTIME_TARGET_NEW)],
        )?;
    }
    fs::write(
        source
            .join("sycl")
            .join("src")
            .join("platform")
            .join("winstrint_windows_runtime.c"),
        WINDOWS_RUNTIME_SOURCE,
    )
    .context("writing the Windows Strata runtime compatibility source")?;
    Ok(())
}

fn apply_windows_patch(source: &Path) -> Result<()> {
    if !cfg!(windows) {
        return Ok(());
    }
    let marker = source.join(".winstrint-windows-sycl");
    let expected = format!("{STRATA_COMMIT}\n{WINDOWS_PATCH_VERSION}\n");
    if fs::read_to_string(&marker).ok().as_deref() == Some(expected.as_str()) {
        return Ok(());
    }
    let previous = format!("{STRATA_COMMIT}\n{PREVIOUS_WINDOWS_PATCH_VERSION}\n");
    if fs::read_to_string(&marker).ok().as_deref() == Some(previous.as_str()) {
        apply_windows_esimd_patch(source)?;
        fs::write(&marker, expected)?;
        return Ok(());
    }
    let older = format!("{STRATA_COMMIT}\n{OLDER_WINDOWS_PATCH_VERSION}\n");
    if fs::read_to_string(&marker).ok().as_deref() == Some(older.as_str()) {
        apply_windows_esimd_patch(source)?;
        fs::write(&marker, expected)?;
        return Ok(());
    }
    let legacy = format!("{STRATA_COMMIT}\n{LEGACY_WINDOWS_PATCH_VERSION}\n");
    if fs::read_to_string(&marker).ok().as_deref() == Some(legacy.as_str()) {
        patch_text(
            &source
                .join("sycl")
                .join("src")
                .join("program")
                .join("generate.cpp"),
            &[(WINDOWS_GENERATE_MEMMEM_OLD, WINDOWS_GENERATE_MEMMEM_NEW)],
        )?;
        patch_text(
            &source
                .join("sycl")
                .join("src")
                .join("program")
                .join("generate.cpp"),
            WINDOWS_GENERATE_SMALL_OLD,
        )?;
        apply_windows_esimd_patch(source)?;
        fs::write(&marker, expected)?;
        return Ok(());
    }
    patch_text(
        &source.join("sycl").join("CMakeLists.txt"),
        &[
            (
                "project(strata_sycl VERSION ${STRATA_SYCL_VERSION} LANGUAGES C CXX)\n",
                "project(strata_sycl VERSION ${STRATA_SYCL_VERSION} LANGUAGES C CXX)\nif(WIN32 AND NOT CMAKE_CXX_COMPILER_ID MATCHES \"IntelLLVM\")\n  message(FATAL_ERROR \"Windows Strata SYCL requires Intel oneAPI icx/icpx (IntelLLVM)\")\nendif()\n",
            ),
            (
                "add_compile_options(\"$<$<COMPILE_LANGUAGE:CXX>:${_sycl_flags}>\" -fp-model=precise -Wno-unused-parameter\n                    -Wno-unused-variable -Wno-deprecated-declarations)\n",
                "if(WIN32)\n  # icx/icpx on Windows use the MSVC spelling for the precise FP model.\n  set(_strata_fp_model /fp:precise)\nelse()\n  set(_strata_fp_model -fp-model=precise)\nendif()\nadd_compile_options(\"$<$<COMPILE_LANGUAGE:CXX>:${_sycl_flags}>\" ${_strata_fp_model} -Wno-unused-parameter\n                    -Wno-unused-variable -Wno-deprecated-declarations)\n",
            ),
            (
                "target_compile_options(strata_prefill PUBLIC -qmkl=sequential)\ntarget_link_options(strata_prefill PUBLIC -qmkl=sequential)\n",
                "if(WIN32)\n  set(_strata_mkl_flag /Qmkl:sequential)\nelse()\n  set(_strata_mkl_flag -qmkl=sequential)\nendif()\ntarget_compile_options(strata_prefill PUBLIC ${_strata_mkl_flag})\ntarget_link_options(strata_prefill PUBLIC ${_strata_mkl_flag})\n",
            ),
            (WINDOWS_ESIMD_SOURCE_OLD, WINDOWS_ESIMD_SOURCE_NEW),
            (WINDOWS_RUNTIME_TARGET_OLD, WINDOWS_RUNTIME_TARGET_NEW),
        ],
    )?;

    patch_text(
        &source
            .join("sycl")
            .join("src")
            .join("core")
            .join("gguf_expert_source.cpp"),
        &[
            (
                "#include <utility>\n#include <fcntl.h>\n#include <unistd.h>\n#include <dpct/dpct.hpp>\n",
                "#include <utility>\n#if defined(_WIN32)\n#define WIN32_LEAN_AND_MEAN\n#define NOMINMAX\n#include <windows.h>\n#else\n#include <fcntl.h>\n#include <unistd.h>\n#endif\n#include <dpct/dpct.hpp>\n",
            ),
            (
                "constexpr size_t kRing = 512;   // blobs alive at once: the prompt path holds a layer's worth of streamed experts\n}",
                "constexpr size_t kRing = 512;   // blobs alive at once: the prompt path holds a layer's worth of streamed experts\n#if defined(_WIN32)\nstd::wstring wide_path(const std::string& path) {\n    const int size = MultiByteToWideChar(CP_UTF8, 0, path.data(), static_cast<int>(path.size()), nullptr, 0);\n    if (size <= 0) return {};\n    std::wstring result(static_cast<size_t>(size), L'\\0');\n    MultiByteToWideChar(CP_UTF8, 0, path.data(), static_cast<int>(path.size()), result.data(), size);\n    return result;\n}\n\nbool read_at(HANDLE handle, uint64_t offset, uint8_t* dst, size_t bytes) {\n    if (bytes == 0) return true;\n    HANDLE event = CreateEventW(nullptr, TRUE, FALSE, nullptr);\n    if (event == nullptr) return false;\n    size_t done = 0;\n    bool ok = true;\n    while (done < bytes) {\n        OVERLAPPED overlapped{};\n        overlapped.hEvent = event;\n        const uint64_t at = offset + done;\n        overlapped.Offset = static_cast<DWORD>(at & 0xffffffffu);\n        overlapped.OffsetHigh = static_cast<DWORD>(at >> 32);\n        ResetEvent(event);\n        const DWORD request = static_cast<DWORD>(std::min<size_t>(bytes - done, 1u << 30));\n        DWORD got = 0;\n        if (!ReadFile(handle, dst + done, request, &got, &overlapped)) {\n            if (GetLastError() != ERROR_IO_PENDING || !GetOverlappedResult(handle, &overlapped, &got, TRUE)) {\n                ok = false;\n                break;\n            }\n        }\n        if (got == 0) {\n            ok = false;\n            break;\n        }\n        done += got;\n    }\n    CloseHandle(event);\n    return ok && done == bytes;\n}\n#endif\n}",
            ),
            (
                "    for (int fd : fds_) if (fd >= 0) ::close(fd);\n",
                "#if defined(_WIN32)\n    for (intptr_t fd : fds_) if (fd != 0) CloseHandle(reinterpret_cast<HANDLE>(fd));\n#else\n    for (intptr_t fd : fds_) if (fd >= 0) ::close(static_cast<int>(fd));\n#endif\n",
            ),
            (
                "    if (layer_fd_[i] >= 0) return fds_[(size_t) layer_fd_[i]];\n",
                "    if (layer_fd_[i] >= 0) {\n#if defined(_WIN32)\n        return 0;\n#else\n        return static_cast<int>(fds_[(size_t) layer_fd_[i]]);\n#endif\n    }\n",
            ),
            (
                "    if (names_[k] == name) { layer_fd_[i] = (int) k; return fds_[k]; }\n    const int fd = ::open(name.c_str(), O_RDONLY | O_CLOEXEC);\n    if (fd < 0) { err = \"--stream-experts: cannot open \" + name; return -1; }\n    names_.push_back(name);\n    fds_.push_back(fd);\n    layer_fd_[i] = (int) fds_.size() - 1;\n    return fd;\n",
                "    if (names_[k] == name) { layer_fd_[i] = (int) k; return 0; }\n#if defined(_WIN32)\n    const std::wstring wide = wide_path(name);\n    const HANDLE handle = wide.empty() ? INVALID_HANDLE_VALUE : CreateFileW(\n        wide.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,\n        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED, nullptr);\n    if (handle == INVALID_HANDLE_VALUE) { err = \"--stream-experts: cannot open \" + name; return -1; }\n    fds_.push_back(reinterpret_cast<intptr_t>(handle));\n#else\n    const int fd = ::open(name.c_str(), O_RDONLY | O_CLOEXEC);\n    if (fd < 0) { err = \"--stream-experts: cannot open \" + name; return -1; }\n    fds_.push_back(static_cast<intptr_t>(fd));\n#endif\n    names_.push_back(name);\n    layer_fd_[i] = (int) fds_.size() - 1;\n    return 0;\n",
            ),
            (
                "        const int fd = fds_[(size_t) layer_fd_[(size_t) (3 * layer + r)]];\n        const uint64_t src = lay.gguf_off[(size_t) (3 * layer + r)] + per[r] * (uint64_t) expert;\n        uint64_t done = 0;\n        while (done < per[r]) {\n            const ssize_t n = ::pread(fd, buf.data() + at[r] + done, (size_t) (per[r] - done), (off_t) (src + done));\n            if (n <= 0) { std::lock_guard<std::mutex> lk(mu_); where_.erase(key); ring_key_[slot] = -1; return nullptr; }\n            done += (uint64_t) n;\n        }\n",
                "        const uint64_t src = lay.gguf_off[(size_t) (3 * layer + r)] + per[r] * (uint64_t) expert;\n#if defined(_WIN32)\n        const HANDLE handle = reinterpret_cast<HANDLE>(fds_[(size_t) layer_fd_[(size_t) (3 * layer + r)]]);\n        if (!read_at(handle, src, buf.data() + at[r], (size_t) per[r])) { std::lock_guard<std::mutex> lk(mu_); where_.erase(key); ring_key_[slot] = -1; return nullptr; }\n#else\n        const int fd = static_cast<int>(fds_[(size_t) layer_fd_[(size_t) (3 * layer + r)]]);\n        uint64_t done = 0;\n        while (done < per[r]) {\n            const ssize_t n = ::pread(fd, buf.data() + at[r] + done, (size_t) (per[r] - done), (off_t) (src + done));\n            if (n <= 0) { std::lock_guard<std::mutex> lk(mu_); where_.erase(key); ring_key_[slot] = -1; return nullptr; }\n            done += (uint64_t) n;\n        }\n#endif\n",
            ),
            (
                "        const int fd = fds_[(size_t) layer_fd_[(size_t) (3 * layer + r)]];\n        const uint64_t src = lay.gguf_off[(size_t) (3 * layer + r)] + per[r] * (uint64_t) expert;\n        uint64_t done = 0;\n        while (done < per[r]) {\n            const ssize_t n = ::pread(fd, dst + at[r] + done, (size_t) (per[r] - done), (off_t) (src + done));\n            if (n <= 0) return false;\n            done += (uint64_t) n;\n        }\n",
                "        const uint64_t src = lay.gguf_off[(size_t) (3 * layer + r)] + per[r] * (uint64_t) expert;\n#if defined(_WIN32)\n        const HANDLE handle = reinterpret_cast<HANDLE>(fds_[(size_t) layer_fd_[(size_t) (3 * layer + r)]]);\n        if (!read_at(handle, src, dst + at[r], (size_t) per[r])) return false;\n#else\n        const int fd = static_cast<int>(fds_[(size_t) layer_fd_[(size_t) (3 * layer + r)]]);\n        uint64_t done = 0;\n        while (done < per[r]) {\n            const ssize_t n = ::pread(fd, dst + at[r] + done, (size_t) (per[r] - done), (off_t) (src + done));\n            if (n <= 0) return false;\n            done += (uint64_t) n;\n        }\n#endif\n",
            ),
        ],
    )?;

    patch_text(
        &source
            .join("sycl")
            .join("include")
            .join("strata")
            .join("core")
            .join("gguf_expert_source.hpp"),
        &[(
            "    std::vector<int> fds_;              ///< per distinct file name, opened once\n",
            "    std::vector<intptr_t> fds_;         ///< POSIX file descriptors or Windows HANDLE values\n",
        )],
    )?;

    patch_text(
        &source
            .join("sycl")
            .join("src")
            .join("prefill")
            .join("prefill.cpp"),
        &[
            ("#include <sys/stat.h>\n", "#include <filesystem>\n"),
            (
                "        ::mkdir(dir.c_str(), 0755);\n",
                "        std::error_code directory_error;\n        std::filesystem::create_directories(dir, directory_error);\n",
            ),
        ],
    )?;
    patch_text(
        &source
            .join("sycl")
            .join("src")
            .join("core")
            .join("pinned.dp.cpp"),
        &[(WINDOWS_PINNED_LIMIT_OLD, WINDOWS_PINNED_LIMIT_NEW)],
    )?;
    patch_text(
        &source
            .join("sycl")
            .join("include")
            .join("dpct")
            .join("util.hpp"),
        &[
            (WINDOWS_GROUP_ENTANGLE_OLD, WINDOWS_GROUP_ENTANGLE_NEW),
            (WINDOWS_GROUP_PARTITION_OLD, WINDOWS_GROUP_PARTITION_NEW),
            (WINDOWS_GROUP_ENTANGLE_OLD, WINDOWS_GROUP_ENTANGLE_NEW),
            (WINDOWS_GROUP_PARTITION_OLD, WINDOWS_GROUP_PARTITION_NEW),
        ],
    )?;
    patch_text(
        &source
            .join("sycl")
            .join("src")
            .join("kernels")
            .join("cuda")
            .join("verify_kernels.dp.cpp"),
        &[
            (WINDOWS_VERIFY_TEMPLATE_OLD, WINDOWS_VERIFY_TEMPLATE_NEW),
            (WINDOWS_VERIFY_CONDITION_OLD, WINDOWS_VERIFY_CONDITION_NEW),
        ],
    )?;
    patch_text(
        &source
            .join("sycl")
            .join("src")
            .join("program")
            .join("generate.cpp"),
        &[(WINDOWS_GENERATE_MEMMEM_OLD, WINDOWS_GENERATE_MEMMEM_NEW)],
    )?;
    patch_text(
        &source
            .join("sycl")
            .join("src")
            .join("program")
            .join("generate.cpp"),
        WINDOWS_GENERATE_SMALL_OLD,
    )?;
    fs::write(
        source
            .join("sycl")
            .join("src")
            .join("platform")
            .join("winstrint_windows_runtime.c"),
        WINDOWS_RUNTIME_SOURCE,
    )
    .context("writing the Windows Strata runtime compatibility source")?;
    fs::write(marker, expected)?;
    Ok(())
}

fn patch_text(path: &Path, replacements: &[(&str, &str)]) -> Result<()> {
    let mut text = fs::read_to_string(path)
        .with_context(|| format!("reading upstream file {}", path.display()))?
        .replace("\r\n", "\n");
    for (needle, replacement) in replacements {
        if !text.contains(needle) {
            bail!(
                "Windows Strata patch anchor is missing in {}; source revision or patch is stale",
                path.display()
            );
        }
        text = text.replacen(needle, replacement, 1);
    }
    fs::write(path, text)
        .with_context(|| format!("writing Windows patch to {}", path.display()))?;
    Ok(())
}

fn ensure_gguf_py(source: &Path) -> Result<()> {
    let llama = source.join("third_party").join("llama.cpp");
    if llama.join("gguf-py").join("gguf").is_dir() {
        return Ok(());
    }
    if llama.exists() {
        bail!(
            "managed llama.cpp checkout lacks gguf-py: {}; remove it and reprovision",
            llama.display()
        );
    }
    if let Some(parent) = llama.parent() {
        fs::create_dir_all(parent)?;
    }
    run_checked(
        OsStr::new("git"),
        &[
            os("clone"),
            os("--filter=blob:none"),
            os("--no-checkout"),
            os(LLAMA_CPP_REPOSITORY),
            llama.clone().into_os_string(),
        ],
        None,
        "cloning the pinned llama.cpp tooling checkout",
    )?;
    run_checked(
        OsStr::new("git"),
        &[
            os("fetch"),
            os("--depth=1"),
            os("origin"),
            os(LLAMA_CPP_COMMIT),
        ],
        Some(&llama),
        "fetching the pinned llama.cpp tooling revision",
    )?;
    run_checked(
        OsStr::new("git"),
        &[os("sparse-checkout"), os("init"), os("--cone")],
        Some(&llama),
        "initializing the llama.cpp sparse checkout",
    )?;
    run_checked(
        OsStr::new("git"),
        &[os("sparse-checkout"), os("set"), os("gguf-py")],
        Some(&llama),
        "selecting llama.cpp GGUF tooling",
    )?;
    run_checked(
        OsStr::new("git"),
        &[os("checkout"), os("--detach"), os(LLAMA_CPP_COMMIT)],
        Some(&llama),
        "checking out the pinned llama.cpp tooling revision",
    )?;
    Ok(())
}

fn ensure_engine(config: &AppConfig, source: &Path, rebuild: bool) -> Result<PathBuf> {
    let engine = engine_path(config);
    let marker = engine_marker(config);
    let expected_marker =
        format!("{STRATA_COMMIT}\n{WINDOWS_PATCH_VERSION}\n{WINDOWS_ENGINE_BUILD_VERSION}\n");
    let engine_ready =
        fs::read_to_string(&marker).ok().as_deref() == Some(expected_marker.as_str());
    if engine.is_file() && !rebuild && engine_ready {
        return Ok(engine);
    }
    if (rebuild || !engine_ready) && build_root(config).exists() {
        fs::remove_dir_all(build_root(config)).with_context(|| {
            format!(
                "removing the stale managed Strata build {}",
                build_root(config).display()
            )
        })?;
    }
    let setvars = find_setvars().context("locating Intel oneAPI setvars.bat")?;
    let build = build_root(config);
    fs::create_dir_all(&build)?;
    let cmake_rules = build.join("windows-cxx-rules.cmake");
    fs::write(&cmake_rules, WINDOWS_CMAKE_RULES)?;
    let script = strata_root(config).join("build-windows.cmd");
    let content = format!(
        "@echo off\r\nsetlocal\r\ncall {} intel64 vs2022\r\nif errorlevel 1 exit /b %errorlevel%\r\nwhere icx >nul 2>nul\r\nif errorlevel 1 exit /b 1\r\nwhere icpx >nul 2>nul\r\nif errorlevel 1 exit /b 1\r\ncmake -S {} -B {} -G Ninja -DCMAKE_USER_MAKE_RULES_OVERRIDE_CXX={} -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx -DCMAKE_CXX_FLAGS= -DCMAKE_CXX_FLAGS_DEBUG= -DCMAKE_CXX_FLAGS_RELEASE= -DCMAKE_CXX_FLAGS_RELWITHDEBINFO= -DCMAKE_CXX_FLAGS_MINSIZEREL= -DCMAKE_BUILD_TYPE=Release -DSTRATA_SYCL_PARITY=OFF -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_SYCL_JIT_STATELESS=ON -DSTRATA_WINDOWS_ESIMD=ON\r\nif errorlevel 1 exit /b %errorlevel%\r\ncmake --build {} --target strata --config Release\r\nexit /b %errorlevel%\r\n",
        batch_quote(&setvars),
        batch_quote(&source.join("sycl")),
        batch_quote(&build),
        batch_quote(&cmake_rules),
        batch_quote(&build),
    );
    fs::write(&script, content)?;
    println!("Building Strata native SYCL engine; build output is attached to this terminal.");
    run_cmd_status(
        &format!("call {}", batch_quote(&script)),
        source,
        "building the Strata native Windows SYCL engine",
    )?;
    if !engine.is_file() {
        let found = find_named_file(&build, "strata.exe")?;
        if found != engine {
            fs::copy(&found, &engine)
                .with_context(|| format!("copying Strata engine from {}", found.display()))?;
        }
    }
    if !engine.is_file() {
        bail!(
            "Strata build completed without producing {}",
            engine.display()
        );
    }
    fs::write(&marker, &expected_marker)
        .with_context(|| format!("writing Strata engine marker {}", marker.display()))?;
    Ok(engine)
}

fn model_manifest_revision(artifact: &ModelArtifact) -> Result<String> {
    let manifest = artifact.directory.join("manifest.json");
    let value: Value = serde_json::from_slice(
        &fs::read(&manifest)
            .with_context(|| format!("reading model manifest {}", manifest.display()))?,
    )
    .with_context(|| format!("parsing model manifest {}", manifest.display()))?;
    value
        .get("revision")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .context("model manifest has no revision")
}

fn prepare_model(
    config: &AppConfig,
    source: &Path,
    python: &Path,
    artifact: &ModelArtifact,
) -> Result<PreparedModel> {
    let first_shard = fs::canonicalize(&artifact.first_shard).with_context(|| {
        format!(
            "canonicalizing model shard {}",
            artifact.first_shard.display()
        )
    })?;
    let shards = canonical_model_shards(artifact)?;
    let ple_shard = select_ple_shard(python, source, &shards)?;
    let pack_root = strata_root(config).join("packs").join(config.model.key());
    fs::create_dir_all(&pack_root)?;
    let pack = pack_root.join("native");
    let tokenizer = pack_root.join("tokenizer");
    let marker = pack_root.join(".winstrint-prepared");
    let expected_marker = format!(
        "model_revision={}\nstrata_commit={STRATA_COMMIT}\nwindows_patch={WINDOWS_PATCH_VERSION}\n",
        model_manifest_revision(artifact)?
    );
    let marker_matches =
        fs::read_to_string(&marker).ok().as_deref() == Some(expected_marker.as_str());
    if !marker_matches {
        if pack.exists() {
            fs::remove_dir_all(&pack)
                .with_context(|| format!("removing stale Strata pack {}", pack.display()))?;
        }
        if tokenizer.exists() {
            fs::remove_dir_all(&tokenizer).with_context(|| {
                format!("removing stale Strata tokenizer {}", tokenizer.display())
            })?;
        }
    }
    if !pack.join("native_experts.txt").is_file() {
        run_python(
            python,
            [
                os("-u"),
                source.join("tools").join("iq_pack.py").into_os_string(),
                os("--gguf"),
                first_shard.clone().into_os_string(),
                os("--out"),
                pack.clone().into_os_string(),
            ],
            source,
            "creating the Strata native IQ pack",
        )?;
    }
    if !tokenizer.join("vocab.json").is_file() || !tokenizer.join("chat_template.jinja").is_file() {
        run_python(
            python,
            [
                os("-u"),
                source
                    .join("tools")
                    .join("strata_tokenizer.py")
                    .into_os_string(),
                os("--gguf"),
                first_shard.clone().into_os_string(),
                os("--out"),
                pack_root.clone().into_os_string(),
            ],
            source,
            "creating the Strata tokenizer bundle",
        )?;
    }
    let mtp_root = strata_root(config).join("mtp");
    let mtp_gguf = mtp_root.join("mtp-q2_0.gguf");
    let mtp_runtime = mtp_root.join("rt");
    if !mtp_runtime.join("experts.bin").is_file() {
        fs::create_dir_all(&mtp_root)?;
        run_python(
            python,
            [
                os("-u"),
                source.join("tools").join("mtp_fetch.py").into_os_string(),
                os("fetch"),
                os("--out"),
                mtp_root.clone().into_os_string(),
            ],
            source,
            "fetching the Strata MTP model",
        )?;
        run_python(
            python,
            [
                os("-u"),
                source.join("tools").join("mtp_pack.py").into_os_string(),
                os("--src"),
                mtp_root.clone().into_os_string(),
                os("--experts"),
                os("q2_0"),
                os("--out"),
                mtp_gguf.clone().into_os_string(),
            ],
            source,
            "packing the Strata MTP model",
        )?;
        run_python(
            python,
            [
                os("-u"),
                source.join("tools").join("mtp_rt.py").into_os_string(),
                os("--gguf"),
                mtp_gguf.clone().into_os_string(),
                os("--out"),
                mtp_runtime.clone().into_os_string(),
            ],
            source,
            "building the Strata MTP runtime",
        )?;
    }
    if !pack.join("native_experts.txt").is_file() || !mtp_runtime.join("experts.bin").is_file() {
        bail!("Strata model preparation completed without required runtime artifacts");
    }
    let profile_name = if matches!(config.model, ModelChoice::Coder) {
        "expert-profile-coder.bin"
    } else {
        "expert-profile.bin"
    };
    let profile = source.join("data").join(profile_name);
    if !profile.is_file() {
        bail!("Strata expert profile is missing: {}", profile.display());
    }
    fs::write(&marker, expected_marker)?;
    Ok(PreparedModel {
        pack,
        tokenizer,
        mtp_runtime,
        profile,
        first_shard,
        ple_shard,
    })
}

struct PreparedModel {
    pack: PathBuf,
    tokenizer: PathBuf,
    mtp_runtime: PathBuf,
    profile: PathBuf,
    first_shard: PathBuf,
    ple_shard: PathBuf,
}

fn host_memory_gib() -> Option<f64> {
    if !cfg!(windows) {
        return None;
    }
    if let Ok(output) = Command::new("powershell.exe")
        .args([
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
        ])
        .output()
    {
        if output.status.success() {
            if let Ok(bytes) = String::from_utf8_lossy(&output.stdout)
                .trim()
                .parse::<u64>()
            {
                return Some(bytes as f64 / 1_073_741_824.0);
            }
        }
    }
    let output = Command::new("cmd.exe")
        .args(["/D", "/C", "systeminfo"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    parse_systeminfo_memory_mib(&String::from_utf8_lossy(&output.stdout))
        .map(|mib| mib as f64 / 1024.0)
}

fn parse_systeminfo_memory_mib(text: &str) -> Option<u64> {
    let value = text.lines().find_map(|line| {
        let (key, value) = line.split_once(':')?;
        key.trim()
            .eq_ignore_ascii_case("Total Physical Memory")
            .then_some(value.trim())
    })?;
    let mut fields = value.split_whitespace();
    let number = fields.next()?;
    let unit = fields.next().unwrap_or("MB").to_ascii_lowercase();
    let digits: String = number.chars().filter(char::is_ascii_digit).collect();
    let amount = digits.parse::<u64>().ok()?;
    match unit.as_str() {
        "kb" => Some(amount / 1024),
        "mb" => Some(amount),
        "gb" => amount.checked_mul(1024),
        "tb" => amount.checked_mul(1024 * 1024),
        _ => None,
    }
}

fn use_streamed_kv(context: u32, host_ram_gib: Option<f64>) -> bool {
    if context < 65_536 {
        return false;
    }
    let kv_gib = context as f64 * 13.0 * 1056.0 / 1_000_000_000.0;
    host_ram_gib.is_some_and(|ram| ram >= kv_gib + 6.0)
}

fn write_server_config(
    config: &AppConfig,
    source: &Path,
    engine: &Path,
    model: &PreparedModel,
    alchemist: bool,
    arc140t: bool,
) -> Result<PathBuf> {
    let directory = strata_root(config).join("configs");
    fs::create_dir_all(&directory)?;
    let path = directory.join(format!("{}.json", config.model.key()));
    let log = log_root(config).join("strata-engine.log");
    fs::create_dir_all(log_root(config))?;
    let reserve = if arc140t {
        "2048"
    } else if config.context <= 32_768 {
        "1024"
    } else {
        "2048"
    };
    let prefill = if config.context > 32_768 {
        "4096"
    } else {
        "auto"
    };
    let host_ram_gib = host_memory_gib();
    let streamed_kv = use_streamed_kv(config.context, host_ram_gib);
    if config.context >= 65_536 && !streamed_kv {
        eprintln!(
            "Strata: keeping KV resident for {} tokens; host RAM check did not allow --kv-resident 32768",
            config.context
        );
    }
    let mut environment = Map::new();
    for (key, value) in [
        ("ONEAPI_DEVICE_SELECTOR", "level_zero:0"),
        ("SYCL_CACHE_PERSISTENT", "0"),
        ("ZES_ENABLE_SYSMAN", "1"),
        ("STRATA_VERIFY_DEVICE_PLAN", "1"),
        ("STRATA_VERIFY_NO_HOST", "1"),
        ("STRATA_STAGER_THREADS", "12"),
        ("STRATA_WARM_GRAPHS", "0"),
    ] {
        environment.insert(key.to_string(), json!(value));
    }
    if arc140t {
        // Arc 140T is an integrated 8-Xe-core GPU with shared system memory.
        // Map the GGUF expert spans instead of committing a 23 GiB resident
        // arena; this leaves Windows enough RAM to keep serving requests.
        environment.insert("STRATA_VERIFY_DEVICE_PLAN".into(), json!("0"));
        environment.insert("STRATA_VERIFY_EAGER".into(), json!("1"));
        // Shared-memory verification can take longer than the stock 20-second
        // ring timeout while the iGPU and CPU pool drain a window.
        environment.insert("STRATA_RING_TIMEOUT_S".into(), json!("120"));
        environment.insert("STRATA_VERIFY_NO_HOST".into(), json!("0"));
    }
    let mut args = vec![
        "--pack".to_string(),
        path_string(&model.pack),
        "--native".to_string(),
        path_string(&model.first_shard),
        "--ple-gguf".to_string(),
        path_string(&model.ple_shard),
        "--expert-profile".to_string(),
        path_string(&model.profile),
        "--expert-cache".to_string(),
        "auto".to_string(),
        "--vram-reserve-mib".to_string(),
        reserve.to_string(),
        "--prefill".to_string(),
        prefill.to_string(),
        "--spec".to_string(),
        "4".to_string(),
    ];
    if arc140t {
        args.push("--mmap-experts".to_string());
    }
    if !arc140t {
        args.extend([
            "--spec-min-p".to_string(),
            "0.5".to_string(),
            "--mtp".to_string(),
            path_string(&model.mtp_runtime),
        ]);
    }
    args.extend(["--kv".to_string(), "int8".to_string()]);
    if alchemist {
        environment.insert("IGC_EnableDPEmulation".into(), json!("1"));
        environment.insert("OverrideDefaultFP64Settings".into(), json!("1"));
        environment.insert("NEOReadDebugKeys".into(), json!("1"));
    }
    if streamed_kv {
        args.extend(["--kv-resident".to_string(), "32768".to_string()]);
    }
    if arc140t {
        // Keep expert misses on the mapped GGUF file tier; the Windows
        // Level Zero pinned-host mirror path is unstable on this iGPU.
        args.extend(["--pcie-frac".to_string(), "0".to_string()]);
    }
    args.extend(["--max-context".to_string(), config.context.to_string()]);
    let value = json!({
        "backend": "sycl",
        "exe": path_string(engine),
        "args": args,
        "cwd": path_string(source),
        "tokenizer": path_string(&model.tokenizer),
        "model_name": config.model.key(),
        "arch": "qwen4exp",
        "reasoning_effort": "high",
        "log": path_string(&log),
        "gpu": 0,
        "env": environment
    });
    fs::write(&path, serde_json::to_vec_pretty(&value)?)?;
    Ok(path)
}

fn write_runtime_script(
    config: &AppConfig,
    source: &Path,
    python: &Path,
    setvars: &Path,
    server_config: &Path,
    engine: &Path,
) -> Result<PathBuf> {
    let directory = strata_root(config);
    fs::create_dir_all(&directory)?;
    let script = directory.join("run-windows.cmd");
    let engine_parent = engine
        .parent()
        .context("Strata engine has no parent directory")?;
    let content = format!(
        "@echo off\r\nsetlocal\r\ncall {} intel64 vs2022\r\nif errorlevel 1 exit /b %errorlevel%\r\nset \"PATH={};%PATH%\"\r\n{} -u {} --engine strata --config {} --host 127.0.0.1 --port {}\r\nexit /b %errorlevel%\r\n",
        batch_quote(setvars),
        engine_parent.display(),
        batch_quote(python),
        batch_quote(&source.join("serve").join("server.py")),
        batch_quote(server_config),
        config.port,
    );
    fs::write(&script, content)?;
    Ok(script)
}
fn canonical_model_shards(artifact: &ModelArtifact) -> Result<Vec<PathBuf>> {
    let mut shards = artifact.shards.clone();
    for shard in &mut shards {
        *shard = fs::canonicalize(&*shard)
            .with_context(|| format!("canonicalizing model shard {}", shard.display()))?;
    }
    if shards.len() < 2 {
        bail!(
            "Strata streaming launch needs at least two GGUF shards in {}",
            artifact.directory.display()
        );
    }
    Ok(shards)
}

fn select_ple_shard(python: &Path, source: &Path, shards: &[PathBuf]) -> Result<PathBuf> {
    let script = r#"
import pathlib
import sys

sys.path.insert(0, sys.argv[1])
from gguf_reader import GGUFFile

for raw in sys.argv[2:]:
    path = pathlib.Path(raw)
    if any(t.name.endswith("per_layer_token_embd.weight") for t in GGUFFile(path).tensors):
        print(path)
        raise SystemExit(0)
raise SystemExit("no GGUF shard contains per_layer_token_embd.weight")
"#;
    let mut args = vec![
        os("-c"),
        OsString::from(script),
        source.join("tools").into_os_string(),
    ];
    args.extend(shards.iter().map(|path| path.clone().into_os_string()));
    let output = run_python_output(python, args, source, "locating the GGUF PLE shard")?;
    let selected = String::from_utf8_lossy(&output.stdout)
        .lines()
        .find(|line| !line.trim().is_empty())
        .map(PathBuf::from)
        .context("GGUF PLE detector returned no shard")?;
    if shards.iter().any(|path| path == &selected) {
        Ok(selected)
    } else {
        bail!(
            "GGUF PLE detector returned an unexpected path: {}",
            selected.display()
        );
    }
}

fn find_named_file(root: &Path, name: &str) -> Result<PathBuf> {
    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(&directory)
            .with_context(|| format!("reading build directory {}", directory.display()))?
        {
            let path = entry?.path();
            if path.is_dir() {
                pending.push(path);
            } else if path.file_name().is_some_and(|item| item == name) {
                return Ok(path);
            }
        }
    }
    bail!("could not find {name} below {}", root.display())
}

fn create_virtualenv(config: &AppConfig, program: &OsString, prefix: &[OsString]) -> Result<()> {
    if diagnostics::command_exists("uv") {
        let mut args = vec![os("venv"), os("--python")];
        if prefix.is_empty() {
            args.push(program.clone());
        } else if prefix == [os("-3.12")] {
            args.push(os("3.12"));
        } else {
            args.push(program.clone());
        }
        args.push(python_venv_arg(config));
        if run_checked(
            OsStr::new("uv"),
            &args,
            None,
            "creating the managed Strata Python environment with uv",
        )
        .is_ok()
        {
            return Ok(());
        }
    }
    let mut args = prefix.to_vec();
    args.extend([os("-m"), os("venv"), python_venv_arg(config)]);
    run_checked(
        program,
        &args,
        None,
        "creating the managed Strata Python environment",
    )?;
    Ok(())
}

fn find_bootstrap_python() -> Result<(OsString, Vec<OsString>)> {
    let candidates = [
        (OsString::from("python"), Vec::new()),
        (OsString::from("py"), vec![os("-3.12")]),
    ];
    for (program, prefix) in candidates {
        let mut args = prefix.clone();
        args.extend([
            os("-c"),
            os("import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"),
        ]);
        if run_checked(&program, &args, None, "checking the bootstrap Python").is_ok() {
            return Ok((program, prefix));
        }
    }
    bail!("Python 3.10 or newer is required to create the managed Strata environment")
}

fn python_venv_arg(config: &AppConfig) -> OsString {
    virtualenv_root(config).into_os_string()
}

fn install_requirements(python: &Path, requirements: &Path, cwd: &Path) -> Result<()> {
    if diagnostics::command_exists("uv") {
        run_checked(
            OsStr::new("uv"),
            &[
                os("pip"),
                os("install"),
                os("--python"),
                python.to_path_buf().into_os_string(),
                os("--requirement"),
                requirements.to_path_buf().into_os_string(),
            ],
            Some(cwd),
            "installing pinned Strata Python requirements with uv",
        )?;
        return Ok(());
    }
    run_python(
        python,
        [
            os("-m"),
            os("pip"),
            os("install"),
            os("--requirement"),
            requirements.to_path_buf().into_os_string(),
        ],
        cwd,
        "installing pinned Strata Python requirements",
    )
}

fn run_python<I>(python: &Path, args: I, cwd: &Path, label: &str) -> Result<()>
where
    I: IntoIterator<Item = OsString>,
{
    let mut command = Command::new(python);
    command.args(args).current_dir(cwd);
    command.env(
        "STRATA_GGUF_PY",
        cwd.join("third_party").join("llama.cpp").join("gguf-py"),
    );
    let status = command
        .status()
        .with_context(|| format!("{label}: launching {}", python.display()))?;
    if !status.success() {
        bail!("{label} failed with status {status}");
    }
    Ok(())
}
fn run_python_output(
    python: &Path,
    args: Vec<OsString>,
    cwd: &Path,
    label: &str,
) -> Result<std::process::Output> {
    let mut command = Command::new(python);
    command.args(args).current_dir(cwd).env(
        "STRATA_GGUF_PY",
        cwd.join("third_party").join("llama.cpp").join("gguf-py"),
    );
    let output = command
        .output()
        .with_context(|| format!("{label}: launching {}", python.display()))?;
    if !output.status.success() {
        bail!(
            "{label} failed with status {}: {}",
            output.status,
            command_output(&output)
        );
    }
    Ok(output)
}

fn run_checked(
    program: &OsStr,
    args: &[OsString],
    cwd: Option<&Path>,
    label: &str,
) -> Result<String> {
    let mut command = Command::new(program);
    command.args(args);
    if let Some(cwd) = cwd {
        command.current_dir(cwd);
    }
    let output = command
        .output()
        .with_context(|| format!("{label}: launching {}", program.to_string_lossy()))?;
    if !output.status.success() {
        bail!(
            "{label} failed with status {}: {}",
            output.status,
            command_output(&output)
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

fn write_cmd_script(command_text: &str) -> Result<PathBuf> {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let path = env::temp_dir().join(format!("winstrint-cmd-{}-{stamp}.cmd", std::process::id()));
    let content = format!("@echo off\r\n{command_text}\r\nexit /b %errorlevel%\r\n");
    fs::write(&path, content).with_context(|| {
        format!(
            "writing temporary Windows command script {}",
            path.display()
        )
    })?;
    Ok(path)
}

fn run_cmd_capture(command_text: &str, cwd: &Path, label: &str) -> Result<std::process::Output> {
    let script = write_cmd_script(command_text)?;
    let result = Command::new("cmd.exe")
        .args(["/D", "/S", "/C", "call"])
        .arg(&script)
        .current_dir(cwd)
        .output()
        .with_context(|| format!("{label}: launching cmd.exe"));
    let _ = fs::remove_file(&script);
    result
}

fn run_cmd_allow_failure(
    command_text: &str,
    cwd: &Path,
    label: &str,
) -> Result<std::process::Output> {
    run_cmd_capture(command_text, cwd, label)
}

fn run_cmd_output(command_text: &str, cwd: &Path, label: &str) -> Result<std::process::Output> {
    let output = run_cmd_capture(command_text, cwd, label)?;
    if !output.status.success() {
        bail!(
            "{label} failed with status {}: {}",
            output.status,
            command_output(&output)
        );
    }
    Ok(output)
}

fn run_cmd_status(command_text: &str, cwd: &Path, label: &str) -> Result<()> {
    let script = write_cmd_script(command_text)?;
    let status = Command::new("cmd.exe")
        .args(["/D", "/S", "/C", "call"])
        .arg(&script)
        .current_dir(cwd)
        .status()
        .with_context(|| format!("{label}: launching cmd.exe"));
    let _ = fs::remove_file(&script);
    let status = status?;
    if !status.success() {
        bail!("{label} failed with status {status}");
    }
    Ok(())
}

fn command_output(output: &std::process::Output) -> String {
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    format!("{} {}", stdout.trim(), stderr.trim())
        .trim()
        .to_string()
}

fn batch_quote(path: &Path) -> String {
    format!("\"{}\"", path.display())
}
fn path_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

fn os(value: &str) -> OsString {
    OsString::from(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streamed_kv_requires_context_and_ram_headroom() {
        assert!(!use_streamed_kv(32_768, Some(64.0)));
        assert!(use_streamed_kv(131_072, Some(32.0)));
        assert!(!use_streamed_kv(131_072, Some(6.0)));
        assert!(!use_streamed_kv(131_072, None));
    }

    #[test]
    fn systeminfo_memory_parser_handles_windows_units() {
        assert_eq!(
            parse_systeminfo_memory_mib("Total Physical Memory: 32,189 MB"),
            Some(32_189)
        );
        assert_eq!(
            parse_systeminfo_memory_mib("Total Physical Memory: 64 GB"),
            Some(65_536)
        );
        assert_eq!(
            parse_systeminfo_memory_mib("Available Physical Memory: 8,000 MB"),
            None
        );
    }

    #[test]
    fn generated_config_keeps_model_template_and_runtime_defaults() {
        let root = env::temp_dir().join(format!(
            "winstrint-strata-config-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        let config = AppConfig {
            data_dir: root.clone(),
            model_dir: root.join("models"),
            runtime_dir: root.join("runtime"),
            queue_path: root.join("queue.json"),
            backend: crate::config::BackendPreference::Strata,
            model: ModelChoice::Coder,
            model_path: None,
            port: 8080,
            context: 4096,
            decode_profile: crate::config::DecodeProfile::Auto,
            force: false,
            no_download: true,
            yes: true,
        };
        let model = PreparedModel {
            pack: PathBuf::from("pack"),
            tokenizer: PathBuf::from("tokenizer"),
            mtp_runtime: PathBuf::from("mtp"),
            profile: PathBuf::from("profile"),
            first_shard: PathBuf::from("first.gguf"),
            ple_shard: PathBuf::from("second.gguf"),
        };
        let path = write_server_config(
            &config,
            Path::new("source"),
            Path::new("strata.exe"),
            &model,
            false,
            false,
        )
        .expect("config generation");
        let value: Value = serde_json::from_slice(&fs::read(&path).expect("generated config"))
            .expect("valid JSON");
        assert_eq!(value["arch"], "qwen4exp");
        assert_eq!(value["reasoning_effort"], "high");
        assert_eq!(value["model_name"], "coder-iq1_m");
        assert_eq!(value["tokenizer"], "tokenizer");
        assert!(value.get("sampling").is_none());
        assert_eq!(value["env"]["SYCL_CACHE_PERSISTENT"], "0");
        assert_eq!(value["env"]["STRATA_WARM_GRAPHS"], "0");
        let args = value["args"]
            .as_array()
            .expect("args array")
            .iter()
            .map(|item| item.as_str().expect("string arg"))
            .collect::<Vec<_>>();
        assert!(args.windows(2).any(|pair| pair == ["--kv", "int8"]));
        assert!(!args.contains(&"--kv-resident"));
        assert!(args.windows(2).any(|pair| pair == ["--prefill", "auto"]));
        assert!(!args.contains(&"--mmap-experts"));
        let arc_path = write_server_config(
            &config,
            Path::new("source"),
            Path::new("strata.exe"),
            &model,
            false,
            true,
        )
        .expect("Arc 140T config generation");
        let arc_value: Value =
            serde_json::from_slice(&fs::read(arc_path).expect("Arc 140T generated config"))
                .expect("valid Arc 140T JSON");
        assert_eq!(arc_value["env"]["STRATA_VERIFY_DEVICE_PLAN"], "0");
        assert_eq!(arc_value["env"]["STRATA_VERIFY_EAGER"], "1");
        assert_eq!(arc_value["env"]["STRATA_RING_TIMEOUT_S"], "120");
        assert_eq!(arc_value["env"]["STRATA_VERIFY_NO_HOST"], "0");
        let arc_args = arc_value["args"]
            .as_array()
            .expect("Arc 140T args array")
            .iter()
            .map(|item| item.as_str().expect("Arc 140T string arg"))
            .collect::<Vec<_>>();
        assert!(!arc_args.contains(&"--stream-experts"));
        assert!(arc_args.contains(&"--mmap-experts"));
        assert!(!arc_args.contains(&"--mtp"));
        assert!(arc_args.windows(2).any(|pair| pair == ["--pcie-frac", "0"]));
        let _ = fs::remove_dir_all(root);
    }
    #[test]
    fn prepared_model_ignores_unlisted_gguf_files() {
        let root = env::temp_dir().join(format!("winstrint-strata-shards-{}", std::process::id()));
        fs::create_dir_all(&root).expect("test directory");
        let first = root.join("model-00001.gguf");
        let second = root.join("model-00002.gguf");
        let stale = root.join("stale-00003.gguf");
        for path in [&first, &second, &stale] {
            fs::write(path, b"fixture").expect("test shard");
        }
        let expected = vec![
            fs::canonicalize(&first).expect("first shard"),
            fs::canonicalize(&second).expect("second shard"),
        ];
        let artifact = ModelArtifact {
            directory: root.clone(),
            first_shard: first.clone(),
            shards: vec![first, second],
            supports_moe_cache: false,
        };
        let shards = canonical_model_shards(&artifact).expect("manifest shards");
        assert_eq!(shards, expected);
        assert!(!shards.contains(&fs::canonicalize(stale).expect("stale shard")));
        let _ = fs::remove_dir_all(root);
    }
}
