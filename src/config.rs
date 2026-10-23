use std::env;
use std::path::PathBuf;

use anyhow::Result;
use clap::{Args, Parser, Subcommand, ValueEnum};

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum BackendPreference {
    Auto,
    Cpu,
    Sycl,
    Vulkan,
    Strata,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum ModelChoice {
    Coder,
    Q2_0,
    Iq2Xs,
    Iq3Xxs,
    Iq3S,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum StrataTarget {
    Auto,
    A770,
    #[value(name = "arc140t", alias = "arc-140t")]
    Arc140T,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum DecodeProfile {
    Auto,
    #[value(name = "no-kv")]
    NoKv,
    #[value(name = "q8-kv")]
    Q8Kv,
    #[value(name = "q4-kv")]
    Q4Kv,
    #[value(name = "q8-no-op")]
    Q8NoOp,
    #[value(name = "q8-ngram")]
    Q8Ngram,
}

impl DecodeProfile {
    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::NoKv => "no-kv",
            Self::Q8Kv => "q8-kv",
            Self::Q4Kv => "q4-kv",
            Self::Q8NoOp => "q8-no-op",
            Self::Q8Ngram => "q8-ngram",
        }
    }
}

#[derive(Args, Debug)]
pub struct DecodeTuneArgs {
    #[arg(long, value_name = "PATH", required = true)]
    pub prompt_file: PathBuf,
    #[arg(long, conflicts_with = "profiles")]
    pub compare: bool,
    #[arg(long, value_enum, value_delimiter = ',', value_name = "PROFILE,...")]
    pub profiles: Option<Vec<DecodeProfile>>,
    #[arg(long, default_value_t = 192)]
    pub max_tokens: u32,
    #[arg(long, default_value_t = 3600)]
    pub timeout_secs: u64,
    #[arg(long, value_name = "PATH")]
    pub output: Option<PathBuf>,
    #[arg(long)]
    pub json: bool,
}

#[derive(Args, Debug)]
pub struct RebarUefiArgs {
    #[arg(
        long,
        value_name = "EXPONENT",
        value_parser = parse_rebar_state,
        conflicts_with_all = ["native_status", "native_enable"],
        help = "Write ReBarState: 0 disables, 20 is 1 TiB, 32 is unlimited"
    )]
    pub set: Option<u8>,
    #[arg(
        long,
        help = "Confirm that Above 4G Decoding is enabled; Windows cannot reliably read the BIOS toggle"
    )]
    pub confirm_4g_decoding: bool,
    #[arg(
        long,
        help = "Confirm that CSM is disabled; UEFI boot is checked separately"
    )]
    pub confirm_csm_disabled: bool,
    #[arg(
        long,
        conflicts_with_all = ["set", "download", "native_enable"],
        help = "Read native PchSetup and SaSetup ReBAR-related values"
    )]
    pub native_status: bool,
    #[arg(
        long,
        conflicts_with_all = ["set", "download", "native_status"],
        help = "Enable native firmware ReBAR through the PchSetup NVRAM byte"
    )]
    pub native_enable: bool,
    #[arg(
        long,
        requires = "native_enable",
        help = "Confirm the one-byte native PchSetup NVRAM change and reboot requirement"
    )]
    pub confirm_native_rebar: bool,
    #[arg(
        long,
        conflicts_with_all = ["set", "native_status", "native_enable"],
        help = "Download and verify official ReBarUEFI 0.3 assets"
    )]
    pub download: bool,
    #[arg(
        long,
        value_name = "PATH",
        help = "Asset or backup directory (default: command-specific subdirectory under <data-dir>)"
    )]
    pub workspace: Option<PathBuf>,
}

#[derive(Args, Debug)]
pub struct FirmwareInspectArgs {
    #[arg(long, value_name = "PATH", required = true)]
    pub image: PathBuf,
    #[arg(long, value_name = "PATH")]
    pub output: Option<PathBuf>,
    #[arg(long)]
    pub json: bool,
}

#[derive(Args, Debug)]
pub struct BiosDowngradeArgs {
    #[arg(long, value_name = "PATH", required = true)]
    pub package: PathBuf,
    #[arg(
        long,
        value_name = "PATH",
        help = "Persistent staging directory (default: <data-dir>/bios-downgrade)"
    )]
    pub workspace: Option<PathBuf>,
    #[arg(
        long,
        help = "Copy the verified package and manifest without launching it"
    )]
    pub prepare: bool,
    #[arg(
        long,
        help = "Launch the verified Lenovo updater after all explicit safety confirmations"
    )]
    pub execute: bool,
    #[arg(long, requires = "execute")]
    pub confirm_downgrade: bool,
    #[arg(long, requires = "execute")]
    pub confirm_bitlocker_recovery: bool,
    #[arg(long, requires = "execute")]
    pub confirm_ac_power: bool,
    #[arg(long, requires = "execute")]
    pub confirm_recovery_path: bool,
    #[arg(long)]
    pub json: bool,
}

#[derive(Args, Clone, Debug)]
pub struct CommonArgs {
    #[arg(long, value_enum, default_value = "auto")]
    pub backend: BackendPreference,
    #[arg(long, value_enum, default_value = "coder")]
    pub model: ModelChoice,
    #[arg(
        long,
        value_name = "PATH",
        help = "Use an existing GGUF file instead of the managed model cache"
    )]
    pub model_path: Option<PathBuf>,
    #[arg(long, value_name = "PATH")]
    pub data_dir: Option<PathBuf>,
    #[arg(long, value_name = "PATH")]
    pub queue_path: Option<PathBuf>,
    #[arg(long, default_value_t = 8080)]
    pub port: u16,
    #[arg(long, default_value_t = 4096)]
    pub context: u32,
    #[arg(
        long,
        value_enum,
        default_value = "auto",
        help = "llama.cpp decode profile; auto uses q8 KV on a discrete A770"
    )]
    pub decode_profile: DecodeProfile,
    #[arg(long)]
    pub force: bool,
    #[arg(long)]
    pub no_download: bool,
    #[arg(long, short = 'y')]
    pub yes: bool,
}

impl ModelChoice {
    pub fn key(self) -> &'static str {
        match self {
            Self::Coder => "coder-iq1_m",
            Self::Q2_0 => "base-q2_0",
            Self::Iq2Xs => "base-iq2_xs",
            Self::Iq3Xxs => "base-iq3_xxs",
            Self::Iq3S => "base-iq3_s",
        }
    }
}

#[derive(Subcommand, Debug)]
pub enum Command {
    #[command(about = "Inspect Windows, Intel Arc, runtime, and GPU activity")]
    Check,
    #[command(about = "Discover and download the selected Hugging Face model")]
    Download,
    #[command(about = "Show the shared A770/GPU queue")]
    QueueStatus,
    #[command(about = "Acquire the shared queue and hold it until Enter")]
    QueueAcquire {
        #[arg(long, default_value = "manual")]
        purpose: String,
    },
    #[command(about = "Release the active queue ticket")]
    QueueRelease {
        #[arg(long)]
        ticket: Option<String>,
    },
    #[command(about = "Run a benchmark only after acquiring the GPU queue ticket")]
    Ticketed {
        #[arg(long, default_value = "benchmark")]
        purpose: String,
        #[arg(last = true, required = true)]
        command: Vec<String>,
    },
    #[command(about = "Inspect the managed Windows Strata SYCL environment")]
    StrataEnv,
    #[command(about = "Clone, patch, build, and provision the managed Windows Strata SYCL engine")]
    StrataProvision {
        #[arg(long, value_enum, default_value = "auto")]
        target: StrataTarget,
        #[arg(long)]
        rebuild: bool,
        #[arg(long)]
        skip_probe: bool,
    },
    #[command(about = "Probe the managed Windows Strata SYCL engine")]
    StrataProbe,
    #[command(about = "Read-only scan of a BIOS image for IFR menu evidence")]
    FirmwareInspect(FirmwareInspectArgs),
    #[command(about = "Validate and optionally launch the signed Lenovo BIOS downgrade package")]
    BiosDowngrade(BiosDowngradeArgs),
    #[command(about = "Show Strata source, Windows patch, and toolchain provenance")]
    StrataStatus {
        #[arg(
            long,
            help = "Run the oneAPI/CMake/Ninja preflight in addition to printing paths"
        )]
        verify: bool,
    },
    #[command(about = "Start the selected model and open the terminal chat")]
    Chat,
    #[command(about = "Benchmark decode profiles and recommend the fastest one")]
    TuneDecode(DecodeTuneArgs),
    #[command(about = "Inspect or set ReBarUEFI state; never flashes or patches BIOS firmware")]
    RebarUefi(RebarUefiArgs),
}

#[derive(Parser, Debug)]
#[command(
    name = "winstrint",
    version,
    about = "WINSTRINT Intel Arc model launcher"
)]
pub struct Cli {
    #[command(flatten)]
    pub common: CommonArgs,
    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Clone, Debug)]
pub struct AppConfig {
    pub data_dir: PathBuf,
    pub model_dir: PathBuf,
    pub runtime_dir: PathBuf,
    pub queue_path: PathBuf,
    pub backend: BackendPreference,
    pub model: ModelChoice,
    pub model_path: Option<PathBuf>,
    pub port: u16,
    pub context: u32,
    pub decode_profile: DecodeProfile,
    pub force: bool,
    pub no_download: bool,
    pub yes: bool,
}

impl AppConfig {
    pub fn from_cli(cli: &Cli) -> Result<Self> {
        let data_dir = cli
            .common
            .data_dir
            .clone()
            .or_else(default_data_dir)
            .unwrap_or_else(|| PathBuf::from(".winstrint"));
        let queue_path = cli
            .common
            .queue_path
            .clone()
            .or_else(|| env::var_os("WINSTRINT_A770_QUEUE").map(PathBuf::from))
            .unwrap_or_else(|| data_dir.join("a770-queue.json"));
        Ok(Self {
            model_dir: data_dir.join("models"),
            runtime_dir: data_dir.join("runtime"),
            data_dir,
            queue_path,
            backend: cli.common.backend,
            model: cli.common.model,
            model_path: cli.common.model_path.clone(),
            port: cli.common.port,
            context: cli.common.context,
            decode_profile: cli.common.decode_profile,
            force: cli.common.force,
            no_download: cli.common.no_download,
            yes: cli.common.yes,
        })
    }

    pub fn ensure_dirs(&self) -> Result<()> {
        std::fs::create_dir_all(&self.data_dir)?;
        std::fs::create_dir_all(&self.model_dir)?;
        std::fs::create_dir_all(&self.runtime_dir)?;
        if let Some(parent) = self.queue_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        Ok(())
    }
}

fn default_data_dir() -> Option<PathBuf> {
    dirs::data_local_dir().map(|path| path.join("WINSTRINT"))
}

fn parse_rebar_state(value: &str) -> std::result::Result<u8, String> {
    let parsed = value
        .parse::<u8>()
        .map_err(|_| format!("ReBarState must be an integer from 0 to 32, got {value:?}"))?;
    if parsed > 32 {
        return Err(format!(
            "ReBarState must be an integer from 0 to 32, got {parsed}"
        ));
    }
    Ok(parsed)
}
