use std::env;
use std::path::PathBuf;

use anyhow::{Result, bail};
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
pub enum ToggleMode {
    Auto,
    On,
    Off,
}

impl ToggleMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::On => "on",
            Self::Off => "off",
        }
    }
}

impl Default for ToggleMode {
    fn default() -> Self {
        Self::Auto
    }
}

#[derive(Clone, Debug, Default)]
pub struct RuntimeTuning {
    pub flash_attn: ToggleMode,
    pub op_offload: ToggleMode,
    pub threads: Option<usize>,
    pub threads_batch: Option<usize>,
    pub batch_size: Option<usize>,
    pub ubatch_size: Option<usize>,
    pub gpu_layers: Option<usize>,
    pub cpu_moe: bool,
    pub no_host: bool,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum ReasoningEffort {
    #[value(name = "none", alias = "off")]
    None,
    Low,
    Medium,
    High,
}

impl ReasoningEffort {
    pub fn label(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
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
    #[arg(long, default_value_t = 1, value_name = "N")]
    pub repeats: u32,
    #[arg(long, default_value_t = 1, value_name = "N")]
    pub warmup: u32,
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
    #[arg(
        long,
        value_enum,
        default_value = "auto",
        help = "Flash Attention mode for llama.cpp"
    )]
    pub flash_attn: ToggleMode,
    #[arg(
        long,
        value_enum,
        default_value = "auto",
        help = "Host operation offload mode for llama.cpp"
    )]
    pub op_offload: ToggleMode,
    #[arg(long, value_name = "N", help = "Override llama.cpp generation threads")]
    pub threads: Option<usize>,
    #[arg(long, value_name = "N", help = "Override llama.cpp prompt threads")]
    pub threads_batch: Option<usize>,
    #[arg(long, value_name = "N", help = "Override llama.cpp logical batch size")]
    pub batch_size: Option<usize>,
    #[arg(
        long,
        value_name = "N",
        help = "Override llama.cpp physical batch size"
    )]
    pub ubatch_size: Option<usize>,
    #[arg(long, value_name = "N", help = "Override llama.cpp GPU layer count")]
    pub gpu_layers: Option<usize>,
    #[arg(long, help = "Keep mixture-of-experts weights on the CPU")]
    pub cpu_moe: bool,
    #[arg(long, help = "Bypass host buffers for Vulkan memory experiments")]
    pub no_host: bool,
    #[arg(
        long,
        value_enum,
        help = "Set the default reasoning budget; none is lowest latency"
    )]
    pub reasoning_effort: Option<ReasoningEffort>,
    #[arg(
        long,
        help = "Prefer Strata's direct SSD expert streaming and throughput path"
    )]
    pub speedup: bool,
    #[arg(
        long,
        help = "Enable Strata MTP on Arc 140T; disabled by default for safety"
    )]
    pub strata_mtp: bool,
    #[arg(long, value_name = "N", help = "Override Strata speculative window")]
    pub strata_spec: Option<u32>,
    #[arg(
        long,
        value_name = "P",
        help = "Override Strata speculative minimum probability"
    )]
    pub strata_spec_min_p: Option<f32>,
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
    #[command(about = "Inspect OS, Intel Arc, runtime, and GPU activity")]
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
    pub runtime_tuning: RuntimeTuning,
    pub reasoning_effort: Option<ReasoningEffort>,
    pub strata_mtp: bool,
    pub strata_spec: Option<u32>,
    pub strata_spec_min_p: Option<f32>,
    pub speedup: bool,
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
        let runtime_tuning = RuntimeTuning {
            flash_attn: cli.common.flash_attn,
            op_offload: cli.common.op_offload,
            threads: cli.common.threads,
            threads_batch: cli.common.threads_batch,
            batch_size: cli.common.batch_size,
            ubatch_size: cli.common.ubatch_size,
            gpu_layers: cli.common.gpu_layers,
            cpu_moe: cli.common.cpu_moe,
            no_host: cli.common.no_host,
        };
        for (name, value) in [
            ("threads", runtime_tuning.threads),
            ("threads-batch", runtime_tuning.threads_batch),
            ("batch-size", runtime_tuning.batch_size),
            ("ubatch-size", runtime_tuning.ubatch_size),
        ] {
            if value == Some(0) {
                bail!("--{name} must be greater than zero");
            }
        }
        if cli.common.strata_spec == Some(0) {
            bail!("--strata-spec must be greater than zero");
        }
        if let Some(value) = cli.common.strata_spec_min_p
            && (!value.is_finite() || !(0.0..=1.0).contains(&value))
        {
            bail!("--strata-spec-min-p must be finite and between 0 and 1");
        }
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
            runtime_tuning,
            reasoning_effort: cli.common.reasoning_effort,
            strata_mtp: cli.common.strata_mtp,
            strata_spec: cli.common.strata_spec,
            strata_spec_min_p: cli.common.strata_spec_min_p,
            speedup: cli.common.speedup,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speedup_flag_reaches_app_config() {
        let cli = Cli::try_parse_from([
            "winstrint",
            "--speedup",
            "--reasoning-effort",
            "none",
            "--data-dir",
            "speedup-test",
            "chat",
        ])
        .expect("speedup CLI");
        let config = AppConfig::from_cli(&cli).expect("app config");
        assert!(config.speedup);
        assert_eq!(config.reasoning_effort, Some(ReasoningEffort::None));
    }
    #[test]
    fn runtime_tuning_flags_reach_app_config() {
        let cli = Cli::try_parse_from([
            "winstrint",
            "--flash-attn",
            "on",
            "--op-offload",
            "off",
            "--threads",
            "12",
            "--threads-batch",
            "16",
            "--batch-size",
            "8192",
            "--ubatch-size",
            "2048",
            "--gpu-layers",
            "1",
            "--cpu-moe",
            "--no-host",
            "--strata-mtp",
            "--strata-spec",
            "3",
            "--strata-spec-min-p",
            "0.5",
            "chat",
        ])
        .expect("runtime tuning CLI");
        let config = AppConfig::from_cli(&cli).expect("app config");
        assert_eq!(config.runtime_tuning.flash_attn, ToggleMode::On);
        assert_eq!(config.runtime_tuning.op_offload, ToggleMode::Off);
        assert_eq!(config.runtime_tuning.threads, Some(12));
        assert_eq!(config.runtime_tuning.threads_batch, Some(16));
        assert_eq!(config.runtime_tuning.batch_size, Some(8192));
        assert_eq!(config.runtime_tuning.ubatch_size, Some(2048));
        assert_eq!(config.runtime_tuning.gpu_layers, Some(1));
        assert!(config.runtime_tuning.cpu_moe);
        assert!(config.runtime_tuning.no_host);
        assert!(config.strata_mtp);
        assert_eq!(config.strata_spec, Some(3));
        assert_eq!(config.strata_spec_min_p, Some(0.5));
    }

    #[test]
    fn runtime_tuning_rejects_zero_thread_count() {
        let cli = Cli::try_parse_from(["winstrint", "--threads", "0", "chat"])
            .expect("runtime tuning CLI");
        let error = AppConfig::from_cli(&cli).expect_err("zero threads must fail");
        assert!(
            error
                .to_string()
                .contains("--threads must be greater than zero")
        );
    }
}
