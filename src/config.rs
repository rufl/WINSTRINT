use std::env;
use std::path::PathBuf;

use anyhow::Result;
use clap::{Args, Parser, Subcommand, ValueEnum};

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum BackendPreference {
    Auto,
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

#[derive(Args, Clone, Debug)]
pub struct CommonArgs {
    #[arg(long, value_enum, default_value = "auto")]
    pub backend: BackendPreference,
    #[arg(long, value_enum, default_value = "coder")]
    pub model: ModelChoice,
    #[arg(long, value_name = "PATH")]
    pub data_dir: Option<PathBuf>,
    #[arg(long, value_name = "PATH")]
    pub queue_path: Option<PathBuf>,
    #[arg(long, default_value_t = 8080)]
    pub port: u16,
    #[arg(long, default_value_t = 32768)]
    pub context: u32,
    #[arg(long)]
    pub force: bool,
    #[arg(long)]
    pub no_download: bool,
    #[arg(long, short = 'y')]
    pub yes: bool,
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
    #[command(about = "Start the selected model and open the terminal chat")]
    Chat,
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
    pub port: u16,
    pub context: u32,
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
            port: cli.common.port,
            context: cli.common.context,
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
