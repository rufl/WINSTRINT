mod config;
mod diagnostics;
mod model;
mod queue;
mod runtime;
mod tui;

use std::io::{self, Write};

use anyhow::{Context, Result};
use clap::Parser;

use config::{Cli, Command};

fn main() {
    if let Err(error) = run() {
        eprintln!("WINSTRINT error: {error:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    let config = config::AppConfig::from_cli(&cli)?;
    match cli.command {
        Some(Command::Check) => {
            let diagnostics = diagnostics::collect()?;
            diagnostics::print_report(&diagnostics);
            model::print_catalog();
            Ok(())
        }
        Some(Command::Download) => {
            config.ensure_dirs()?;
            let diagnostics = diagnostics::collect()?;
            diagnostics::print_report(&diagnostics);
            let artifact = model::ensure_model(&config)?;
            println!("Model ready: {}", artifact.directory.display());
            Ok(())
        }
        Some(Command::QueueStatus) => {
            let queue = queue::GpuQueue::new(&config.queue_path);
            let state = queue.status()?;
            println!("Queue: {}", queue.path().display());
            println!("{}", serde_json::to_string_pretty(&state)?);
            Ok(())
        }
        Some(Command::QueueAcquire { purpose }) => {
            config.ensure_dirs()?;
            let queue = queue::GpuQueue::new(&config.queue_path);
            let lease = queue.acquire(purpose, true)?;
            let ticket = lease.ticket().context("queue lease has no ticket")?;
            println!(
                "Queue ticket {} acquired; press Enter to release it.",
                ticket.id
            );
            let mut line = String::new();
            io::stdin().read_line(&mut line)?;
            lease.release()?;
            Ok(())
        }
        Some(Command::QueueRelease { ticket }) => {
            let queue = queue::GpuQueue::new(&config.queue_path);
            if queue.release(ticket.as_deref())? {
                println!("Queue ticket released.");
            } else {
                println!("Queue was already idle.");
            }
            Ok(())
        }
        Some(Command::Ticketed { purpose, command }) => {
            config.ensure_dirs()?;
            let queue = queue::GpuQueue::new(&config.queue_path);
            let status = queue.run_ticketed(&purpose, &command, config.force)?;
            std::process::exit(status.code().unwrap_or(1));
        }
        Some(Command::Chat) | None => start_chat(&config),
    }
}

fn start_chat(config: &config::AppConfig) -> Result<()> {
    config.ensure_dirs()?;
    let diagnostics = diagnostics::collect()?;
    diagnostics::print_report(&diagnostics);
    let backend = runtime::select_backend(config, &diagnostics)?;
    println!("Selected backend: {}", backend.label());
    model::print_catalog();
    if config.no_download {
        anyhow::bail!(
            "--no-download was supplied; download the selected model first with `winstrint download`"
        );
    }
    if !config.yes && !model_exists(config) {
        print!("Download the selected model now? It is tens of gigabytes. [y/N] ");
        io::stdout().flush()?;
        let mut answer = String::new();
        io::stdin().read_line(&mut answer)?;
        if !answer.trim().eq_ignore_ascii_case("y") {
            anyhow::bail!("model download cancelled; rerun with --yes to confirm");
        }
    }
    let artifact = model::ensure_model(config)?;
    let fresh = diagnostics::collect()?;
    diagnostics::require_idle(&fresh, config.force, false)?;
    let process = runtime::start(config, &fresh, backend, &artifact)?;
    let result = tui::run(&process.endpoint, "winstrint");
    drop(process);
    result
}

fn model_exists(config: &config::AppConfig) -> bool {
    config
        .model_dir
        .join(config.model.key())
        .read_dir()
        .ok()
        .map(|entries| {
            entries.flatten().any(|entry| {
                entry.path().file_name().is_some_and(|name| {
                    name.to_string_lossy().contains("00001")
                        && name.to_string_lossy().ends_with(".gguf")
                })
            })
        })
        .unwrap_or(false)
}
