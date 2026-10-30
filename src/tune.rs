use std::fs;
use std::net::TcpListener;
use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::config::{AppConfig, DecodeProfile, DecodeTuneArgs};
use crate::diagnostics;
use crate::model;
use crate::queue::GpuQueue;
use crate::runtime::{self, ResolvedBackend};

#[derive(Debug, Deserialize)]
struct ServerResponse {
    timings: Option<ServerTimings>,
}

#[derive(Debug, Deserialize)]
struct ServerTimings {
    prompt_n: usize,
    prompt_per_second: f64,
    predicted_n: usize,
    predicted_per_second: f64,
}

#[derive(Debug, Serialize)]
struct ProfileResult {
    profile: String,
    success: bool,
    prompt_tokens: Option<usize>,
    generated_tokens: Option<usize>,
    prompt_tokens_per_second: Option<f64>,
    decode_tokens_per_second: Option<f64>,
    prompt_tokens_per_second_samples: Vec<f64>,
    decode_tokens_per_second_samples: Vec<f64>,
    elapsed_ms: Option<u128>,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct TuneReport {
    backend: String,
    model: String,
    prompt_file: String,
    prompt_bytes: usize,
    context: u32,
    max_tokens: u32,
    repeats: u32,
    warmup: u32,
    profiles: Vec<ProfileResult>,
    recommendation: Option<String>,
}

pub fn run(config: &AppConfig, args: &DecodeTuneArgs) -> Result<()> {
    if args.max_tokens == 0 {
        anyhow::bail!("--max-tokens must be greater than zero");
    }
    if args.timeout_secs == 0 {
        anyhow::bail!("--timeout-secs must be greater than zero");
    }
    if args.repeats == 0 {
        anyhow::bail!("--repeats must be greater than zero");
    }
    let prompt = fs::read_to_string(&args.prompt_file)
        .with_context(|| format!("reading prompt file {}", args.prompt_file.display()))?;
    if prompt.trim().is_empty() {
        anyhow::bail!("prompt file {} is empty", args.prompt_file.display());
    }
    config.ensure_dirs()?;
    let diagnostics = diagnostics::collect()?;
    diagnostics::print_report(&diagnostics);
    let backend = runtime::select_backend(config, &diagnostics)?;
    let profiles = selected_profiles(args)?;
    if matches!(backend, ResolvedBackend::Strata)
        && profiles
            .iter()
            .any(|profile| !matches!(profile, DecodeProfile::Auto))
    {
        anyhow::bail!("Strata supports only the auto decode profile");
    }
    println!("Selected backend: {}", backend.label());
    println!("ReBAR is not required; this tool compares software KV and offload profiles.");

    let artifact = if let Some(path) = config.model_path.as_deref() {
        model::external_model(path)?
    } else if matches!(backend, ResolvedBackend::Strata) {
        model::cached_model(config).context(
            "tune-decode does not download model weights; run `winstrint download` first",
        )?
    } else {
        model::cached_model(config).context(
            "tune-decode does not download model weights; run `winstrint download` first or pass --model-path",
        )?
    };
    let queue = GpuQueue::new(&config.queue_path);
    let _lease = queue.acquire("decode-tune", true)?;
    let fresh = diagnostics::collect()?;
    diagnostics::require_idle(&fresh, config.force, true)?;
    if !matches!(backend, ResolvedBackend::Strata) {
        runtime::ensure_llama_bundle(config, backend)?;
    }

    let client = Client::builder()
        .timeout(Duration::from_secs(args.timeout_secs))
        .build()
        .context("creating benchmark HTTP client")?;
    let reasoning_effort = config
        .reasoning_effort
        .map(|effort| effort.label())
        .unwrap_or("low");
    let mut results = Vec::with_capacity(profiles.len());
    for (index, profile) in profiles.into_iter().enumerate() {
        let port = benchmark_port(config.port, index)?;
        if !port_available(port) {
            anyhow::bail!(
                "benchmark port {} is already in use; stop the other server or choose another --port",
                port
            );
        }
        println!("Benchmarking {} on port {}...", profile.label(), port);
        match benchmark_profile(
            config,
            &fresh,
            backend,
            &artifact,
            &client,
            &prompt,
            args.max_tokens,
            reasoning_effort,
            args.repeats,
            args.warmup,
            profile,
            port,
        ) {
            Ok(result) => {
                println!(
                    "  prompt {:.2} tok/s | decode {:.2} tok/s | {} generated | median of {} runs",
                    result.prompt_tokens_per_second.unwrap_or_default(),
                    result.decode_tokens_per_second.unwrap_or_default(),
                    result.generated_tokens.unwrap_or_default(),
                    result.decode_tokens_per_second_samples.len()
                );
                results.push(result);
            }
            Err(error) => {
                eprintln!("  {} failed: {error:#}", profile.label());
                results.push(ProfileResult {
                    profile: profile.label().to_owned(),
                    success: false,
                    prompt_tokens: None,
                    generated_tokens: None,
                    prompt_tokens_per_second: None,
                    decode_tokens_per_second: None,
                    prompt_tokens_per_second_samples: Vec::new(),
                    decode_tokens_per_second_samples: Vec::new(),
                    elapsed_ms: None,
                    error: Some(format!("{error:#}")),
                });
            }
        }
    }

    let recommendation = best_profile(&results);
    let report = TuneReport {
        backend: backend.label().to_owned(),
        model: config.model.key().to_owned(),
        prompt_file: args.prompt_file.display().to_string(),
        prompt_bytes: prompt.len(),
        context: config.context,
        max_tokens: args.max_tokens,
        repeats: args.repeats,
        warmup: args.warmup,
        profiles: results,
        recommendation: recommendation.clone(),
    };
    if let Some(path) = args.output.as_deref() {
        write_report(path, &report)?;
    }
    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_report(&report);
    }
    if recommendation.is_none() {
        anyhow::bail!("all decode profiles failed; inspect the runtime log and benchmark report");
    }
    Ok(())
}

fn selected_profiles(args: &DecodeTuneArgs) -> Result<Vec<DecodeProfile>> {
    let requested = if let Some(profiles) = args.profiles.as_ref() {
        if profiles.is_empty() {
            anyhow::bail!("--profiles must contain at least one profile");
        }
        profiles.clone()
    } else if args.compare {
        vec![
            DecodeProfile::Auto,
            DecodeProfile::Q4Kv,
            DecodeProfile::Q8NoOp,
            DecodeProfile::Q8Ngram,
            DecodeProfile::NoKv,
        ]
    } else {
        vec![DecodeProfile::Auto]
    };
    let mut selected = Vec::with_capacity(requested.len());
    for profile in requested {
        if !selected.contains(&profile) {
            selected.push(profile);
        }
    }
    Ok(selected)
}

fn benchmark_profile(
    config: &AppConfig,
    diagnostics: &diagnostics::Diagnostics,
    backend: ResolvedBackend,
    artifact: &model::ModelArtifact,
    client: &Client,
    prompt: &str,
    max_tokens: u32,
    reasoning_effort: &str,
    repeats: u32,
    warmup: u32,
    profile: DecodeProfile,
    port: u16,
) -> Result<ProfileResult> {
    let mut benchmark_config = config.clone();
    benchmark_config.decode_profile = profile;
    benchmark_config.port = port;
    let process =
        runtime::start_with_profile(&benchmark_config, diagnostics, backend, artifact, profile)?;
    let endpoint = process.endpoint.clone();
    for _ in 0..warmup {
        let _ = benchmark_request(
            client,
            &endpoint,
            config.model.key(),
            prompt,
            max_tokens,
            reasoning_effort,
        )?;
    }

    let mut prompt_samples = Vec::with_capacity(repeats as usize);
    let mut decode_samples = Vec::with_capacity(repeats as usize);
    let mut elapsed_samples = Vec::with_capacity(repeats as usize);
    let mut last_timings = None;
    for _ in 0..repeats {
        let (timings, elapsed_ms) = benchmark_request(
            client,
            &endpoint,
            config.model.key(),
            prompt,
            max_tokens,
            reasoning_effort,
        )?;
        prompt_samples.push(timings.prompt_per_second);
        decode_samples.push(timings.predicted_per_second);
        elapsed_samples.push(elapsed_ms);
        last_timings = Some(timings);
    }
    let timings = last_timings.context("benchmark produced no measured timings")?;
    let prompt_per_second = median(&prompt_samples)
        .context("benchmark produced no finite prompt throughput samples")?;
    let decode_per_second = median(&decode_samples)
        .context("benchmark produced no finite decode throughput samples")?;
    let elapsed_ms =
        median_u128(&elapsed_samples).context("benchmark produced no elapsed samples")?;
    drop(process);
    Ok(ProfileResult {
        profile: profile.label().to_owned(),
        success: true,
        prompt_tokens: Some(timings.prompt_n),
        generated_tokens: Some(timings.predicted_n),
        prompt_tokens_per_second: Some(prompt_per_second),
        decode_tokens_per_second: Some(decode_per_second),
        prompt_tokens_per_second_samples: prompt_samples,
        decode_tokens_per_second_samples: decode_samples,
        elapsed_ms: Some(elapsed_ms),
        error: None,
    })
}

fn benchmark_request(
    client: &Client,
    endpoint: &str,
    model: &str,
    prompt: &str,
    max_tokens: u32,
    reasoning_effort: &str,
) -> Result<(ServerTimings, u128)> {
    let started = Instant::now();
    let response = client
        .post(format!("{endpoint}/v1/chat/completions"))
        .json(&json!({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": false,
            "max_tokens": max_tokens,
            "temperature": 0,
            "reasoning_effort": reasoning_effort
        }))
        .send()
        .context("sending decode benchmark request")?;
    let status = response.status();
    let body: Value = response
        .json()
        .context("parsing decode benchmark response")?;
    let elapsed_ms = started.elapsed().as_millis();
    if !status.is_success() {
        anyhow::bail!("runtime returned {status}: {body}");
    }
    let parsed: ServerResponse = serde_json::from_value(body)
        .context("decode benchmark response did not contain timings")?;
    let timings = parsed
        .timings
        .context("decode benchmark response omitted timings")?;
    Ok((timings, elapsed_ms))
}

fn median(values: &[f64]) -> Option<f64> {
    let mut finite = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if finite.is_empty() {
        return None;
    }
    finite.sort_by(f64::total_cmp);
    Some(finite[finite.len() / 2])
}

fn median_u128(values: &[u128]) -> Option<u128> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    Some(sorted[sorted.len() / 2])
}

fn best_profile(results: &[ProfileResult]) -> Option<String> {
    results
        .iter()
        .filter(|result| result.success)
        .filter(|result| result.decode_tokens_per_second.is_some_and(f64::is_finite))
        .max_by(|left, right| {
            let speed_order = left
                .decode_tokens_per_second
                .unwrap_or_default()
                .total_cmp(&right.decode_tokens_per_second.unwrap_or_default());
            if speed_order == std::cmp::Ordering::Equal {
                right
                    .elapsed_ms
                    .unwrap_or(u128::MAX)
                    .cmp(&left.elapsed_ms.unwrap_or(u128::MAX))
            } else {
                speed_order
            }
        })
        .map(|result| result.profile.clone())
}

fn print_report(report: &TuneReport) {
    println!();
    println!("Decode tuning report (ReBAR-independent)");
    println!("Prompt: {} bytes", report.prompt_bytes);
    println!(
        "Context: {} | max tokens: {}",
        report.context, report.max_tokens
    );
    println!(
        "{:<12} {:>14} {:>14} {:>12}",
        "profile", "prompt tok/s", "decode tok/s", "elapsed"
    );
    for result in &report.profiles {
        if result.success {
            println!(
                "{:<12} {:>14.2} {:>14.2} {:>12}",
                result.profile,
                result.prompt_tokens_per_second.unwrap_or_default(),
                result.decode_tokens_per_second.unwrap_or_default(),
                format!("{} ms", result.elapsed_ms.unwrap_or_default())
            );
        } else {
            println!("{:<12} {:>14} {:>14} failed", result.profile, "-", "-");
        }
    }
    if let Some(profile) = report.recommendation.as_deref() {
        println!();
        println!("Fastest measured decode profile: {profile}");
        println!("Apply it to chat with: --decode-profile {profile}");
    }
}

fn write_report(path: &Path, report: &TuneReport) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_vec_pretty(report)?)
        .with_context(|| format!("writing decode tuning report {}", path.display()))?;
    println!("Report written: {}", path.display());
    Ok(())
}

fn benchmark_port(base: u16, index: usize) -> Result<u16> {
    let offset = u16::try_from(index).context("too many decode profiles for the port range")?;
    base.checked_add(offset)
        .context("decode benchmark ports exceed 65535; choose a lower --port")
}

fn port_available(port: u16) -> bool {
    TcpListener::bind(("127.0.0.1", port)).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_ignores_non_finite_samples_and_sorts() {
        assert_eq!(median(&[f64::NAN, 9.0, 1.0, 5.0]), Some(5.0));
        assert_eq!(median(&[f64::INFINITY]), None);
        assert_eq!(median_u128(&[9, 1, 5, 3]), Some(5));
    }
}
