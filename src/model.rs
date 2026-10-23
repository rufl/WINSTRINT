use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};
use reqwest::blocking::{Client, Response};
use reqwest::header::{HeaderValue, RANGE};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::config::{AppConfig, ModelChoice};

const HF_API: &str = "https://huggingface.co/api/models";
const CODER_MODEL_REVISION: &str = "bba88856f8c3c0c9e40e45f82d05e228b814ac43";
// Keep the previously downloaded Coder pack usable.  The GGUF payload is content-addressed by
// its manifest records; a revision-only mismatch must not force another 58 GiB transfer.
const LEGACY_CODER_MODEL_REVISION: &str = "5348543e0147355ac9cbcb031184a3546350988e";
const BASE_MODEL_REVISION: &str = "ed59f92082b1e93c0e96d60a8b11aab089b52f09";

fn compatible_revision(spec: ModelSpec, revision: &str) -> bool {
    revision == spec.revision
        || (spec.choice == ModelChoice::Coder && revision == LEGACY_CODER_MODEL_REVISION)
}

#[derive(Clone, Copy, Debug)]
pub struct ModelSpec {
    pub choice: ModelChoice,
    pub repo: &'static str,
    pub revision: &'static str,
    pub prefix: &'static str,
    pub resident_bytes: u64,
    pub total_bytes: u64,
    pub note: &'static str,
}

pub const MODEL_SPECS: [ModelSpec; 5] = [
    ModelSpec {
        choice: ModelChoice::Coder,
        repo: "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF",
        revision: CODER_MODEL_REVISION,
        prefix: "IQ1_M",
        resident_bytes: 29_608_446_496,
        total_bytes: 58_408_584_928,
        note: "the only exact Coder GGUF currently published by ISTA-DASLab",
    },
    ModelSpec {
        choice: ModelChoice::Q2_0,
        repo: "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
        revision: BASE_MODEL_REVISION,
        prefix: "Q2_0",
        resident_bytes: 37_600_000_000,
        total_bytes: 66_400_000_000,
        note: "unpruned base model; not the Coder checkpoint",
    },
    ModelSpec {
        choice: ModelChoice::Iq2Xs,
        repo: "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
        revision: BASE_MODEL_REVISION,
        prefix: "IQ2_XS",
        resident_bytes: 39_200_000_000,
        total_bytes: 68_000_000_000,
        note: "unpruned base model; not the Coder checkpoint",
    },
    ModelSpec {
        choice: ModelChoice::Iq3Xxs,
        repo: "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
        revision: BASE_MODEL_REVISION,
        prefix: "IQ3_XXS",
        resident_bytes: 47_000_000_000,
        total_bytes: 75_800_000_000,
        note: "unpruned base model; not the Coder checkpoint",
    },
    ModelSpec {
        choice: ModelChoice::Iq3S,
        repo: "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
        revision: BASE_MODEL_REVISION,
        prefix: "IQ3_S",
        resident_bytes: 54_800_000_000,
        total_bytes: 83_600_000_000,
        note: "unpruned base model; not the Coder checkpoint",
    },
];

#[derive(Clone, Debug)]
pub struct ModelArtifact {
    pub directory: PathBuf,
    pub first_shard: PathBuf,
    pub shards: Vec<PathBuf>,
    pub supports_moe_cache: bool,
}

pub fn external_model(path: &Path) -> Result<ModelArtifact> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("checking external model {}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        anyhow::bail!("external model {} is not a normal file", path.display());
    }
    if !path
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
    {
        anyhow::bail!("external model {} is not a GGUF file", path.display());
    }
    if metadata.len() == 0 {
        anyhow::bail!("external model {} is empty", path.display());
    }
    let first_shard = fs::canonicalize(path)
        .with_context(|| format!("canonicalizing external model {}", path.display()))?;
    let directory = first_shard
        .parent()
        .map(Path::to_path_buf)
        .context("external model has no parent directory")?;
    Ok(ModelArtifact {
        directory,
        first_shard: first_shard.clone(),
        shards: vec![first_shard],
        supports_moe_cache: false,
    })
}

#[derive(Deserialize)]
struct TreeEntry {
    path: String,
    size: Option<u64>,
    lfs: Option<LfsInfo>,
}
#[derive(Deserialize)]
struct LfsInfo {
    oid: Option<String>,
    size: Option<u64>,
}
#[derive(Deserialize, Serialize)]
struct Manifest {
    repo: String,
    revision: String,
    variant: String,
    files: Vec<ManifestFile>,
}
#[derive(Deserialize, Serialize)]
struct ManifestFile {
    path: String,
    size: u64,
    sha256: String,
}

pub fn spec(choice: ModelChoice) -> ModelSpec {
    MODEL_SPECS
        .iter()
        .copied()
        .find(|item| item.choice == choice)
        .expect("all model choices have a specification")
}

fn prepare_model_directory(directory: &Path, create: bool) -> Result<()> {
    match fs::symlink_metadata(directory) {
        Ok(metadata) => {
            if metadata.file_type().is_symlink() || !metadata.is_dir() {
                anyhow::bail!(
                    "model cache directory {} is not a normal directory",
                    directory.display()
                );
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound && create => {
            fs::create_dir_all(directory)?;
            let metadata = fs::symlink_metadata(directory)?;
            if metadata.file_type().is_symlink() || !metadata.is_dir() {
                anyhow::bail!(
                    "model cache directory {} is not a normal directory",
                    directory.display()
                );
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(error).with_context(|| {
                format!("checking model cache directory {}", directory.display())
            });
        }
    }
    Ok(())
}

pub fn print_catalog() {
    println!("Available Qwen3.8 Flash Next GGUF choices:");
    for item in MODEL_SPECS {
        println!(
            "  {:<14} {:>5.1} GiB shard-1 / {:>5.1} GiB total - {}",
            item.choice.key(),
            item.resident_bytes as f64 / 1e9,
            item.total_bytes as f64 / 1e9,
            item.note
        );
    }
    println!(
        "The exact Coder repository currently publishes IQ1_M only; the other entries are unpruned base variants."
    );
    println!(
        "Each completed file is checked against its Hugging Face LFS SHA-256; scanner status is not treated as a trust guarantee."
    );
}

pub fn ensure_model(config: &AppConfig) -> Result<ModelArtifact> {
    let spec = spec(config.model);
    if let Ok(cached) = cached_model(config) {
        println!("Using cached model: {}", cached.directory.display());
        return Ok(cached);
    }
    let client = Client::builder()
        .user_agent("WINSTRINT/0.1")
        .timeout(Duration::from_secs(300))
        .build()
        .context("creating HTTPS client")?;
    let entries = discover(&client, spec)?;
    let selected = select_shards(&entries, spec)?;
    let directory = config.model_dir.join(config.model.key());
    prepare_model_directory(&directory, true)?;
    let mut files = Vec::new();
    for entry in selected {
        let filename = Path::new(&entry.path)
            .file_name()
            .context("Hugging Face tree entry has no filename")?;
        let destination = directory.join(filename);
        let url = format!(
            "https://huggingface.co/{}/resolve/{}/{}?download=true",
            spec.repo, spec.revision, entry.path
        );
        println!(
            "Model file: {} ({:.2} GiB)",
            entry.path,
            expected_size(entry) as f64 / 1e9
        );
        download_verified(
            &client,
            &url,
            &destination,
            expected_size(entry),
            expected_sha(entry),
        )?;
        files.push(destination);
    }
    files.sort();
    let first_shard = files
        .iter()
        .find(|path| {
            path.file_name()
                .is_some_and(|name| name.to_string_lossy().contains("00001"))
        })
        .cloned()
        .or_else(|| files.first().cloned())
        .context("Hugging Face model has no first shard")?;
    let manifest_files = files
        .iter()
        .map(|path| {
            Ok(ManifestFile {
                path: path
                    .file_name()
                    .context("model file has no filename")?
                    .to_string_lossy()
                    .into_owned(),
                size: fs::metadata(path)?.len(),
                sha256: sha256_file(path)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    fs::write(
        directory.join("manifest.json"),
        serde_json::to_vec_pretty(&Manifest {
            repo: spec.repo.into(),
            revision: spec.revision.into(),
            variant: spec.prefix.into(),
            files: manifest_files,
        })?,
    )?;
    Ok(ModelArtifact {
        directory,
        first_shard,
        shards: files,
        supports_moe_cache: true,
    })
}

pub fn cached_model(config: &AppConfig) -> Result<ModelArtifact> {
    let spec = spec(config.model);
    let directory = config.model_dir.join(config.model.key());
    prepare_model_directory(&directory, false)?;
    let manifest_path = directory.join("manifest.json");
    let manifest_metadata = fs::symlink_metadata(&manifest_path).with_context(|| {
        format!(
            "offline startup needs the pinned model manifest at {}; run `winstrint download` first",
            manifest_path.display()
        )
    })?;
    if manifest_metadata.file_type().is_symlink() || !manifest_metadata.is_file() {
        anyhow::bail!(
            "pinned model manifest {} is not a normal file; run `winstrint download` again",
            manifest_path.display()
        );
    }
    let manifest: Manifest = serde_json::from_slice(&fs::read(&manifest_path)?)
        .with_context(|| format!("parsing pinned model manifest {}", manifest_path.display()))?;
    if manifest.repo != spec.repo
        || !compatible_revision(spec, &manifest.revision)
        || manifest.variant != spec.prefix
    {
        anyhow::bail!(
            "cached model at {} is not a compatible {} revision; run `winstrint download` again",
            directory.display(),
            spec.revision
        );
    }
    if manifest.revision != spec.revision {
        eprintln!(
            "Using cached Coder GGUF revision {}; current launcher pin is {}; no redownload required",
            manifest.revision, spec.revision
        );
    }
    let mut files = manifest
        .files
        .iter()
        .map(|record| {
            let relative = Path::new(&record.path);
            if relative.components().count() != 1
                || relative.file_name().is_none()
                || !relative
                    .extension()
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
                || record.size == 0
                || record.sha256.len() != 64
                || !record.sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
            {
                anyhow::bail!(
                    "pinned model manifest contains an unsafe file record {:?}",
                    record.path
                );
            }
            let path = directory.join(relative);
            let metadata = fs::symlink_metadata(&path)
                .with_context(|| format!("cached model file {} is missing", path.display()))?;
            if metadata.file_type().is_symlink() || !metadata.is_file() {
                anyhow::bail!("cached model file {} is not a normal file", path.display());
            }
            if metadata.len() != record.size {
                anyhow::bail!(
                    "cached model file {} has size {}, expected {}; run `winstrint download` again",
                    path.display(),
                    metadata.len(),
                    record.size
                );
            }
            Ok(path)
        })
        .collect::<Result<Vec<_>>>()?;
    files.sort();
    if files.windows(2).any(|pair| pair[0] == pair[1]) {
        anyhow::bail!(
            "pinned model manifest contains duplicate files; run `winstrint download` again"
        );
    }
    if files.len() < 2 {
        anyhow::bail!(
            "offline startup requires at least two pinned GGUF shards in {}; run `winstrint download` first",
            directory.display()
        );
    }
    let first_shard = files
        .iter()
        .find(|path| {
            path.file_name()
                .is_some_and(|name| name.to_string_lossy().contains("00001"))
        })
        .cloned()
        .context("cached model has no first GGUF shard")?;
    Ok(ModelArtifact {
        directory,
        first_shard,
        shards: files,
        supports_moe_cache: true,
    })
}

pub fn download_verified(
    client: &Client,
    url: &str,
    destination: &Path,
    expected_size: u64,
    expected_sha: Option<&str>,
) -> Result<()> {
    if expected_size == 0 {
        anyhow::bail!("refusing download with unknown expected size: {url}");
    }
    if let Ok(metadata) = fs::symlink_metadata(destination) {
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            anyhow::bail!(
                "refusing linked or non-file download destination {}",
                destination.display()
            );
        }
        if metadata.len() == expected_size {
            if let Some(expected) = expected_sha {
                let actual = sha256_file(destination)?;
                if actual.eq_ignore_ascii_case(expected) {
                    println!("Already verified: {}", destination.display());
                    return Ok(());
                }
                eprintln!("Hash mismatch; replacing {}", destination.display());
            } else {
                println!(
                    "Already present with expected size: {}",
                    destination.display()
                );
                return Ok(());
            }
        }
    }
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)?;
    }
    let part = PathBuf::from(format!("{}.part", destination.display()));
    if let Ok(metadata) = fs::symlink_metadata(&part)
        && (metadata.file_type().is_symlink() || !metadata.is_file())
    {
        anyhow::bail!(
            "refusing linked or non-file partial download {}",
            part.display()
        );
    }
    let mut offset = fs::metadata(&part)
        .map(|metadata| metadata.len())
        .unwrap_or(0);
    let mut response = request_range(client, url, offset)?;
    if offset > 0 && response.status() != reqwest::StatusCode::PARTIAL_CONTENT {
        fs::remove_file(&part).ok();
        offset = 0;
        response = request_range(client, url, 0)?;
    }
    if !response.status().is_success() {
        anyhow::bail!("download failed with {}: {}", response.status(), url);
    }
    let mut file = if offset > 0 {
        OpenOptions::new().append(true).open(&part)?
    } else {
        File::create(&part)?
    };
    let mut downloaded = offset;
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = response.read(&mut buffer).with_context(|| {
            format!(
                "reading model download response; partial file is kept at {}",
                part.display()
            )
        })?;
        if read == 0 {
            break;
        }
        file.write_all(&buffer[..read])?;
        downloaded += read as u64;
        if expected_size > 0 {
            print!(
                "\r  {:.1}% ({:.2}/{:.2} GiB)",
                downloaded as f64 * 100.0 / expected_size as f64,
                downloaded as f64 / 1e9,
                expected_size as f64 / 1e9
            );
            std::io::stdout().flush().ok();
        }
    }
    println!();
    file.sync_all()?;
    if downloaded != expected_size {
        anyhow::bail!(
            "download size mismatch for {}: got {}, expected {} (partial file kept at {})",
            url,
            downloaded,
            expected_size,
            part.display()
        );
    }
    if let Some(expected) = expected_sha {
        let actual = sha256_file(&part)?;
        if !actual.eq_ignore_ascii_case(expected) {
            anyhow::bail!(
                "SHA-256 mismatch for {}: got {}, expected {}",
                url,
                actual,
                expected
            );
        }
    }
    if fs::symlink_metadata(destination).is_ok() {
        fs::remove_file(destination)?;
    }
    fs::rename(&part, destination).with_context(|| {
        format!(
            "installing verified download {} -> {}",
            part.display(),
            destination.display()
        )
    })?;
    Ok(())
}

pub fn sha256_file(path: &Path) -> Result<String> {
    let mut file =
        File::open(path).with_context(|| format!("opening {} for hashing", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex::encode(hasher.finalize()))
}

fn discover(client: &Client, spec: ModelSpec) -> Result<Vec<TreeEntry>> {
    let url = format!(
        "{HF_API}/{}/tree/{}?recursive=true&expand=true",
        spec.repo, spec.revision
    );
    let response = client
        .get(&url)
        .send()
        .with_context(|| format!("querying Hugging Face tree {url}"))?;
    if !response.status().is_success() {
        anyhow::bail!(
            "Hugging Face tree query failed with {}: {}",
            response.status(),
            url
        );
    }
    response
        .json()
        .context("parsing Hugging Face tree response")
}

fn select_shards(entries: &[TreeEntry], spec: ModelSpec) -> Result<Vec<&TreeEntry>> {
    let mut selected = entries
        .iter()
        .filter(|entry| {
            entry.path.starts_with(&format!("{}/", spec.prefix))
                && entry.path.to_ascii_lowercase().ends_with(".gguf")
                && !entry.path.to_ascii_lowercase().contains("mmproj")
        })
        .collect::<Vec<_>>();
    selected.sort_by(|left, right| left.path.cmp(&right.path));
    if selected.len() < 2 {
        anyhow::bail!(
            "Hugging Face repository {} does not expose two GGUF shards under {}; refusing an incomplete model",
            spec.repo,
            spec.prefix
        );
    }
    Ok(selected)
}

fn expected_size(entry: &TreeEntry) -> u64 {
    entry
        .size
        .or_else(|| entry.lfs.as_ref().and_then(|lfs| lfs.size))
        .unwrap_or_default()
}
fn expected_sha(entry: &TreeEntry) -> Option<&str> {
    entry
        .lfs
        .as_ref()
        .and_then(|lfs| lfs.oid.as_deref())
        .filter(|value| value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit()))
}
fn request_range(client: &Client, url: &str, offset: u64) -> Result<Response> {
    let mut request = client.get(url);
    if offset > 0 {
        request = request.header(RANGE, HeaderValue::from_str(&format!("bytes={offset}-"))?);
    }
    Ok(request.send()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cached_coder_accepts_the_downloaded_legacy_revision_only() {
        let coder = spec(ModelChoice::Coder);
        assert!(compatible_revision(coder, CODER_MODEL_REVISION));
        assert!(compatible_revision(coder, LEGACY_CODER_MODEL_REVISION));
        assert!(!compatible_revision(coder, "not-a-model-revision"));

        let base = spec(ModelChoice::Q2_0);
        assert!(!compatible_revision(base, LEGACY_CODER_MODEL_REVISION));
    }

    #[test]
    fn external_model_disables_moe_cache_and_canonicalizes_path() {
        let path = std::env::temp_dir().join(format!(
            "winstrint-external-model-{}.gguf",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        std::fs::write(&path, b"fixture").expect("external model fixture");
        let artifact = external_model(&path).expect("external GGUF");
        assert_eq!(
            artifact.first_shard,
            std::fs::canonicalize(&path).expect("canonical fixture")
        );
        assert!(!artifact.supports_moe_cache);
        let _ = std::fs::remove_file(path);
    }
}
