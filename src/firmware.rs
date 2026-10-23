use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use serde::Serialize;

use crate::config::FirmwareInspectArgs;

const MAX_IMAGE_BYTES: u64 = 256 * 1024 * 1024;
const KEYWORDS: [&str; 10] = [
    "above 4g decoding",
    "above 4g",
    "resizable bar",
    "rebar",
    "compatibility support module",
    "csm",
    "pci express",
    "pcie",
    "large bar",
    "4g decoding",
];

#[derive(Debug, Serialize)]
struct TextHit {
    offset: usize,
    encoding: &'static str,
    text: String,
    matches: Vec<&'static str>,
}

#[derive(Debug, Serialize)]
struct OpcodeEvidence {
    offset: usize,
    opcode: u8,
    length: usize,
    name: &'static str,
}

#[derive(Debug, Serialize)]
struct ScanReport {
    image: String,
    bytes: usize,
    utf16_hits: Vec<TextHit>,
    ascii_hits: Vec<TextHit>,
    ifr_like_opcodes_near_hits: Vec<OpcodeEvidence>,
    notes: Vec<&'static str>,
}

pub fn run(args: &FirmwareInspectArgs) -> Result<()> {
    let image = fs::symlink_metadata(&args.image)
        .with_context(|| format!("checking firmware image {}", args.image.display()))?;
    if image.file_type().is_symlink() || !image.is_file() {
        anyhow::bail!(
            "firmware input {} is not a normal file; refusing to inspect it",
            args.image.display()
        );
    }
    if image.len() > MAX_IMAGE_BYTES {
        anyhow::bail!(
            "firmware input {} is {} bytes; refusing files over {} MiB",
            args.image.display(),
            image.len(),
            MAX_IMAGE_BYTES / (1024 * 1024)
        );
    }
    let bytes = fs::read(&args.image)
        .with_context(|| format!("reading firmware image {}", args.image.display()))?;
    let utf16_hits = scan_utf16(&bytes);
    let ascii_hits = scan_ascii(&bytes);
    let mut all_hits = utf16_hits
        .iter()
        .chain(ascii_hits.iter())
        .collect::<Vec<_>>();
    all_hits.sort_by_key(|hit| hit.offset);
    let report = ScanReport {
        image: args.image.display().to_string(),
        bytes: bytes.len(),
        ifr_like_opcodes_near_hits: scan_opcode_evidence(&bytes, &all_hits),
        utf16_hits,
        ascii_hits,
        notes: vec![
            "This is a read-only heuristic scan; raw BIOS containers can produce false positives.",
            "Offsets are file offsets, not safe setup_var offsets or flash addresses.",
            "A full SPI dump or vendor capsule is required for meaningful firmware analysis.",
            "No BIOS image, NVRAM variable, or PCI configuration is modified.",
        ],
    };
    if let Some(path) = args.output.as_deref() {
        write_report(path, &report)?;
    }
    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_report(&report);
    }
    Ok(())
}

fn scan_utf16(bytes: &[u8]) -> Vec<TextHit> {
    let mut hits = Vec::new();
    for alignment in [0usize, 1usize] {
        let mut offset = alignment;
        while offset + 1 < bytes.len() {
            let start = offset;
            let mut units = Vec::new();
            while offset + 1 < bytes.len() {
                let unit = u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
                if !is_text_unit(unit) {
                    break;
                }
                units.push(unit);
                offset += 2;
            }
            if units.len() >= 4 {
                let text = String::from_utf16_lossy(&units);
                let matches = keyword_matches(&text);
                if !matches.is_empty() {
                    hits.push(TextHit {
                        offset: start,
                        encoding: "utf-16le",
                        text,
                        matches,
                    });
                }
            }
            offset = if offset > start { offset } else { start + 2 };
        }
    }
    deduplicate_hits(hits)
}

fn scan_ascii(bytes: &[u8]) -> Vec<TextHit> {
    let mut hits = Vec::new();
    let mut offset = 0;
    while offset < bytes.len() {
        let start = offset;
        while offset < bytes.len() && is_ascii_text_byte(bytes[offset]) {
            offset += 1;
        }
        if offset - start >= 4 {
            let text = String::from_utf8_lossy(&bytes[start..offset]).into_owned();
            let matches = keyword_matches(&text);
            if !matches.is_empty() {
                hits.push(TextHit {
                    offset: start,
                    encoding: "ascii",
                    text,
                    matches,
                });
            }
        }
        if offset == start {
            offset += 1;
        }
    }
    deduplicate_hits(hits)
}

fn deduplicate_hits(mut hits: Vec<TextHit>) -> Vec<TextHit> {
    hits.sort_by_key(|hit| (hit.offset, hit.encoding));
    hits.dedup_by(|left, right| {
        left.offset == right.offset && left.encoding == right.encoding && left.text == right.text
    });
    hits
}

fn keyword_matches(text: &str) -> Vec<&'static str> {
    let lower = text.to_ascii_lowercase();
    KEYWORDS
        .into_iter()
        .filter(|keyword| lower.contains(keyword))
        .collect()
}

fn is_text_unit(unit: u16) -> bool {
    matches!(unit, 0x20..=0x7e | 0xa0..=0xff)
}

fn is_ascii_text_byte(byte: u8) -> bool {
    matches!(byte, 0x20..=0x7e)
}

fn scan_opcode_evidence(bytes: &[u8], hits: &[&TextHit]) -> Vec<OpcodeEvidence> {
    let mut offsets = BTreeSet::new();
    for hit in hits {
        let start = hit.offset.saturating_sub(512);
        let end = hit
            .offset
            .saturating_add(hit.text.len().saturating_mul(2))
            .saturating_add(512)
            .min(bytes.len().saturating_sub(1));
        for offset in start..end {
            offsets.insert(offset);
        }
    }

    let mut evidence = Vec::new();
    for offset in offsets {
        if offset + 1 >= bytes.len() {
            continue;
        }
        let opcode = bytes[offset];
        let length = usize::from(bytes[offset + 1] & 0x7f);
        if !(2..=64).contains(&length) || offset + length > bytes.len() {
            continue;
        }
        let Some(name) = ifr_opcode_name(opcode) else {
            continue;
        };
        evidence.push(OpcodeEvidence {
            offset,
            opcode,
            length,
            name,
        });
        if evidence.len() >= 256 {
            break;
        }
    }
    evidence
}

fn ifr_opcode_name(opcode: u8) -> Option<&'static str> {
    Some(match opcode {
        0x01 => "FORM",
        0x02 => "SUBTITLE",
        0x03 => "TEXT",
        0x04 => "IMAGE",
        0x05 => "ONE_OF",
        0x06 => "CHECKBOX",
        0x07 => "NUMERIC",
        0x08 => "PASSWORD",
        0x09 => "ORDERED_LIST",
        0x0a => "SUPPRESS_IF",
        0x0b => "GRAY_OUT_IF",
        0x0c => "DISABLE_IF",
        0x0d => "HIDDEN",
        0x0e => "FORM_SET",
        0x0f => "REF",
        0x29 => "END",
        _ => return None,
    })
}

fn print_report(report: &ScanReport) {
    println!("Firmware evidence scan: {}", report.image);
    println!("Input bytes: {}", report.bytes);
    println!("UTF-16 keyword hits: {}", report.utf16_hits.len());
    for hit in &report.utf16_hits {
        println!(
            "  0x{:x} [{}] {} => {}",
            hit.offset,
            hit.encoding,
            hit.matches.join(", "),
            hit.text
        );
    }
    println!("ASCII keyword hits: {}", report.ascii_hits.len());
    for hit in &report.ascii_hits {
        println!(
            "  0x{:x} [{}] {} => {}",
            hit.offset,
            hit.encoding,
            hit.matches.join(", "),
            hit.text
        );
    }
    println!(
        "IFR-like opcode evidence near hits: {}",
        report.ifr_like_opcodes_near_hits.len()
    );
    if report.utf16_hits.is_empty() && report.ascii_hits.is_empty() {
        println!("No matching menu text found; this does not prove the settings are absent.");
    }
    println!("Read-only: no firmware, NVRAM, or PCI state was changed.");
}

fn write_report(path: &Path, report: &ScanReport) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_vec_pretty(report)?)
        .with_context(|| format!("writing firmware scan report {}", path.display()))?;
    println!("Report written: {}", path.display());
    Ok(())
}
