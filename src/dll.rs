//! Non-executing extraction of an authorized DLL's `WEIGHTS_HT` PE resource.

use anyhow::{Context, Result, bail, ensure};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, fs, path::Path};

fn u16_at(b: &[u8], p: usize) -> Result<u16> {
    Ok(u16::from_le_bytes(
        b.get(p..p + 2).context("truncated PE")?.try_into().unwrap(),
    ))
}
fn u32_at(b: &[u8], p: usize) -> Result<u32> {
    Ok(u32::from_le_bytes(
        b.get(p..p + 4).context("truncated PE")?.try_into().unwrap(),
    ))
}
fn u64_at(b: &[u8], p: usize) -> Result<u64> {
    Ok(u64::from_le_bytes(
        b.get(p..p + 8).context("truncated PE")?.try_into().unwrap(),
    ))
}

fn weights_resource(pe: &[u8]) -> Result<&[u8]> {
    let pe_at = u32_at(pe, 0x3c)? as usize;
    ensure!(pe.get(pe_at..pe_at + 4) == Some(b"PE\0\0"), "not a PE DLL");
    let coff = pe_at + 4;
    let sections = u16_at(pe, coff + 2)? as usize;
    let optional_size = u16_at(pe, coff + 16)? as usize;
    let optional = coff + 20;
    let data_directory = match u16_at(pe, optional)? {
        0x20b => optional + 112,
        0x10b => optional + 96,
        _ => bail!("unknown PE optional header"),
    };
    let resource_rva = u32_at(pe, data_directory + 16)?;
    ensure!(resource_rva != 0, "PE has no resource directory");
    let section_table = optional + optional_size;
    let rva_file = |rva: u32| -> Result<usize> {
        for i in 0..sections {
            let p = section_table + 40 * i;
            let virtual_size = u32_at(pe, p + 8)?;
            let virtual_address = u32_at(pe, p + 12)?;
            let raw_size = u32_at(pe, p + 16)?;
            let raw_offset = u32_at(pe, p + 20)?;
            if rva >= virtual_address && rva - virtual_address < virtual_size.max(raw_size) {
                return Ok((raw_offset + rva - virtual_address) as usize);
            }
        }
        bail!("resource RVA outside PE sections")
    };
    let base = rva_file(resource_rva)?;
    let tree = pe.get(base..).context("truncated PE resource tree")?;
    let find_child = |directory: usize, id: Option<u32>, name: Option<&str>| -> Result<usize> {
        let count = u16_at(tree, directory + 12)? as usize + u16_at(tree, directory + 14)? as usize;
        for i in 0..count {
            let entry = directory + 16 + i * 8;
            let key = u32_at(tree, entry)?;
            let found = if let Some(id) = id {
                key == id
            } else if let Some(name) = name {
                if key & 0x8000_0000 == 0 {
                    false
                } else {
                    let at = (key & 0x7fff_ffff) as usize;
                    let len = u16_at(tree, at)? as usize;
                    let utf16 = tree
                        .get(at + 2..at + 2 + 2 * len)
                        .context("truncated resource name")?;
                    let (chunks, _) = utf16.as_chunks::<2>();
                    let units = chunks
                        .iter()
                        .map(|c| u16::from_le_bytes(*c))
                        .collect::<Vec<_>>();
                    String::from_utf16(&units).ok().as_deref() == Some(name)
                }
            } else {
                false
            };
            if found {
                let child = u32_at(tree, entry + 4)?;
                ensure!(child & 0x8000_0000 != 0, "malformed resource directory");
                return Ok((child & 0x7fff_ffff) as usize);
            }
        }
        bail!("WEIGHTS_HT resource not found")
    };
    let names = find_child(0, Some(10), None)?;
    let languages = find_child(names, None, Some("WEIGHTS_HT"))?;
    let data_entry = (u32_at(tree, languages + 20)? & 0x7fff_ffff) as usize;
    let data_rva = u32_at(tree, data_entry)?;
    let data_size = u32_at(tree, data_entry + 4)? as usize;
    let data_at = rva_file(data_rva)?;
    pe.get(data_at..data_at + data_size)
        .context("truncated WEIGHTS_HT data")
}

#[derive(Debug)]
struct Record {
    name: String,
    block: u32,
    layer: u32,
    start: usize,
    len: usize,
}

fn parse_records(bytes: &[u8]) -> Result<Vec<Record>> {
    ensure!(
        u64_at(bytes, 0)? as usize == bytes.len(),
        "unexpected WEIGHTS_HT size header"
    );
    let mut out = Vec::new();
    for at in 0..bytes.len().saturating_sub(64) {
        if !bytes[at..].starts_with(b"block") {
            continue;
        }
        let mut p = at + 5;
        let number = |p: &mut usize| -> Option<u32> {
            let first = *p;
            while bytes.get(*p).is_some_and(u8::is_ascii_digit) {
                *p += 1;
            }
            if first == *p {
                return None;
            }
            std::str::from_utf8(&bytes[first..*p]).ok()?.parse().ok()
        };
        let Some(block) = number(&mut p) else {
            continue;
        };
        if !bytes.get(p..).is_some_and(|v| v.starts_with(b".layer")) {
            continue;
        }
        p += 6;
        let Some(layer) = number(&mut p) else {
            continue;
        };
        if !bytes.get(p..).is_some_and(|v| v.starts_with(b".layer")) {
            continue;
        }
        p += 6;
        let outer = u64_at(bytes, p)? as usize;
        let raw = u64_at(bytes, p + 16)? as usize;
        if outer != u64_at(bytes, p + 8)? as usize
            || outer != raw + 40
            || u32_at(bytes, p + 24)? != 1
        {
            continue;
        }
        let start = p + 28;
        if start.checked_add(raw).is_none_or(|end| end > bytes.len()) {
            continue;
        }
        out.push(Record {
            name: format!("block{block}.layer{layer}.layer"),
            block,
            layer,
            start,
            len: raw,
        });
    }
    out.sort_by_key(|v| (v.block, v.layer));
    let names = out.iter().map(|v| &v.name).collect::<BTreeSet<_>>();
    ensure!(names.len() == out.len(), "duplicate tensor records");
    let missing = (0..=70)
        .filter(|block| !out.iter().any(|record| record.block == *block))
        .collect::<Vec<_>>();
    ensure!(
        missing.is_empty(),
        "missing graph blocks in resource: {missing:?}"
    );
    Ok(out)
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Manifest {
    totals: Totals,
    stages: Vec<Stage>,
    tensors: Vec<Tensor>,
}
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Totals {
    block_count: u32,
}
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Stage {
    id: String,
    file: String,
    packed_byte_length: usize,
    sha256: String,
}
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Tensor {
    name: String,
    block: u32,
    layer: u32,
    parameter: &'static str,
    stage: String,
    stage_offset: usize,
    byte_length: usize,
}

pub struct ExtractReport {
    pub records: usize,
    pub packed_bytes: usize,
}

/// Writes a portable model directory. The output must not already exist.
pub fn extract_model(dll: impl AsRef<Path>, output: impl AsRef<Path>) -> Result<ExtractReport> {
    let output = output.as_ref();
    ensure!(
        !output.exists(),
        "refusing to overwrite {}",
        output.display()
    );
    let pe = fs::read(dll.as_ref())
        .with_context(|| format!("cannot read {}", dll.as_ref().display()))?;
    let weights = weights_resource(&pe)?;
    let records = parse_records(weights)?;
    let mut stages = Vec::new();
    let mut tensors = Vec::new();
    fs::create_dir_all(output.join("model"))?;
    for (i, record) in records.iter().enumerate() {
        let id = format!("tensor-{i:03}");
        let file = format!("{i:03}.bin");
        let data = &weights[record.start..record.start + record.len];
        fs::write(output.join("model").join(&file), data)?;
        stages.push(Stage {
            id: id.clone(),
            file,
            packed_byte_length: data.len(),
            sha256: format!("{:x}", Sha256::digest(data)),
        });
        tensors.push(Tensor {
            name: record.name.clone(),
            block: record.block,
            layer: record.layer,
            parameter: "layer",
            stage: id,
            stage_offset: 0,
            byte_length: data.len(),
        });
    }
    fs::write(
        output.join("manifest.json"),
        serde_json::to_vec_pretty(&Manifest {
            totals: Totals { block_count: 71 },
            stages,
            tensors,
        })?,
    )?;
    Ok(ExtractReport {
        records: records.len(),
        packed_bytes: records.iter().map(|v| v.len).sum(),
    })
}
