use anyhow::{Context, Result, bail};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

pub const BLOCK_COUNT: u32 = 71;

#[derive(Debug, Deserialize)]
pub struct Manifest {
    pub totals: Totals,
    pub stages: Vec<StageSpec>,
    pub tensors: Vec<TensorSpec>,
}

#[derive(Debug, Deserialize)]
pub struct Totals {
    #[serde(rename = "blockCount")]
    pub block_count: u32,
}
#[derive(Debug, Deserialize)]
pub struct StageSpec {
    pub id: String,
    pub file: PathBuf,
    #[serde(rename = "packedByteLength")]
    pub packed_byte_length: usize,
    pub sha256: String,
}
#[derive(Debug, Deserialize)]
pub struct TensorSpec {
    pub name: String,
    pub block: i32,
    pub layer: i32,
    pub parameter: String,
    pub stage: String,
    #[serde(rename = "stageOffset")]
    pub stage_offset: usize,
    #[serde(rename = "byteLength")]
    pub byte_length: usize,
}

#[derive(Debug)]
pub struct Model {
    pub manifest: Manifest,
    pub stage_bytes: HashMap<String, Vec<u8>>,
}

impl Model {
    pub fn load(directory: impl AsRef<Path>, verify_hashes: bool) -> Result<Self> {
        let directory = directory.as_ref();
        let text = fs::read_to_string(directory.join("manifest.json"))
            .with_context(|| format!("cannot read {}/manifest.json", directory.display()))?;
        let manifest: Manifest = serde_json::from_str(&text).context("invalid model manifest")?;
        if manifest.totals.block_count != BLOCK_COUNT {
            bail!(
                "model has {} blocks; this graph requires {BLOCK_COUNT}",
                manifest.totals.block_count
            );
        }
        let mut stage_bytes = HashMap::new();
        for stage in &manifest.stages {
            if stage.id.is_empty()
                || stage.file.is_absolute()
                || stage
                    .file
                    .components()
                    .any(|c| matches!(c, std::path::Component::ParentDir))
            {
                bail!("invalid stage path for {:?}", stage.id);
            }
            let path = directory.join("model").join(&stage.file);
            let bytes =
                fs::read(&path).with_context(|| format!("cannot read stage {}", path.display()))?;
            if bytes.len() != stage.packed_byte_length {
                bail!(
                    "stage {} has {} bytes; manifest requires {}",
                    stage.id,
                    bytes.len(),
                    stage.packed_byte_length
                );
            }
            if verify_hashes {
                let actual = format!("{:x}", Sha256::digest(&bytes));
                if !actual.eq_ignore_ascii_case(&stage.sha256) {
                    bail!("stage {} SHA-256 mismatch", stage.id);
                }
            }
            if stage_bytes.insert(stage.id.clone(), bytes).is_some() {
                bail!("duplicate stage id {}", stage.id);
            }
        }
        for tensor in &manifest.tensors {
            let bytes = stage_bytes.get(&tensor.stage).with_context(|| {
                format!(
                    "tensor {} references unknown stage {}",
                    tensor.name, tensor.stage
                )
            })?;
            if tensor
                .stage_offset
                .checked_add(tensor.byte_length)
                .is_none_or(|end| end > bytes.len())
            {
                bail!("tensor {} exceeds stage {}", tensor.name, tensor.stage);
            }
        }
        Ok(Self {
            manifest,
            stage_bytes,
        })
    }

    pub fn packed_input_index(k: u32) -> u32 {
        let base = k & !31;
        let within = k & 31;
        let half = within & 16;
        let quarter = within & 15;
        base + half + (quarter >> 2) * 2 + (quarter & 1) + u32::from((quarter & 2) != 0) * 8
    }

    pub fn inverse_packed_input_index(k: u32) -> u32 {
        let base = k & !31;
        let within = k & 31;
        base + (within & 17) + ((within & 2) << 1) + ((within & 4) << 1) + ((within & 8) >> 2)
    }

    pub fn packed_weight_index(k: u32, n: u32, output_channels: u32) -> u32 {
        let k_tile = k >> 5;
        let k_in = k & 31;
        let n_tile = n >> 7;
        let n_in = n & 127;
        let n_half = n_in >> 6;
        let n_group = (n_in & 63) >> 4;
        let n_in_group = n_in & 15;
        let lane = ((n_in_group & 7) << 2) | ((k_in & 15) >> 2);
        let byte_in_lane = ((n_in_group >> 3) << 3) | ((k_in >> 4) << 2) | (k_in & 3);
        k_tile * output_channels * 32
            + n_tile * 4096
            + n_half * 2048
            + n_group * 512
            + lane * 16
            + byte_in_lane
    }

    /// Converts a native packed E4M3 matrix to the K32-tile-major layout used
    /// by the Vulkan cooperative-matrix kernels. The source slice is a tensor
    /// subrange beginning at its matrix offset. `column_source`, when given,
    /// permutes the output columns: plain column c holds tensor column
    /// `column_source[c]` (the PTX MLP's register-resident hidden layer).
    #[allow(clippy::too_many_arguments)]
    pub fn relayout_fp8_matrix(
        source: &[u8],
        offset: usize,
        k: u32,
        n: u32,
        swizzle_k: bool,
        batch_k: Option<u32>,
        tile_major: bool,
        column_source: Option<&[u32]>,
    ) -> Result<Vec<u8>> {
        let batch_k = batch_k.unwrap_or(k);
        if batch_k == 0
            || !k.is_multiple_of(32)
            || !n.is_multiple_of(16)
            || !batch_k.is_multiple_of(32)
            || !k.is_multiple_of(batch_k)
        {
            bail!(
                "FP8 matrix requires K and batchK multiples of 32, N multiple of 16, and batchK | K"
            );
        }
        let byte_count = (k as usize) * (n as usize);
        let end = offset
            .checked_add(byte_count)
            .context("FP8 matrix offset overflow")?;
        let packed = source
            .get(offset..end)
            .context("FP8 matrix exceeds its tensor")?;
        if column_source.is_some_and(|columns| columns.len() != n as usize) {
            bail!("column permutation size mismatch");
        }
        let mut plain = vec![0; byte_count];
        for j in 0..k {
            let source_k = if swizzle_k {
                Self::inverse_packed_input_index(j)
            } else {
                j
            };
            for column in 0..n {
                let source_column =
                    column_source.map_or(column, |columns| columns[column as usize]);
                let source_index = Self::packed_weight_index(source_k, source_column, n) as usize;
                let mut code = packed[source_index];
                // E4M3FN's sole NaN encoding is defined as zero by the reference route.
                if code & 0x7f == 0x7f {
                    code = 0;
                }
                let destination = if tile_major {
                    (((j / batch_k * (batch_k / 32) + (j % batch_k) / 32) * n + column) * 32
                        + j % 32) as usize
                } else {
                    ((j / batch_k * n + column) * batch_k + j % batch_k) as usize
                };
                plain[destination] = code;
            }
        }
        Ok(plain)
    }
}

#[cfg(test)]
mod tests {
    use super::Model;
    #[test]
    fn input_permutation_is_invertible() {
        for k in 0..64 {
            assert_eq!(
                Model::inverse_packed_input_index(Model::packed_input_index(k)),
                k
            );
        }
    }
    #[test]
    fn packed_weight_indices_cover_a_tile() {
        let mut positions = std::collections::HashSet::new();
        for k in 0..32 {
            for n in 0..128 {
                positions.insert(Model::packed_weight_index(k, n, 128));
            }
        }
        assert_eq!(positions.len(), 4096);
    }

    #[test]
    fn relayout_reverses_native_weight_packing() {
        let mut packed = vec![0; 32 * 16];
        for k in 0..32 {
            for n in 0..16 {
                packed[Model::packed_weight_index(k, n, 16) as usize] = ((k * 16 + n) % 126) as u8;
            }
        }
        let plain =
            Model::relayout_fp8_matrix(&packed, 0, 32, 16, false, None, true, None).unwrap();
        for k in 0..32 {
            for n in 0..16 {
                assert_eq!(plain[(n * 32 + k) as usize], ((k * 16 + n) % 126) as u8);
            }
        }
    }
}
