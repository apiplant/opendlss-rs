//! The weights on the device, as `ports/browser-webgpu/src/model.js` of the reference lays them out.
//!
//! Every FP8 matrix stays where the model file put it, in tensor-core fragment order: the GEMM addresses that
//! order directly, so a stage file is uploaded as it is and a matrix is a byte offset into it. What is rebuilt
//! on the host is what no kernel addresses in place: the two f16 matrices, the attention prior (stored as
//! accumulator tiles, wanted in natural token order) and a few small scale vectors.

use anyhow::{Context, Result, bail};
use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
};

use crate::{model::Model, runtime::Gpu};

/// One tensor record of the manifest, with its bytes.
#[derive(Clone, Copy)]
pub struct TensorRef<'a> {
    pub name: &'a str,
    pub stage: &'a str,
    pub stage_offset: u32,
    pub bytes: &'a [u8],
}

impl TensorRef<'_> {
    pub fn byte_length(&self) -> u32 {
        self.bytes.len() as u32
    }
}

/// An FP8 matrix in place: its stage buffer and where it starts.
#[derive(Clone)]
pub struct Fp8Matrix {
    pub buffer: wgpu::Buffer,
    pub byte_offset: u32,
    pub k: u32,
    pub matrix_channels: u32,
    pub batch_k: u32,
}

pub struct GpuModel<'m> {
    model: &'m Model,
    index: HashMap<&'m str, usize>,
    stages: HashMap<String, wgpu::Buffer>,
    derived: RefCell<HashMap<String, wgpu::Buffer>>,
    checked: RefCell<HashSet<String>>,
    pub bytes_uploaded: u64,
}

/// Natural window token (row-major in the 8x8 window) -> physical token (4x4 tiles of 16).
pub fn tiled_token(token: u32) -> u32 {
    let (x, y) = (token & 7, token >> 3);
    (y >> 2) * 32 + (x >> 2) * 16 + (y & 3) * 4 + (x & 3)
}

pub fn inverse_tiled_token(token: u32) -> u32 {
    let (tile, within) = (token >> 4, token & 15);
    let x = (tile & 1) * 4 + (within & 3);
    let y = (tile >> 1) * 4 + (within >> 2);
    y * 8 + x
}

/// Half index of an f16 weight inside a packed matrix: 16x16 tiles in (k, n) order.
fn packed_f16_weight_index(input: u32, output: u32, output_channels: u32) -> u32 {
    let n_tiles = output_channels.div_ceil(16);
    let tile = (input >> 4) * n_tiles + (output >> 4);
    let (k, n) = (input & 15, output & 15);
    let lane = ((n & 7) << 2) | ((k & 7) >> 1);
    let fragment = if k >= 8 { 2 } else { 0 } + (k & 1);
    tile * 256 + lane * 8 + ((n >> 3) & 1) * 4 + fragment
}

fn half_at(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([bytes[at], bytes[at + 1]])
}

pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = (bits >> 10) & 0x1f;
    let mantissa = (bits & 0x3ff) as f32;
    match exponent {
        0 => sign * mantissa * 2f32.powi(-24),
        0x1f if mantissa != 0.0 => f32::NAN,
        0x1f => sign * f32::INFINITY,
        e => sign * (1.0 + mantissa / 1024.0) * 2f32.powi(e as i32 - 15),
    }
}

impl<'m> GpuModel<'m> {
    pub fn upload(gpu: &Gpu, model: &'m Model) -> Result<Self> {
        let mut stages = HashMap::new();
        let mut bytes_uploaded = 0;
        for stage in &model.manifest.stages {
            let bytes = &model.stage_bytes[&stage.id];
            // Four bytes of slack: an unaligned weight fragment is read as two words, the second of which can
            // be one word past the last matrix byte.
            let mut padded = bytes.clone();
            padded.resize((bytes.len() + 4).next_multiple_of(4), 0);
            stages.insert(
                stage.id.clone(),
                gpu.storage_from(&format!("stage {}", stage.id), &padded),
            );
            bytes_uploaded += bytes.len() as u64;
        }
        let index = model
            .manifest
            .tensors
            .iter()
            .enumerate()
            .map(|(i, t)| (t.name.as_str(), i))
            .collect();
        Ok(Self {
            model,
            index,
            stages,
            derived: RefCell::new(HashMap::new()),
            checked: RefCell::new(HashSet::new()),
            bytes_uploaded,
        })
    }

    pub fn tensor(&self, block: u32, layer: u32) -> Result<TensorRef<'m>> {
        self.named(&format!("block{block}.layer{layer}.layer"))
    }

    pub fn named(&self, name: &str) -> Result<TensorRef<'m>> {
        let spec = &self.model.manifest.tensors[*self
            .index
            .get(name)
            .with_context(|| format!("missing tensor {name}"))?];
        let stage = &self.model.stage_bytes[&spec.stage];
        Ok(TensorRef {
            name: &spec.name,
            stage: &spec.stage,
            stage_offset: spec.stage_offset as u32,
            bytes: &stage[spec.stage_offset..spec.stage_offset + spec.byte_length],
        })
    }

    /// One FP8 matrix, where it already is. Every weight must satisfy |w| <= 9, which is what makes the
    /// product of two operands scaled by four an exact normal half; the whole GEMM reduction rests on it.
    pub fn fp8_matrix(
        &self,
        tensor: TensorRef<'_>,
        byte_offset: u32,
        k: u32,
        n: u32,
        batch_k: Option<u32>,
    ) -> Result<Fp8Matrix> {
        let batch = batch_k.unwrap_or(k);
        if !k.is_multiple_of(32)
            || !n.is_multiple_of(16)
            || !batch.is_multiple_of(32)
            || !k.is_multiple_of(batch)
        {
            bail!(
                "FP8 matrix shape must be K%32==0, N%16==0, batchK | K: {}",
                tensor.name
            );
        }
        let end = byte_offset as usize + (k * n) as usize;
        if end > tensor.bytes.len() {
            bail!("FP8 matrix exceeds tensor {}", tensor.name);
        }
        let key = format!("{}/{byte_offset}/{k}x{n}", tensor.name);
        if self.checked.borrow_mut().insert(key) {
            for (i, byte) in tensor.bytes[byte_offset as usize..end].iter().enumerate() {
                let magnitude = byte & 0x7f;
                // 0x51 is 9; 0x7f is the NaN code, which the weight table decodes to zero.
                if magnitude > 0x51 && magnitude != 0x7f {
                    bail!(
                        "weight {i} of {} is outside the bounded-half range",
                        tensor.name
                    );
                }
            }
        }
        Ok(Fp8Matrix {
            buffer: self.stages[tensor.stage].clone(),
            byte_offset: tensor.stage_offset + byte_offset,
            k,
            matrix_channels: n,
            batch_k: batch,
        })
    }

    /// The per-channel skip scales, as the GEMM wants them: an offset into the stage its weights come from.
    pub fn aux_offset(&self, tensor: TensorRef<'_>, byte_offset: u32, count: u32) -> Result<u32> {
        if !byte_offset.is_multiple_of(2) || byte_offset + count * 2 > tensor.byte_length() {
            bail!("bad skip scales in {}", tensor.name);
        }
        Ok(tensor.stage_offset + byte_offset)
    }

    fn derived(
        &self,
        gpu: &Gpu,
        key: String,
        build: impl FnOnce() -> Result<Vec<u8>>,
    ) -> Result<wgpu::Buffer> {
        if let Some(buffer) = self.derived.borrow().get(&key) {
            return Ok(buffer.clone());
        }
        let buffer = gpu.storage_from(&key, &build()?);
        self.derived.borrow_mut().insert(key, buffer.clone());
        Ok(buffer)
    }

    /// A plain `[K][paddedN]` f16 matrix, for the input adapter and the head.
    pub fn f16_matrix(
        &self,
        gpu: &Gpu,
        tensor: TensorRef<'_>,
        byte_offset: u32,
        k: u32,
        n: u32,
    ) -> Result<(wgpu::Buffer, u32)> {
        let padded_n = n.next_multiple_of(16);
        let buffer = self.derived(
            gpu,
            format!("{}/f16/{byte_offset}/{k}x{n}", tensor.name),
            || {
                if !k.is_multiple_of(16) {
                    bail!("f16 matrix K must be a multiple of 16");
                }
                let mut plain = vec![0u16; (k * padded_n) as usize];
                for i in 0..k {
                    for j in 0..n {
                        let half = (byte_offset >> 1) + packed_f16_weight_index(i, j, n);
                        let at = half as usize * 2;
                        if at + 1 >= tensor.bytes.len() {
                            bail!("f16 matrix exceeds tensor {}", tensor.name);
                        }
                        plain[(i * padded_n + j) as usize] = half_at(tensor.bytes, at);
                    }
                }
                Ok(bytemuck::cast_slice(&plain).to_vec())
            },
        )?;
        Ok((buffer, padded_n))
    }

    /// The learned attention prior as f16 `[heads][64 query][64 key]`, both in natural window order. The model
    /// stores it as the score MMA's C accumulator: both axes 4x4-tiled inside 16x16 fragments.
    pub fn relative_bias(
        &self,
        gpu: &Gpu,
        tensor: TensorRef<'_>,
        offset: u32,
        heads: u32,
    ) -> Result<wgpu::Buffer> {
        self.derived(
            gpu,
            format!("{}/prior/{offset}/{heads}", tensor.name),
            || {
                let mut prior = vec![0u16; (heads * 64 * 64) as usize];
                for head in 0..heads {
                    for query in 0..64 {
                        let q = tiled_token(query);
                        for k in 0..64u32 {
                            let (m, n) = (q & 15, k & 15);
                            let lane = ((m & 7) << 2) | ((n & 7) >> 1);
                            let fragment = if m >= 8 { 2 } else { 0 } + (n & 1);
                            let half = (q >> 4) * 1024
                                + (k >> 4) * 256
                                + lane * 8
                                + (n >> 3) * 4
                                + fragment;
                            let at = (offset + head * 8192 + half * 2) as usize;
                            if at + 1 >= tensor.bytes.len() {
                                bail!("relative bias exceeds tensor {}", tensor.name);
                            }
                            prior[((head * 64 + query) * 64 + inverse_tiled_token(k)) as usize] =
                                half_at(tensor.bytes, at);
                        }
                    }
                }
                Ok(bytemuck::cast_slice(&prior).to_vec())
            },
        )
    }

    /// The per-head f32 attention scales.
    pub fn head_scales(
        &self,
        gpu: &Gpu,
        tensor: TensorRef<'_>,
        offset: u32,
        heads: u32,
    ) -> Result<wgpu::Buffer> {
        self.derived(
            gpu,
            format!("{}/heads/{offset}/{heads}", tensor.name),
            || {
                let range = offset as usize..(offset + heads * 4) as usize;
                Ok(tensor
                    .bytes
                    .get(range)
                    .with_context(|| format!("head scales exceed {}", tensor.name))?
                    .to_vec())
            },
        )
    }

    /// `count` per-channel halves, as a buffer the elementwise kernels index by column.
    pub fn aux_vector(
        &self,
        gpu: &Gpu,
        tensor: TensorRef<'_>,
        offset: u32,
        count: u32,
    ) -> Result<wgpu::Buffer> {
        self.aux_pair(gpu, tensor, &[offset], count)
    }

    /// Several per-channel half vectors end to end (the post blend reads two).
    pub fn aux_pair(
        &self,
        gpu: &Gpu,
        tensor: TensorRef<'_>,
        offsets: &[u32],
        count: u32,
    ) -> Result<wgpu::Buffer> {
        self.derived(
            gpu,
            format!("{}/aux/{offsets:?}/{count}", tensor.name),
            || {
                let mut values = Vec::new();
                for offset in offsets {
                    let range = *offset as usize..(offset + count * 2) as usize;
                    values.extend_from_slice(
                        tensor
                            .bytes
                            .get(range)
                            .with_context(|| format!("scales exceed {}", tensor.name))?,
                    );
                }
                Ok(values)
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiled_tokens_invert() {
        for token in 0..64 {
            assert_eq!(inverse_tiled_token(tiled_token(token)), token);
        }
    }

    #[test]
    fn half_decode() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
    }
}
