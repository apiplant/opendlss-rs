//! The weights as the PTX kernels read them: the reference's `nr::Model` (src/nr_model.cpp).
//!
//! Every tensor's raw bytes are on the device, one allocation each, because the kernels read their per-channel
//! scales straight out of them. The matrices are re-laid out on the host: E4M3 as plain K32-tile-major
//! `[K/32][N][32]` with the chained K permutation undone (the MMA's A operand is loaded in natural order), the
//! MLP's first layer with its columns permuted for a register-resident hidden layer, the two f16 matrices plain,
//! and the attention prior with the query in natural and the key in physical (4x4-tiled) order.

use anyhow::{Context, Result, bail};
use std::collections::HashMap;

use super::driver::{Cuda, DeviceBuffer, DevicePtr};
use crate::model::Model;

#[derive(Clone, Copy)]
pub struct Tensor<'m> {
    pub name: &'m str,
    pub bytes: &'m [u8],
    /// The tensor's raw bytes on the device.
    pub raw: DevicePtr,
}

pub struct CudaModel<'m> {
    model: &'m Model,
    raw: HashMap<&'m str, DeviceBuffer>,
    matrices: HashMap<String, DeviceBuffer>,
}

/// Natural window token (row-major in the 8x8 window) -> physical token (4x4 tiles of 16).
fn tiled_token(token: u32) -> u32 {
    let (x, y) = (token & 7, token >> 3);
    (y >> 2) * 32 + (x >> 2) * 16 + (y & 3) * 4 + (x & 3)
}

fn packed_f16_weight_index(input: u32, output: u32, output_channels: u32) -> u32 {
    let n_tiles = output_channels.div_ceil(16);
    let tile = (input >> 4) * n_tiles + (output >> 4);
    let (k, n) = (input & 15, output & 15);
    let lane = ((n & 7) << 2) | ((k & 7) >> 1);
    let fragment = if k >= 8 { 2 } else { 0 } + (k & 1);
    tile * 256 + lane * 8 + ((n >> 3) & 1) * 4 + fragment
}

/// W1' column c holds hidden unit u(c) (mlp_e4m3.py hidden_permutation): per 32-block and 16-half, columns
/// 2t + i -> units 4t + i, columns 8 + 2t + i -> units 4t + 2 + i.
pub fn mlp_hidden_permutation(hidden: u32) -> Vec<u32> {
    let mut permutation = vec![0; hidden as usize];
    for block in 0..hidden / 32 {
        for half in 0..2 {
            for t in 0..4 {
                for i in 0..2 {
                    let base = block * 32 + half * 16;
                    permutation[(base + 2 * t + i) as usize] = base + 4 * t + i;
                    permutation[(base + 8 + 2 * t + i) as usize] = base + 4 * t + 2 + i;
                }
            }
        }
    }
    permutation
}

impl<'m> CudaModel<'m> {
    pub fn upload(cuda: &Cuda, model: &'m Model) -> Result<Self> {
        let mut raw = HashMap::new();
        for spec in &model.manifest.tensors {
            let stage = &model.stage_bytes[&spec.stage];
            let bytes = &stage[spec.stage_offset..spec.stage_offset + spec.byte_length];
            raw.insert(spec.name.as_str(), cuda.upload(bytes)?);
        }
        Ok(Self {
            model,
            raw,
            matrices: HashMap::new(),
        })
    }

    pub fn tensor(&self, block: u32, layer: u32) -> Result<Tensor<'m>> {
        self.named(&format!("block{block}.layer{layer}.layer"))
    }

    pub fn named(&self, name: &str) -> Result<Tensor<'m>> {
        let spec = self
            .model
            .manifest
            .tensors
            .iter()
            .find(|t| t.name == name)
            .with_context(|| format!("missing tensor {name}"))?;
        let stage = &self.model.stage_bytes[&spec.stage];
        Ok(Tensor {
            name: &spec.name,
            bytes: &stage[spec.stage_offset..spec.stage_offset + spec.byte_length],
            raw: self.raw[spec.name.as_str()].ptr,
        })
    }

    pub fn bytes_on_device(&self) -> usize {
        self.raw
            .values()
            .chain(self.matrices.values())
            .map(|b| b.size)
            .sum()
    }

    fn cached(
        &mut self,
        cuda: &Cuda,
        key: String,
        build: impl FnOnce() -> Result<Vec<u8>>,
    ) -> Result<DevicePtr> {
        if let Some(buffer) = self.matrices.get(&key) {
            return Ok(buffer.ptr);
        }
        let buffer = cuda.upload(&build()?)?;
        let ptr = buffer.ptr;
        self.matrices.insert(key, buffer);
        Ok(ptr)
    }

    /// `Model::fp8Matrix`: plain E4M3, K swizzled back to natural order. `batch_k` 0 means one batch;
    /// `tile_major` false is the N-major `[K/batchK][N][batchK]` form.
    #[allow(clippy::too_many_arguments)]
    pub fn fp8_matrix(
        &mut self,
        cuda: &Cuda,
        tensor: Tensor<'_>,
        offset: u32,
        k: u32,
        n: u32,
        batch_k: u32,
        tile_major: bool,
    ) -> Result<DevicePtr> {
        let batch = if batch_k == 0 { k } else { batch_k };
        let key = format!("{}/fp8/{offset}/{k}x{n}/{batch}/{tile_major}", tensor.name);
        let bytes = tensor.bytes;
        self.cached(cuda, key, || {
            Model::relayout_fp8_matrix(
                bytes,
                offset as usize,
                k,
                n,
                true,
                Some(batch),
                tile_major,
                None,
            )
        })
    }

    /// A tile-major `[K/32][padded_n][32]` matrix whose columns past `n` are zero: the PTX GEMM needs N to be a
    /// multiple of 64, and a zero column changes no other column.
    #[allow(clippy::too_many_arguments)]
    pub fn fp8_matrix_padded(
        &mut self,
        cuda: &Cuda,
        tensor: Tensor<'_>,
        offset: u32,
        k: u32,
        n: u32,
        padded_n: u32,
    ) -> Result<DevicePtr> {
        let key = format!("{}/fp8pad/{offset}/{k}x{n}/{padded_n}", tensor.name);
        let bytes = tensor.bytes;
        self.cached(cuda, key, || {
            let plain = Model::relayout_fp8_matrix(
                bytes,
                offset as usize,
                k,
                n,
                true,
                Some(k),
                true,
                None,
            )?;
            let mut padded = vec![0u8; (k * padded_n) as usize];
            for tile in 0..(k / 32) as usize {
                let (from, to) = (tile * n as usize * 32, tile * padded_n as usize * 32);
                padded[to..to + n as usize * 32]
                    .copy_from_slice(&plain[from..from + n as usize * 32]);
            }
            Ok(padded)
        })
    }

    /// `Model::fp8MatrixPermuted`: tile-major with the output columns permuted.
    #[allow(clippy::too_many_arguments)]
    pub fn fp8_matrix_permuted(
        &mut self,
        cuda: &Cuda,
        tensor: Tensor<'_>,
        offset: u32,
        k: u32,
        n: u32,
        batch_k: u32,
        columns: &[u32],
    ) -> Result<DevicePtr> {
        let key = format!("{}/fp8perm/{offset}/{k}x{n}/{batch_k}", tensor.name);
        let bytes = tensor.bytes;
        self.cached(cuda, key, || {
            Model::relayout_fp8_matrix(
                bytes,
                offset as usize,
                k,
                n,
                true,
                Some(batch_k),
                true,
                Some(columns),
            )
        })
    }

    /// A plain `[K][paddedN]` f16 matrix (the input adapter and the head).
    pub fn f16_matrix(
        &mut self,
        cuda: &Cuda,
        tensor: Tensor<'_>,
        offset: u32,
        k: u32,
        n: u32,
    ) -> Result<DevicePtr> {
        let padded_n = n.next_multiple_of(16);
        let bytes = tensor.bytes;
        let name = tensor.name;
        self.cached(cuda, format!("{name}/f16/{offset}/{k}x{n}"), || {
            if !k.is_multiple_of(16) {
                bail!("f16 matrix K must be a multiple of 16");
            }
            let mut plain = vec![0u16; (k * padded_n) as usize];
            for i in 0..k {
                for j in 0..n {
                    let at = (((offset >> 1) + packed_f16_weight_index(i, j, n)) * 2) as usize;
                    if at + 1 >= bytes.len() {
                        bail!("f16 matrix exceeds tensor {name}");
                    }
                    plain[(i * padded_n + j) as usize] =
                        u16::from_le_bytes([bytes[at], bytes[at + 1]]);
                }
            }
            Ok(bytemuck::cast_slice(&plain).to_vec())
        })
    }

    /// `Model::relativeBias`: f16 `[heads][64 query][64 key]`, the query natural and the key physical.
    pub fn relative_bias(
        &mut self,
        cuda: &Cuda,
        tensor: Tensor<'_>,
        offset: u32,
        heads: u32,
    ) -> Result<DevicePtr> {
        let bytes = tensor.bytes;
        let name = tensor.name;
        self.cached(cuda, format!("{name}/prior/{offset}/{heads}"), || {
            let mut prior = vec![0u16; (heads * 64 * 64) as usize];
            for head in 0..heads {
                for query in 0..64 {
                    for physical in 0..64u32 {
                        let (q, k) = (tiled_token(query), physical);
                        let (m, n) = (q & 15, k & 15);
                        let lane = ((m & 7) << 2) | ((n & 7) >> 1);
                        let fragment = if m >= 8 { 2 } else { 0 } + (n & 1);
                        let half =
                            (q >> 4) * 1024 + (k >> 4) * 256 + lane * 8 + (n >> 3) * 4 + fragment;
                        let at = (offset + head * 8192 + half * 2) as usize;
                        if at + 1 >= bytes.len() {
                            bail!("relative bias exceeds tensor {name}");
                        }
                        prior[((head * 64 + query) * 64 + physical) as usize] =
                            u16::from_le_bytes([bytes[at], bytes[at + 1]]);
                    }
                }
            }
            Ok(bytemuck::cast_slice(&prior).to_vec())
        })
    }
}
