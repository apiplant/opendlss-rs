//! The network on the CUDA backend: the reference's `nr::Graph::record` (src/nr_graph.cpp) in its production
//! configuration - every 32-channel block one fused launch with the input adapter, the pools, the decoder
//! merge, the post blend and the head folded in; the expert stages' FFN and W3 in one launch with the previous
//! block's projection computed on chip; QKV, normalization and window attention in one launch; the ViT on the
//! streamed attention route - recorded once and replayed as one CUDA graph per frame.
//!
//! One deliberate difference: the 512-channel stage's branch MLP, a GLSL kernel upstream with no PTX twin, runs
//! as its two GEMMs per branch on the PTX GEMM. It publishes the hidden layer as E4M3 either way, so the result
//! is the same bytes (the upstream unfused route runs exactly these GEMMs).

use anyhow::{Context, Result, bail};
use std::{collections::HashMap, time::Instant};

use super::{
    driver::{Cuda, DeviceBuffer, DevicePtr, Graph as CudaGraph, HostBuffer},
    kernels::{Activation, Block32Args, FfnArgs, Format, GemmArgs, Kernels, align_rows},
    weights::{CudaModel, Tensor, mlp_hidden_permutation},
};
use crate::{
    geometry::{BlockLayout, Geometry, Level, WindowPhases, window_phase},
    model::Model,
    network::{Conditioning, Image},
};

/// The previous block's projection, deferred into the next block's FFN kernel.
struct Pending {
    block: u32,
    attended: Activation,
    ffn_out: Activation,
    wproj: DevicePtr,
    aux_attn: u32,
    aux: DevicePtr,
}

struct Temps {
    ffn_quantized: Activation,
    ffn_quantized2: Option<Activation>,
    attended: Activation,
}

struct SplitTemps {
    branch: Activation,
    middle: Activation,
    layer0: Activation,
    ffn_residual: Activation,
    attended: Activation,
}

struct Recorder<'c, 'm> {
    cuda: &'c Cuda,
    kernels: Kernels<'c>,
    model: CudaModel<'m>,
    geometry: Geometry,
    buffers: HashMap<String, (DeviceBuffer, Activation)>,
    phases: WindowPhases,
    pending: Option<Pending>,
    permutation: Vec<u32>,
    /// Keep a copy of every block output (`block-N`), for checking against another route.
    capture: bool,
}

const FULL_LEVEL: usize = 6;

fn rows(level: Level) -> u32 {
    level.width * level.height
}

impl<'c, 'm> Recorder<'c, 'm> {
    fn capture(&mut self, name: &str, source: &Activation) -> Result<()> {
        if self.capture {
            let copy = self.allocate(
                &format!("boundary {name}"),
                source.rows,
                source.channels,
                source.format,
            )?;
            self.kernels.copy(source, &copy);
        }
        Ok(())
    }

    fn allocate(
        &mut self,
        label: &str,
        rows: u32,
        channels: u32,
        format: Format,
    ) -> Result<Activation> {
        let key = format!("{label}/{rows}x{channels}/{format:?}");
        if let Some((_, activation)) = self.buffers.get(&key) {
            return Ok(activation.clone());
        }
        let alloc_rows = align_rows(rows);
        let buffer = self
            .cuda
            .alloc(alloc_rows as usize * channels as usize * format.bytes() as usize)?;
        let activation = Activation {
            ptr: buffer.ptr,
            rows,
            channels,
            alloc_rows,
            format,
            label: label.to_string(),
        };
        self.buffers.insert(key, (buffer, activation.clone()));
        Ok(activation)
    }

    fn fp8(&mut self, tensor: Tensor<'_>, offset: u32, k: u32, n: u32) -> Result<DevicePtr> {
        self.model
            .fp8_matrix(self.cuda, tensor, offset, k, n, 0, true)
    }

    /// The fused 32-channel block's weights (the PTX kernel's W1 permuted, W2 tile-major).
    fn block32_weights(
        &mut self,
        tensor: Tensor<'_>,
        layout: &BlockLayout,
        args: &mut Block32Args<'_>,
    ) -> Result<()> {
        let permutation = self.permutation.clone();
        args.w1 = self.model.fp8_matrix_permuted(
            self.cuda,
            tensor,
            layout.expand,
            32,
            layout.hidden,
            32,
            &permutation,
        )?;
        args.w2 = self.model.fp8_matrix(
            self.cuda,
            tensor,
            layout.contract_weights,
            layout.hidden,
            32,
            layout.hidden,
            true,
        )?;
        args.wqkv = self.fp8(tensor, layout.qkv, 32, 96)?;
        args.wproj = self.fp8(tensor, layout.projection, 32, 32)?;
        args.prior = self
            .model
            .relative_bias(self.cuda, tensor, layout.relative, 1)?;
        args.aux = tensor.raw;
        args.ffn_scale_byte_offset = layout.ffn_cos_skip;
        args.attn_scale_byte_offset = layout.attn_cos_skip;
        args.attention_scale_byte_offset = layout.scale;
        Ok(())
    }

    fn temporaries(&mut self, label: &str, rows: u32, channels: u32) -> Result<Temps> {
        Ok(Temps {
            ffn_quantized: self.allocate(
                &format!("{label} FFN quantized"),
                rows,
                channels,
                Format::E4,
            )?,
            // The expert stages alternate their FFN publication between two buffers, so a projection deferred into
            // the next block's FFN can still read the previous one; c256 runs its projection separately.
            ffn_quantized2: if channels <= 128 {
                Some(self.allocate(
                    &format!("{label} FFN quantized B"),
                    rows,
                    channels,
                    Format::E4,
                )?)
            } else {
                None
            },
            attended: self.allocate(&format!("{label} attended"), rows, channels, Format::E4)?,
        })
    }

    fn split_temporaries(&mut self, label: &str, rows: u32) -> Result<SplitTemps> {
        Ok(SplitTemps {
            branch: self.allocate(&format!("{label} split branch"), rows, 512, Format::E4)?,
            middle: self.allocate(&format!("{label} split middle"), rows, 2048, Format::E4)?,
            layer0: self.allocate(&format!("{label} split layer0"), rows, 512, Format::E4)?,
            ffn_residual: self.allocate(
                &format!("{label} split residual"),
                rows,
                512,
                Format::E4,
            )?,
            attended: self.allocate(&format!("{label} split attended"), rows, 512, Format::E4)?,
        })
    }

    /// `Graph::encodeFusedBlock`: FFN -> QKV + window attention -> projection.
    #[allow(clippy::too_many_arguments)]
    fn block(
        &mut self,
        temps: &Temps,
        state: &Activation,
        output: &Activation,
        block: u32,
        channels: u32,
        level: Level,
        phase: u32,
        layout: &BlockLayout,
        raw_output: Option<&Activation>,
        pooled: Option<(&Activation, u32)>,
        defer_projection: bool,
    ) -> Result<()> {
        let tensor = self.model.tensor(block, 0)?;
        let (shift_x, shift_y) = window_phase(phase);
        let (width, height) = (level.width, level.height);
        let row_count = width * height;
        if channels == 32 {
            if raw_output.is_some() {
                bail!("the PTX fused block has no raw f16 output");
            }
            let mut f = Block32Args {
                state: Some(state),
                out_e4: Some(output),
                pooled: pooled.map(|(p, _)| p),
                pooled_width: pooled.map_or(0, |(_, w)| w),
                width,
                height,
                shift_x,
                shift_y,
                ..Default::default()
            };
            self.block32_weights(tensor, layout, &mut f)?;
            return self.kernels.fused_block32(&f);
        }
        if pooled.is_some() {
            bail!("pooled output needs the fused 32-channel block");
        }
        let ffn_out = match (&temps.ffn_quantized2, block & 1) {
            (Some(second), 1) => second.clone(),
            _ => temps.ffn_quantized.clone(),
        };
        // C/32 expert paths, each C -> 128 -> 32, then W3 over their concatenation carrying the skip: one launch.
        let experts = layout.expert_count;
        let w2_base = layout.expand + experts * channels * 128;
        let w3_base = w2_base + experts * 128 * 32;
        let permutation = self.permutation.clone();
        let mut ffn = FfnArgs {
            input: Some(state),
            w1: self.model.fp8_matrix_permuted(
                self.cuda,
                tensor,
                layout.expand,
                experts * channels,
                128,
                channels,
                &permutation,
            )?,
            w2: self
                .model
                .fp8_matrix(self.cuda, tensor, w2_base, experts * 128, 32, 128, true)?,
            w3: self.fp8(tensor, w3_base, channels, channels)?,
            aux: tensor.raw,
            aux_byte_offset: layout.ffn_cos_skip,
            output: Some(&ffn_out),
            rows: row_count,
            channels,
            width,
            ..Default::default()
        };
        let pending = self.pending.take();
        // A deferred block's output only exists inside this kernel; a capture asks the kernel to store it.
        let state_out = match &pending {
            Some(p) if self.capture => Some(self.allocate(
                &format!("boundary block-{}", p.block),
                state.rows,
                state.channels,
                Format::E4,
            )?),
            _ => None,
        };
        ffn.state_out = state_out.as_ref();
        if let Some(p) = &pending {
            ffn.attended = Some(&p.attended);
            ffn.ffn_prev = Some(&p.ffn_out);
            ffn.wproj = p.wproj;
            ffn.aux_attn_byte_offset = p.aux_attn;
            ffn.aux_prev = p.aux;
        }
        self.kernels.expert_ffn(&ffn)?;
        self.capture(&format!("block-{block} ffn"), &ffn_out)?;

        let wqkv = self.fp8(tensor, layout.qkv, channels, channels * 3)?;
        let prior = self
            .model
            .relative_bias(self.cuda, tensor, layout.relative, layout.heads)?;
        self.kernels.qkv_attention(
            &ffn_out,
            wqkv,
            tensor.raw,
            layout.scale,
            prior,
            &temps.attended,
            width,
            height,
            layout.heads,
            shift_x,
            shift_y,
        )?;
        self.capture(&format!("block-{block} attended"), &temps.attended)?;

        let wproj = self.fp8(tensor, layout.projection, channels, channels)?;
        if defer_projection {
            if raw_output.is_some() {
                bail!("a deferred projection has no raw output");
            }
            self.pending = Some(Pending {
                block,
                attended: temps.attended.clone(),
                ffn_out,
                wproj,
                aux_attn: layout.attn_cos_skip,
                aux: tensor.raw,
            });
            return Ok(());
        }
        // The expert blocks reuse the E4 FFN publication as the attention skip.
        self.kernels.gemm_fp8(&GemmArgs {
            input: Some(&temps.attended),
            weights: wproj,
            n_matrix: channels,
            output: Some(raw_output.unwrap_or(output)),
            dual_output: raw_output.map(|_| output),
            quantize: raw_output.is_none(),
            residual: Some(&ffn_out),
            scale_residual: true,
            aux: tensor.raw,
            aux_byte_offset: layout.attn_cos_skip,
            rows: row_count,
            k: channels,
            n: channels,
            ..Default::default()
        })
    }

    /// `Graph::encodeSplitBlock`: eight 64 -> 256 -> 64 branches over one 512 -> 512 layer, 16-head attention.
    #[allow(clippy::too_many_arguments)]
    fn split_block(
        &mut self,
        temps: &SplitTemps,
        state: &Activation,
        output: &Activation,
        block: u32,
        level: Level,
        phase: u32,
        raw_output: Option<&Activation>,
    ) -> Result<()> {
        let row_count = rows(level);
        let (channels, branches, branch_channels, middle_channels, heads) = (512, 8, 64, 256, 16);
        let branch_tensor = self.model.tensor(block, 0)?;
        let contract = self.model.tensor(block, 1)?;
        let qkv_tensor = self.model.tensor(block, 2)?;
        let projection = self.model.tensor(block, 3)?;
        let w2_base = branches * channels * branch_channels;
        let w3_base = w2_base + branches * branch_channels * middle_channels;
        let qkv_relative = channels * channels * 3;
        let qkv_scale = qkv_relative + heads * 8192;

        let w1 = self.fp8(branch_tensor, 0, channels, channels)?;
        self.kernels.gemm_fp8(&GemmArgs {
            input: Some(state),
            weights: w1,
            n_matrix: channels,
            output: Some(&temps.branch),
            quantize: true,
            rows: row_count,
            k: channels,
            n: channels,
            ..Default::default()
        })?;
        // The branch MLP: per branch, a SiLU GEMM into the 256-wide middle and a GEMM back to 64, each branch's
        // slice addressed by column offsets and its matrix by a byte offset into the batched tile-major matrix.
        let w2 = self.model.fp8_matrix(
            self.cuda,
            branch_tensor,
            w2_base,
            branches * branch_channels,
            middle_channels,
            branch_channels,
            true,
        )?;
        let w3 = self.model.fp8_matrix(
            self.cuda,
            branch_tensor,
            w3_base,
            branches * middle_channels,
            branch_channels,
            middle_channels,
            true,
        )?;
        // Every branch in one launch per layer; `NR_UNBATCHED_MLP=1` runs one launch per branch instead (the same
        // bytes: tools/compare_kernels.sh style check).
        let layers = [
            (
                &temps.branch,
                w2,
                branch_channels,
                middle_channels,
                &temps.middle,
                true,
            ),
            (
                &temps.middle,
                w3,
                middle_channels,
                branch_channels,
                &temps.layer0,
                false,
            ),
        ];
        for (input, weights, k, n, output, silu) in layers {
            let args = |branch: u32| GemmArgs {
                input: Some(input),
                input_column_base: branch * k,
                weights: weights + (branch * k * n) as u64,
                n_matrix: n,
                output: Some(output),
                output_column_offset: branch * n,
                silu,
                quantize: true,
                rows: row_count,
                k,
                n,
                ..Default::default()
            };
            if std::env::var("NR_UNBATCHED_MLP").is_ok_and(|v| v == "1") {
                for branch in 0..branches {
                    self.kernels.gemm_fp8(&args(branch))?;
                }
            } else {
                self.kernels.gemm_fp8_batched(&args(0), branches)?;
            }
        }
        let wc = self.fp8(contract, 0, channels, channels)?;
        self.kernels.gemm_fp8(&GemmArgs {
            input: Some(&temps.layer0),
            weights: wc,
            n_matrix: channels,
            output: Some(&temps.ffn_residual),
            quantize: true,
            residual: Some(state),
            scale_residual: true,
            aux: contract.raw,
            aux_byte_offset: channels * channels,
            rows: row_count,
            k: channels,
            n: channels,
            ..Default::default()
        })?;
        let (shift_x, shift_y) = window_phase(phase);
        let wqkv = self.fp8(qkv_tensor, 0, channels, channels * 3)?;
        let prior = self
            .model
            .relative_bias(self.cuda, qkv_tensor, qkv_relative, heads)?;
        self.kernels.qkv_attention(
            &temps.ffn_residual,
            wqkv,
            qkv_tensor.raw,
            qkv_scale,
            prior,
            &temps.attended,
            level.width,
            level.height,
            heads,
            shift_x,
            shift_y,
        )?;
        let wp = self.fp8(projection, 0, channels, channels)?;
        self.kernels.gemm_fp8(&GemmArgs {
            input: Some(&temps.attended),
            weights: wp,
            n_matrix: channels,
            output: Some(raw_output.unwrap_or(output)),
            dual_output: raw_output.map(|_| output),
            quantize: raw_output.is_none(),
            residual: Some(&temps.ffn_residual),
            scale_residual: true,
            aux: projection.raw,
            aux_byte_offset: channels * channels,
            rows: row_count,
            k: channels,
            n: channels,
            ..Default::default()
        })
    }

    /// `Graph::encodeVit`: eight 1024-channel blocks attending over every token of the coarsest level.
    fn vit(&mut self, state: &Activation, tokens: u32) -> Result<()> {
        let (channels, heads, ffn_channels) = (1024, 32, 4096);
        let padded = self.geometry.padded_vit_tokens();
        let expanded = self.allocate("ViT FFN 4096", tokens, ffn_channels, Format::E4)?;
        let ffn_residual = self.allocate("ViT FFN residual", tokens, channels, Format::E4)?;
        let qkv = self.allocate("ViT QKV", tokens, channels * 3, Format::F16)?;
        let normalized = self.allocate("ViT normalized QKV", padded, channels * 3, Format::E4)?;
        let attended = self.allocate("ViT attended", tokens, channels, Format::E4)?;
        for block in 31..=38 {
            let expand = self.model.tensor(block, 0)?;
            let contract = self.model.tensor(block, 1)?;
            let qkv_tensor = self.model.tensor(block, 2)?;
            let projection = self.model.tensor(block, 4)?;
            let we = self.fp8(expand, 0, channels, ffn_channels)?;
            self.kernels.gemm_fp8(&GemmArgs {
                input: Some(state),
                weights: we,
                n_matrix: ffn_channels,
                output: Some(&expanded),
                silu: true,
                quantize: true,
                rows: tokens,
                k: channels,
                n: ffn_channels,
                ..Default::default()
            })?;
            let wc = self.fp8(contract, 0, ffn_channels, channels)?;
            self.kernels.gemm_fp8(&GemmArgs {
                input: Some(&expanded),
                weights: wc,
                n_matrix: channels,
                output: Some(&ffn_residual),
                quantize: true,
                residual: Some(state),
                scale_residual: true,
                aux: contract.raw,
                aux_byte_offset: ffn_channels * channels,
                rows: tokens,
                k: ffn_channels,
                n: channels,
                partition: 1024,
                ..Default::default()
            })?;
            // The ViT's qkv tensor puts its per-head scales before the weights.
            let wq = self.fp8(qkv_tensor, heads * 4, channels, channels * 3)?;
            self.kernels.gemm_fp8(&GemmArgs {
                input: Some(&ffn_residual),
                weights: wq,
                n_matrix: channels * 3,
                output: Some(&qkv),
                quantize: false,
                rows: tokens,
                k: channels,
                n: channels * 3,
                partition: 512,
                ..Default::default()
            })?;
            if padded <= 256 {
                self.kernels.global_attention(
                    &qkv,
                    qkv_tensor.raw,
                    0,
                    &attended,
                    tokens,
                    padded,
                    heads,
                )?;
            } else {
                self.kernels.global_normalize(
                    &qkv,
                    qkv_tensor.raw,
                    0,
                    &normalized,
                    tokens,
                    padded,
                    heads,
                )?;
                self.kernels.global_attention_stream(
                    &normalized,
                    &attended,
                    tokens,
                    padded,
                    heads,
                )?;
            }
            let wp = self.fp8(projection, 0, channels, channels)?;
            self.kernels.gemm_fp8(&GemmArgs {
                input: Some(&attended),
                weights: wp,
                n_matrix: channels,
                output: Some(state),
                quantize: true,
                residual: Some(&ffn_residual),
                scale_residual: true,
                aux: projection.raw,
                aux_byte_offset: channels * channels,
                rows: tokens,
                k: channels,
                n: channels,
                partition: 256,
                ..Default::default()
            })?;
        }
        Ok(())
    }

    /// `Graph::record`, the production route. `features` is f32 `[full rows][16]`; returns the f32 head.
    fn record(&mut self, features: &Activation) -> Result<Activation> {
        let g = self.geometry.clone();
        let full_rows = g.full_width * g.full_height;
        let [d0, d1, d2, d3, d4, d5] = g.levels;
        let full = Level {
            width: g.full_width,
            height: g.full_height,
        };

        // ---- Block 0 with the input adapter in front and the 2x2 pool behind, in one launch.
        let pre = BlockLayout::pre();
        let pre_tensor = self.model.tensor(0, 0)?;
        if pre_tensor.bytes.len() as u32 != pre.end_without_padding + 16 {
            bail!("unexpected block 0 layout");
        }
        let adapter = self.allocate("retained full block0", full_rows, 32, Format::E4)?;
        let resized = self.allocate("block0 downsample", rows(d0), 32, Format::E4)?;
        {
            let adapter_weights =
                self.model
                    .f16_matrix(self.cuda, pre_tensor, pre.input_adapter, 16, 32)?;
            let (shift_x, shift_y) = window_phase(self.phases.take(FULL_LEVEL));
            let mut f = Block32Args {
                features: Some(features),
                adapter_weights,
                out_e4: Some(&adapter),
                pooled: Some(&resized),
                pooled_width: d0.width,
                width: g.full_width,
                height: g.full_height,
                shift_x,
                shift_y,
                ..Default::default()
            };
            self.block32_weights(pre_tensor, &pre, &mut f)?;
            self.kernels.fused_block32(&f)?;
        }
        self.capture("block-0", &adapter)?;

        // ---- Level 0: four 32-channel blocks, the last pooling into level 1.
        let mut state = resized;
        let mut scratch = self.allocate("encoder 32 state", rows(d0), 32, Format::E4)?;
        let latent = self.temporaries("encoder 32", rows(d0), 32)?;
        let downsampled32 = self.allocate("encoder 32 downsample", rows(d1), 32, Format::E4)?;
        let fused32 = BlockLayout::fused(32);
        for block in 1..=4 {
            let phase = self.phases.take(0);
            self.block(
                &latent,
                &state,
                &scratch,
                block,
                32,
                d0,
                phase,
                &fused32,
                None,
                (block == 4).then_some((&downsampled32, d1.width)),
                false,
            )?;
            std::mem::swap(&mut state, &mut scratch);
            if self.pending.is_none() {
                self.capture(&format!("block-{block}"), &state)?;
            }
        }
        let skip32 = state;
        let next64 = self.allocate("encoder 64 input", rows(d1), 64, Format::E4)?;
        let wt = self.fp8(
            self.model.tensor(4, 0)?,
            fused32.end_without_padding,
            32,
            64,
        )?;
        self.kernels.gemm_fp8(&GemmArgs {
            input: Some(&downsampled32),
            weights: wt,
            n_matrix: 64,
            output: Some(&next64),
            quantize: true,
            rows: rows(d1),
            k: 32,
            n: 64,
            ..Default::default()
        })?;

        // ---- Encoder stages 64 / 128 / 256.
        let mut skips = Vec::new();
        let mut stage_input = next64;
        for (level, next, channels, first, last, level_index) in [
            (d1, d2, 64, 5, 8, 1),
            (d2, d3, 128, 9, 14, 2),
            (d3, d4, 256, 15, 22, 3),
        ] {
            let label = format!("encoder {channels}");
            let mut st = stage_input.clone();
            let mut sc =
                self.allocate(&format!("{label} state"), rows(level), channels, Format::E4)?;
            let raw = self.allocate(
                &format!("{label} raw transition"),
                rows(level),
                channels,
                Format::F16,
            )?;
            let temps = self.temporaries(&label, rows(level), channels)?;
            let layout = BlockLayout::fused(channels);
            for block in first..=last {
                let phase = self.phases.take(level_index);
                self.block(
                    &temps,
                    &st,
                    &sc,
                    block,
                    channels,
                    level,
                    phase,
                    &layout,
                    (block == last).then_some(&raw),
                    None,
                    temps.ffn_quantized2.is_some() && block != last,
                )?;
                std::mem::swap(&mut st, &mut sc);
                if self.pending.is_none() {
                    self.capture(&format!("block-{block}"), &st)?;
                }
            }
            skips.push(st);
            let pooled = self.allocate(
                &format!("{label} downsample"),
                rows(next),
                channels,
                Format::E4,
            )?;
            self.kernels.downsample(
                &raw,
                &pooled,
                level.width,
                level.height,
                next.width,
                next.height,
            )?;
            let next_input = self.allocate(
                &format!("{label} next stage"),
                rows(next),
                channels * 2,
                Format::E4,
            )?;
            let wt = self.fp8(
                self.model.tensor(last, 0)?,
                layout.end_without_padding,
                channels,
                channels * 2,
            )?;
            self.kernels.gemm_fp8(&GemmArgs {
                input: Some(&pooled),
                weights: wt,
                n_matrix: channels * 2,
                output: Some(&next_input),
                quantize: true,
                rows: rows(next),
                k: channels,
                n: channels * 2,
                ..Default::default()
            })?;
            stage_input = next_input;
        }
        let [skip64, skip128, skip256]: [Activation; 3] = skips.try_into().unwrap();

        // ---- Encoder 512, the ViT, decoder 512.
        {
            let mut st = stage_input.clone();
            let mut sc = self.allocate("encoder 512 state", rows(d4), 512, Format::E4)?;
            let raw = self.allocate("encoder 512 raw transition", rows(d4), 512, Format::F16)?;
            let temps = self.split_temporaries("encoder 512", rows(d4))?;
            for block in 23..=30 {
                let phase = self.phases.take(4);
                self.split_block(
                    &temps,
                    &st,
                    &sc,
                    block,
                    d4,
                    phase,
                    (block == 30).then_some(&raw),
                )?;
                std::mem::swap(&mut st, &mut sc);
                if self.pending.is_none() {
                    self.capture(&format!("block-{block}"), &st)?;
                }
            }
            let skip512 = st;
            let tokens = g.vit_tokens();
            let pooled = self.allocate("encoder 512 pooled", tokens, 512, Format::E4)?;
            self.kernels
                .downsample(&raw, &pooled, d4.width, d4.height, d5.width, d5.height)?;
            let vit_state = self.allocate("ViT state", tokens, 1024, Format::E4)?;
            let wt = self.fp8(self.model.tensor(30, 4)?, 0, 512, 1024)?;
            self.kernels.gemm_fp8(&GemmArgs {
                input: Some(&pooled),
                weights: wt,
                n_matrix: 1024,
                output: Some(&vit_state),
                quantize: true,
                rows: tokens,
                k: 512,
                n: 1024,
                ..Default::default()
            })?;
            self.vit(&vit_state, tokens)?;

            let projected = self.allocate("decoder 512 projection", tokens, 512, Format::F16)?;
            let t39 = self.model.tensor(39, 0)?;
            let wp = self.fp8(t39, 0, 1024, 512)?;
            self.kernels.gemm_fp8(&GemmArgs {
                input: Some(&vit_state),
                weights: wp,
                n_matrix: 512,
                output: Some(&projected),
                quantize: false,
                rows: tokens,
                k: 1024,
                n: 512,
                partition: 256,
                ..Default::default()
            })?;
            let merged = self.allocate("decoder 512 skip merge", rows(d4), 512, Format::E4)?;
            self.kernels.upsample_residual(
                &projected,
                &skip512,
                t39.raw,
                1024 * 512,
                &merged,
                None,
                d5.width,
                d4.width,
            )?;
            let mut dst = merged;
            let mut dsc = self.allocate("decoder 512 state", rows(d4), 512, Format::E4)?;
            let dtemps = self.split_temporaries("decoder 512", rows(d4))?;
            for block in 40..=47 {
                let phase = self.phases.take(4);
                self.split_block(&dtemps, &dst, &dsc, block, d4, phase, None)?;
                std::mem::swap(&mut dst, &mut dsc);
                if self.pending.is_none() {
                    self.capture(&format!("block-{block}"), &dst)?;
                }
            }
            stage_input = dst;
        }

        // ---- Decoder stages 256 / 128 / 64 / 32.
        for (low, high, channels, first, last, level_index, skip) in [
            (d4, d3, 256, 48, 55, 3, skip256),
            (d3, d2, 128, 56, 61, 2, skip128),
            (d2, d1, 64, 62, 65, 1, skip64),
            (d1, d0, 32, 66, 69, 0, skip32),
        ] {
            let label = format!("decoder {channels}");
            let transition = self.model.tensor(first, 0)?;
            let layout = BlockLayout::upsample(channels);
            if transition.bytes.len() as u32 != layout.end_without_padding + 16 {
                bail!("unexpected upsample layout for block {first}");
            }
            let projection = self.allocate(
                &format!("{label} projection"),
                rows(low),
                channels,
                Format::F16,
            )?;
            // Every PTX GEMM wants N % 64 == 0; the 64 -> 32 transition runs padded to 64 zero columns and keeps
            // the first 32 (upstream runs it on the GLSL GEMM).
            let padded_n = channels.next_multiple_of(64);
            let wp = if padded_n == channels {
                self.fp8(transition, layout.upsample_weight, channels * 2, channels)?
            } else {
                self.model.fp8_matrix_padded(
                    self.cuda,
                    transition,
                    layout.upsample_weight,
                    channels * 2,
                    channels,
                    padded_n,
                )?
            };
            let wide = if padded_n == channels {
                projection.clone()
            } else {
                self.allocate(
                    &format!("{label} projection padded"),
                    rows(low),
                    padded_n,
                    Format::F16,
                )?
            };
            self.kernels.gemm_fp8(&GemmArgs {
                input: Some(&stage_input),
                weights: wp,
                n_matrix: padded_n,
                output: Some(&wide),
                quantize: false,
                rows: rows(low),
                k: channels * 2,
                n: padded_n,
                ..Default::default()
            })?;
            if padded_n != channels {
                self.kernels.narrow_f16(&wide, &projection)?;
            }
            let merged = self.allocate(
                &format!("{label} skip merge"),
                rows(high),
                channels,
                Format::E4,
            )?;
            if channels != 32 {
                self.kernels.upsample_residual(
                    &projection,
                    &skip,
                    transition.raw,
                    layout.transition_scale,
                    &merged,
                    None,
                    low.width,
                    high.width,
                )?;
            }
            let mut st = merged;
            let mut sc =
                self.allocate(&format!("{label} state"), rows(high), channels, Format::E4)?;
            let temps = self.temporaries(&label, rows(high), channels)?;
            for block in first..=last {
                let phase = self.phases.take(level_index);
                if block == first && channels == 32 {
                    // The 2x upsample and scaled skip merge run inside the first block's input path.
                    let (shift_x, shift_y) = window_phase(phase);
                    let mut f = Block32Args {
                        state: Some(&skip),
                        low_projection: Some(&projection),
                        low_width: low.width,
                        input_scale_byte_offset: layout.transition_scale,
                        out_e4: Some(&sc),
                        width: high.width,
                        height: high.height,
                        shift_x,
                        shift_y,
                        ..Default::default()
                    };
                    self.block32_weights(transition, &layout, &mut f)?;
                    self.kernels.fused_block32(&f)?;
                } else {
                    self.block(
                        &temps,
                        &st,
                        &sc,
                        block,
                        channels,
                        high,
                        phase,
                        &if block == first {
                            layout
                        } else {
                            BlockLayout::fused(channels)
                        },
                        None,
                        None,
                        temps.ffn_quantized2.is_some() && block != last,
                    )?;
                }
                std::mem::swap(&mut st, &mut sc);
                if self.pending.is_none() {
                    self.capture(&format!("block-{block}"), &st)?;
                }
            }
            stage_input = st;
        }

        // ---- Block 70: the post blend in its input path and the RGBA head in its epilogue.
        let tensor = self.model.tensor(70, 0)?;
        let post = BlockLayout::post();
        if tensor.bytes.len() as u32 != post.end_without_padding {
            bail!("unexpected block 70 layout");
        }
        let head = self.allocate("RGBA neural head", full_rows, 4, Format::F32)?;
        let head_weights = self
            .model
            .f16_matrix(self.cuda, tensor, post.post_weights, 32, 4)?;
        let (shift_x, shift_y) = window_phase(self.phases.take(FULL_LEVEL));
        let mut f = Block32Args {
            state: Some(&adapter),
            low_res: Some(&stage_input),
            low_width: d0.width,
            input_scale_byte_offset: post.input_scale,
            adapter_scale_byte_offset: post.adapter_scale,
            head_weights,
            head: Some(&head),
            width: full.width,
            height: full.height,
            shift_x,
            shift_y,
            ..Default::default()
        };
        self.block32_weights(tensor, &post, &mut f)?;
        self.kernels.fused_block32(&f)?;
        Ok(head)
    }
}

pub struct Timings {
    pub gpu_ms: f64,
    pub frame_ms: f64,
}

/// The network for one resolution on the CUDA backend.
pub struct CudaNetwork<'c, 'm> {
    cuda: &'c Cuda,
    /// Owns every weight buffer the recorded launches point into.
    model: CudaModel<'m>,
    pub geometry: Geometry,
    pub launches: usize,
    pub modules: usize,
    pub activation_bytes: usize,
    pub weight_bytes: usize,
    pub setup_seconds: f64,
    kernels: Kernels<'c>,
    graph: CudaGraph,
    _proxy: DeviceBuffer,
    rgb_in: DeviceBuffer,
    rgb_out: DeviceBuffer,
    /// Page-locked staging for the 8-bit image in and out.
    rgb_host: HostBuffer,
    head: Activation,
    buffers: HashMap<String, (DeviceBuffer, Activation)>,
}

impl<'c, 'm> CudaNetwork<'c, 'm> {
    pub fn new(
        cuda: &'c Cuda,
        model: &'m Model,
        width: u32,
        height: u32,
        conditioning: Conditioning,
    ) -> Result<Self> {
        Self::with_captures(cuda, model, width, height, conditioning, false)
    }

    /// `capture`: also keep every block's output as `boundary block-N` (`read_tensor`).
    pub fn with_captures(
        cuda: &'c Cuda,
        model: &'m Model,
        width: u32,
        height: u32,
        conditioning: Conditioning,
        capture: bool,
    ) -> Result<Self> {
        let start = Instant::now();
        cuda.bind()?;
        let geometry = Geometry::from_valid(width, height)?;
        let mut recorder = Recorder {
            cuda,
            kernels: Kernels::new(cuda)?,
            model: CudaModel::upload(cuda, model)?,
            geometry: geometry.clone(),
            buffers: HashMap::new(),
            phases: WindowPhases::default(),
            pending: None,
            permutation: mlp_hidden_permutation(128),
            capture,
        };
        let full_rows = geometry.full_width * geometry.full_height;
        let (pixels, rgb_bytes) = (width * height, (width * height * 3) as usize);
        let proxy = cuda.alloc(pixels as usize * 16)?;
        let rgb_in = cuda.alloc(rgb_bytes)?;
        let rgb_out = cuda.alloc(rgb_bytes)?;
        let features = recorder.allocate("input features", full_rows, 16, Format::F32)?;
        recorder.kernels.begin();
        recorder
            .kernels
            .proxy_from_rgb8(rgb_in.ptr, proxy.ptr, pixels)?;
        recorder.kernels.preprocess(
            proxy.ptr,
            &features,
            (geometry.full_width, geometry.full_height),
            (width, height),
            conditioning.seed,
            [
                if conditioning.auto_mask { 1.0 } else { -1.0 },
                conditioning.local_tone,
                conditioning.local_structure,
                conditioning.skin_structure,
                conditioning.style,
            ],
        )?;
        let head = recorder.record(&features)?;
        recorder.kernels.compose(
            &head,
            rgb_in.ptr,
            rgb_out.ptr,
            width,
            height,
            geometry.full_width,
        )?;
        let Recorder {
            kernels,
            model: cuda_model,
            buffers,
            ..
        } = recorder;
        let graph = cuda
            .capture(|| kernels.replay())
            .context("capturing the frame as a CUDA graph")?;
        let activation_bytes = buffers.values().map(|(b, _)| b.size).sum();
        Ok(Self {
            cuda,
            launches: kernels.launch_count(),
            modules: kernels.module_count(),
            activation_bytes,
            weight_bytes: cuda_model.bytes_on_device(),
            model: cuda_model,
            setup_seconds: start.elapsed().as_secs_f64(),
            rgb_host: cuda.host_buffer(rgb_bytes)?,
            geometry,
            kernels,
            graph,
            _proxy: proxy,
            rgb_in,
            rgb_out,
            head,
            buffers,
        })
    }

    /// One frame: the 8-bit image up, the graph (proxy, features, network, composition), the 8-bit image down.
    pub fn process(&mut self, image: &Image) -> Result<(Image, Timings)> {
        if (image.width, image.height) != (self.geometry.valid_width, self.geometry.valid_height) {
            bail!("the graph was recorded for a different size");
        }
        let start = Instant::now();
        self.cuda.bind()?;
        self.rgb_host.as_mut_slice().copy_from_slice(&image.rgb);
        let bytes = image.rgb.len();
        let (cuda, graph, rgb_in, rgb_out) =
            (self.cuda, &self.graph, self.rgb_in.ptr, self.rgb_out.ptr);
        let rgb_host = &mut self.rgb_host;
        // NR_CUDA_EAGER=1: the launches one by one with a synchronization after each, so a fault names its kernel.
        if std::env::var("NR_CUDA_EAGER").is_ok_and(|v| v == "1") {
            cuda.copy_to_device_async(rgb_in, rgb_host, bytes)?;
            self.kernels.profile()?;
        }
        let gpu_ms = cuda.time(|| {
            cuda.copy_to_device_async(rgb_in, rgb_host, bytes)?;
            cuda.launch_graph(graph)?;
            cuda.copy_to_host_async(rgb_host, rgb_out, bytes)
        })?;
        let timeouts = self.kernels.chain_timeouts()?;
        if timeouts != 0 {
            bail!("{timeouts} chained wait(s) timed out");
        }
        let output = Image {
            width: image.width,
            height: image.height,
            rgb: self.rgb_host.as_slice().to_vec(),
        };
        Ok((
            output,
            Timings {
                gpu_ms,
                frame_ms: start.elapsed().as_secs_f64() * 1000.0,
            },
        ))
    }

    /// One frame, returning the f32 `[full rows][4]` head (RGB residual and blend logit).
    pub fn run(&mut self, image: &Image) -> Result<(Vec<f32>, Timings)> {
        let (_, timings) = self.process(image)?;
        let head = self.cuda.read(self.head.ptr, self.head.valid_bytes())?;
        Ok((bytemuck::pod_collect_to_vec(&head), timings))
    }

    /// Per-launch GPU times of one frame, `(kernel file, label, milliseconds)`.
    pub fn profile(&mut self) -> Result<Vec<(String, String, f64)>> {
        self.cuda.bind()?;
        self.kernels.profile()
    }

    /// Runs an instrumented build of the fused QKV + window attention kernel (tools/ptx/qkv_debug.py) for one
    /// (window, head) item of `block`, on the FFN publication the last frame captured for it (`with_captures`), with
    /// the production launch's arguments. Returns the debug words, `[slot][128 threads]`.
    #[allow(clippy::too_many_arguments)]
    pub fn debug_qkv(
        &mut self,
        block: u32,
        channels: u32,
        level: Level,
        phase: u32,
        ptx: &str,
        entry: &str,
        item: u32,
    ) -> Result<Vec<u32>> {
        use super::driver::Arg;
        self.cuda.bind()?;
        let (_, input) = self
            .buffers
            .values()
            .find(|(_, a)| a.label == format!("boundary block-{block} ffn"))
            .context("no captured FFN publication; run with captures")?;
        let input = input.clone();
        let tensor = self.model.tensor(block, 0)?;
        let layout = BlockLayout::fused(channels);
        let wqkv = self.model.fp8_matrix(
            self.cuda,
            tensor,
            layout.qkv,
            channels,
            channels * 3,
            0,
            true,
        )?;
        let prior = self
            .model
            .relative_bias(self.cuda, tensor, layout.relative, layout.heads)?;
        let (shift_x, shift_y) = window_phase(phase);
        let windows_x = (level.width + shift_x).div_ceil(8);
        let windows = windows_x * (level.height + shift_y).div_ceil(8);
        let attended = self
            .cuda
            .alloc(input.alloc_rows as usize * channels as usize)?;
        let debug = self.cuda.alloc(400 * 512)?;
        let status = self.cuda.alloc(16)?;
        let module = self.cuda.module(ptx)?;
        let function = self.cuda.function(&module, entry)?;
        let args = [
            Arg::Ptr(input.ptr),
            Arg::Ptr(wqkv),
            Arg::Ptr(prior),
            Arg::Ptr(tensor.raw),
            Arg::Ptr(attended.ptr),
            Arg::U32(level.width),
            Arg::U32(level.height),
            Arg::U32(shift_x),
            Arg::U32(shift_y),
            Arg::U32(windows_x),
            Arg::U32(layout.scale / 4),
            Arg::U32(windows),
            Arg::U32(item + 1),
            Arg::Ptr(0),
            Arg::Ptr(0),
            Arg::U32(1),
            Arg::U32(64),
            Arg::Ptr(status.ptr),
            Arg::Ptr(debug.ptr),
            Arg::U32(item),
        ];
        // grid = itemCount = item + 1: every CTA runs exactly its own item, and CTA `item` records.
        self.cuda
            .launch(&function, [item + 1, 1, 1], 128, 0, &args)?;
        let words = self.cuda.read(debug.ptr, 400 * 512)?;
        Ok(bytemuck::pod_collect_to_vec(&words))
    }

    /// Any activation of the last frame by label, as raw bytes.
    pub fn read_tensor(&self, label: &str) -> Result<Vec<u8>> {
        let (_, activation) = self
            .buffers
            .values()
            .find(|(_, a)| a.label == label)
            .with_context(|| format!("no activation labelled {label:?}"))?;
        self.cuda.read(activation.ptr, activation.valid_bytes())
    }
}
