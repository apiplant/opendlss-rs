//! The network: 71 blocks over a six-level encoder/decoder with a global ViT at the bottom, recorded once into
//! a list of dispatches. A port of `ports/browser-webgpu/src/graph.js` of the reference (which mirrors
//! `src/nr_graph.cpp`); `docs/network.md` there explains the shape and why it is that shape.

use anyhow::{Result, bail};

use crate::{
    geometry::{BlockLayout, Geometry, WindowPhases, window_phase},
    runtime::{Format, Gpu, MAX_GROUPS, PipelineKey, Recorder, Tensor, Tensors},
    weights::{Fp8Matrix, GpuModel, TensorRef},
};

// The FP8 GEMM's flag word (src/matmul/index.js).
const FLAG_RESIDUAL: u32 = 2;
const FLAG_SILU: u32 = 4;
const FLAG_SCALE_RESIDUAL: u32 = 8;
const FLAG_SWIZZLE_INPUT: u32 = 16;
const FLAG_QUANTIZE_OUTPUT: u32 = 32;
const FLAG_BROADCAST_INPUT: u32 = 131072;
const FLAG_RESIDUAL_E4: u32 = 262144;

// Flags of the hand-written kernels.
const WRITE_E4: u32 = 2;
const WRITE_F16: u32 = 4;
const WRITE_F32: u32 = 32;
const DUAL: u32 = 1;

/// 32 queries per window-attention workgroup.
const WINDOW_QUERIES: u32 = 32;

/// The K span whose sums are published to half and added separately, as the flag that selects it.
fn partition_flag(span: u32) -> u32 {
    match span {
        0 => 1024,
        1024 => 8192,
        512 => 16384,
        256 => 32768,
        _ => unreachable!("no K partition of {span}"),
    }
}

/// A 1D dispatch of `count` invocations at 64 per workgroup, folded into two dimensions past the limit.
fn grid1d(count: u32) -> [u32; 3] {
    let groups = count.div_ceil(64);
    if groups <= MAX_GROUPS {
        [groups, 1, 1]
    } else {
        [MAX_GROUPS, groups.div_ceil(MAX_GROUPS), 1]
    }
}

/// The lookup tables every GEMM reads.
pub struct Tables {
    pub silu: wgpu::Buffer,
    pub packed_silu: wgpu::Buffer,
    pub weight_metadata: wgpu::Buffer,
}

#[derive(Default)]
struct Gemm<'a> {
    input: Option<&'a Tensor>,
    weights: Option<Fp8Matrix>,
    output: Option<&'a Tensor>,
    output_f16: Option<&'a Tensor>,
    rows: u32,
    k: u32,
    n: u32,
    batches: u32,
    broadcast: bool,
    partition: u32,
    silu: bool,
    residual: Option<&'a Tensor>,
    aux: u32,
}

#[derive(Default)]
struct Op<'a> {
    count: u32,
    channels: u32,
    in_width: u32,
    in_height: u32,
    out_width: u32,
    out_height: u32,
    aux_a: u32,
    aux_b: u32,
    dual: bool,
    in_f32: Option<&'a Tensor>,
    in_f16: Option<&'a Tensor>,
    in_e4: Option<&'a Tensor>,
    skip_e4: Option<&'a Tensor>,
    aux: Option<wgpu::Buffer>,
    out_e4: Option<&'a Tensor>,
    out_f16: Option<&'a Tensor>,
}

struct Temps {
    ffn: Tensor,
    ffn_narrow: Option<Tensor>,
    ffn_residual: Tensor,
    ffn_quantized: Tensor,
    qkv: Tensor,
    attended: Tensor,
}

struct SplitTemps {
    branch: Tensor,
    middle: Tensor,
    layer0: Tensor,
    ffn_residual: Tensor,
    qkv: Tensor,
    attended: Tensor,
}

struct BlockArgs<'a> {
    block: u32,
    channels: u32,
    width: u32,
    height: u32,
    layout: BlockLayout,
    tensor: TensorRef<'a>,
    state: &'a Tensor,
    output: Option<&'a Tensor>,
    output_f16: Option<&'a Tensor>,
    ffn_skip_override: Option<&'a Tensor>,
    phase: u32,
}

pub struct Graph<'g, 'm> {
    gpu: &'g Gpu,
    model: &'g GpuModel<'m>,
    tables: &'g Tables,
    tensors: &'g mut Tensors,
    recorder: &'g mut Recorder,
    geometry: Geometry,
    phases: WindowPhases,
    /// `(name, tensor)` of every block output, when capturing boundaries.
    pub boundaries: Option<Vec<(String, Tensor)>>,
}

impl<'g, 'm> Graph<'g, 'm> {
    pub fn new(
        gpu: &'g Gpu,
        model: &'g GpuModel<'m>,
        tables: &'g Tables,
        tensors: &'g mut Tensors,
        recorder: &'g mut Recorder,
        geometry: Geometry,
        capture_boundaries: bool,
    ) -> Self {
        Self {
            gpu,
            model,
            tables,
            tensors,
            recorder,
            geometry,
            phases: WindowPhases::default(),
            boundaries: capture_boundaries.then(Vec::new),
        }
    }

    fn tensor(&mut self, label: &str, rows: u32, channels: u32, format: Format) -> Tensor {
        self.tensors
            .allocate(self.gpu, label, rows, channels, format)
    }

    fn capture(&mut self, name: &str, source: &Tensor) {
        if self.boundaries.is_none() {
            return;
        }
        let copy = self.tensor(
            &format!("boundary {name}"),
            source.rows,
            source.channels,
            source.format,
        );
        self.recorder
            .copy(&source.buffer, &copy.buffer, copy.byte_length);
        self.boundaries
            .as_mut()
            .unwrap()
            .push((name.to_string(), copy));
    }

    // -----------------------------------------------------------------------------------------------------
    // The kernels, as the graph wants to call them.
    // -----------------------------------------------------------------------------------------------------

    /// One FP8 matrix multiply. `residual`, when given, is not added afterwards: it is the value the
    /// accumulator starts from, scaled by the block's learned per-channel vector.
    fn gemm(&mut self, g: Gemm<'_>, label: &str) -> Result<()> {
        let input = g.input.unwrap();
        let weights = g.weights.unwrap();
        let batches = g.batches.max(1);
        let target = g.output.or(g.output_f16).unwrap();
        let mode = match (g.output.is_some(), g.output_f16.is_some()) {
            (true, true) => "dual",
            (true, false) => "e4",
            _ => "half",
        };
        if let (Some(a), Some(b)) = (g.output, g.output_f16)
            && a.channels != b.channels
        {
            bail!("{label} publishes both tensors at one index, so they must share a stride");
        }
        if mode != "e4" && g.silu {
            bail!("{label} activates on a boundary that is not E4M3");
        }
        if let Some(residual) = g.residual
            && residual.channels != target.channels
        {
            bail!("{label} reads its skip at the output index, so the strides must agree");
        }
        let batched = batches > 1 || input.channels != g.k;
        if weights.k != g.k * batches || weights.batch_k != g.k {
            bail!(
                "{label} dispatches {batches}x{} against a {} matrix",
                g.k,
                weights.k
            );
        }
        let mut flags = partition_flag(g.partition) | FLAG_SWIZZLE_INPUT;
        if g.silu {
            flags |= FLAG_SILU;
        }
        if mode != "half" {
            flags |= FLAG_QUANTIZE_OUTPUT;
        }
        if g.broadcast {
            flags |= FLAG_BROADCAST_INPUT;
        }
        if let Some(residual) = g.residual {
            flags |= FLAG_RESIDUAL | FLAG_SCALE_RESIDUAL;
            if residual.format == Format::E4 {
                flags |= FLAG_RESIDUAL_E4;
            }
        }
        let module: &'static str = match (mode, batched) {
            ("e4", false) => "gemm_e4",
            ("e4", true) => "gemm_e4_batched",
            ("half", false) => "gemm_half",
            ("half", true) => "gemm_half_batched",
            ("dual", false) => "gemm_dual",
            _ => "gemm_dual_batched",
        };
        let mut constants = vec![
            ("MATMUL_FLAGS", flags),
            ("MATMUL_ROWS", g.rows),
            ("MATMUL_K", g.k),
            ("MATMUL_N", g.n),
            ("LAYOUT_WEIGHT_BYTE_OFFSET", weights.byte_offset),
            ("LAYOUT_BIAS_BYTE_OFFSET", g.aux),
            ("LAYOUT_WEIGHT_MATRIX_CHANNELS", weights.matrix_channels),
            ("LAYOUT_WEIGHT_COLUMN_OFFSET", 0),
            ("LAYOUT_OUTPUT_MATRIX_CHANNELS", target.channels),
            ("LAYOUT_OUTPUT_COLUMN_OFFSET", 0),
        ];
        if batched {
            constants.push(("LAYOUT_INPUT_MATRIX_CHANNELS", input.channels));
        }
        let params = [
            g.rows,
            g.k,
            g.n,
            weights.byte_offset,
            g.aux,
            flags,
            weights.matrix_channels,
            0,
            target.channels,
            0,
            input.channels,
            batches,
        ];
        let row_tiles = g.rows.div_ceil(32);
        let silu = if mode == "half" {
            &self.tables.silu
        } else {
            &self.tables.packed_silu
        };
        let mut buffers = vec![
            (0, &input.buffer),
            (1, &weights.buffer),
            (2, &target.buffer),
            (5, silu),
            (6, &self.tables.weight_metadata),
        ];
        if let Some(residual) = g.residual {
            buffers.push((3, &residual.buffer));
        }
        if mode == "dual" {
            buffers.push((7, &g.output_f16.unwrap().buffer));
        }
        self.recorder.dispatch(
            PipelineKey::Specialized {
                module,
                entry: "main",
                gemm: true,
                constants,
            },
            &buffers,
            &params,
            [
                g.n.div_ceil(32) * batches,
                row_tiles.min(MAX_GROUPS),
                row_tiles.div_ceil(MAX_GROUPS),
            ],
            label,
        )
    }

    /// The f16 matrix multiply: only the input adapter and the head.
    #[allow(clippy::too_many_arguments)]
    fn gemm_f16(
        &mut self,
        input: &Tensor,
        weights: &wgpu::Buffer,
        padded_n: u32,
        outputs: [Option<&Tensor>; 3],
        rows: u32,
        k: u32,
        n: u32,
        label: &str,
    ) -> Result<()> {
        let [output, output_f16, output_f32] = outputs;
        let mut flags = 0;
        if output.is_some() {
            flags |= WRITE_E4;
        }
        if output_f16.is_some() {
            flags |= WRITE_F16;
        }
        if output_f32.is_some() {
            flags |= WRITE_F32;
        }
        let target = output_f32.or(output_f16).or(output).unwrap();
        let params = [
            rows,
            k,
            n,
            padded_n,
            input.channels,
            target.channels,
            flags,
            0,
        ];
        let mut buffers = vec![(1, &input.buffer), (2, weights)];
        for (binding, tensor) in [(5, output), (6, output_f16), (7, output_f32)] {
            if let Some(tensor) = tensor {
                buffers.push((binding, &tensor.buffer));
            }
        }
        self.recorder.dispatch(
            PipelineKey::Shared("gemm_f16"),
            &buffers,
            &params,
            grid1d(rows * (n / 4)),
            label,
        )
    }

    fn op(&mut self, entry: &'static str, o: Op<'_>, label: &str) -> Result<()> {
        let params = [
            o.count,
            o.channels,
            o.in_width,
            o.in_height,
            o.out_width,
            o.out_height,
            o.aux_a,
            o.aux_b,
            if o.dual { DUAL } else { 0 },
            0,
            0,
            0,
        ];
        let mut buffers = Vec::new();
        for (binding, tensor) in [
            (1, o.in_f32),
            (2, o.in_f16),
            (3, o.in_e4),
            (4, o.skip_e4),
            (6, o.out_e4),
            (7, o.out_f16),
        ] {
            if let Some(tensor) = tensor {
                buffers.push((binding, &tensor.buffer));
            }
        }
        if let Some(aux) = &o.aux {
            buffers.push((8, aux));
        }
        self.recorder.dispatch(
            PipelineKey::Shared(entry),
            &buffers,
            &params,
            grid1d(o.count / 4),
            label,
        )
    }

    /// One shifted-window attention, with the cosine normalization fused in.
    #[allow(clippy::too_many_arguments)]
    fn window_attention(
        &mut self,
        qkv: &Tensor,
        attended: &Tensor,
        prior: &wgpu::Buffer,
        scales: &wgpu::Buffer,
        width: u32,
        height: u32,
        heads: u32,
        phase: u32,
        label: &str,
    ) -> Result<()> {
        let (shift_x, shift_y) = window_phase(phase);
        let windows_x = (width + shift_x).div_ceil(8);
        let windows_y = (height + shift_y).div_ceil(8);
        let tasks = windows_x * windows_y * (64 / WINDOW_QUERIES);
        let params = [
            width * height,
            heads,
            width,
            height,
            heads * 32,
            shift_x,
            shift_y,
            1,
        ];
        self.recorder.dispatch(
            PipelineKey::Specialized {
                module: "window_attention",
                entry: "attend_window_tiled",
                gemm: false,
                constants: vec![
                    ("WINDOW_WIDTH", width),
                    ("WINDOW_HEIGHT", height),
                    ("WINDOW_CHANNELS", heads * 32),
                    ("WINDOW_SHIFT_X", shift_x),
                    ("WINDOW_SHIFT_Y", shift_y),
                    ("WINDOW_USE_RELATIVE_BIAS", 1),
                ],
            },
            &[
                (1, &qkv.buffer),
                (2, scales),
                (3, prior),
                (6, &attended.buffer),
            ],
            &params,
            [heads, tasks.min(MAX_GROUPS), tasks.div_ceil(MAX_GROUPS)],
            format!("{label} attend"),
        )
    }

    // -----------------------------------------------------------------------------------------------------
    // One block.
    // -----------------------------------------------------------------------------------------------------

    /// FFN -> QKV -> window attention -> projection, with the two scaled skips that make it a residual block.
    /// Under 64 channels the FFN is one dense C -> 128 -> C path; at 64 and above it is C/32 parallel paths,
    /// each C -> 128 -> 32, concatenated and run through one more C -> C layer that carries the FFN's skip.
    fn block(&mut self, a: BlockArgs<'_>, temps: &Temps) -> Result<()> {
        let rows = a.width * a.height;
        let model = self.model;
        let label = format!("block {}", a.block);
        let residual = a.ffn_skip_override.unwrap_or(a.state);
        let ffn_aux = model.aux_offset(a.tensor, a.layout.ffn_cos_skip, a.channels)?;
        let l = a.layout;
        let c = a.channels;

        if l.expert_ffn {
            let experts = l.expert_count;
            let w2 = l.expand + experts * c * 128;
            let w3 = w2 + experts * 128 * 32;
            self.gemm(
                Gemm {
                    input: Some(a.state),
                    weights: Some(model.fp8_matrix(
                        a.tensor,
                        l.expand,
                        experts * c,
                        128,
                        Some(c),
                    )?),
                    output: Some(&temps.ffn),
                    rows,
                    k: c,
                    n: 128,
                    batches: experts,
                    broadcast: true,
                    silu: true,
                    ..Default::default()
                },
                &format!("{label} expert expand"),
            )?;
            self.gemm(
                Gemm {
                    input: Some(&temps.ffn),
                    weights: Some(model.fp8_matrix(a.tensor, w2, experts * 128, 32, Some(128))?),
                    output: temps.ffn_narrow.as_ref(),
                    rows,
                    k: 128,
                    n: 32,
                    batches: experts,
                    ..Default::default()
                },
                &format!("{label} expert contract"),
            )?;
            self.gemm(
                Gemm {
                    input: temps.ffn_narrow.as_ref(),
                    weights: Some(model.fp8_matrix(a.tensor, w3, c, c, None)?),
                    output: Some(&temps.ffn_quantized),
                    output_f16: Some(&temps.ffn_residual),
                    rows,
                    k: c,
                    n: c,
                    residual: Some(residual),
                    aux: ffn_aux,
                    ..Default::default()
                },
                &format!("{label} expert merge"),
            )?;
        } else {
            self.gemm(
                Gemm {
                    input: Some(a.state),
                    weights: Some(model.fp8_matrix(a.tensor, l.expand, c, l.hidden, None)?),
                    output: Some(&temps.ffn),
                    rows,
                    k: c,
                    n: l.hidden,
                    silu: true,
                    ..Default::default()
                },
                &format!("{label} expand"),
            )?;
            self.gemm(
                Gemm {
                    input: Some(&temps.ffn),
                    weights: Some(model.fp8_matrix(
                        a.tensor,
                        l.contract_weights,
                        l.hidden,
                        c,
                        None,
                    )?),
                    output: Some(&temps.ffn_quantized),
                    output_f16: Some(&temps.ffn_residual),
                    rows,
                    k: l.hidden,
                    n: c,
                    residual: Some(residual),
                    aux: ffn_aux,
                    ..Default::default()
                },
                &format!("{label} contract"),
            )?;
        }

        self.gemm(
            Gemm {
                input: Some(&temps.ffn_quantized),
                weights: Some(model.fp8_matrix(a.tensor, l.qkv, c, c * 3, None)?),
                output_f16: Some(&temps.qkv),
                rows,
                k: c,
                n: c * 3,
                ..Default::default()
            },
            &format!("{label} qkv"),
        )?;
        let prior = model.relative_bias(self.gpu, a.tensor, l.relative, l.heads)?;
        let scales = model.head_scales(self.gpu, a.tensor, l.scale, l.heads)?;
        self.window_attention(
            &temps.qkv,
            &temps.attended,
            &prior,
            &scales,
            a.width,
            a.height,
            l.heads,
            a.phase,
            &label,
        )?;

        // The attention skip: the E4M3 publication of the FFN for the expert blocks, the raw half otherwise.
        self.gemm(
            Gemm {
                input: Some(&temps.attended),
                weights: Some(model.fp8_matrix(a.tensor, l.projection, c, c, None)?),
                output: a.output,
                output_f16: a.output_f16,
                rows,
                k: c,
                n: c,
                residual: Some(if l.expert_ffn {
                    &temps.ffn_quantized
                } else {
                    &temps.ffn_residual
                }),
                aux: model.aux_offset(a.tensor, l.attn_cos_skip, c)?,
                ..Default::default()
            },
            &format!("{label} projection"),
        )
    }

    /// The 512 stage. Its FFN is eight independent 64-wide branches, each widened to 256 and brought back,
    /// with the activation on the middle layer only, all on top of one 512 -> 512 layer.
    #[allow(clippy::too_many_arguments)]
    fn split_block(
        &mut self,
        block: u32,
        width: u32,
        height: u32,
        temps: &SplitTemps,
        state: &Tensor,
        output: &Tensor,
        output_f16: Option<&Tensor>,
        phase: u32,
    ) -> Result<()> {
        let rows = width * height;
        let model = self.model;
        let label = format!("block {block}");
        let (channels, branches, branch_channels, middle, heads) = (512, 8, 64, 256, 16);
        let branch_tensor = model.tensor(block, 0)?;
        let contract = model.tensor(block, 1)?;
        let qkv_tensor = model.tensor(block, 2)?;
        let projection = model.tensor(block, 3)?;
        let w2 = branches * channels * branch_channels;
        let w3 = w2 + branches * branch_channels * middle;
        let qkv_relative = channels * channels * 3;
        let qkv_scale = qkv_relative + heads * 8192;

        self.gemm(
            Gemm {
                input: Some(state),
                weights: Some(model.fp8_matrix(branch_tensor, 0, channels, channels, None)?),
                output: Some(&temps.branch),
                rows,
                k: channels,
                n: channels,
                ..Default::default()
            },
            &format!("{label} split layer0"),
        )?;
        self.gemm(
            Gemm {
                input: Some(&temps.branch),
                weights: Some(model.fp8_matrix(
                    branch_tensor,
                    w2,
                    branches * branch_channels,
                    middle,
                    Some(branch_channels),
                )?),
                output: Some(&temps.middle),
                rows,
                k: branch_channels,
                n: middle,
                batches: branches,
                silu: true,
                ..Default::default()
            },
            &format!("{label} split expand"),
        )?;
        self.gemm(
            Gemm {
                input: Some(&temps.middle),
                weights: Some(model.fp8_matrix(
                    branch_tensor,
                    w3,
                    branches * middle,
                    branch_channels,
                    Some(middle),
                )?),
                output: Some(&temps.layer0),
                rows,
                k: middle,
                n: branch_channels,
                batches: branches,
                ..Default::default()
            },
            &format!("{label} split contract"),
        )?;
        self.gemm(
            Gemm {
                input: Some(&temps.layer0),
                weights: Some(model.fp8_matrix(contract, 0, channels, channels, None)?),
                output: Some(&temps.ffn_residual),
                rows,
                k: channels,
                n: channels,
                residual: Some(state),
                aux: model.aux_offset(contract, channels * channels, channels)?,
                ..Default::default()
            },
            &format!("{label} split merge"),
        )?;
        self.gemm(
            Gemm {
                input: Some(&temps.ffn_residual),
                weights: Some(model.fp8_matrix(qkv_tensor, 0, channels, channels * 3, None)?),
                output_f16: Some(&temps.qkv),
                rows,
                k: channels,
                n: channels * 3,
                ..Default::default()
            },
            &format!("{label} qkv"),
        )?;
        let prior = model.relative_bias(self.gpu, qkv_tensor, qkv_relative, heads)?;
        let scales = model.head_scales(self.gpu, qkv_tensor, qkv_scale, heads)?;
        self.window_attention(
            &temps.qkv,
            &temps.attended,
            &prior,
            &scales,
            width,
            height,
            heads,
            phase,
            &label,
        )?;
        self.gemm(
            Gemm {
                input: Some(&temps.attended),
                weights: Some(model.fp8_matrix(projection, 0, channels, channels, None)?),
                output: Some(output),
                output_f16,
                rows,
                k: channels,
                n: channels,
                residual: Some(&temps.ffn_residual),
                aux: model.aux_offset(projection, channels * channels, channels)?,
                ..Default::default()
            },
            &format!("{label} projection"),
        )
    }

    /// The global ViT: eight blocks whose attention spans every token of the coarsest level.
    fn vit(&mut self, state: &Tensor, tokens: u32) -> Result<()> {
        let (channels, heads, ffn_channels) = (1024, 32, 4096);
        let padded = self.geometry.padded_vit_tokens();
        let model = self.model;
        let expanded = self.tensor("vit expand", tokens, ffn_channels, Format::E4);
        let ffn_residual = self.tensor("vit residual", tokens, channels, Format::E4);
        let qkv = self.tensor("vit qkv", tokens, channels * 3, Format::F16);
        let normalized = self.tensor("vit normalized", padded, channels * 3, Format::E4);
        let attended = self.tensor("vit attended", tokens, channels, Format::E4);

        for block in 31..=38 {
            let expand = model.tensor(block, 0)?;
            let contract = model.tensor(block, 1)?;
            let qkv_tensor = model.tensor(block, 2)?;
            let projection = model.tensor(block, 4)?;
            let label = format!("block {block}");
            self.gemm(
                Gemm {
                    input: Some(state),
                    weights: Some(model.fp8_matrix(expand, 0, channels, ffn_channels, None)?),
                    output: Some(&expanded),
                    rows: tokens,
                    k: channels,
                    n: ffn_channels,
                    silu: true,
                    ..Default::default()
                },
                &format!("{label} expand"),
            )?;
            self.gemm(
                Gemm {
                    input: Some(&expanded),
                    weights: Some(model.fp8_matrix(contract, 0, ffn_channels, channels, None)?),
                    output: Some(&ffn_residual),
                    rows: tokens,
                    k: ffn_channels,
                    n: channels,
                    partition: 1024,
                    residual: Some(state),
                    aux: model.aux_offset(contract, ffn_channels * channels, channels)?,
                    ..Default::default()
                },
                &format!("{label} contract"),
            )?;
            // The ViT's qkv tensor puts its per-head scales before the weights.
            self.gemm(
                Gemm {
                    input: Some(&ffn_residual),
                    weights: Some(model.fp8_matrix(
                        qkv_tensor,
                        heads * 4,
                        channels,
                        channels * 3,
                        None,
                    )?),
                    output_f16: Some(&qkv),
                    rows: tokens,
                    k: channels,
                    n: channels * 3,
                    partition: 512,
                    ..Default::default()
                },
                &format!("{label} qkv"),
            )?;
            let params = [tokens, heads, channels, padded];
            let scales = model.head_scales(self.gpu, qkv_tensor, 0, heads)?;
            self.recorder.dispatch(
                PipelineKey::Shared("vit_normalize"),
                &[(1, &qkv.buffer), (2, &scales), (5, &normalized.buffer)],
                &params,
                grid1d(tokens * heads * 8),
                format!("{label} normalize"),
            )?;
            // The rewrite takes two queries per workgroup (shaders/vit_attend_parallel.wgsl).
            let (attend, queries) = if crate::network::reference_kernels() {
                ("vit_attend", 1)
            } else {
                ("vit_attend_parallel", 2)
            };
            self.recorder.dispatch(
                PipelineKey::Shared(attend),
                &[(4, &normalized.buffer), (6, &attended.buffer)],
                &params,
                [heads, tokens.div_ceil(queries), 1],
                format!("{label} attend"),
            )?;
            self.gemm(
                Gemm {
                    input: Some(&attended),
                    weights: Some(model.fp8_matrix(projection, 0, channels, channels, None)?),
                    output: Some(state),
                    rows: tokens,
                    k: channels,
                    n: channels,
                    partition: 256,
                    residual: Some(&ffn_residual),
                    aux: model.aux_offset(projection, channels * channels, channels)?,
                    ..Default::default()
                },
                &format!("{label} projection"),
            )?;
            self.capture(&format!("block-{block}"), state);
        }
        Ok(())
    }

    fn temporaries(
        &mut self,
        label: &str,
        rows: u32,
        channels: u32,
        layout: &BlockLayout,
    ) -> Temps {
        let hidden = if layout.expert_ffn {
            layout.expert_count * 128
        } else {
            layout.hidden
        };
        Temps {
            ffn: self.tensor(&format!("{label} ffn"), rows, hidden, Format::E4),
            ffn_narrow: layout
                .expert_ffn
                .then(|| self.tensor(&format!("{label} ffn narrow"), rows, channels, Format::E4)),
            ffn_residual: self.tensor(
                &format!("{label} ffn residual"),
                rows,
                channels,
                Format::F16,
            ),
            ffn_quantized: self.tensor(
                &format!("{label} ffn quantized"),
                rows,
                channels,
                Format::E4,
            ),
            qkv: self.tensor(&format!("{label} qkv"), rows, channels * 3, Format::F16),
            attended: self.tensor(&format!("{label} attended"), rows, channels, Format::E4),
        }
    }

    fn split_temporaries(&mut self, label: &str, rows: u32) -> SplitTemps {
        SplitTemps {
            branch: self.tensor(&format!("{label} branch"), rows, 512, Format::E4),
            middle: self.tensor(&format!("{label} middle"), rows, 2048, Format::E4),
            layer0: self.tensor(&format!("{label} layer0"), rows, 512, Format::E4),
            ffn_residual: self.tensor(&format!("{label} split residual"), rows, 512, Format::E4),
            qkv: self.tensor(&format!("{label} split qkv"), rows, 1536, Format::F16),
            attended: self.tensor(&format!("{label} split attended"), rows, 512, Format::E4),
        }
    }

    /// Record the whole network. `features` is f32 `[fullRows][16]`; returns the f32 `[fullRows][4]` head.
    pub fn record(&mut self, features: &Tensor) -> Result<Tensor> {
        let g = self.geometry.clone();
        let model = self.model;
        let full_rows = g.full_width * g.full_height;
        let [d0, d1, d2, d3, d4, d5] = g.levels;
        let rows_of = |l: crate::geometry::Level| l.width * l.height;

        // ---- Block 0 at full resolution, behind the f16 input adapter.
        let pre_tensor = model.tensor(0, 0)?;
        let pre = BlockLayout::pre();
        if pre_tensor.byte_length() != pre.end_without_padding + 16 {
            bail!("unexpected block 0 layout");
        }
        let features_half = self.tensor("features f16", full_rows, 16, Format::F16);
        self.op(
            "convert_f32_to_f16",
            Op {
                count: full_rows * 16,
                channels: 16,
                in_f32: Some(features),
                out_f16: Some(&features_half),
                ..Default::default()
            },
            "features to half",
        )?;
        let adapter_f16 = self.tensor("adapter f16", full_rows, 32, Format::F16);
        let adapter_e4 = self.tensor("adapter e4", full_rows, 32, Format::E4);
        let (weights, padded_n) =
            model.f16_matrix(self.gpu, pre_tensor, pre.input_adapter, 16, 32)?;
        self.gemm_f16(
            &features_half,
            &weights,
            padded_n,
            [Some(&adapter_e4), Some(&adapter_f16), None],
            full_rows,
            16,
            32,
            "input adapter",
        )?;
        let block0 = self.tensor("block 0 out", full_rows, 32, Format::E4);
        let block0_raw = self.tensor("block 0 raw", full_rows, 32, Format::F16);
        let full_temps = self.temporaries("full", full_rows, 32, &pre);
        let phase = self.phases.take(6);
        self.block(
            BlockArgs {
                block: 0,
                channels: 32,
                width: g.full_width,
                height: g.full_height,
                layout: pre,
                tensor: pre_tensor,
                state: &adapter_e4,
                output: Some(&block0),
                output_f16: Some(&block0_raw),
                ffn_skip_override: Some(&adapter_f16),
                phase,
            },
            &full_temps,
        )?;
        self.capture("block-0", &block0);

        // ---- Down to level 0 and the four 32-channel blocks there.
        let rows0 = rows_of(d0);
        let mut state = self.tensor("level0 in", rows0, 32, Format::E4);
        self.op(
            "downsample",
            Op {
                count: rows0 * 32,
                channels: 32,
                in_width: g.full_width,
                in_height: g.full_height,
                out_width: d0.width,
                out_height: d0.height,
                in_f16: Some(&block0_raw),
                out_e4: Some(&state),
                ..Default::default()
            },
            "pool 0",
        )?;
        self.capture("transition-0-1", &state);
        let mut scratch = self.tensor("level0 state", rows0, 32, Format::E4);
        let level0_raw = self.tensor("level0 raw", rows0, 32, Format::F16);
        let fused32 = BlockLayout::fused(32);
        let level0_temps = self.temporaries("level0", rows0, 32, &fused32);
        for block in 1..=4 {
            let phase = self.phases.take(0);
            self.block(
                BlockArgs {
                    block,
                    channels: 32,
                    width: d0.width,
                    height: d0.height,
                    layout: fused32,
                    tensor: model.tensor(block, 0)?,
                    state: &state,
                    output: Some(&scratch),
                    output_f16: (block == 4).then_some(&level0_raw),
                    ffn_skip_override: None,
                    phase,
                },
                &level0_temps,
            )?;
            std::mem::swap(&mut state, &mut scratch);
            self.capture(&format!("block-{block}"), &state);
        }
        let skip32 = state;

        // ---- Encoder stages 64 / 128 / 256, each ending in a pool and a widening transition.
        let pooled32 = self.tensor("pool 4", rows_of(d1), 32, Format::E4);
        self.op(
            "downsample",
            Op {
                count: rows_of(d1) * 32,
                channels: 32,
                in_width: d0.width,
                in_height: d0.height,
                out_width: d1.width,
                out_height: d1.height,
                in_f16: Some(&level0_raw),
                out_e4: Some(&pooled32),
                ..Default::default()
            },
            "pool 4",
        )?;
        self.capture("pooled-4-5", &pooled32);
        let mut stage_input = self.tensor("stage 64 in", rows_of(d1), 64, Format::E4);
        self.gemm(
            Gemm {
                input: Some(&pooled32),
                weights: Some(model.fp8_matrix(
                    model.tensor(4, 0)?,
                    fused32.end_without_padding,
                    32,
                    64,
                    None,
                )?),
                output: Some(&stage_input),
                rows: rows_of(d1),
                k: 32,
                n: 64,
                ..Default::default()
            },
            "transition 4-5",
        )?;
        self.capture("transition-4-5", &stage_input);

        let encoder_stages = [
            (d1, d2, 64, 5, 8, 1),
            (d2, d3, 128, 9, 14, 2),
            (d3, d4, 256, 15, 22, 3),
        ];
        let mut skips = Vec::new();
        for (level, next, channels, first, last, level_index) in encoder_stages {
            let rows = rows_of(level);
            let label = format!("encoder {channels}");
            let layout = BlockLayout::fused(channels);
            let mut st = stage_input.clone();
            let mut sc = self.tensor(&format!("{label} state"), rows, channels, Format::E4);
            let raw = self.tensor(&format!("{label} raw"), rows, channels, Format::F16);
            let temps = self.temporaries(&label, rows, channels, &layout);
            for block in first..=last {
                let phase = self.phases.take(level_index);
                self.block(
                    BlockArgs {
                        block,
                        channels,
                        width: level.width,
                        height: level.height,
                        layout,
                        tensor: model.tensor(block, 0)?,
                        state: &st,
                        output: Some(&sc),
                        output_f16: (block == last).then_some(&raw),
                        ffn_skip_override: None,
                        phase,
                    },
                    &temps,
                )?;
                std::mem::swap(&mut st, &mut sc);
                self.capture(&format!("block-{block}"), &st);
            }
            skips.push(st);
            let pooled = self.tensor(
                &format!("{label} pooled"),
                rows_of(next),
                channels,
                Format::E4,
            );
            self.op(
                "downsample",
                Op {
                    count: rows_of(next) * channels,
                    channels,
                    in_width: level.width,
                    in_height: level.height,
                    out_width: next.width,
                    out_height: next.height,
                    in_f16: Some(&raw),
                    out_e4: Some(&pooled),
                    ..Default::default()
                },
                &format!("pool {last}"),
            )?;
            self.capture(&format!("pooled-{last}-{}", last + 1), &pooled);
            let next_tensor = self.tensor(
                &format!("{label} next"),
                rows_of(next),
                channels * 2,
                Format::E4,
            );
            self.gemm(
                Gemm {
                    input: Some(&pooled),
                    weights: Some(model.fp8_matrix(
                        model.tensor(last, 0)?,
                        layout.end_without_padding,
                        channels,
                        channels * 2,
                        None,
                    )?),
                    output: Some(&next_tensor),
                    rows: rows_of(next),
                    k: channels,
                    n: channels * 2,
                    ..Default::default()
                },
                &format!("transition {last}"),
            )?;
            self.capture(&format!("transition-{last}-{}", last + 1), &next_tensor);
            stage_input = next_tensor;
        }
        let [skip64, skip128, skip256]: [Tensor; 3] = skips.try_into().ok().unwrap();

        // ---- The 512 stage, the pool into the ViT, and the ViT.
        {
            let rows = rows_of(d4);
            let mut st = stage_input.clone();
            let mut sc = self.tensor("encoder 512 state", rows, 512, Format::E4);
            let raw = self.tensor("encoder 512 raw", rows, 512, Format::F16);
            let temps = self.split_temporaries("encoder 512", rows);
            for block in 23..=30 {
                let phase = self.phases.take(4);
                self.split_block(
                    block,
                    d4.width,
                    d4.height,
                    &temps,
                    &st,
                    &sc,
                    (block == 30).then_some(&raw),
                    phase,
                )?;
                std::mem::swap(&mut st, &mut sc);
                self.capture(&format!("block-{block}"), &st);
            }
            let skip512 = st;
            let tokens = g.vit_tokens();
            let pooled = self.tensor("vit pooled", tokens, 512, Format::E4);
            self.op(
                "downsample",
                Op {
                    count: tokens * 512,
                    channels: 512,
                    in_width: d4.width,
                    in_height: d4.height,
                    out_width: d5.width,
                    out_height: d5.height,
                    in_f16: Some(&raw),
                    out_e4: Some(&pooled),
                    ..Default::default()
                },
                "pool 30",
            )?;
            let vit_state = self.tensor("vit state", tokens, 1024, Format::E4);
            self.gemm(
                Gemm {
                    input: Some(&pooled),
                    weights: Some(model.fp8_matrix(model.tensor(30, 4)?, 0, 512, 1024, None)?),
                    output: Some(&vit_state),
                    rows: tokens,
                    k: 512,
                    n: 1024,
                    ..Default::default()
                },
                "transition 30-31",
            )?;
            self.vit(&vit_state, tokens)?;

            // ---- Decoder 512: the ViT output projected, doubled, and merged onto the encoder skip.
            let projected = self.tensor("decoder 512 projection", tokens, 512, Format::F16);
            self.gemm(
                Gemm {
                    input: Some(&vit_state),
                    weights: Some(model.fp8_matrix(model.tensor(39, 0)?, 0, 1024, 512, None)?),
                    output_f16: Some(&projected),
                    rows: tokens,
                    k: 1024,
                    n: 512,
                    partition: 256,
                    ..Default::default()
                },
                "transition 38-39",
            )?;
            let merged = self.tensor("decoder 512 merge", rows, 512, Format::E4);
            let aux = model.aux_vector(self.gpu, model.tensor(39, 0)?, 1024 * 512, 512)?;
            self.op(
                "upsample_residual",
                Op {
                    count: rows * 512,
                    channels: 512,
                    in_width: d5.width,
                    in_height: d5.height,
                    out_width: d4.width,
                    out_height: d4.height,
                    in_f16: Some(&projected),
                    skip_e4: Some(&skip512),
                    aux: Some(aux),
                    out_e4: Some(&merged),
                    ..Default::default()
                },
                "block 39 merge",
            )?;
            self.capture("block-39", &merged);

            let mut dst = merged;
            let mut dsc = self.tensor("decoder 512 state", rows, 512, Format::E4);
            let dtemps = self.split_temporaries("decoder 512", rows);
            for block in 40..=47 {
                let phase = self.phases.take(4);
                self.split_block(block, d4.width, d4.height, &dtemps, &dst, &dsc, None, phase)?;
                std::mem::swap(&mut dst, &mut dsc);
                self.capture(&format!("block-{block}"), &dst);
            }
            stage_input = dst;
        }

        // ---- Decoder stages 256 / 128 / 64 / 32.
        let decoder_stages = [
            (d4, d3, 256, 48, 55, 3, skip256),
            (d3, d2, 128, 56, 61, 2, skip128),
            (d2, d1, 64, 62, 65, 1, skip64),
            (d1, d0, 32, 66, 69, 0, skip32),
        ];
        for (low, high, channels, first, last, level_index, skip) in decoder_stages {
            let rows = rows_of(high);
            let label = format!("decoder {channels}");
            let transition = model.tensor(first, 0)?;
            let layout = BlockLayout::upsample(channels);
            if transition.byte_length() != layout.end_without_padding + 16 {
                bail!("unexpected upsample layout for block {first}");
            }
            let projection = self.tensor(
                &format!("{label} projection"),
                rows_of(low),
                channels,
                Format::F16,
            );
            self.gemm(
                Gemm {
                    input: Some(&stage_input),
                    weights: Some(model.fp8_matrix(
                        transition,
                        layout.upsample_weight,
                        channels * 2,
                        channels,
                        None,
                    )?),
                    output_f16: Some(&projection),
                    rows: rows_of(low),
                    k: channels * 2,
                    n: channels,
                    ..Default::default()
                },
                &format!("transition ->{first}"),
            )?;
            let merged = self.tensor(&format!("{label} merge"), rows, channels, Format::E4);
            let merged_raw = (channels == 32)
                .then(|| self.tensor(&format!("{label} merge raw"), rows, 32, Format::F16));
            let aux = model.aux_vector(self.gpu, transition, layout.transition_scale, channels)?;
            self.op(
                "upsample_residual",
                Op {
                    count: rows * channels,
                    channels,
                    in_width: low.width,
                    in_height: low.height,
                    out_width: high.width,
                    out_height: high.height,
                    dual: merged_raw.is_some(),
                    in_f16: Some(&projection),
                    skip_e4: Some(&skip),
                    aux: Some(aux),
                    out_e4: Some(&merged),
                    out_f16: merged_raw.as_ref(),
                    ..Default::default()
                },
                &format!("block {first} merge"),
            )?;

            let mut st = merged;
            let mut sc = self.tensor(&format!("{label} state"), rows, channels, Format::E4);
            let temps = self.temporaries(&label, rows, channels, &layout);
            for block in first..=last {
                let phase = self.phases.take(level_index);
                self.block(
                    BlockArgs {
                        block,
                        channels,
                        width: high.width,
                        height: high.height,
                        layout: if block == first {
                            layout
                        } else {
                            BlockLayout::fused(channels)
                        },
                        tensor: model.tensor(block, 0)?,
                        state: &st,
                        output: Some(&sc),
                        output_f16: None,
                        ffn_skip_override: if block == first {
                            merged_raw.as_ref()
                        } else {
                            None
                        },
                        phase,
                    },
                    &temps,
                )?;
                std::mem::swap(&mut st, &mut sc);
                self.capture(&format!("block-{block}"), &st);
            }
            stage_input = st;
        }

        // ---- The post block at full resolution, and the head.
        let tensor = model.tensor(70, 0)?;
        let post = BlockLayout::post();
        if tensor.byte_length() != post.end_without_padding {
            bail!("unexpected block 70 layout");
        }
        let merged_raw = self.tensor("post merge raw", full_rows, 32, Format::F16);
        let merged = self.tensor("post merge", full_rows, 32, Format::E4);
        let aux = model.aux_pair(
            self.gpu,
            tensor,
            &[post.input_scale, post.adapter_scale],
            32,
        )?;
        self.op(
            "post_blend",
            Op {
                count: full_rows * 32,
                channels: 32,
                in_width: d0.width,
                in_height: d0.height,
                out_width: g.full_width,
                out_height: g.full_height,
                dual: true,
                aux_a: 0,
                aux_b: 32,
                in_e4: Some(&stage_input),
                skip_e4: Some(&block0),
                aux: Some(aux),
                out_e4: Some(&merged),
                out_f16: Some(&merged_raw),
                ..Default::default()
            },
            "post blend",
        )?;
        let block_raw = self.tensor("post block raw", full_rows, 32, Format::F16);
        let post_temps = self.temporaries("post", full_rows, 32, &post);
        let phase = self.phases.take(6);
        self.block(
            BlockArgs {
                block: 70,
                channels: 32,
                width: g.full_width,
                height: g.full_height,
                layout: post,
                tensor,
                state: &merged,
                output: None,
                output_f16: Some(&block_raw),
                ffn_skip_override: Some(&merged_raw),
                phase,
            },
            &post_temps,
        )?;
        let head = self.tensor("head", full_rows, 4, Format::F32);
        let (weights, padded_n) = model.f16_matrix(self.gpu, tensor, post.post_weights, 32, 4)?;
        self.gemm_f16(
            &block_raw,
            &weights,
            padded_n,
            [None, None, Some(&head)],
            full_rows,
            32,
            4,
            "head",
        )?;
        Ok(head)
    }
}
