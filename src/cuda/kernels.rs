//! The PTX kernels of the reference's fast route (scripts/ptx/), launched through the CUDA driver: a port of the
//! PTX half of its `nr::Kernels` (src/kernels.cpp), plus the three kernels that exist upstream only as GLSL
//! (cuda/ops.cu). Every route choice, argument list and grid below is the reference's.
//!
//! Counter chaining is not used. On one CUDA stream a launch starts only when the previous one has finished
//! (overlapping them would need programmatic dependent launch, sm_90), so every wait address is zero: the
//! kernels skip their waits, and the arithmetic does not depend on scheduling. The launches are recorded once and
//! replayed as a CUDA graph.

use anyhow::{Context, Result, bail};
use std::collections::HashMap;

use super::driver::{Arg, Cuda, DeviceBuffer, DevicePtr, Function, Module};

include!(concat!(env!("OUT_DIR"), "/ptx_table.rs"));

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    E4,
    F16,
    F32,
}

impl Format {
    pub fn bytes(self) -> u32 {
        match self {
            Format::E4 => 1,
            Format::F16 => 2,
            Format::F32 => 4,
        }
    }
}

pub fn align_rows(rows: u32) -> u32 {
    rows.next_multiple_of(64)
}

/// A `[rows][channels]` activation, rows allocated padded to 64 and zeroed.
#[derive(Clone, Debug)]
pub struct Activation {
    pub ptr: DevicePtr,
    pub rows: u32,
    pub channels: u32,
    pub alloc_rows: u32,
    pub format: Format,
    pub label: String,
}

impl Activation {
    pub fn valid_bytes(&self) -> usize {
        self.rows as usize * self.channels as usize * self.format.bytes() as usize
    }
}

#[derive(Default)]
pub struct GemmArgs<'a> {
    pub input: Option<&'a Activation>,
    pub input_column_base: u32,
    /// Plain `[K/32][Nmatrix][32]` E4M3 (`CudaModel::fp8_matrix`).
    pub weights: DevicePtr,
    pub n_matrix: u32,
    pub weight_column_offset: u32,
    /// F16 unless `quantize` (E4).
    pub output: Option<&'a Activation>,
    pub output_column_offset: u32,
    /// E4 copy of the published value, beside an f16 output.
    pub dual_output: Option<&'a Activation>,
    pub residual: Option<&'a Activation>,
    pub scale_residual: bool,
    /// The raw tensor holding the skip scales.
    pub aux: DevicePtr,
    pub aux_byte_offset: u32,
    pub silu: bool,
    pub quantize: bool,
    pub rows: u32,
    pub k: u32,
    pub n: u32,
    pub partition: u32,
}

#[derive(Default)]
pub struct Block32Args<'a> {
    pub features: Option<&'a Activation>,
    pub adapter_weights: DevicePtr,
    pub low_res: Option<&'a Activation>,
    pub low_projection: Option<&'a Activation>,
    pub input_scale_byte_offset: u32,
    pub adapter_scale_byte_offset: u32,
    pub low_width: u32,
    pub head_weights: DevicePtr,
    pub head: Option<&'a Activation>,
    pub pooled: Option<&'a Activation>,
    pub pooled_width: u32,
    pub state: Option<&'a Activation>,
    pub w1: DevicePtr,
    pub w2: DevicePtr,
    pub wqkv: DevicePtr,
    pub wproj: DevicePtr,
    pub prior: DevicePtr,
    pub aux: DevicePtr,
    pub ffn_scale_byte_offset: u32,
    pub attn_scale_byte_offset: u32,
    pub attention_scale_byte_offset: u32,
    pub out_e4: Option<&'a Activation>,
    pub width: u32,
    pub height: u32,
    pub shift_x: u32,
    pub shift_y: u32,
}

#[derive(Default)]
pub struct FfnArgs<'a> {
    /// The block state, E4 `[rows][channels]`; also the skip.
    pub input: Option<&'a Activation>,
    pub w1: DevicePtr,
    pub w2: DevicePtr,
    pub w3: DevicePtr,
    pub aux: DevicePtr,
    pub aux_byte_offset: u32,
    pub output: Option<&'a Activation>,
    pub rows: u32,
    pub channels: u32,
    /// The previous block's projection, computed in this kernel as the block state.
    pub attended: Option<&'a Activation>,
    pub ffn_prev: Option<&'a Activation>,
    pub wproj: DevicePtr,
    pub aux_prev: DevicePtr,
    pub aux_attn_byte_offset: u32,
    pub state_out: Option<&'a Activation>,
    pub width: u32,
}

struct Kernel {
    function: Function,
    dynamic_shared: u32,
    threads: u32,
    file: String,
}

enum Op {
    Launch {
        kernel: usize,
        grid: [u32; 3],
        block: u32,
        shared: u32,
        args: Vec<Arg>,
        label: String,
    },
    Zero {
        ptr: DevicePtr,
        bytes: usize,
    },
    Copy {
        from: DevicePtr,
        to: DevicePtr,
        bytes: usize,
    },
}

const SYNC_SLOT_BYTES: u32 = 6144;
const SYNC_SLOTS: u32 = 96;
const TILE_COUNTER_SLOT: u32 = 72;

pub struct Kernels<'c> {
    cuda: &'c Cuda,
    modules: Vec<Module>,
    kernels: Vec<Kernel>,
    /// Kernels by `file:entry`, modules by file.
    by_file: HashMap<String, usize>,
    module_of: HashMap<String, usize>,
    sync: DeviceBuffer,
    tile_cursor: u32,
    split_scratch: Option<DeviceBuffer>,
    chain_status: DeviceBuffer,
    ops: Vec<Op>,
}

fn check(condition: bool, message: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        bail!("{message}")
    }
}

fn ptx_exists(file: &str) -> bool {
    PTX.iter().any(|(name, _)| *name == file)
}

fn ptr(activation: Option<&Activation>) -> DevicePtr {
    activation.map_or(0, |a| a.ptr)
}

impl<'c> Kernels<'c> {
    pub fn new(cuda: &'c Cuda) -> Result<Self> {
        Ok(Self {
            cuda,
            modules: Vec::new(),
            kernels: Vec::new(),
            by_file: HashMap::new(),
            module_of: HashMap::new(),
            sync: cuda.alloc((SYNC_SLOTS * SYNC_SLOT_BYTES) as usize)?,
            tile_cursor: 0,
            split_scratch: None,
            chain_status: cuda.alloc(16)?,
            ops: Vec::new(),
        })
    }

    pub fn launch_count(&self) -> usize {
        self.ops
            .iter()
            .filter(|op| matches!(op, Op::Launch { .. }))
            .count()
    }

    pub fn module_count(&self) -> usize {
        self.modules.len()
    }

    /// Starts a recording: the split-K tile counters are zeroed at the top of every frame.
    pub fn begin(&mut self) {
        self.ops.clear();
        self.tile_cursor = 0;
        self.ops.push(Op::Zero {
            ptr: self.sync.ptr,
            bytes: self.sync.size,
        });
    }

    /// Issues the recorded frame on the stream.
    pub fn replay(&self) -> Result<()> {
        for op in &self.ops {
            match op {
                Op::Launch {
                    kernel,
                    grid,
                    block,
                    shared,
                    args,
                    label,
                } => self
                    .cuda
                    .launch(
                        &self.kernels[*kernel].function,
                        *grid,
                        *block,
                        *shared,
                        args,
                    )
                    .with_context(|| format!("launching {label}"))?,
                Op::Zero { ptr, bytes } => self.cuda.memset_async(*ptr, *bytes)?,
                Op::Copy { from, to, bytes } => self.cuda.copy_async(*to, *from, *bytes)?,
            }
        }
        Ok(())
    }

    /// Each launch of the recording on its own, timed, as `(kernel file, label, milliseconds)`.
    pub fn profile(&self) -> Result<Vec<(String, String, f64)>> {
        let mut timings = Vec::new();
        for op in &self.ops {
            match op {
                Op::Launch {
                    kernel,
                    grid,
                    block,
                    shared,
                    args,
                    label,
                } => {
                    let k = &self.kernels[*kernel];
                    let ms = self
                        .cuda
                        .time(|| self.cuda.launch(&k.function, *grid, *block, *shared, args))
                        .with_context(|| format!("launch {} ({label}, grid {grid:?}, block {block}, shared {shared})", k.file))?;
                    timings.push((k.file.clone(), label.clone(), ms));
                }
                Op::Zero { ptr, bytes } => self.cuda.memset_async(*ptr, *bytes)?,
                Op::Copy { from, to, bytes } => self.cuda.copy_async(*to, *from, *bytes)?,
            }
        }
        Ok(timings)
    }

    /// Waits that timed out in the last frame; nonzero means a chained wait gave up (never, without chaining).
    pub fn chain_timeouts(&self) -> Result<u32> {
        let bytes = self.cuda.read(self.chain_status.ptr, 4)?;
        Ok(u32::from_le_bytes(bytes[..4].try_into().unwrap()))
    }

    fn kernel(&mut self, file: &str, entry: &str) -> Result<usize> {
        let key = format!("{file}:{entry}");
        if let Some(&index) = self.by_file.get(&key) {
            return Ok(index);
        }
        let text = PTX
            .iter()
            .find(|(name, _)| *name == file)
            .map(|(_, text)| *text)
            .with_context(|| {
                format!("no PTX kernel {file}; tools/gen_ptx.sh generates the variant set")
            })?;
        let comment = |tag: &str| {
            text.find(tag)
                .and_then(|at| {
                    text[at + tag.len()..]
                        .split_whitespace()
                        .next()?
                        .parse::<u32>()
                        .ok()
                })
                .unwrap_or(0)
        };
        let module = match self.module_of.get(file) {
            Some(&index) => index,
            None => {
                let module = self
                    .cuda
                    .module(text)
                    .with_context(|| format!("JIT-compiling {file}"))?;
                self.modules.push(module);
                self.module_of
                    .insert(file.to_string(), self.modules.len() - 1);
                self.modules.len() - 1
            }
        };
        let function = self.cuda.function(&self.modules[module], entry)?;
        self.kernels.push(Kernel {
            function,
            dynamic_shared: comment("// dynamic_shared "),
            threads: comment("// threads "),
            file: file.to_string(),
        });
        self.by_file.insert(key, self.kernels.len() - 1);
        Ok(self.kernels.len() - 1)
    }

    fn launch(
        &mut self,
        kernel: usize,
        grid: [u32; 3],
        block: u32,
        shared: u32,
        args: Vec<Arg>,
        label: String,
    ) -> Result<()> {
        check(grid.iter().all(|&g| g > 0), &format!("{label}: empty grid"))?;
        self.cuda
            .allow_shared(&mut self.kernels[kernel].function, shared)?;
        self.ops.push(Op::Launch {
            kernel,
            grid,
            block,
            shared,
            args,
            label,
        });
        Ok(())
    }

    /// Copies a tensor as it stands at this point of the frame (boundary captures).
    pub fn copy(&mut self, from: &Activation, to: &Activation) {
        self.ops.push(Op::Copy {
            from: from.ptr,
            to: to.ptr,
            bytes: from.valid_bytes(),
        });
    }

    /// Per-frame zeroed counters for the split-K last-arrival reduction.
    fn tile_counters(&mut self, count: u32) -> Result<DevicePtr> {
        let bytes = (count * 4).next_multiple_of(16);
        check(
            self.tile_cursor + bytes <= (SYNC_SLOTS - TILE_COUNTER_SLOT) * SYNC_SLOT_BYTES,
            "tile counters exhausted (per frame)",
        )?;
        let address =
            self.sync.ptr + (TILE_COUNTER_SLOT * SYNC_SLOT_BYTES + self.tile_cursor) as u64;
        self.tile_cursor += bytes;
        Ok(address)
    }

    /// One fixed split-K scratch: recorded launches hold its address, so it is sized once.
    fn split_scratch(&mut self, bytes: usize) -> Result<DevicePtr> {
        if self.split_scratch.is_none() {
            self.split_scratch = Some(self.cuda.alloc(bytes.max(16 << 20))?);
        }
        let scratch = self.split_scratch.as_ref().unwrap();
        check(
            bytes <= scratch.size,
            "split-K partials exceed the scratch buffer",
        )?;
        Ok(scratch.ptr)
    }

    fn error_word(&self) -> DevicePtr {
        self.chain_status.ptr
    }

    /// The SM count the persistent grids are sized for; `NR_GRID_SMS` overrides it (a diagnostic: the output
    /// must not depend on it).
    fn grid_sms(sm_count: u32) -> u32 {
        std::env::var("NR_GRID_SMS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(sm_count)
    }

    /// `Kernels::vitGemmTileRows`: 192-row tiles for one row group, 96 for a few.
    fn vit_gemm_tile_rows(rows: u32) -> u32 {
        if rows <= 192 { 192 } else { 96 }
    }

    /// The FP8 GEMM, through the first PTX route that applies (gemmv, gemmt, gemm2), as `Kernels::gemmFp8`.
    pub fn gemm_fp8(&mut self, a: &GemmArgs<'_>) -> Result<()> {
        let input = a.input.context("GEMM input")?;
        let output = a.output.context("GEMM output")?;
        check(input.format == Format::E4, "FP8 GEMM input must be E4M3")?;
        check(
            a.k.is_multiple_of(32),
            "FP8 GEMM K must be a multiple of 32",
        )?;
        check(
            output.format == if a.quantize { Format::E4 } else { Format::F16 },
            "FP8 GEMM output format mismatch",
        )?;
        check(
            a.dual_output
                .is_none_or(|d| d.format == Format::E4 && !a.quantize),
            "dual output must be E4 beside f16",
        )?;
        check(
            output.alloc_rows >= align_rows(a.rows) && input.alloc_rows >= align_rows(a.rows),
            "GEMM rows",
        )?;
        check(
            a.input_column_base + a.k <= input.channels,
            "GEMM input columns",
        )?;
        check(
            a.output_column_offset + a.n <= output.channels,
            "GEMM output columns",
        )?;
        check(
            a.weight_column_offset + a.n <= a.n_matrix,
            "GEMM weight columns",
        )?;
        check(
            a.residual.is_none_or(|r| r.channels == output.channels),
            "residual must share the output layout",
        )?;
        check(
            a.partition == 0 || (a.partition.is_multiple_of(32) && a.k.is_multiple_of(a.partition)),
            "GEMM partition",
        )?;
        check(
            input.channels % 16 == 0 && a.input_column_base.is_multiple_of(16),
            "GEMM input rows must be 16-byte aligned",
        )?;
        let residual_ok = a
            .residual
            .is_none_or(|r| a.scale_residual && r.format == Format::E4);
        let label = format!(
            "{}x{}->{}{}{}",
            a.rows,
            a.k,
            a.n,
            if a.silu { " silu" } else { "" },
            if a.partition > 0 {
                format!(" p{}", a.partition)
            } else {
                String::new()
            }
        );
        let p_a = input.ptr;
        let p_res = a.residual.map_or(p_a, |r| r.ptr);
        let p_aux = if a.aux != 0 { a.aux } else { p_a };
        let p_out = if a.quantize {
            output.ptr
        } else {
            a.dual_output.map_or(p_a, |d| d.ptr)
        };
        let p_out16 = if a.quantize { p_a } else { output.ptr };
        let sm = self.cuda.sm_count;

        // ViT GEMM (gemmv_e4m3.py): 192- or 96-row tiles, one chain per workgroup (the whole K, or one partition
        // per split-K workgroup with the reduce folded into the last-arriving one).
        if a.n.is_multiple_of(128) && residual_ok && a.k / 32 >= 3 {
            let bm = Self::vit_gemm_tile_rows(a.rows);
            let warps = bm / 24;
            let (row_groups, col_groups) = (a.rows.div_ceil(bm), a.n / 128);
            // The partition count is the network's K split: part of the arithmetic, not a performance choice.
            let splits = if a.partition > 0 && a.k > a.partition {
                a.k / a.partition
            } else {
                1
            };
            let flags = u32::from(a.residual.is_some())
                | if a.silu { 2 } else { 0 }
                | if a.quantize || a.dual_output.is_some() {
                    4
                } else {
                    0
                }
                | if a.quantize { 0 } else { 8 };
            let entry = format!(
                "gemmv_e4m3_K{}_f{flags}_s{splits}{}",
                a.k,
                if bm != 192 {
                    format!("_m{bm}")
                } else {
                    String::new()
                }
            );
            if ptx_exists(&format!("{entry}.ptx")) && row_groups <= 65535 {
                let kernel = self.kernel(&format!("{entry}.ptx"), &entry)?;
                let partial_bytes = (splits * row_groups * col_groups * warps * 48 * 128) as usize;
                let p_partial = if splits > 1 {
                    let existing = self.split_scratch.as_ref().map_or(0, |s| s.size);
                    self.split_scratch(if existing == 0 {
                        4 * partial_bytes
                    } else {
                        partial_bytes
                    })?
                } else {
                    p_a
                };
                let p_count = if splits > 1 {
                    self.tile_counters(row_groups * col_groups)?
                } else {
                    p_a
                };
                let shared = self.kernels[kernel].dynamic_shared;
                check(shared > 0, "gemmv PTX without a dynamic_shared size")?;
                let args = vec![
                    Arg::Ptr(p_a),
                    Arg::Ptr(a.weights),
                    Arg::Ptr(p_res),
                    Arg::Ptr(p_aux),
                    Arg::Ptr(p_out),
                    Arg::Ptr(p_out16),
                    Arg::Ptr(p_partial),
                    Arg::Ptr(p_count),
                    Arg::U32(a.rows),
                    Arg::U32(input.channels),
                    Arg::U32(a.input_column_base),
                    Arg::U32(a.n_matrix),
                    Arg::U32(a.weight_column_offset),
                    Arg::U32(output.channels),
                    Arg::U32(a.output_column_offset),
                    Arg::U32(a.aux_byte_offset / 2),
                    Arg::Ptr(0),
                    Arg::U32(0),
                    Arg::Ptr(0),
                    Arg::Ptr(self.error_word()),
                ];
                return self.launch(
                    kernel,
                    [col_groups, row_groups, splits],
                    32 * warps,
                    shared,
                    args,
                    format!("gemmv {label}"),
                );
            }
        }

        // Tall-tile GEMM (gemmt_e4m3.py): 192-row tiles, partitions in the workgroup or one partition per split-K
        // workgroup plus a PTX reduce, for the few-row partitioned stages.
        if a.n.is_multiple_of(64)
            && residual_ok
            && a.partition != 0
            && a.rows.div_ceil(192) <= 65535
        {
            let part_steps = a.partition / 32;
            let (row_groups, col_groups) = (a.rows.div_ceil(192), a.n / 64);
            let split_stride = align_rows(a.rows) * a.n;
            let splits = if a.k > a.partition
                && row_groups * col_groups < 2 * sm
                && (a.k / a.partition) as usize * align_rows(a.rows) as usize * a.n as usize * 2
                    <= 16 << 20
            {
                a.k / a.partition
            } else {
                1
            };
            let out_flags = if a.silu { 2 } else { 0 }
                | if a.quantize || a.dual_output.is_some() {
                    4
                } else {
                    0
                }
                | if a.quantize { 0 } else { 8 };
            let flags = u32::from(a.residual.is_some()) | if splits == 1 { out_flags } else { 0 };
            let entry = format!("gemmt_e4m3_K{}_f{flags}_p{part_steps}_s{splits}", a.k);
            if ptx_exists(&format!("{entry}.ptx")) {
                let kernel = self.kernel(&format!("{entry}.ptx"), &entry)?;
                let p_scratch = if splits > 1 {
                    Some(self.split_scratch(splits as usize * split_stride as usize * 2)?)
                } else {
                    None
                };
                let p_out16_t = p_scratch.unwrap_or(p_out16);
                let (output_stride, output_column) = if splits > 1 {
                    (a.n, 0)
                } else {
                    (output.channels, a.output_column_offset)
                };
                let args = vec![
                    Arg::Ptr(p_a),
                    Arg::Ptr(a.weights),
                    Arg::Ptr(p_res),
                    Arg::Ptr(p_aux),
                    Arg::Ptr(p_out),
                    Arg::Ptr(p_out16_t),
                    Arg::U32(a.rows),
                    Arg::U32(input.channels),
                    Arg::U32(a.input_column_base),
                    Arg::U32(a.n_matrix),
                    Arg::U32(a.weight_column_offset),
                    Arg::U32(output_stride),
                    Arg::U32(output_column),
                    Arg::U32(a.aux_byte_offset / 2),
                    Arg::U32(split_stride),
                ];
                // = gemmt_e4m3.py shared_bytes: a 3-stage ring (8 KB per stage) or the f16 tile, then the E4 tile
                // only next to an f16 tile.
                let has_f16 = splits > 1 || !a.quantize;
                let has_e4 = splits == 1 && (a.quantize || a.dual_output.is_some());
                let shared = (a.k / 32 / splits).min(3) * 8192;
                let shared = shared.max(if has_f16 { 192 * 128 } else { 0 })
                    + if has_e4 && has_f16 { 192 * 64 } else { 0 };
                self.launch(
                    kernel,
                    [col_groups, row_groups, splits],
                    384,
                    shared,
                    args,
                    format!("gemmt {label}"),
                )?;
                if let Some(p_partial) = p_scratch {
                    let entry = format!("reduce_e4m3_s{splits}_f{out_flags}");
                    let reduce = self.kernel(&format!("{entry}.ptx"), &entry)?;
                    let args = vec![
                        Arg::Ptr(p_partial),
                        Arg::Ptr(p_out),
                        Arg::Ptr(p_out16),
                        Arg::U32(a.rows),
                        Arg::U32(a.n),
                        Arg::U32(split_stride),
                        Arg::U32(output.channels),
                        Arg::U32(a.output_column_offset),
                    ];
                    self.launch(
                        reduce,
                        [(a.rows * a.n / 8).div_ceil(256), 1, 1],
                        256,
                        0,
                        args,
                        format!("reduce {label}"),
                    )?;
                }
                return Ok(());
            }
        }

        // The general PTX GEMM (gemm2_e4m3.py): 64x64 tiles, the residual prologue and the publication epilogue.
        if a.partition == 0 && a.n.is_multiple_of(64) && residual_ok && a.rows.div_ceil(64) <= 65535
        {
            let flags = u32::from(a.residual.is_some())
                | if a.silu { 2 } else { 0 }
                | if a.quantize || a.dual_output.is_some() {
                    4
                } else {
                    0
                }
                | if a.quantize { 0 } else { 8 };
            let entry = format!("gemm2_e4m3_K{}_f{flags}", a.k);
            let kernel = self.kernel(&format!("{entry}.ptx"), &entry)?;
            let threads = match self.kernels[kernel].threads {
                0 => 128,
                t => t,
            };
            let args = vec![
                Arg::Ptr(p_a),
                Arg::Ptr(a.weights),
                Arg::Ptr(p_res),
                Arg::Ptr(p_aux),
                Arg::Ptr(p_out),
                Arg::Ptr(p_out16),
                Arg::U32(a.rows),
                Arg::U32(input.channels),
                Arg::U32(a.input_column_base),
                Arg::U32(a.n_matrix),
                Arg::U32(a.weight_column_offset),
                Arg::U32(output.channels),
                Arg::U32(a.output_column_offset),
                Arg::U32(a.aux_byte_offset / 2),
                Arg::Ptr(0),
                Arg::U32(0),
                Arg::U32(0),
                Arg::Ptr(0),
                Arg::U32(0),
                Arg::Ptr(0),
                Arg::U32(1),
                Arg::U32(64),
                Arg::Ptr(self.error_word()),
            ];
            return self.launch(
                kernel,
                [a.n / 64, a.rows.div_ceil(64), 1],
                threads,
                0,
                args,
                format!("gemm2 {label}"),
            );
        }
        bail!("no PTX GEMM route for {label}")
    }

    /// `batches` independent GEMMs in one launch of the batched gemm2 (tools/ptx/gemm2_batched.py): batch b reads
    /// input columns `input_column_base + b K`, writes output columns `output_column_offset + b N`, and its
    /// matrix starts `b K Nmatrix` bytes after `weights` (a batched tile-major matrix). Plain E4 outputs only.
    pub fn gemm_fp8_batched(&mut self, a: &GemmArgs<'_>, batches: u32) -> Result<()> {
        let input = a.input.context("GEMM input")?;
        let output = a.output.context("GEMM output")?;
        check(
            input.format == Format::E4 && output.format == Format::E4 && a.quantize,
            "batched GEMM: E4 in and out",
        )?;
        check(
            a.residual.is_none() && a.partition == 0 && a.dual_output.is_none(),
            "batched GEMM: no skip, partition or dual",
        )?;
        check(
            a.n.is_multiple_of(64) && a.k.is_multiple_of(32),
            "batched GEMM shape",
        )?;
        check(
            a.input_column_base + batches * a.k <= input.channels,
            "batched GEMM input columns",
        )?;
        check(
            a.output_column_offset + batches * a.n <= output.channels,
            "batched GEMM output columns",
        )?;
        check(
            output.alloc_rows >= align_rows(a.rows) && input.alloc_rows >= align_rows(a.rows),
            "GEMM rows",
        )?;
        let flags = if a.silu { 6 } else { 4 };
        let entry = format!("gemm2b_e4m3_K{}_f{flags}", a.k);
        let kernel = self.kernel(&format!("{entry}.ptx"), &entry)?;
        let threads = match self.kernels[kernel].threads {
            0 => 128,
            t => t,
        };
        let args = vec![
            Arg::Ptr(input.ptr),
            Arg::Ptr(a.weights),
            Arg::Ptr(input.ptr),
            Arg::Ptr(input.ptr),
            Arg::Ptr(output.ptr),
            Arg::Ptr(input.ptr),
            Arg::U32(a.rows),
            Arg::U32(input.channels),
            Arg::U32(a.input_column_base),
            Arg::U32(a.n_matrix),
            Arg::U32(a.weight_column_offset),
            Arg::U32(output.channels),
            Arg::U32(a.output_column_offset),
            Arg::U32(0),
            Arg::Ptr(0),
            Arg::U32(0),
            Arg::U32(0),
            Arg::Ptr(0),
            Arg::U32(0),
            Arg::Ptr(0),
            Arg::U32(1),
            Arg::U32(64),
            Arg::Ptr(self.error_word()),
            Arg::U32(a.k),
            Arg::U32(a.n),
            Arg::U32(a.k * a.n_matrix),
        ];
        self.launch(
            kernel,
            [a.n / 64, a.rows.div_ceil(64), batches],
            threads,
            0,
            args,
            format!(
                "gemm2b {}x{}->{} x{batches}{}",
                a.rows,
                a.k,
                a.n,
                if a.silu { " silu" } else { "" }
            ),
        )
    }

    /// The whole 32-channel block in one persistent launch (block32_e4m3.py), with the optional fusions: the input
    /// adapter (block 0), the 2x2 pool of the output, the decoder upsample merge, the post blend and the head.
    pub fn fused_block32(&mut self, a: &Block32Args<'_>) -> Result<()> {
        let flags = if a.out_e4.is_some() { 2 } else { 0 }
            | if a.features.is_some() { 8 } else { 0 }
            | if a.low_res.is_some() { 16 } else { 0 }
            | if a.head.is_some() { 32 } else { 0 }
            | if a.pooled.is_some() { 64 } else { 0 }
            | if a.low_projection.is_some() { 128 } else { 0 };
        let entry = format!("block32_e4m3_f{flags}");
        let kernel = self.kernel(&format!("{entry}.ptx"), &entry)?;
        let windows_x = (a.width + a.shift_x).div_ceil(8);
        let windows = windows_x * (a.height + a.shift_y).div_ceil(8);
        let p_state = ptr(a.features.or(a.state));
        let p_low = a.low_res.or(a.low_projection).map_or(p_state, |l| l.ptr);
        let p_out_e4 = a.out_e4.map_or(p_state, |o| o.ptr);
        let p_out2 = a.pooled.or(a.head).map_or(p_state, |o| o.ptr);
        let p_wf16 = if a.features.is_some() {
            a.adapter_weights
        } else if a.head.is_some() {
            a.head_weights
        } else {
            p_state
        };
        let low_width = if a.pooled.is_some() {
            a.pooled_width
        } else {
            a.low_width
        };
        let args = vec![
            Arg::Ptr(p_state),
            Arg::Ptr(p_low),
            Arg::Ptr(a.w1),
            Arg::Ptr(a.w2),
            Arg::Ptr(a.wqkv),
            Arg::Ptr(a.wproj),
            Arg::Ptr(a.aux),
            Arg::Ptr(a.prior),
            Arg::Ptr(p_out_e4),
            Arg::Ptr(p_out2),
            Arg::Ptr(p_wf16),
            Arg::U32(a.width),
            Arg::U32(a.height),
            Arg::U32(a.shift_x),
            Arg::U32(a.shift_y),
            Arg::U32(windows_x),
            Arg::U32(windows),
            Arg::U32(a.ffn_scale_byte_offset / 2),
            Arg::U32(a.attn_scale_byte_offset / 2),
            Arg::U32(a.attention_scale_byte_offset / 4),
            Arg::U32(a.input_scale_byte_offset / 2),
            Arg::U32(a.adapter_scale_byte_offset / 2),
            Arg::U32(low_width),
            Arg::Ptr(0),
            Arg::U32(0),
            Arg::U32(0),
            Arg::U32(0),
            Arg::Ptr(0),
            Arg::Ptr(self.error_word()),
        ];
        // Persistent: eight resident workgroups per SM, each walking windows.
        let groups = windows.min(8 * Self::grid_sms(self.cuda.sm_count));
        self.launch(
            kernel,
            [groups, 1, 1],
            128,
            0,
            args,
            format!("block32 {windows}w f{flags}"),
        )
    }

    /// `Kernels::ffnRowTiles`: 16-row tiles per workgroup.
    pub fn ffn_row_tiles(channels: u32) -> u32 {
        if channels == 256 { 3 } else { 4 }
    }

    /// The expert FFN and W3 in one launch (ffn_e4m3.py), optionally computing the previous block's projection
    /// as its input on chip.
    pub fn expert_ffn(&mut self, a: &FfnArgs<'_>) -> Result<()> {
        let input = a.input.context("FFN input")?;
        let output = a.output.context("FFN output")?;
        check(
            input.format == Format::E4 && output.format == Format::E4,
            "PTX FFN formats",
        )?;
        check(matches!(a.channels, 64 | 128 | 256), "PTX FFN channels")?;
        check(
            input.channels == a.channels && output.channels == a.channels,
            "PTX FFN strides",
        )?;
        let experts = a.channels / 32;
        let row_tiles = Self::ffn_row_tiles(a.channels);
        let shared = 2 * (experts * 4096 + 2048) + 16 * row_tiles * (a.channels + 16);
        let proj = a.attended.is_some();
        let entry = format!(
            "ffn_e4m3_C{}_R{row_tiles}{}",
            a.channels,
            if proj { "_proj" } else { "" }
        );
        let kernel = self.kernel(&format!("{entry}.ptx"), &entry)?;
        let p_a = input.ptr;
        let args = vec![
            Arg::Ptr(p_a),
            Arg::Ptr(a.w1),
            Arg::Ptr(a.w2),
            Arg::Ptr(a.w3),
            Arg::Ptr(a.aux),
            Arg::Ptr(output.ptr),
            Arg::U32(a.rows),
            Arg::U32(a.aux_byte_offset / 2),
            Arg::Ptr(a.attended.map_or(p_a, |t| t.ptr)),
            Arg::Ptr(a.ffn_prev.map_or(p_a, |t| t.ptr)),
            Arg::Ptr(if proj { a.wproj } else { p_a }),
            Arg::U32(a.aux_attn_byte_offset / 2),
            Arg::Ptr(if proj { a.aux_prev } else { a.aux }),
            Arg::Ptr(a.state_out.map_or(p_a, |t| t.ptr)),
            Arg::U32(u32::from(a.state_out.is_some())),
            Arg::Ptr(0),
            Arg::U32(0),
            Arg::U32(0),
            Arg::Ptr(0),
            Arg::Ptr(0),
            Arg::U32(a.width),
            Arg::U32(1),
            Arg::U32(64),
            Arg::Ptr(self.error_word()),
        ];
        let groups = a.rows.div_ceil(16 * row_tiles);
        self.launch(
            kernel,
            [groups, 1, 1],
            32 * experts * row_tiles,
            shared,
            args,
            format!(
                "ffn {}x{}{}",
                a.rows,
                a.channels,
                if proj { " +proj" } else { "" }
            ),
        )
    }

    /// QKV projection, normalization and window attention per (window, head) in one launch (qkv_e4m3.py).
    #[allow(clippy::too_many_arguments)]
    pub fn qkv_attention(
        &mut self,
        input: &Activation,
        weights: DevicePtr,
        aux: DevicePtr,
        scale_byte_offset: u32,
        prior: DevicePtr,
        attended: &Activation,
        width: u32,
        height: u32,
        heads: u32,
        shift_x: u32,
        shift_y: u32,
    ) -> Result<()> {
        check(
            input.format == Format::E4 && input.channels == heads * 32,
            "qkv attention input",
        )?;
        check(
            attended.format == Format::E4 && attended.channels == heads * 32,
            "qkv attention output",
        )?;
        let windows_x = (width + shift_x).div_ceil(8);
        let windows = windows_x * (height + shift_y).div_ceil(8);
        let items = heads * windows;
        check(
            heads.is_power_of_two() && items < (1 << 24),
            "PTX qkv item indexing",
        )?;
        let entry = format!("qkv_e4m3_K{}", heads * 32);
        let kernel = self.kernel(&format!("{entry}.ptx"), &entry)?;
        let args = vec![
            Arg::Ptr(input.ptr),
            Arg::Ptr(weights),
            Arg::Ptr(prior),
            Arg::Ptr(aux),
            Arg::Ptr(attended.ptr),
            Arg::U32(width),
            Arg::U32(height),
            Arg::U32(shift_x),
            Arg::U32(shift_y),
            Arg::U32(windows_x),
            Arg::U32(scale_byte_offset / 4),
            Arg::U32(windows),
            Arg::U32(items),
            Arg::Ptr(0),
            Arg::Ptr(0),
            Arg::U32(1),
            Arg::U32(64),
            Arg::Ptr(self.error_word()),
        ];
        // Persistent, each workgroup walking (window, head) items.
        let groups = items.min(12 * Self::grid_sms(self.cuda.sm_count));
        self.launch(
            kernel,
            [groups, 1, 1],
            128,
            0,
            args,
            format!("qkv {windows}w x{heads}"),
        )
    }

    /// The ViT's normalize + global attention, resident route (global_attention_e4m3.py), for <= 256 padded tokens.
    #[allow(clippy::too_many_arguments)]
    pub fn global_attention(
        &mut self,
        qkv: &Activation,
        aux: DevicePtr,
        scale_byte_offset: u32,
        attended: &Activation,
        tokens: u32,
        padded: u32,
        heads: u32,
    ) -> Result<()> {
        let entry = format!("global_attention_e4m3_p{padded}");
        let kernel = self.kernel(&format!("{entry}.ptx"), &entry)?;
        let shared = self.kernels[kernel].dynamic_shared;
        let args = vec![
            Arg::Ptr(qkv.ptr),
            Arg::Ptr(aux),
            Arg::Ptr(attended.ptr),
            Arg::U32(tokens),
            Arg::U32(heads),
            Arg::U32(scale_byte_offset / 4),
            Arg::Ptr(0),
            Arg::U32(0),
            Arg::Ptr(0),
            Arg::Ptr(self.error_word()),
        ];
        self.launch(
            kernel,
            [heads, padded / 64, 1],
            128,
            shared,
            args,
            format!("global attention {tokens}t"),
        )
    }

    /// The streamed ViT route for any token count: normalize once into the per-head E4 layout...
    #[allow(clippy::too_many_arguments)]
    pub fn global_normalize(
        &mut self,
        qkv: &Activation,
        aux: DevicePtr,
        scale_byte_offset: u32,
        normalized: &Activation,
        tokens: u32,
        padded: u32,
        heads: u32,
    ) -> Result<()> {
        check(
            normalized.alloc_rows as usize * normalized.channels as usize
                >= 3 * (heads * padded * 32) as usize,
            "global normalize buffer",
        )?;
        let kernel = self.kernel("global_normalize_e4m3.ptx", "global_normalize_e4m3")?;
        let args = vec![
            Arg::Ptr(qkv.ptr),
            Arg::Ptr(aux),
            Arg::Ptr(normalized.ptr),
            Arg::U32(tokens),
            Arg::U32(padded),
            Arg::U32(heads),
            Arg::U32(scale_byte_offset / 4),
            Arg::Ptr(0),
            Arg::U32(0),
            Arg::Ptr(0),
            Arg::Ptr(self.error_word()),
        ];
        self.launch(
            kernel,
            [padded / 64, heads, 1],
            64,
            0,
            args,
            format!("global normalize {tokens}t"),
        )
    }

    /// ...then attention with the key blocks streamed through shared memory.
    pub fn global_attention_stream(
        &mut self,
        normalized: &Activation,
        attended: &Activation,
        tokens: u32,
        padded: u32,
        heads: u32,
    ) -> Result<()> {
        let kernel = self.kernel(
            "global_attention_stream_e4m3.ptx",
            "global_attention_stream_e4m3",
        )?;
        let args = vec![
            Arg::Ptr(normalized.ptr),
            Arg::Ptr(attended.ptr),
            Arg::U32(tokens),
            Arg::U32(padded),
            Arg::U32(heads),
            Arg::Ptr(0),
            Arg::U32(0),
            Arg::Ptr(0),
            Arg::Ptr(self.error_word()),
        ];
        self.launch(
            kernel,
            [heads, padded / 64, 1],
            128,
            0,
            args,
            format!("global attention stream {tokens}t"),
        )
    }

    fn ops_launch(&mut self, entry: &str, count: u32, args: Vec<Arg>) -> Result<()> {
        check(
            count.is_multiple_of(8),
            "ops: element count must be a multiple of 8",
        )?;
        let kernel = self.kernel("ops_cuda.ptx", entry)?;
        self.launch(
            kernel,
            [(count / 8).div_ceil(256), 1, 1],
            256,
            0,
            args,
            entry.to_string(),
        )
    }

    /// Raw f16 2x2 box pool -> E4M3 (ops.comp MODE_DOWNSAMPLE_FP8).
    #[allow(clippy::too_many_arguments)]
    pub fn downsample(
        &mut self,
        input: &Activation,
        output: &Activation,
        in_width: u32,
        in_height: u32,
        out_width: u32,
        out_height: u32,
    ) -> Result<()> {
        check(
            input.format == Format::F16 && output.format == Format::E4,
            "downsample formats",
        )?;
        check(
            input.rows == in_width * in_height && output.rows == out_width * out_height,
            "downsample geometry",
        )?;
        let count = output.rows * output.channels;
        self.ops_launch(
            "nr_downsample",
            count,
            vec![
                Arg::Ptr(input.ptr),
                Arg::Ptr(output.ptr),
                Arg::U32(count),
                Arg::U32(output.channels),
                Arg::U32(in_width),
                Arg::U32(in_height),
                Arg::U32(out_width),
            ],
        )
    }

    /// Decoder merge: E4 (and optionally f16) of round_f16(projection[low] + skip * scale).
    #[allow(clippy::too_many_arguments)]
    pub fn upsample_residual(
        &mut self,
        projection: &Activation,
        skip: &Activation,
        aux: DevicePtr,
        scale_byte_offset: u32,
        output: &Activation,
        raw: Option<&Activation>,
        in_width: u32,
        out_width: u32,
    ) -> Result<()> {
        check(
            projection.format == Format::F16
                && skip.format == Format::E4
                && output.format == Format::E4,
            "upsample residual formats",
        )?;
        let count = output.rows * output.channels;
        self.ops_launch(
            "nr_upsample_residual",
            count,
            vec![
                Arg::Ptr(projection.ptr),
                Arg::Ptr(skip.ptr),
                Arg::Ptr(aux),
                Arg::Ptr(output.ptr),
                Arg::Ptr(raw.map_or(output.ptr, |r| r.ptr)),
                Arg::U32(count),
                Arg::U32(output.channels),
                Arg::U32(in_width),
                Arg::U32(out_width),
                Arg::U32(scale_byte_offset / 2),
                Arg::U32(u32::from(raw.is_some())),
            ],
        )
    }

    /// The 8-bit RGB image -> the RGBA f32 proxy (code / 255).
    pub fn proxy_from_rgb8(&mut self, rgb: DevicePtr, proxy: DevicePtr, pixels: u32) -> Result<()> {
        let kernel = self.kernel("ops_cuda.ptx", "nr_proxy_from_rgb8")?;
        let args = vec![Arg::Ptr(rgb), Arg::Ptr(proxy), Arg::U32(pixels)];
        self.launch(
            kernel,
            [pixels.div_ceil(256), 1, 1],
            256,
            0,
            args,
            "proxy from rgb8".into(),
        )
    }

    /// The head's residual composed onto the image, as 8-bit RGB (network::compose on the device).
    pub fn compose(
        &mut self,
        head: &Activation,
        rgb: DevicePtr,
        out: DevicePtr,
        width: u32,
        height: u32,
        full_width: u32,
    ) -> Result<()> {
        let kernel = self.kernel("ops_cuda.ptx", "nr_compose")?;
        let count = width * height * 3;
        let args = vec![
            Arg::Ptr(head.ptr),
            Arg::Ptr(rgb),
            Arg::Ptr(out),
            Arg::U32(width),
            Arg::U32(height),
            Arg::U32(full_width),
        ];
        self.launch(
            kernel,
            [count.div_ceil(256), 1, 1],
            256,
            0,
            args,
            "compose".into(),
        )
    }

    /// The first `target.channels` f16 channels of every row of `source`.
    pub fn narrow_f16(&mut self, source: &Activation, target: &Activation) -> Result<()> {
        check(
            source.format == Format::F16
                && target.format == Format::F16
                && source.rows == target.rows,
            "narrow formats",
        )?;
        let kernel = self.kernel("ops_cuda.ptx", "nr_narrow_f16")?;
        let count = target.rows * target.channels;
        let args = vec![
            Arg::Ptr(source.ptr),
            Arg::Ptr(target.ptr),
            Arg::U32(target.rows),
            Arg::U32(source.channels),
            Arg::U32(target.channels),
            Arg::U32(target.channels),
        ];
        self.launch(
            kernel,
            [count.div_ceil(256), 1, 1],
            256,
            0,
            args,
            "narrow f16".into(),
        )
    }

    /// Input features from an RGBA f32 proxy (preprocess.comp).
    #[allow(clippy::too_many_arguments)]
    pub fn preprocess(
        &mut self,
        proxy: DevicePtr,
        features: &Activation,
        full: (u32, u32),
        valid: (u32, u32),
        seed: u32,
        conditioning: [f32; 5],
    ) -> Result<()> {
        let kernel = self.kernel("ops_cuda.ptx", "nr_preprocess")?;
        let mut args = vec![
            Arg::Ptr(proxy),
            Arg::Ptr(features.ptr),
            Arg::U32(full.0),
            Arg::U32(full.1),
            Arg::U32(valid.0),
            Arg::U32(valid.1),
            Arg::U32(valid.0),
            Arg::U32(valid.1),
            Arg::U32(seed),
        ];
        args.extend(conditioning.iter().map(|&v| Arg::F32(v)));
        self.launch(
            kernel,
            [(full.0 * full.1).div_ceil(256), 1, 1],
            256,
            0,
            args,
            "preprocess".into(),
        )
    }
}
