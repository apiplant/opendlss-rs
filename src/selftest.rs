//! The reference port's arithmetic self-test kernels (`shaders/selftest.wgsl`), for `tools/check_numerics.mjs`.
//!
//! Every scalar primitive the network's kernels are made of is evaluated here, through the same naga and
//! driver path as the network, so the check covers what a browser-validated port cannot vouch for on another
//! compiler: contraction, denormals, and the conversions the naga rewrites introduced.

use anyhow::{Context, Result};
use std::{fs, path::Path};

use crate::runtime::Gpu;

const NUMERICS: &str = include_str!("../shaders/numerics.wgsl");
const SELFTEST: &str = include_str!("../shaders/selftest.wgsl");
const NORM_FMA: &str = include_str!("../shaders/norm_fma.wgsl");

/// The kernels round with WGSL's native `f16(x)` as well as with numerics.wgsl's bit-exact `f16_bits`; this
/// checks the native conversion against the same reference. `f32(h)` is exact, and `f16_bits` of a value
/// already on the half grid returns its pattern, so the result is the bits of `f16(x)`.
const NATIVE: &str = "
enable f16;
@group(0) @binding(0) var<storage, read> inputs : array<u32>;
@group(0) @binding(1) var<storage, read_write> results : array<u32>;
@compute @workgroup_size(64)
fn case_native_f16_bits(@builtin(global_invocation_id) id : vec3<u32>) {
  let i = id.x;
  if (i >= arrayLength(&results)) { return; }
  results[i] = f16_bits(f32(f16(bitcast<f32>(inputs[i]))));
}

// The window kernel's half fma (shaders/norm_fma.wgsl, the text the generator puts in the kernel), with the
// test's own opaque zero: the last input word, always 0.
fn nr_opaque_zero() -> u32 { return inputs[arrayLength(&inputs) - 1u]; }
@compute @workgroup_size(64)
fn case_native_norm_fma(@builtin(global_invocation_id) id : vec3<u32>) {
  let i = id.x;
  let count = arrayLength(&results);
  if (i >= count) { return; }
  let word = inputs[i];
  let a = f16(f16_to_f32(word & 0xffffu));
  let b = a;
  let c = f16_to_f32(word >> 16u);
  results[i] = f16_bits(f32(nr_norm_fma(a, b, f16(c))));
}
";

/// Runs every case listed in `dir/cases.txt` (`index entry_point count` per line), reading `index.in` as the
/// u32 inputs and writing `index.out` as the u32 results.
pub fn run(gpu: &Gpu, dir: &Path) -> Result<()> {
    let module = gpu.module("selftest.wgsl", &format!("{NUMERICS}\n{SELFTEST}"));
    let native = gpu.module(
        "native f16",
        &format!("enable f16;\n{NUMERICS}\n{}\n{NORM_FMA}", &NATIVE[13..]),
    );
    let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let layout = gpu
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("selftest"),
            entries: &[storage(0, true), storage(1, false)],
        });
    let pipeline_layout = gpu
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("selftest"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
    let list = fs::read_to_string(dir.join("cases.txt")).context("reading cases.txt")?;
    for line in list.lines().filter(|l| !l.trim().is_empty()) {
        let mut fields = line.split_whitespace();
        let (index, entry, count) = (
            fields.next().context("case index")?,
            fields.next().context("entry point")?,
            fields.next().context("count")?.parse::<u32>()?,
        );
        let inputs = gpu.storage_from("inputs", &fs::read(dir.join(format!("{index}.in")))?);
        // Exactly `count` words: the kernels bound themselves by arrayLength(&results).
        let results = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(entry),
            size: count as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let pipeline = gpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: if entry.starts_with("case_native") {
                    &native
                } else {
                    &module
                },
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            });
        let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(entry),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: inputs.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: results.as_entire_binding(),
                },
            ],
        });
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(count.div_ceil(64), 1, 1);
        }
        gpu.queue.submit([encoder.finish()]);
        let bytes = gpu.read_back(&results, count as u64 * 4)?;
        fs::write(dir.join(format!("{index}.out")), bytes)?;
    }
    // The two GEMM lookup tables, as the network builds them, for the script to check against the oracle.
    let tables = crate::network::tables(gpu)?;
    fs::write(
        dir.join("silu.bin"),
        gpu.read_back(&tables.silu, 65536 * 4)?,
    )?;
    fs::write(
        dir.join("packed_silu.bin"),
        gpu.read_back(&tables.packed_silu, 65536)?,
    )?;
    gpu.check()
}
