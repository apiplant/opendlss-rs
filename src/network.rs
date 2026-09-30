//! Putting it together: kernels, lookup tables, the recorded graph for one resolution, and a frame that goes
//! from an 8-bit sRGB image to the network's re-rendered image.
//!
//! A frame is: the image's code values uploaded as the display proxy, the input features built from them
//! (`shaders/preprocess.wgsl`: three Gaussian noise lanes, the centred proxy, the conditioning), the 71 blocks,
//! and the head. The head's RGB residual is then composed onto the proxy on the host, exactly as the reference
//! port's `composeImage` does: a single frame, so no history to blend and no display transform.

use anyhow::{Context, Result, bail};
use std::{collections::HashMap, time::Instant};

use crate::{
    geometry::Geometry,
    graph::{Graph, Tables},
    model::Model,
    runtime::{Format, Gpu, Modules, PipelineKey, Recorder, Tensor, Tensors},
    weights::GpuModel,
};

const NUMERICS: &str = include_str!("../shaders/numerics.wgsl");
const GEMM_F16: &str = include_str!("../shaders/gemm_f16.wgsl");
const VIT: &str = include_str!("../shaders/vit.wgsl");
const OPS: &str = include_str!("../shaders/ops.wgsl");
const VIT_ATTEND_PARALLEL: &str = include_str!("../shaders/vit_attend_parallel.wgsl");
const PREPROCESS: &str = include_str!("../shaders/preprocess.wgsl");
const WINDOW: &str = include_str!("../shaders/window_attention.wgsl");
const SILU_TABLE: &str = include_str!("../shaders/silu_table.wgsl");
const PACKED_SILU_TABLE: &str = include_str!("../shaders/packed_silu_table.wgsl");
const WEIGHT_METADATA: &[u8] = include_bytes!("../shaders/weight_metadata.bin");
const GEMMS: [(&str, &str); 6] = [
    ("gemm_e4", include_str!("../shaders/gemm_e4.wgsl")),
    (
        "gemm_e4_batched",
        include_str!("../shaders/gemm_e4_batched.wgsl"),
    ),
    ("gemm_half", include_str!("../shaders/gemm_half.wgsl")),
    (
        "gemm_half_batched",
        include_str!("../shaders/gemm_half_batched.wgsl"),
    ),
    ("gemm_dual", include_str!("../shaders/gemm_dual.wgsl")),
    (
        "gemm_dual_batched",
        include_str!("../shaders/gemm_dual_batched.wgsl"),
    ),
];

/// The five conditioning inputs and the noise seed. The defaults are the ones the reference uses for a
/// recorded proxy with no conditioning of its own.
#[derive(Debug, Clone, Copy)]
pub struct Conditioning {
    pub seed: u32,
    pub auto_mask: bool,
    pub local_tone: f32,
    pub local_structure: f32,
    pub skin_structure: f32,
    pub style: f32,
}

impl Default for Conditioning {
    fn default() -> Self {
        Self {
            seed: 0,
            auto_mask: false,
            local_tone: 1.0,
            local_structure: 1.0,
            skin_structure: -1.0,
            style: 0.0,
        }
    }
}

/// An 8-bit RGB image, row-major.
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
}

pub struct Timings {
    pub compile_seconds: f64,
    /// Upload, the network and the head's read back.
    pub frame_seconds: f64,
    /// The frame plus composing the output image.
    pub total_seconds: f64,
}

pub struct Network {
    pub geometry: Geometry,
    pub dispatches: usize,
    pub pipelines: usize,
    pub activation_bytes: u64,
    pub weight_bytes: u64,
    pub compile_seconds: f64,
    recorder: Recorder,
    proxy: Tensor,
    features: Tensor,
    head: Tensor,
    /// Where the frame's command buffer leaves the head, so reading it back needs no second submission.
    head_staging: wgpu::Buffer,
    boundaries: Vec<(String, Tensor)>,
    tensors: Tensors,
}

/// `NR_REFERENCE_KERNELS=1` runs the reference port's kernels wherever this crate has a faster rewrite of
/// one, so the two can be compared byte for byte (tools/compare_kernels.sh).
pub fn reference_kernels() -> bool {
    std::env::var("NR_REFERENCE_KERNELS").is_ok_and(|v| v == "1")
}

fn with_numerics(source: &str) -> String {
    format!("{NUMERICS}\n{source}")
}

fn modules(gpu: &Gpu, geometry: &Geometry) -> Result<Modules> {
    let mut by_name = HashMap::new();
    let mut shared_entry = HashMap::new();
    // The ViT's shared arrays are sized by the padded token count, which only the geometry knows. It is an
    // array length, so it is substituted as a constant rather than passed as an override.
    let override_line = "override PADDED_TOKENS : u32 = 64u;";
    if !VIT.contains(override_line) {
        bail!("vit.wgsl no longer declares {override_line}");
    }
    let vit = VIT.replace(
        override_line,
        &format!(
            "const PADDED_TOKENS : u32 = {}u;",
            geometry.padded_vit_tokens()
        ),
    );
    let shared: [(&str, String, &[&str]); 4] = [
        ("gemm_f16.wgsl", with_numerics(GEMM_F16), &["gemm_f16"]),
        (
            "vit.wgsl",
            with_numerics(&format!("{vit}\n{VIT_ATTEND_PARALLEL}")),
            &["vit_normalize", "vit_attend", "vit_attend_parallel"],
        ),
        (
            "ops.wgsl",
            with_numerics(OPS),
            &[
                "convert_f32_to_f16",
                "downsample",
                "upsample_residual",
                "post_blend",
            ],
        ),
        (
            "preprocess.wgsl",
            with_numerics(PREPROCESS),
            &["preprocess"],
        ),
    ];
    for (name, code, entries) in shared {
        by_name.insert(name, gpu.module(name, &code));
        for entry in entries {
            shared_entry.insert(*entry, name);
        }
    }
    for (name, code) in GEMMS {
        by_name.insert(name, gpu.module(name, code));
    }
    by_name.insert("window_attention", gpu.module("window_attention", WINDOW));
    gpu.check().context("compiling the shader modules")?;
    Ok(Modules {
        by_name,
        shared_entry,
    })
}

/// Run a one-off table builder over `groups` workgroups of 256.
fn build_table(
    gpu: &Gpu,
    label: &str,
    code: &str,
    inputs: &[&wgpu::Buffer],
    size: u64,
    groups: u32,
) -> Result<wgpu::Buffer> {
    let module = gpu.module(label, code);
    let pipeline = gpu
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
    let output = gpu.storage(label, size);
    let mut entries: Vec<_> = inputs
        .iter()
        .enumerate()
        .map(|(i, b)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: b.as_entire_binding(),
        })
        .collect();
    entries.push(wgpu::BindGroupEntry {
        binding: inputs.len() as u32,
        resource: output.as_entire_binding(),
    });
    let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    });
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
    gpu.queue.submit([encoder.finish()]);
    gpu.check().with_context(|| format!("building {label}"))?;
    Ok(output)
}

pub fn tables(gpu: &Gpu) -> Result<Tables> {
    // Every half input's SiLU, as a half and as its E4M3 publication; then the same as packed E4M3 bytes.
    let silu = build_table(gpu, "silu table", SILU_TABLE, &[], 65536 * 4, 256)?;
    let packed_silu = build_table(
        gpu,
        "packed silu table",
        PACKED_SILU_TABLE,
        &[&silu],
        65536,
        64,
    )?;
    let weight_metadata = gpu.storage_from("weight metadata", WEIGHT_METADATA);
    Ok(Tables {
        silu,
        packed_silu,
        weight_metadata,
    })
}

impl Network {
    /// Uploads the weights and records the graph for a `width` x `height` image.
    pub fn new(
        gpu: &Gpu,
        model: &Model,
        width: u32,
        height: u32,
        conditioning: Conditioning,
        capture_boundaries: bool,
    ) -> Result<Self> {
        let geometry = Geometry::from_valid(width, height)?;
        let start = Instant::now();
        let modules = modules(gpu, &geometry)?;
        let tables = tables(gpu)?;
        let weights = GpuModel::upload(gpu, model)?;
        let mut tensors = Tensors::new();
        let mut recorder = Recorder::new();
        let full_rows = geometry.full_width * geometry.full_height;
        let proxy = tensors.allocate(gpu, "proxy", width * height, 4, Format::F32);
        let features = tensors.allocate(gpu, "input features", full_rows, 16, Format::F32);

        // The input features, first in the frame.
        let params = [
            geometry.full_width,
            geometry.full_height,
            width,
            height,
            width,
            height,
            conditioning.seed,
            (if conditioning.auto_mask { 1.0f32 } else { -1.0 }).to_bits(),
            conditioning.local_tone.to_bits(),
            conditioning.local_structure.to_bits(),
            conditioning.skin_structure.to_bits(),
            conditioning.style.to_bits(),
        ];
        recorder.dispatch(
            PipelineKey::Shared("preprocess"),
            &[(1, &proxy.buffer), (5, &features.buffer)],
            &params,
            [
                geometry.full_width.div_ceil(8),
                geometry.full_height.div_ceil(8),
                1,
            ],
            "preprocess",
        )?;

        let mut graph = Graph::new(
            gpu,
            &weights,
            &tables,
            &mut tensors,
            &mut recorder,
            geometry.clone(),
            capture_boundaries,
        );
        let head = graph.record(&features)?;
        let boundaries = graph.boundaries.take().unwrap_or_default();
        recorder.finish(gpu, &modules)?;
        Ok(Self {
            geometry,
            dispatches: recorder.dispatch_count(),
            pipelines: recorder.pipeline_count(),
            activation_bytes: tensors.total,
            weight_bytes: weights.bytes_uploaded,
            compile_seconds: start.elapsed().as_secs_f64(),
            recorder,
            proxy,
            features,
            head_staging: gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("head read back"),
                size: head.valid_bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            head,
            boundaries,
            tensors,
        })
    }

    /// Runs the network on `image` and returns the f32 `[fullRows][4]` head.
    pub fn run(&self, gpu: &Gpu, image: &Image) -> Result<Vec<f32>> {
        if (image.width, image.height) != (self.geometry.valid_width, self.geometry.valid_height) {
            bail!(
                "the graph was recorded for {}x{}, not {}x{}",
                self.geometry.valid_width,
                self.geometry.valid_height,
                image.width,
                image.height
            );
        }
        let proxy = proxy_of(image);
        gpu.queue
            .write_buffer(&self.proxy.buffer, 0, bytemuck::cast_slice(&proxy));
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("nr frame"),
            });
        self.recorder.encode(&mut encoder);
        encoder.copy_buffer_to_buffer(
            &self.head.buffer,
            0,
            &self.head_staging,
            0,
            self.head.valid_bytes,
        );
        gpu.queue.submit([encoder.finish()]);
        let (sender, receiver) = std::sync::mpsc::channel();
        self.head_staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
        gpu.wait()?;
        receiver.recv()?.context("mapping the head")?;
        let head = bytemuck::pod_collect_to_vec(
            &self
                .head_staging
                .slice(..)
                .get_mapped_range()
                .context("reading the head")?,
        );
        self.head_staging.unmap();
        gpu.check().context("running the frame")?;
        Ok(head)
    }

    /// Runs the network and composes its output image.
    pub fn process(&self, gpu: &Gpu, image: &Image) -> Result<(Image, Timings)> {
        let start = Instant::now();
        let head = self.run(gpu, image)?;
        let frame_seconds = start.elapsed().as_secs_f64();
        let composed = compose(&head, image, &self.geometry);
        Ok((
            composed,
            Timings {
                compile_seconds: self.compile_seconds,
                frame_seconds,
                total_seconds: start.elapsed().as_secs_f64(),
            },
        ))
    }

    /// Per-dispatch GPU times of one frame on `image`.
    pub fn profile(&self, gpu: &Gpu, image: &Image) -> Result<Vec<crate::runtime::Timing>> {
        self.run(gpu, image)?;
        self.recorder.profile(gpu)
    }

    /// The f32 input features of the last frame, `[fullRows][16]`.
    pub fn read_features(&self, gpu: &Gpu) -> Result<Vec<f32>> {
        let bytes = gpu.read_back(&self.features.buffer, self.features.valid_bytes)?;
        Ok(bytemuck::pod_collect_to_vec(&bytes))
    }

    /// Any intermediate tensor of the last frame, by the label the graph allocated it under, as raw bytes
    /// with its `(rows, channels, format)`. For checking a kernel against a reference.
    pub fn read_tensor(&self, gpu: &Gpu, label: &str) -> Result<(Vec<u8>, u32, u32, Format)> {
        let tensor = self.tensors.find(label).with_context(|| {
            format!(
                "no tensor labelled {label:?}; have {}",
                self.tensors.labels().join(", ")
            )
        })?;
        let bytes = gpu.read_back(&tensor.buffer, tensor.valid_bytes)?;
        Ok((bytes, tensor.rows, tensor.channels, tensor.format))
    }

    /// Captured block outputs of the last frame (only with `capture_boundaries`).
    pub fn read_boundary(&self, gpu: &Gpu, name: &str) -> Result<Vec<u8>> {
        let (_, tensor) = self
            .boundaries
            .iter()
            .find(|(n, _)| n == name)
            .with_context(|| format!("no captured boundary {name}"))?;
        gpu.read_back(&tensor.buffer, tensor.valid_bytes)
    }

    pub fn boundary_names(&self) -> impl Iterator<Item = &str> {
        self.boundaries.iter().map(|(n, _)| n.as_str())
    }
}

/// Toward zero to the half grid, back as an f32: what the publication does, and not the same as rounding.
pub fn truncate_to_half(value: f32) -> f32 {
    let bits = value.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7fffff;
    let half = if exponent == 0xff {
        sign | if mantissa != 0 { 0x7e00 } else { 0x7c00 }
    } else {
        let e = exponent - 112;
        if e >= 31 {
            sign | 0x7c00
        } else if e <= 0 {
            if e < -10 {
                sign
            } else {
                sign | ((mantissa | 0x800000) >> (14 - e))
            }
        } else {
            sign | ((e as u32) << 10) | (mantissa >> 13)
        }
    };
    crate::weights::f16_to_f32(half as u16)
}

/// The head's residual on the centred proxy, back to a code value, truncated to the half grid: RGB f32 per
/// pixel. `inner * 8` is exact, so there is one rounding whether or not the last step is contracted.
pub fn compose_values(head: &[f32], image: &Image, geometry: &Geometry) -> Vec<f32> {
    let (width, height) = (image.width as usize, image.height as usize);
    let full_width = geometry.full_width as usize;
    let mut values = vec![0f32; width * height * 3];
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let rows_per_chunk = height.div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        for (chunk, out) in values.chunks_mut(rows_per_chunk * width * 3).enumerate() {
            scope.spawn(move || {
                for (i, value) in out.iter_mut().enumerate() {
                    let at = chunk * rows_per_chunk * width * 3 + i;
                    let (pixel, c) = (at / 3, at % 3);
                    let (y, x) = (pixel / width, pixel % width);
                    let code = image.rgb[at] as f32 / 255.0;
                    let centred = code.mul_add(0.125, -0.0625);
                    let inner = head[(y * full_width + x) * 4 + c].mul_add(0.03125, centred);
                    *value = truncate_to_half((inner * 8.0 + 0.5).clamp(0.0, 1.0));
                }
            });
        }
    });
    values
}

/// The composed image in eight bits.
pub fn compose(head: &[f32], image: &Image, geometry: &Geometry) -> Image {
    let rgb = compose_values(head, image, geometry)
        .into_iter()
        .map(|value| {
            // Evaluated in f64 and rounded once to f32, as the reference's `quantizeByte` does.
            let scaled = (value as f64 * 255.0 + 0.5) as f32;
            scaled.floor().clamp(0.0, 255.0) as u8
        })
        .collect();
    Image {
        width: image.width,
        height: image.height,
        rgb,
    }
}

/// The display proxy the network reads: sRGB code values in 0..1, RGBA f32.
pub fn proxy_of(image: &Image) -> Vec<f32> {
    image
        .rgb
        .chunks_exact(3)
        .flat_map(|p| {
            [
                p[0] as f32 / 255.0,
                p[1] as f32 / 255.0,
                p[2] as f32 / 255.0,
                1.0,
            ]
        })
        .collect()
}

/// Writes a parity fixture in the reference port's format (its README, "Fixtures"): this run's proxy as the
/// input, its head and composed image as the references. The reference's `parity` page then says whether it
/// reproduces them bit for bit.
pub fn write_fixture(
    directory: &std::path::Path,
    image: &Image,
    head: &[f32],
    geometry: &Geometry,
    conditioning: &Conditioning,
) -> Result<()> {
    std::fs::create_dir_all(directory)?;
    std::fs::write(
        directory.join("proxy.bin"),
        bytemuck::cast_slice(&proxy_of(image)),
    )?;
    std::fs::write(directory.join("head.bin"), bytemuck::cast_slice(head))?;
    let output: Vec<f32> = compose_values(head, image, geometry)
        .chunks_exact(3)
        .flat_map(|p| [p[0], p[1], p[2], 1.0])
        .collect();
    std::fs::write(directory.join("output.bin"), bytemuck::cast_slice(&output))?;
    let manifest = serde_json::json!({
        "sourceDimensions": [image.width, image.height],
        "fullDimensions": [geometry.full_width, geometry.full_height],
        "proxy": { "file": "proxy.bin", "width": image.width, "height": image.height },
        "seed": conditioning.seed,
        "autoMask": conditioning.auto_mask,
        "conditioning": {
            "localTone": conditioning.local_tone,
            "localStructure": conditioning.local_structure,
            "skinStructure": conditioning.skin_structure,
            "style": conditioning.style,
        },
        "checks": ["head", "output"],
        "referenceHead": { "file": "head.bin" },
        "nativeOutput": { "file": "output.bin", "width": image.width, "height": image.height, "dtype": "f32" },
    });
    std::fs::write(
        directory.join("manifest.json"),
        serde_json::to_string_pretty(&manifest)?,
    )?;
    Ok(())
}

pub fn load_image(path: &std::path::Path) -> Result<Image> {
    let image = image::open(path)
        .with_context(|| format!("cannot read {}", path.display()))?
        .to_rgb8();
    Ok(Image {
        width: image.width(),
        height: image.height(),
        rgb: image.into_raw(),
    })
}

pub fn save_image(path: &std::path::Path, image: &Image) -> Result<()> {
    image::RgbImage::from_raw(image.width, image.height, image.rgb.clone())
        .context("image buffer has the wrong size")?
        .save(path)
        .with_context(|| format!("cannot write {}", path.display()))
}
