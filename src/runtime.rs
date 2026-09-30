//! The thin layer the network sits on: a device with the limits the kernels need, the two bind group
//! layouts, activation tensors, and a recorder that turns the graph into a replayable list of dispatches.
//!
//! This mirrors `ports/browser-webgpu/src/{gpu,passes}.js` of the reference. The hand-written kernels share
//! one bind group layout (a dynamic uniform at 0, four read-only buffers, three writable, one more read);
//! the FP8 GEMM has its own. Every GEMM and window-attention dispatch is its own pipeline, because shapes,
//! strides and offsets are pipeline overrides rather than uniform reads.

use anyhow::{Context, Result, bail};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

pub const BINDING_COUNT: u32 = 9;
pub const GEMM_BINDING_COUNT: u32 = 8;
/// minUniformBufferOffsetAlignment, and the largest parameter block.
pub const PARAMS_STRIDE: u64 = 256;
pub const MAX_GROUPS: u32 = 65535;
/// Shared memory the widest window-attention tile needs.
pub const WINDOW_WORKGROUP_STORAGE: u32 = 32768;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Uniform,
    Read,
    Write,
}

const KINDS: [Kind; 9] = [
    Kind::Uniform,
    Kind::Read,
    Kind::Read,
    Kind::Read,
    Kind::Read,
    Kind::Write,
    Kind::Write,
    Kind::Write,
    Kind::Read,
];
const GEMM_KINDS: [Kind; 8] = [
    Kind::Read,
    Kind::Read,
    Kind::Write,
    Kind::Read,
    Kind::Uniform,
    Kind::Read,
    Kind::Read,
    Kind::Write,
];

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_name: String,
    pub backend: String,
    pub layout: wgpu::BindGroupLayout,
    pub pipeline_layout: wgpu::PipelineLayout,
    pub gemm_layout: wgpu::BindGroupLayout,
    pub gemm_pipeline_layout: wgpu::PipelineLayout,
    errors: Arc<Mutex<Vec<String>>>,
}

fn layout_entries(kinds: &[Kind]) -> Vec<wgpu::BindGroupLayoutEntry> {
    kinds
        .iter()
        .enumerate()
        .map(|(binding, kind)| wgpu::BindGroupLayoutEntry {
            binding: binding as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: match kind {
                    Kind::Uniform => wgpu::BufferBindingType::Uniform,
                    Kind::Read => wgpu::BufferBindingType::Storage { read_only: true },
                    Kind::Write => wgpu::BufferBindingType::Storage { read_only: false },
                },
                has_dynamic_offset: *kind == Kind::Uniform,
                min_binding_size: if *kind == Kind::Uniform {
                    wgpu::BufferSize::new(PARAMS_STRIDE)
                } else {
                    None
                },
            },
            count: None,
        })
        .collect()
}

impl Gpu {
    /// A high-performance adapter with `shader-f16` and the largest buffers, workgroups and shared memory it
    /// offers. The kernels need f16 as a type, 512 invocations and 32 KiB of workgroup storage.
    pub async fn request() -> Result<Self> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .context("no WebGPU adapter is available")?;
        let info = adapter.get_info();
        let limits = adapter.limits();
        if !adapter.features().contains(wgpu::Features::SHADER_F16) {
            bail!(
                "{} does not support shader-f16, which the kernels require",
                info.name
            );
        }
        if limits.max_compute_workgroup_storage_size < WINDOW_WORKGROUP_STORAGE {
            bail!(
                "the window attention needs {} KiB of workgroup storage; {} offers {}",
                WINDOW_WORKGROUP_STORAGE / 1024,
                info.name,
                limits.max_compute_workgroup_storage_size
            );
        }
        let required_limits = wgpu::Limits {
            max_buffer_size: limits.max_buffer_size,
            max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
            max_compute_workgroup_storage_size: limits.max_compute_workgroup_storage_size,
            max_compute_invocations_per_workgroup: limits.max_compute_invocations_per_workgroup,
            max_compute_workgroup_size_x: limits.max_compute_workgroup_size_x,
            max_compute_workgroup_size_y: limits.max_compute_workgroup_size_y,
            max_storage_buffers_per_shader_stage: limits.max_storage_buffers_per_shader_stage,
            ..wgpu::Limits::default()
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("opendlss-nr"),
                // Timestamps only for `--profile`; never required.
                required_features: wgpu::Features::SHADER_F16
                    | (adapter.features() & wgpu::Features::TIMESTAMP_QUERY),
                required_limits,
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            })
            .await
            .context("cannot create the WebGPU device")?;
        // A validation error does not stop anything: the call becomes a no-op and the output stays as it was,
        // which reads exactly like a network that computes zeros. Collect them and fail loudly instead.
        let errors = Arc::new(Mutex::new(Vec::new()));
        let sink = errors.clone();
        device.on_uncaptured_error(Arc::new(move |error: wgpu::Error| {
            sink.lock().unwrap().push(error.to_string());
        }));
        let make = |label, kinds: &[Kind]| {
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries: &layout_entries(kinds),
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            (layout, pipeline_layout)
        };
        let (layout, pipeline_layout) = make("nr", &KINDS);
        let (gemm_layout, gemm_pipeline_layout) = make("gemm", &GEMM_KINDS);
        Ok(Self {
            device,
            queue,
            adapter_name: info.name,
            backend: format!("{:?}", info.backend),
            layout,
            pipeline_layout,
            gemm_layout,
            gemm_pipeline_layout,
            errors,
        })
    }

    /// Fails with every validation error raised since the last check.
    pub fn check(&self) -> Result<()> {
        let errors = std::mem::take(&mut *self.errors.lock().unwrap());
        if errors.is_empty() {
            Ok(())
        } else {
            bail!("WebGPU validation failed:\n{}", errors.join("\n"))
        }
    }

    pub fn module(&self, label: &str, code: &str) -> wgpu::ShaderModule {
        self.device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(code.into()),
            })
    }

    pub fn storage(&self, label: &str, size: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size.max(16).next_multiple_of(4),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    pub fn storage_from(&self, label: &str, data: &[u8]) -> wgpu::Buffer {
        let buffer = self.storage(label, data.len() as u64);
        let mut padded = data.to_vec();
        padded.resize(data.len().next_multiple_of(4), 0);
        self.queue.write_buffer(&buffer, 0, &padded);
        buffer
    }

    pub fn wait(&self) -> Result<()> {
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .context("waiting for the GPU")?;
        Ok(())
    }

    /// Copies `byte_length` bytes of a device buffer back to the host.
    pub fn read_back(&self, buffer: &wgpu::Buffer, byte_length: u64) -> Result<Vec<u8>> {
        let size = byte_length.next_multiple_of(4);
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("read back"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        self.queue.submit([encoder.finish()]);
        let (sender, receiver) = std::sync::mpsc::channel();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
        self.wait()?;
        receiver.recv()?.context("mapping the read-back buffer")?;
        let bytes = staging
            .slice(..)
            .get_mapped_range()
            .context("reading the mapped buffer")?[..byte_length as usize]
            .to_vec();
        staging.unmap();
        self.check()?;
        Ok(bytes)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    F32,
    F16,
    E4,
}

impl Format {
    pub fn bytes(self) -> u64 {
        match self {
            Format::F32 => 4,
            Format::F16 => 2,
            Format::E4 => 1,
        }
    }
}

/// An activation tensor: `[rows][channels]` of one format, rows padded to a multiple of 64 and zeroed,
/// because a kernel reading a window or a key block at the end of a tensor reads past the last valid row.
#[derive(Clone)]
pub struct Tensor {
    pub label: String,
    pub rows: u32,
    pub channels: u32,
    pub format: Format,
    pub buffer: wgpu::Buffer,
    pub byte_length: u64,
    pub valid_bytes: u64,
}

pub struct Tensors {
    by_key: HashMap<String, Tensor>,
    pub total: u64,
}

impl Tensors {
    pub fn new() -> Self {
        Self {
            by_key: HashMap::new(),
            total: 0,
        }
    }

    /// The tensor under `label` with this shape, allocated on first use. The graph relies on the reuse:
    /// every block of a stage shares its temporaries.
    pub fn allocate(
        &mut self,
        gpu: &Gpu,
        label: &str,
        rows: u32,
        channels: u32,
        format: Format,
    ) -> Tensor {
        let key = format!("{label}/{rows}x{channels}/{format:?}");
        if let Some(existing) = self.by_key.get(&key) {
            return existing.clone();
        }
        let alloc_rows = rows.next_multiple_of(64) as u64;
        let size = (alloc_rows * channels as u64 * format.bytes()).next_multiple_of(4);
        let buffer = gpu.storage(&key, size);
        let tensor = Tensor {
            label: label.to_string(),
            rows,
            channels,
            format,
            buffer,
            byte_length: size,
            valid_bytes: rows as u64 * channels as u64 * format.bytes(),
        };
        self.total += size;
        self.by_key.insert(key, tensor.clone());
        tensor
    }
}

impl Tensors {
    /// The tensor allocated under `label`, whatever its shape (for inspecting intermediates).
    pub fn find(&self, label: &str) -> Option<&Tensor> {
        self.by_key.values().find(|tensor| tensor.label == label)
    }

    pub fn labels(&self) -> Vec<&str> {
        let mut labels: Vec<_> = self.by_key.values().map(|t| t.label.as_str()).collect();
        labels.sort();
        labels
    }
}

impl Default for Tensors {
    fn default() -> Self {
        Self::new()
    }
}

/// What a dispatch runs: a shared hand-written entry point, or a specialized module with its overrides.
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum PipelineKey {
    Shared(&'static str),
    Specialized {
        module: &'static str,
        entry: &'static str,
        gemm: bool,
        constants: Vec<(&'static str, u32)>,
    },
}

enum Item {
    Dispatch {
        pipeline: usize,
        gemm: bool,
        buffers: Vec<Option<wgpu::Buffer>>,
        params: usize,
        groups: [u32; 3],
        label: String,
    },
    Copy {
        from: wgpu::Buffer,
        to: wgpu::Buffer,
        byte_length: u64,
    },
}

struct Ready {
    pipeline: usize,
    kernel: String,
    label: String,
    bind_group: wgpu::BindGroup,
    offset: u32,
    groups: [u32; 3],
}

enum ReadyItem {
    Dispatch(Ready),
    Copy {
        from: wgpu::Buffer,
        to: wgpu::Buffer,
        byte_length: u64,
    },
}

/// Records the graph once. `finish` compiles every distinct pipeline (in parallel: there are a few hundred),
/// packs every pass's parameters into one uniform buffer, and builds the bind groups, so a frame is one
/// command buffer with no per-frame allocation.
pub struct Recorder {
    keys: Vec<PipelineKey>,
    key_index: HashMap<PipelineKey, usize>,
    items: Vec<Item>,
    params: Vec<[u32; 64]>,
    ready: Vec<ReadyItem>,
    pipelines: Vec<wgpu::ComputePipeline>,
    params_buffer: Option<wgpu::Buffer>,
    dummy: Option<wgpu::Buffer>,
    dummy_writable: Vec<wgpu::Buffer>,
}

pub struct Modules {
    pub by_name: HashMap<&'static str, wgpu::ShaderModule>,
    /// Hand-written entry point -> module name.
    pub shared_entry: HashMap<&'static str, &'static str>,
}

impl Recorder {
    pub fn new() -> Self {
        Self {
            keys: Vec::new(),
            key_index: HashMap::new(),
            items: Vec::new(),
            params: Vec::new(),
            ready: Vec::new(),
            pipelines: Vec::new(),
            params_buffer: None,
            dummy: None,
            dummy_writable: Vec::new(),
        }
    }

    fn key(&mut self, key: PipelineKey) -> usize {
        if let Some(&index) = self.key_index.get(&key) {
            return index;
        }
        self.keys.push(key.clone());
        self.key_index.insert(key, self.keys.len() - 1);
        self.keys.len() - 1
    }

    /// Record one dispatch. `buffers` is keyed by binding index; bindings a kernel does not use get a
    /// placeholder, one per writable binding, since two writable bindings may not alias.
    pub fn dispatch(
        &mut self,
        key: PipelineKey,
        buffers: &[(u32, &wgpu::Buffer)],
        params: &[u32],
        groups: [u32; 3],
        label: impl Into<String>,
    ) -> Result<()> {
        let label = label.into();
        if groups.iter().any(|&g| g > MAX_GROUPS || g == 0) {
            bail!("dispatch {label} has an invalid grid {groups:?}");
        }
        if params.len() > 64 {
            bail!("parameter block of {label} is too large");
        }
        let gemm = matches!(key, PipelineKey::Specialized { gemm: true, .. });
        let count = if gemm {
            GEMM_BINDING_COUNT
        } else {
            BINDING_COUNT
        };
        let mut slots = vec![None; count as usize];
        for (binding, buffer) in buffers {
            slots[*binding as usize] = Some((*buffer).clone());
        }
        let mut block = [0u32; 64];
        block[..params.len()].copy_from_slice(params);
        self.params.push(block);
        let pipeline = self.key(key);
        self.items.push(Item::Dispatch {
            pipeline,
            gemm,
            buffers: slots,
            params: self.params.len() - 1,
            groups,
            label,
        });
        Ok(())
    }

    /// Copy a tensor as it stands at this point in the graph.
    pub fn copy(&mut self, from: &wgpu::Buffer, to: &wgpu::Buffer, byte_length: u64) {
        self.items.push(Item::Copy {
            from: from.clone(),
            to: to.clone(),
            byte_length,
        });
    }

    pub fn dispatch_count(&self) -> usize {
        self.ready
            .iter()
            .filter(|item| matches!(item, ReadyItem::Dispatch(_)))
            .count()
    }

    pub fn pipeline_count(&self) -> usize {
        self.pipelines.len()
    }

    pub fn finish(&mut self, gpu: &Gpu, modules: &Modules) -> Result<()> {
        // Compile every distinct pipeline, a few at a time: each is a driver compile of a large module.
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .clamp(2, 16);
        let next = std::sync::atomic::AtomicUsize::new(0);
        let results: Mutex<Vec<Option<wgpu::ComputePipeline>>> =
            Mutex::new(vec![None; self.keys.len()]);
        std::thread::scope(|scope| {
            for _ in 0..threads {
                scope.spawn(|| {
                    loop {
                        let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(key) = self.keys.get(index) else {
                            break;
                        };
                        let pipeline = compile(gpu, modules, key);
                        results.lock().unwrap()[index] = Some(pipeline);
                    }
                });
            }
        });
        self.pipelines = results
            .into_inner()
            .unwrap()
            .into_iter()
            .map(|p| p.expect("every pipeline compiled"))
            .collect();
        gpu.check().context("compiling the kernels")?;

        let words: Vec<u32> = self.params.iter().flatten().copied().collect();
        let params_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("graph parameters"),
            size: (words.len() as u64 * 4).max(PARAMS_STRIDE),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue
            .write_buffer(&params_buffer, 0, bytemuck::cast_slice(&words));
        let dummy = gpu.storage("unused", 16);
        let dummy_writable: Vec<_> = (0..4)
            .map(|i| gpu.storage(&format!("unused writable {i}"), 16))
            .collect();

        for item in std::mem::take(&mut self.items) {
            match item {
                Item::Copy {
                    from,
                    to,
                    byte_length,
                } => self.ready.push(ReadyItem::Copy {
                    from,
                    to,
                    byte_length,
                }),
                Item::Dispatch {
                    pipeline,
                    gemm,
                    buffers,
                    params,
                    groups,
                    label,
                } => {
                    let (layout, uniform, kinds): (_, usize, &[Kind]) = if gemm {
                        (&gpu.gemm_layout, 4, &GEMM_KINDS)
                    } else {
                        (&gpu.layout, 0, &KINDS)
                    };
                    let mut writable = dummy_writable.iter();
                    let placeholders: Vec<&wgpu::Buffer> = kinds
                        .iter()
                        .map(|kind| match kind {
                            Kind::Write => writable.next().unwrap(),
                            _ => &dummy,
                        })
                        .collect();
                    let entries: Vec<_> = (0..buffers.len())
                        .map(|binding| wgpu::BindGroupEntry {
                            binding: binding as u32,
                            resource: if binding == uniform {
                                wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                    buffer: &params_buffer,
                                    offset: 0,
                                    size: wgpu::BufferSize::new(PARAMS_STRIDE),
                                })
                            } else {
                                buffers[binding]
                                    .as_ref()
                                    .unwrap_or(placeholders[binding])
                                    .as_entire_binding()
                            },
                        })
                        .collect();
                    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some(&label),
                        layout,
                        entries: &entries,
                    });
                    let kernel = match &self.keys[pipeline] {
                        PipelineKey::Shared(entry) => entry.to_string(),
                        PipelineKey::Specialized { module, .. } => module.to_string(),
                    };
                    self.ready.push(ReadyItem::Dispatch(Ready {
                        pipeline,
                        kernel,
                        label: label.clone(),
                        bind_group,
                        offset: (params as u64 * PARAMS_STRIDE) as u32,
                        groups,
                    }));
                }
            }
        }
        self.params_buffer = Some(params_buffer);
        self.dummy = Some(dummy);
        self.dummy_writable = dummy_writable;
        gpu.check().context("building the bind groups")?;
        Ok(())
    }

    /// Replay the recorded graph. One compute pass between copies: WebGPU orders the dispatches in a pass and
    /// inserts the barriers between them, which is exactly the dependency the graph needs.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass: Option<wgpu::ComputePass<'_>> = None;
        for item in &self.ready {
            match item {
                ReadyItem::Copy {
                    from,
                    to,
                    byte_length,
                } => {
                    pass = None;
                    encoder.copy_buffer_to_buffer(from, 0, to, 0, *byte_length);
                }
                ReadyItem::Dispatch(ready) => {
                    let active = pass.get_or_insert_with(|| {
                        encoder
                            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                                label: Some("nr"),
                                timestamp_writes: None,
                            })
                            .forget_lifetime()
                    });
                    active.set_pipeline(&self.pipelines[ready.pipeline]);
                    active.set_bind_group(0, &ready.bind_group, &[ready.offset]);
                    let [x, y, z] = ready.groups;
                    active.dispatch_workgroups(x, y, z);
                }
            }
        }
    }
}

impl Default for Recorder {
    fn default() -> Self {
        Self::new()
    }
}

/// GPU time of one dispatch.
pub struct Timing {
    pub kernel: String,
    pub label: String,
    pub milliseconds: f64,
}

impl Recorder {
    /// Replays the graph with every dispatch in its own pass between two timestamps. The extra passes cost a
    /// little, so the sum runs slightly above a normal frame; it is for finding where the time goes.
    pub fn profile(&self, gpu: &Gpu) -> Result<Vec<Timing>> {
        if !gpu
            .device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
        {
            bail!("this adapter has no timestamp queries");
        }
        let dispatches: Vec<&Ready> = self
            .ready
            .iter()
            .filter_map(|item| match item {
                ReadyItem::Dispatch(ready) => Some(ready),
                ReadyItem::Copy { .. } => None,
            })
            .collect();
        let count = dispatches.len() as u32 * 2;
        let queries = gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("profile"),
            ty: wgpu::QueryType::Timestamp,
            count,
        });
        let resolve = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("profile resolve"),
            size: count as u64 * 8,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        for (index, ready) in dispatches.iter().enumerate() {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&ready.label),
                timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                    query_set: &queries,
                    beginning_of_pass_write_index: Some(index as u32 * 2),
                    end_of_pass_write_index: Some(index as u32 * 2 + 1),
                }),
            });
            pass.set_pipeline(&self.pipelines[ready.pipeline]);
            pass.set_bind_group(0, &ready.bind_group, &[ready.offset]);
            let [x, y, z] = ready.groups;
            pass.dispatch_workgroups(x, y, z);
        }
        encoder.resolve_query_set(&queries, 0..count, &resolve, 0);
        gpu.queue.submit([encoder.finish()]);
        let stamps: Vec<u64> =
            bytemuck::pod_collect_to_vec(&gpu.read_back(&resolve, count as u64 * 8)?);
        let period = gpu.queue.get_timestamp_period() as f64;
        Ok(dispatches
            .iter()
            .enumerate()
            .map(|(i, ready)| Timing {
                kernel: ready.kernel.clone(),
                label: ready.label.clone(),
                milliseconds: stamps[i * 2 + 1].saturating_sub(stamps[i * 2]) as f64 * period / 1e6,
            })
            .collect())
    }
}

fn compile(gpu: &Gpu, modules: &Modules, key: &PipelineKey) -> wgpu::ComputePipeline {
    let (module, entry, layout, constants): (_, _, _, Vec<(&str, f64)>) = match key {
        PipelineKey::Shared(entry) => (
            modules.shared_entry[entry],
            *entry,
            &gpu.pipeline_layout,
            Vec::new(),
        ),
        PipelineKey::Specialized {
            module,
            entry,
            gemm,
            constants,
        } => (
            *module,
            *entry,
            if *gemm {
                &gpu.gemm_pipeline_layout
            } else {
                &gpu.pipeline_layout
            },
            constants.iter().map(|(k, v)| (*k, *v as f64)).collect(),
        ),
    };
    gpu.device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(layout),
            module: &modules.by_name[module],
            entry_point: Some(entry),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &constants,
                ..Default::default()
            },
            cache: None,
        })
}
