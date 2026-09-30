//! Portable GPU substrate for the WebGPU reference route.
//!
//! `wgpu` maps to Vulkan on Linux, Metal on macOS, and DX12/Vulkan/GL on
//! Windows. The network-specific kernels can therefore share one f32 path on
//! every GPU; the NVIDIA route remains an optional acceleration, never a
//! requirement for creating a device or uploading a model.

use anyhow::{Context, Result, bail};
use bytemuck::{Pod, Zeroable};
use std::collections::BTreeMap;
use wgpu::util::DeviceExt;

use crate::model::Model;

#[derive(Debug, Clone)]
pub struct AdapterReport {
    pub name: String,
    pub backend: String,
    pub vendor: u32,
    pub device: u32,
    pub max_buffer_size: u64,
    pub max_storage_binding: u64,
}

pub struct PortableDevice {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub report: AdapterReport,
}

impl PortableDevice {
    /// Requests a high-performance adapter with only baseline WebGPU features.
    /// This deliberately does not require f16, FP8, CUDA, or vendor extensions.
    pub async fn request() -> Result<Self> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .context("no portable WebGPU adapter is available")?;
        let info = adapter.get_info();
        let limits = adapter.limits();
        let required_limits = wgpu::Limits {
            max_buffer_size: limits.max_buffer_size,
            max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
            ..wgpu::Limits::downlevel_defaults()
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("opendlss-nr portable device"),
                required_features: wgpu::Features::empty(),
                required_limits,
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            })
            .await
            .context("cannot create portable WebGPU device")?;
        Ok(Self {
            device,
            queue,
            report: AdapterReport {
                name: info.name,
                backend: format!("{:?}", info.backend),
                vendor: info.vendor,
                device: info.device,
                max_buffer_size: limits.max_buffer_size,
                max_storage_binding: limits.max_storage_buffer_binding_size,
            },
        })
    }

    /// Runs a real f32 GPU dispatch. It is a startup/probe kernel used to
    /// validate the generic compute route before allocating the full graph.
    pub fn self_test(&self) -> Result<()> {
        let input: Vec<f32> = (0..256).map(|v| v as f32 * 0.25).collect();
        let input_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("portable self-test input"),
                contents: bytemuck::cast_slice(&input),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("portable self-test output"),
            size: (input.len() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("portable f32 self-test"), source: wgpu::ShaderSource::Wgsl(
                "@group(0) @binding(0) var<storage, read> src: array<f32>;\n\
                 @group(0) @binding(1) var<storage, read_write> dst: array<f32>;\n\
                 @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {\n\
                   if (id.x < 256u) { dst[id.x] = src[id.x] * 2.0 + 1.0; }\n\
                 }".into()),
        });
        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("portable self-test layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("portable self-test pipeline"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("portable self-test pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("portable self-test bindings"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("portable self-test"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("portable self-test"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(4, 1, 1);
        }
        self.queue.submit([encoder.finish()]);
        Ok(())
    }

    pub fn upload_model(&self, model: &Model) -> Result<PortableModel> {
        let mut stages = BTreeMap::new();
        for stage in &model.manifest.stages {
            let data = model
                .stage_bytes
                .get(&stage.id)
                .context("model stage disappeared after validation")?;
            let buffer = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("NR stage {}", stage.id)),
                    contents: data,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });
            stages.insert(stage.id.clone(), buffer);
        }
        Ok(PortableModel {
            stages,
            tensors: model.manifest.tensors.len(),
        })
    }
}

pub struct PortableModel {
    pub stages: BTreeMap<String, wgpu::Buffer>,
    pub tensors: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct DispatchParams {
    pub words: [u32; 16],
}

pub fn require_model_fits(device: &PortableDevice, model: &Model) -> Result<()> {
    let largest = model.stage_bytes.values().map(Vec::len).max().unwrap_or(0) as u64;
    if largest > device.report.max_storage_binding {
        bail!(
            "largest model stage ({largest} bytes) exceeds this adapter's storage-binding limit ({})",
            device.report.max_storage_binding
        );
    }
    Ok(())
}
