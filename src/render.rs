//! One call from an image to the re-rendered image: the entry point for using this crate as a library.
//!
//! The pieces underneath ([`crate::network`], [`crate::cuda`], [`crate::runtime`]) stay public for callers
//! that want to keep a network resident and run many frames; [`render`] builds one for the image's size,
//! runs a frame and tears it down.

use anyhow::{Result, bail};

use crate::model::Model;
use crate::network::{Conditioning, Image, Network};
use crate::runtime::Gpu;

/// Which GPU path runs the network.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Backend {
    /// CUDA when a driver is present (NVIDIA Ada or newer: FP8 tensor-core PTX kernels), wgpu otherwise.
    #[default]
    Auto,
    /// The CUDA driver API, loaded at run time.
    Cuda,
    /// wgpu (Vulkan, Metal or DX12): any GPU with `shader-f16`.
    Wgpu,
}

/// The re-rendered image and how it was made.
pub struct Rendered {
    pub image: Image,
    /// `"cuda"` or `"wgpu"`: the backend that actually ran.
    pub backend: &'static str,
    /// The adapter or device name.
    pub device: String,
    /// Time of the frame itself (image up, network, composition, image down), in milliseconds.
    pub frame_ms: f64,
}

/// Runs the network once on `image` and returns the same-size re-rendered image.
///
/// `model` comes from [`Model::load`] on a model directory (`manifest.json` + `model/`, as written by
/// `opendlss-nr extract-dll`).
pub fn render(model: &Model, image: &Image, conditioning: Conditioning, backend: Backend) -> Result<Rendered> {
    let use_cuda = match backend {
        Backend::Cuda => true,
        Backend::Wgpu => false,
        Backend::Auto => crate::cuda::driver::Cuda::new().is_ok(),
    };
    if use_cuda {
        let cuda = crate::cuda::driver::Cuda::new()?;
        let mut net = crate::cuda::network::CudaNetwork::new(&cuda, model, image.width, image.height, conditioning)?;
        let (composed, timings) = net.process(image)?;
        return Ok(Rendered { image: composed, backend: "cuda", device: cuda.name.clone(), frame_ms: timings.gpu_ms });
    }
    let gpu = pollster::block_on(Gpu::request())?;
    if image.rgb.len() != (image.width * image.height * 3) as usize {
        bail!("image buffer is {} bytes, expected {}x{}x3", image.rgb.len(), image.width, image.height);
    }
    let net = Network::new(&gpu, model, image.width, image.height, conditioning, false)?;
    let (composed, timings) = net.process(&gpu, image)?;
    Ok(Rendered { image: composed, backend: "wgpu", device: gpu.adapter_name.clone(), frame_ms: timings.frame_seconds * 1000.0 })
}
