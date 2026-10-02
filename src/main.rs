use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use opendlss_nr::{
    config, cuda, dll,
    geometry::Geometry,
    model::Model,
    network::{self, Conditioning, Network},
    portable,
    runtime::Gpu,
    selftest, vulkan,
};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "opendlss", version, about = "Host tools for OpenDLSS-NR: run the network on images, validate models")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    /// Inspect Vulkan devices and the NVIDIA extension gate.
    Doctor,
    /// Create a cross-vendor WebGPU device and execute a small f32 compute probe.
    PortableDoctor,
    /// Probe the optional CUDA driver without requiring CUDA at build time.
    CudaDoctor,
    /// Upload a validated model to the portable GPU backend (no NVIDIA features required).
    PortableLoad {
        /// Model directory (default: the one `opendlss setup` installed in the config directory).
        #[arg(long)]
        model: Option<PathBuf>,
    },
    /// Print the native-compatible padded field geometry.
    Geometry {
        #[arg(long)]
        width: u32,
        #[arg(long)]
        height: u32,
    },
    /// Validate a portable manifest and all staged model bytes.
    ValidateModel {
        /// Model directory (default: the one `opendlss setup` installed in the config directory).
        #[arg(long)]
        model: Option<PathBuf>,
        #[arg(long)]
        skip_hashes: bool,
    },
    /// Run the network on an image: an 8-bit PNG/JPEG in, the re-rendered image out (same size).
    Process {
        /// Model directory (default: the one `opendlss setup` installed in the config directory).
        #[arg(long)]
        model: Option<PathBuf>,
        #[arg(long, short)]
        input: PathBuf,
        #[arg(long, short)]
        output: PathBuf,
        /// `cuda` (NVIDIA Ada or newer: the FP8 tensor-core PTX kernels), `wgpu` (any GPU with shader-f16), or
        /// `auto`: CUDA when it is available, wgpu otherwise.
        #[arg(long, default_value = "auto")]
        backend: String,
        /// Seed of the three Gaussian noise lanes.
        #[arg(long, default_value_t = 0)]
        seed: u32,
        /// Style id (0 = none).
        #[arg(long, default_value_t = 0.0)]
        style: f32,
        #[arg(long, default_value_t = 1.0)]
        local_tone: f32,
        #[arg(long, default_value_t = 1.0)]
        local_structure: f32,
        /// Skin structure; negative follows the local structure.
        #[arg(long, default_value_t = -1.0, allow_hyphen_values = true)]
        skin_structure: f32,
        #[arg(long)]
        auto_mask: bool,
        /// Run the frame this many times and report the fastest.
        #[arg(long, default_value_t = 1)]
        repeat: u32,
        /// Also write the f32 head (RGB residual and blend logit) as raw little-endian floats.
        #[arg(long)]
        head: Option<PathBuf>,
        /// Print where the GPU time of a frame goes, by kernel and by the slowest dispatches.
        #[arg(long)]
        profile: bool,
        /// Write a parity fixture (proxy, head, composed image) for the reference port's `parity` page.
        #[arg(long)]
        fixture: Option<PathBuf>,
    },
    /// Run one frame and write intermediate tensors (raw bytes, `<label>.bin`) for checking against a reference.
    #[command(hide = true)]
    DumpTensors {
        /// Model directory (default: the one `opendlss setup` installed in the config directory).
        #[arg(long)]
        model: Option<PathBuf>,
        #[arg(long, short)]
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// Tensor labels, comma separated.
        #[arg(long, value_delimiter = ',')]
        tensors: Vec<String>,
    },
    /// The CUDA backend's `dump-tensors`.
    #[command(hide = true)]
    CudaDumpTensors {
        /// Model directory (default: the one `opendlss setup` installed in the config directory).
        #[arg(long)]
        model: Option<PathBuf>,
        #[arg(long, short)]
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, value_delimiter = ',')]
        tensors: Vec<String>,
    },
    /// Run one (window, head) item of a block's fused QKV + attention through an instrumented kernel build
    /// (tools/ptx/qkv_debug.py) and write its debug words.
    #[command(hide = true)]
    CudaDebugQkv {
        /// Model directory (default: the one `opendlss setup` installed in the config directory).
        #[arg(long)]
        model: Option<PathBuf>,
        #[arg(long, short)]
        input: PathBuf,
        #[arg(long)]
        ptx: PathBuf,
        #[arg(long)]
        block: u32,
        #[arg(long)]
        channels: u32,
        /// The block's level (0..5) and window phase.
        #[arg(long)]
        level: usize,
        #[arg(long)]
        phase: u32,
        #[arg(long)]
        item: u32,
        #[arg(long)]
        out: PathBuf,
    },
    /// Evaluate the arithmetic self-test cases prepared by tools/check_numerics.mjs.
    #[command(hide = true)]
    NumericsSelftest {
        #[arg(long)]
        dir: PathBuf,
    },
    /// One-time setup: asks for your `nvngx_dlssnr.dll`, extracts the model from it (the DLL is only read as
    /// data, never loaded) and installs it in the config directory, where every other command finds it.
    Setup {
        /// Path to `nvngx_dlssnr.dll`. Asked for when omitted.
        #[arg(long)]
        dll: Option<PathBuf>,
        /// Replace a model that is already set up.
        #[arg(long)]
        force: bool,
    },
    /// Extract a portable model directory without loading or executing the DLL.
    ExtractDll {
        #[arg(long)]
        dll: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Doctor => {
            let devices = vulkan::inspect_devices()?;
            if devices.is_empty() {
                bail!("no Vulkan physical devices found");
            }
            for device in devices {
                println!(
                    "{} (vendor {:#06x}, device {:#06x}; {} compute queues)",
                    device.name, device.vendor_id, device.device_id, device.compute_queues
                );
                if device.supported() {
                    println!("  compatible with the OpenDLSS-NR Vulkan kernel requirements");
                } else {
                    println!(
                        "  unsupported: missing {}",
                        device.missing_extensions.join(", ")
                    );
                }
            }
        }
        Command::PortableDoctor => {
            let device = pollster::block_on(portable::PortableDevice::request())?;
            device.self_test()?;
            println!(
                "{} via {} (vendor {:#06x}, device {:#06x})",
                device.report.name,
                device.report.backend,
                device.report.vendor,
                device.report.device
            );
            println!(
                "portable f32 compute submitted; max buffer {} MiB, max storage binding {} MiB",
                device.report.max_buffer_size / 1048576,
                device.report.max_storage_binding / 1048576
            );
        }
        Command::CudaDoctor => match cuda::probe()? {
            cuda::CudaAvailability::Available {
                devices,
                driver_version,
            } => println!("CUDA driver {driver_version}, {devices} device(s) available"),
            cuda::CudaAvailability::Unavailable(reason) => println!("CUDA unavailable: {reason}"),
        },
        Command::PortableLoad { model: model_path } => {
            let model = Model::load(resolve_model(model_path)?, true)?;
            let device = pollster::block_on(portable::PortableDevice::request())?;
            portable::require_model_fits(&device, &model)?;
            let uploaded = device.upload_model(&model)?;
            println!(
                "uploaded {} tensors in {} stage buffers to {}",
                uploaded.tensors,
                uploaded.stages.len(),
                device.report.name
            );
        }
        Command::Geometry { width, height } => {
            let geometry = Geometry::from_valid(width, height)?;
            println!(
                "valid: {}x{}\nfull: {}x{}",
                geometry.valid_width,
                geometry.valid_height,
                geometry.full_width,
                geometry.full_height
            );
            for (index, level) in geometry.levels.iter().enumerate() {
                println!("level {index}: {}x{}", level.width, level.height);
            }
            println!(
                "ViT tokens: {} (padded {})",
                geometry.vit_tokens(),
                geometry.padded_vit_tokens()
            );
        }
        Command::ValidateModel { model, skip_hashes } => {
            let model = Model::load(resolve_model(model)?, !skip_hashes)?;
            println!(
                "valid: {} blocks, {} stages, {} tensors",
                model.manifest.totals.block_count,
                model.manifest.stages.len(),
                model.manifest.tensors.len()
            );
        }
        Command::Process {
            model,
            input,
            output,
            seed,
            style,
            local_tone,
            local_structure,
            skin_structure,
            auto_mask,
            repeat,
            head,
            fixture,
            profile,
            backend,
        } => {
            let image = network::load_image(&input)?;
            let model = Model::load(resolve_model(model)?, false)?;
            let conditioning = Conditioning {
                seed,
                auto_mask,
                local_tone,
                local_structure,
                skin_structure,
                style,
            };
            let use_cuda = match backend.as_str() {
                "cuda" => true,
                "wgpu" => false,
                "auto" => match opendlss_nr::cuda::driver::Cuda::new() {
                    Ok(_) => true,
                    Err(error) => {
                        println!("CUDA unavailable ({error}); using wgpu");
                        false
                    }
                },
                other => bail!("unknown backend {other}; expected auto, cuda or wgpu"),
            };
            if use_cuda {
                return process_cuda(&model, &image, conditioning, &output, repeat, head, profile);
            }
            let gpu = pollster::block_on(Gpu::request())?;
            println!("{} via {}", gpu.adapter_name, gpu.backend);
            let net = Network::new(&gpu, &model, image.width, image.height, conditioning, false)?;
            println!(
                "{}x{} (field {}x{}): {} dispatches, {} pipelines, {} MiB activations, {} MiB weights, ready in {:.1} s",
                image.width,
                image.height,
                net.geometry.full_width,
                net.geometry.full_height,
                net.dispatches,
                net.pipelines,
                net.activation_bytes >> 20,
                net.weight_bytes >> 20,
                net.compile_seconds
            );
            let (mut best, mut best_total) = (f64::INFINITY, f64::INFINITY);
            let mut result = None;
            for _ in 0..repeat.max(1) {
                let (composed, timings) = net.process(&gpu, &image)?;
                best = best.min(timings.frame_seconds);
                best_total = best_total.min(timings.total_seconds);
                result = Some(composed);
            }
            println!(
                "frame: {:.1} ms, {:.1} ms with the output composed (best of {})",
                best * 1000.0,
                best_total * 1000.0,
                repeat.max(1)
            );
            if profile {
                let timings = net.profile(&gpu, &image)?;
                let total: f64 = timings.iter().map(|t| t.milliseconds).sum();
                let mut by_kernel: Vec<(String, f64, usize)> = Vec::new();
                for t in &timings {
                    match by_kernel.iter_mut().find(|(k, _, _)| *k == t.kernel) {
                        Some(entry) => {
                            entry.1 += t.milliseconds;
                            entry.2 += 1;
                        }
                        None => by_kernel.push((t.kernel.clone(), t.milliseconds, 1)),
                    }
                }
                by_kernel.sort_by(|a, b| b.1.total_cmp(&a.1));
                println!("GPU time {total:.1} ms over {} dispatches:", timings.len());
                for (kernel, ms, n) in by_kernel {
                    println!(
                        "  {kernel:24} {ms:8.2} ms  {:5.1}%  ({n} dispatches)",
                        ms / total * 100.0
                    );
                }
                let mut slowest: Vec<_> = timings.iter().collect();
                slowest.sort_by(|a, b| b.milliseconds.total_cmp(&a.milliseconds));
                println!("slowest dispatches:");
                for t in slowest.iter().take(
                    std::env::var("NR_PROFILE_TOP")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(15),
                ) {
                    println!("  {:7.2} ms  {:20} {}", t.milliseconds, t.kernel, t.label);
                }
            }
            network::save_image(&output, &result.unwrap())?;
            if head.is_some() || fixture.is_some() {
                let values = net.run(&gpu, &image)?;
                if let Some(path) = head {
                    std::fs::write(&path, bytemuck::cast_slice(&values))?;
                }
                if let Some(directory) = fixture {
                    network::write_fixture(
                        &directory,
                        &image,
                        &values,
                        &net.geometry,
                        &conditioning,
                    )?;
                    println!("wrote fixture {}", directory.display());
                }
            }
            println!("wrote {}", output.display());
        }
        Command::DumpTensors {
            model,
            input,
            out,
            tensors,
        } => {
            let image = network::load_image(&input)?;
            let model = Model::load(resolve_model(model)?, false)?;
            let gpu = pollster::block_on(Gpu::request())?;
            let net = Network::new(
                &gpu,
                &model,
                image.width,
                image.height,
                Conditioning::default(),
                true,
            )?;
            net.run(&gpu, &image)?;
            std::fs::create_dir_all(&out)?;
            for label in tensors {
                let (bytes, rows, channels, format) = match label.strip_prefix("boundary ") {
                    Some(name) => (
                        net.read_boundary(&gpu, name)?,
                        0,
                        0,
                        opendlss_nr::runtime::Format::E4,
                    ),
                    None => net.read_tensor(&gpu, &label)?,
                };
                std::fs::write(out.join(format!("{label}.bin")), &bytes)?;
                println!("{label}: {rows}x{channels} {format:?}");
            }
            println!(
                "field {}x{}",
                net.geometry.full_width, net.geometry.full_height
            );
        }
        Command::CudaDumpTensors {
            model,
            input,
            out,
            tensors,
        } => {
            let image = network::load_image(&input)?;
            let model = Model::load(resolve_model(model)?, false)?;
            let cuda = opendlss_nr::cuda::driver::Cuda::new()?;
            let mut net = opendlss_nr::cuda::network::CudaNetwork::with_captures(
                &cuda,
                &model,
                image.width,
                image.height,
                Conditioning::default(),
                true,
            )?;
            net.run(&image)?;
            std::fs::create_dir_all(&out)?;
            for label in tensors {
                std::fs::write(out.join(format!("{label}.bin")), net.read_tensor(&label)?)?;
            }
        }
        Command::CudaDebugQkv {
            model,
            input,
            ptx,
            block,
            channels,
            level,
            phase,
            item,
            out,
        } => {
            let image = network::load_image(&input)?;
            let model = Model::load(resolve_model(model)?, false)?;
            let cuda = opendlss_nr::cuda::driver::Cuda::new()?;
            let mut net = opendlss_nr::cuda::network::CudaNetwork::with_captures(
                &cuda,
                &model,
                image.width,
                image.height,
                Conditioning::default(),
                true,
            )?;
            net.run(&image)?;
            let text = std::fs::read_to_string(&ptx)?;
            let entry = text
                .lines()
                .find_map(|l| l.strip_prefix(".visible .entry "))
                .and_then(|l| l.split('(').next())
                .context("no entry point in the PTX")?
                .to_string();
            let level = net.geometry.levels[level];
            let words = net.debug_qkv(block, channels, level, phase, &text, &entry, item)?;
            std::fs::write(&out, bytemuck::cast_slice(&words))?;
            println!(
                "wrote {} debug words for item {item} of block {block}",
                words.len()
            );
        }
        Command::NumericsSelftest { dir } => {
            let gpu = pollster::block_on(Gpu::request())?;
            selftest::run(&gpu, &dir)?;
        }
        Command::Setup { dll: dll_path, force } => setup(dll_path, force)?,
        Command::ExtractDll {
            dll: source,
            output,
        } => {
            let report = dll::extract_model(source, output)?;
            println!(
                "extracted {} tensors ({} packed bytes)",
                report.records, report.packed_bytes
            );
        }
    }
    Ok(())
}

/// The model directory to use: `--model`, else the one `opendlss setup` installed.
fn resolve_model(explicit: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(dir) = explicit {
        return Ok(dir);
    }
    let dir = config::model_dir().context("no config directory available (set $HOME or $XDG_CONFIG_HOME)")?;
    if !dir.join("manifest.json").is_file() {
        bail!("no model set up yet: run `opendlss setup`, or pass --model DIR");
    }
    Ok(dir)
}

fn setup(dll_path: Option<PathBuf>, force: bool) -> Result<()> {
    use std::io::{BufRead, IsTerminal, Write};
    let target = config::model_dir().context("no config directory available (set $HOME or $XDG_CONFIG_HOME)")?;
    if target.join("manifest.json").is_file() && !force {
        println!("A model is already set up in {}.\nRun `opendlss setup --force` to replace it.", target.display());
        return Ok(());
    }
    let dll_path = match dll_path {
        Some(path) => path,
        None => {
            if !std::io::stdin().is_terminal() {
                bail!("pass --dll PATH (stdin is not a terminal, so there is nobody to ask)");
            }
            println!("OpenDLSS-NR needs the network weights from your own copy of nvngx_dlssnr.dll.");
            println!("The DLL is only read as data (never loaded or run) and is not copied.\n");
            print!("Path to nvngx_dlssnr.dll: ");
            std::io::stdout().flush()?;
            let mut line = String::new();
            std::io::stdin().lock().read_line(&mut line)?;
            config::expand_path(line.trim().trim_matches(|c| c == '"' || c == '\''))
        }
    };
    if !dll_path.is_file() {
        bail!("{} is not a file", dll_path.display());
    }
    if target.exists() {
        std::fs::remove_dir_all(&target).with_context(|| format!("removing {}", target.display()))?;
    }
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let report = dll::extract_model(&dll_path, &target)?;
    // Check what was written the way every later command will read it, hashes included.
    let model = Model::load(&target, true).inspect_err(|_| {
        let _ = std::fs::remove_dir_all(&target);
    })?;
    println!(
        "Extracted {} tensors ({} MiB) to {}\nVerified {} stages. Try: opendlss process -i in.png -o out.png",
        report.records,
        report.packed_bytes >> 20,
        target.display(),
        model.manifest.stages.len()
    );
    Ok(())
}

fn process_cuda(
    model: &Model,
    image: &network::Image,
    conditioning: Conditioning,
    output: &std::path::Path,
    repeat: u32,
    head: Option<PathBuf>,
    profile: bool,
) -> Result<()> {
    let cuda = opendlss_nr::cuda::driver::Cuda::new()?;
    println!("{} via CUDA ({} SMs)", cuda.name, cuda.sm_count);
    let mut net = opendlss_nr::cuda::network::CudaNetwork::new(
        &cuda,
        model,
        image.width,
        image.height,
        conditioning,
    )?;
    println!(
        "{}x{} (field {}x{}): {} launches in one CUDA graph, {} PTX modules, {} MiB activations, {} MiB weights, ready in {:.1} s",
        image.width,
        image.height,
        net.geometry.full_width,
        net.geometry.full_height,
        net.launches,
        net.modules,
        net.activation_bytes >> 20,
        net.weight_bytes >> 20,
        net.setup_seconds
    );
    let (mut best_gpu, mut best_frame) = (f64::INFINITY, f64::INFINITY);
    let mut result = None;
    for _ in 0..repeat.max(1) {
        let (composed, timings) = net.process(image)?;
        best_gpu = best_gpu.min(timings.gpu_ms);
        best_frame = best_frame.min(timings.frame_ms);
        result = Some(composed);
    }
    println!(
        "frame: {best_gpu:.2} ms on the GPU (image up, network, composition, image down), {best_frame:.2} ms end to end (best of {})",
        repeat.max(1)
    );
    if profile {
        let timings = net.profile()?;
        let total: f64 = timings.iter().map(|t| t.2).sum();
        let mut by_kernel: Vec<(String, f64, usize)> = Vec::new();
        for (file, _, ms) in &timings {
            let family = file
                .split("_e4m3")
                .next()
                .unwrap_or(file)
                .trim_end_matches(".ptx")
                .to_string();
            match by_kernel.iter_mut().find(|(k, _, _)| *k == family) {
                Some(entry) => {
                    entry.1 += ms;
                    entry.2 += 1;
                }
                None => by_kernel.push((family, *ms, 1)),
            }
        }
        by_kernel.sort_by(|a, b| b.1.total_cmp(&a.1));
        println!(
            "GPU time {total:.2} ms over {} launches, each timed alone:",
            timings.len()
        );
        for (kernel, ms, n) in by_kernel {
            println!(
                "  {kernel:24} {ms:8.3} ms  {:5.1}%  ({n} launches)",
                ms / total * 100.0
            );
        }
        let mut slowest: Vec<_> = timings.iter().collect();
        slowest.sort_by(|a, b| b.2.total_cmp(&a.2));
        println!("slowest launches:");
        for (file, label, ms) in slowest.iter().take(12) {
            println!(
                "  {ms:7.3} ms  {:36} {label}",
                file.trim_end_matches(".ptx")
            );
        }
    }
    network::save_image(output, &result.unwrap())?;
    if let Some(path) = head {
        let (values, _) = net.run(image)?;
        std::fs::write(&path, bytemuck::cast_slice(&values))?;
    }
    println!("wrote {}", output.display());
    Ok(())
}
