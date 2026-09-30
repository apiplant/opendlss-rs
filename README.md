# opendlss-nr (Rust/Linux)

Rust host-side port of the model format, geometry rules, and Vulkan capability
gate used by `OpenDLSS-NR` project. It is intended to make the
project usable from Linux without a Windows build toolchain.

The original fast compute kernels require an NVIDIA Ada-or-newer Vulkan driver
with `VK_KHR_cooperative_matrix`, `VK_NV_cooperative_matrix2`,
`VK_EXT_shader_float8`, and `VK_NV_cuda_kernel_launch`. `doctor` reports this
requirement before attempting a run.

```bash
cargo run -- doctor
cargo run -- portable-doctor
cargo run -- cuda-doctor
cargo run -- portable-load --model models/nr
cargo run -- geometry --width 1920 --height 1080
cargo run -- validate-model --model /path/to/exported/nr-model
```

The model argument is a portable model directory containing `manifest.json`
and `model/` stage files, exactly as documented in the upstream project. A
Windows `nvngx_dlssnr.dll` is neither loaded nor copied: it is a PE DLL, not
that portable format, and cannot be used directly by a Linux Vulkan process.

This repository deliberately contains no NVIDIA DLLs, weights, or extraction
logic. Supply only model data you are licensed to use.

For a DLL you are authorized to use, the extractor parses only its PE resource
data; it does not load or execute the DLL:

```bash
cargo run -- extract-dll --dll /path/to/nvngx_dlssnr.dll --output models/nr
```

`portable-doctor` uses `wgpu` (Vulkan on Linux, Metal on macOS, and DX12/Vulkan
on Windows) and requires no NVIDIA extension. The CUDA/NVIDIA kernel route is
an optional acceleration path; the portable route is the required baseline.

## Running the network on an image

```bash
cargo run --release -- process -i in.png -o out.png [--seed N] [--style S] [--repeat N]
```

An 8-bit PNG/JPEG goes in; the same-size image the network re-renders comes out.
Its sRGB code values are the display proxy, and the head's RGB residual is
composed back onto it. This is a single frame, so there is no temporal history
and no HDR display transform. The rest of `--help process` is the conditioning
(tone, structure, skin, auto-mask) and `--fixture DIR`, which writes a parity
fixture in the upstream format.

There are two backends. `--backend auto`, the default, uses CUDA when it is
available and wgpu otherwise. `--profile` prints where the GPU time goes.

| backend | needs | 1920x1080 | 2048x1152 | 3840x2160 |
| --- | --- | --- | --- | --- |
| `cuda` | NVIDIA Ada or newer (FP8 tensor cores), a CUDA driver | 5.8 ms | 6.2 ms | 21 ms |
| `wgpu` | any GPU with `shader-f16`, 512-invocation workgroups, 32 KiB shared memory | | 315 ms | |

Times are end to end on an RTX 4090 (image in, network, composition, image
out).

**CUDA** (`src/cuda/`) runs the upstream's production route: its PTX kernels
(`scripts/ptx`, the ones its Vulkan build launches through
`VK_NV_cuda_kernel_launch`), launched through the CUDA driver API. The route
choices, argument lists and fusions are ported from `src/kernels.cpp` and
`src/nr_graph.cpp`. The frame is recorded once and replayed as a single CUDA
graph. `libcuda` is loaded at run time, so the crate builds and runs without
CUDA. It differs from upstream in three ways:

- **No counter chaining.** On one CUDA stream a launch starts only when the
  previous one ends; overlapping them would need programmatic dependent launch,
  which is sm_90 only. The kernels skip their waits, and the arithmetic doesn't
  depend on scheduling.
- **Upstream's GLSL-only kernels are rewritten in CUDA C** (`cuda/ops.cu`):
  the pool, the decoder merge, and the preprocess. So are the 8-bit I/O and the
  composition, so only 8-bit pixels cross the bus.
- **Two upstream GLSL GEMMs run on the PTX GEMM instead.** The 512-stage
  branch MLP runs as two launches per block of a batched `gemm2`
  (`tools/ptx/gemm2_batched.py`: upstream's generator, with grid z selecting
  the branch through three per-batch offsets). The 64→32 transition runs padded
  to 64 zero columns. Both give the same bytes; `NR_UNBATCHED_MLP=1` runs one
  launch per branch for comparison.

**wgpu** is a Rust host for the upstream WebGPU port (`ports/browser-webgpu`):
the same 71-block graph (`src/graph.rs`), the same weight layout
(`src/weights.rs`), and its WGSL kernels.

**The two backends give bit-identical heads.** That covers 512x288 up to
3840x2160, odd sizes, images under 161 pixels, noise, flat, gradient and
photographic content, and non-default seed, style and conditioning.
`tools/compare_backends.sh image... [-- process options]` checks it. The wgpu
route emulates the arithmetic in WGSL; the CUDA route runs upstream's PTX on
the tensor cores. So their agreement checks both.

### Three upstream bugs this found

Getting there took fixing three bugs that are in upstream itself. Each is fixed
here, and each fix is commented where it is applied.

1. **The window attention's cosine norm rounds twice** (upstream CPU
   reference and WebGPU port).
   - The norm sums `fma(v, v, f16(w²))` in half. The hardware, native,
     upstream's GLSL and its PTX all use a half fma, which rounds once.
   - `src/reference.cpp` writes `roundF16((float)((double)x*x + hs))`, and
     the port `f16(f32(a)*f32(b) + f32(c))`. Both round the sum to f32 first,
     which can land it exactly on a half tie that the exact sum is not on.
   - This happens in about one norm per million at 2048x1152. It flips an
     E4M3 byte of K, which spreads to 0.6% of the head.
   - The rewrite is correctly rounded: TwoSum, then a one-f32-step correction
     only when the f32 sum is itself a tie. It is tested against an exact
     oracle in `tools/check_numerics.mjs`.
   - The NVIDIA driver folds a plain TwoSum away (`(p + c) − c` → `p`,
     `s − (s − c)` → `c`), so every intermediate passes through an XOR with
     a zero the compiler can't see.
   - `tools/verify_block0/run.sh` patches its copy of the reference the same
     way.
   - How it was found: `tools/ptx/qkv_debug.py` builds upstream's attention
     kernel with every intermediate stored to memory, and
     `tools/ptx/qkv_debug_compare.py` compares them with the reference's.
     `tools/mma_check` separately confirms that the tensor-core step itself
     matches the reference model (8.4M random cases, plus that query's exact
     operands).
2. **The f16 tensor-core step publishes −0** (WebGPU port and CPU
   reference). When a sum cancels to a negative value too small for a half,
   the model gives −0, but the hardware gives +0
   (`tools/mma_check/run.sh --f16-zero`). It affects the input adapter and the
   head.
3. **The padding mirror overruns for small images** (upstream GLSL and the
   WebGPU port). The field is at least 320 pixels, so an image under about
   161 pixels needs more padding than one reflection covers. Upstream's
   `2·valid − x − 2` then wraps around as an unsigned integer: WGSL clamps
   the read, and CUDA faulted. The mirror now repeats. That is identical to
   upstream wherever its index is in range, but native's behavior at these
   sizes is unknown.

The PTX is generated by upstream's Python emitters, and the CUDA C by nvcc.
Both are committed in `ptx/`, so building needs neither:

```bash
tools/gen_ptx.sh ../OpenDLSS-NR
```

### Shaders

`shaders/` is generated. The upstream port builds its GEMM and window-attention
kernels by source-to-source transforms in JavaScript; `tools/gen_wgsl.mjs` runs
them and commits the output, so building needs no Node:

```bash
node tools/gen_wgsl.mjs ../OpenDLSS-NR/ports/browser-webgpu
```

It applies rewrites of its own, each explained where it happens. The first two
keep every value bit-identical; the last three are fixes for upstream bugs 1–3
above:

- **naga compatibility.** naga rejects `bitcast` between `u32` and
  `vec2<f16>`, and storage pointers passed as function arguments.
- **Exact half rounding.** The NVIDIA Vulkan driver folds a bare
  `f32(f16(x))` round trip, scalar or `vec4`, into nothing: the rounding
  silently never happens. The FP8 GEMM publishes its accumulator and every
  tensor-core step that way, so without this rewrite every GEMM is off by
  one half-ulp in about half its outputs. The round trips are replaced with
  the same rounding done on the bit pattern.
- **Correctly rounded norm fma** (bug 1), **+0 from the f16 step** (bug 2),
  and the **repeated mirror** (bug 3).

### Kernels of our own

`shaders/vit_attend_parallel.wgsl` is hand-written, not generated. It replaces
the port's ViT attention, which sums the softmax on one thread and runs the
value reduction on half the workgroup. It changes only which thread computes
what, not any operand or order of summation, and is twice as fast.
`NR_REFERENCE_KERNELS=1` switches back to the port's kernels, and
`tools/compare_kernels.sh <image>` checks the two give byte-identical output.

### Checking it against the reference

Two checks use the upstream repository's own references:

```bash
node tools/check_numerics.mjs          # every scalar primitive vs the Vulkan CPU reference
tools/verify_block0/run.sh in.png      # every block-0 kernel (+ block 70's shifted attention) vs src/reference.cpp
```

The first runs `web/fixtures/numerics.bin` through this crate's GPU path. It
covers f16 and E4M3 rounding, SiLU, both attention exponentials and both
tensor-core step emulations, exhaustively where the domain allows. It also
checks the native `f16()` conversion at every rounding boundary and the
GEMM's SiLU lookup tables.

The second compiles upstream `src/reference.cpp` and compares every value of
every block-0 kernel against it: the f16 adapter, FFN with SiLU, the
dual-output contract, QKV, window attention and the projection with its
seeded skip. It also covers block 70's phase-1 shifted window attention.
Both currently pass bit for bit, against the reference with its norm fma
corrected (bug 1).
