// The three kernels of the reference's fast route that exist only as GLSL (shaders/ops.comp modes 2 and 4,
// shaders/preprocess.comp), in CUDA C for the CUDA backend. Compiled to ptx/ops_cuda.ptx by tools/gen_ptx.sh
// with -fmad=false, so no multiply and add is ever contracted behind the arithmetic's back.
//
// The rules the GLSL states are kept literally:
//   * every half publication is the half's own bits (cvt.rn.f16.f32), never a float(half(x)) round trip, which
//     the NVIDIA compiler removes;
//   * E4M3 is the saturating hardware conversion of those halves, with a NaN turned into +0 first;
//   * the 2x2 pool is (a + b) + (c + d), then * 0.25, every step a half.
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ uint32_t f32_bits(float value) { return __float_as_uint(value); }

// f32 -> RNE f16 -> f32 on the bit pattern (common.glsl roundF16).
__device__ float round_f16(float value) {
  uint32_t bits = f32_bits(value);
  uint32_t sign = bits & 0x80000000u;
  uint32_t magnitude = bits & 0x7fffffffu;
  if (magnitude >= 0x7f800000u) return value;
  if (magnitude >= 0x477ff000u) return __uint_as_float(sign | 0x7f800000u);
  if (magnitude < 0x38800000u) {
    float scaled = rintf(__uint_as_float(magnitude) * 16777216.0f);
    return __fadd_rn(__uint_as_float(sign), __uint_as_float(sign | f32_bits(__fmul_rn(scaled, 5.9604644775390625e-8f))));
  }
  uint32_t lsb = (magnitude >> 13) & 1u;
  return __uint_as_float(sign | ((magnitude + 0xfffu + lsb) & ~0x1fffu));
}

// E4M3FN value of a code; the NaN code reads as zero (common.glsl e4m3ToF32).
__device__ float e4m3_value(uint32_t bits) {
  bool negative = (bits & 0x80u) != 0u;
  uint32_t exponent = (bits >> 3) & 0x0fu;
  uint32_t mantissa = bits & 0x07u;
  float value;
  if (exponent == 0u) value = __fmul_rn((float)mantissa, 0.001953125f);
  else if (exponent == 15u && mantissa == 7u) value = 0.0f;
  else value = __fmul_rn(__fadd_rn(1.0f, __fmul_rn((float)mantissa, 0.125f)), __uint_as_float((exponent + 120u) << 23));
  return negative ? -value : value;
}

// Two halves (low = first) -> two E4M3 codes, low byte first: cvt.rn.satfinite.e4m3x2.f16x2 with NaN -> +0.
__device__ __forceinline__ uint32_t e4m3_pair(uint32_t halves) {
  uint32_t clean = halves;
  if ((clean & 0x7c00u) == 0x7c00u && (clean & 0x03ffu)) clean &= 0xffff0000u;
  if ((clean & 0x7c000000u) == 0x7c000000u && (clean & 0x03ff0000u)) clean &= 0x0000ffffu;
  __half2_raw raw;
  raw.x = (unsigned short)(clean & 0xffffu);
  raw.y = (unsigned short)(clean >> 16);
  return (uint32_t)__nv_cvt_halfraw2_to_fp8x2(raw, __NV_SATFINITE, __NV_E4M3);
}

__device__ __forceinline__ uint2 quantize8(uint4 halves) {
  return make_uint2(e4m3_pair(halves.x) | (e4m3_pair(halves.y) << 16), e4m3_pair(halves.z) | (e4m3_pair(halves.w) << 16));
}

__device__ __forceinline__ uint32_t half_bits(float value) { return (uint32_t)__half_as_ushort(__float2half_rn(value)); }

__device__ __forceinline__ uint32_t group_index() { return blockIdx.x * blockDim.x + threadIdx.x; }

}  // namespace

// shaders/ops.comp MODE_DOWNSAMPLE_FP8: raw f16 2x2 box pool -> E4M3, one thread per eight channels.
extern "C" __global__ void nr_downsample(const uint4* __restrict__ in16, uint2* __restrict__ out8, uint32_t count,
                                         uint32_t channels, uint32_t inWidth, uint32_t inHeight, uint32_t outWidth) {
  uint32_t group = group_index();
  if (group * 8u >= count) return;
  uint32_t groupsPerPixel = channels / 8u;
  uint32_t c = (group % groupsPerPixel) * 8u;
  uint32_t pixel = group / groupsPerPixel;
  uint32_t ox = pixel % outWidth, oy = pixel / outWidth;
  uint32_t sx = ox * 2u, sy = oy * 2u;
  uint4 result = make_uint4(0u, 0u, 0u, 0u);
  if (sx + 1u < inWidth && sy + 1u < inHeight) {
    uint4 w00 = in16[((sy * inWidth + sx) * channels + c) / 8u];
    uint4 w10 = in16[((sy * inWidth + sx + 1u) * channels + c) / 8u];
    uint4 w01 = in16[(((sy + 1u) * inWidth + sx) * channels + c) / 8u];
    uint4 w11 = in16[(((sy + 1u) * inWidth + sx + 1u) * channels + c) / 8u];
    const uint32_t a[4] = {w00.x, w00.y, w00.z, w00.w}, b[4] = {w10.x, w10.y, w10.z, w10.w};
    const uint32_t d[4] = {w01.x, w01.y, w01.z, w01.w}, e[4] = {w11.x, w11.y, w11.z, w11.w};
    uint32_t r[4];
    const __half2 quarter = __float2half2_rn(0.25f);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      __half2 top = __hadd2(*(const __half2*)&a[i], *(const __half2*)&b[i]);
      __half2 bottom = __hadd2(*(const __half2*)&d[i], *(const __half2*)&e[i]);
      __half2 value = __hmul2_rn(__hadd2(top, bottom), quarter);
      r[i] = *(uint32_t*)&value;
    }
    result = make_uint4(r[0], r[1], r[2], r[3]);
  }
  out8[group] = quantize8(result);
}

// shaders/ops.comp MODE_UPSAMPLE_RESIDUAL: E4M3 (and optionally f16) of round_f16(projection[low] + skip * scale).
extern "C" __global__ void nr_upsample_residual(const uint4* __restrict__ in16, const uint2* __restrict__ skip8,
                                                const uint16_t* __restrict__ aux16, uint2* __restrict__ out8,
                                                uint4* __restrict__ out16, uint32_t count, uint32_t channels,
                                                uint32_t inWidth, uint32_t outWidth, uint32_t auxOffset, uint32_t dual) {
  uint32_t group = group_index();
  if (group * 8u >= count) return;
  uint32_t groupsPerPixel = channels / 8u;
  uint32_t c = (group % groupsPerPixel) * 8u;
  uint32_t pixel = group / groupsPerPixel;
  uint32_t ox = pixel % outWidth, oy = pixel / outWidth;
  uint32_t source = ((oy >> 1) * inWidth + (ox >> 1)) * channels + c;
  uint4 projectedWords = in16[source / 8u];
  uint2 skipCodes = skip8[group];
  const uint32_t pw[4] = {projectedWords.x, projectedWords.y, projectedWords.z, projectedWords.w};
  uint32_t out[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    uint32_t pair = 0u;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      uint32_t k = 2u * i + j;
      float projected = __half2float(__ushort_as_half((unsigned short)((pw[i] >> (16 * j)) & 0xffffu)));
      float skip = e4m3_value(((k < 4u ? skipCodes.x : skipCodes.y) >> ((k & 3u) * 8u)) & 0xffu);
      float scale = __half2float(__ushort_as_half(aux16[auxOffset + c + k]));
      pair |= half_bits(__fadd_rn(projected, __fmul_rn(skip, scale))) << (16 * j);
    }
    out[i] = pair;
  }
  uint4 halves = make_uint4(out[0], out[1], out[2], out[3]);
  out8[group] = quantize8(halves);
  if (dual) out16[group] = halves;
}

// shaders/preprocess.comp: sixteen f32 input lanes per padded pixel from an RGBA f32 proxy.
__device__ float hash_uniform(uint32_t value) {
  uint32_t mixed = value;
  mixed = (mixed >> ((mixed >> 28) + 4u)) ^ mixed;
  mixed *= 0x108ef2d9u;
  uint32_t integer = ((mixed >> 30) ^ (mixed >> 8)) + 1u;
  return __fmul_rn((float)integer, __uint_as_float(0x33800000u));
}

// Reflect-101 (no edge repeat), repeated for padding wider than the image: upstream's 2 n - i - 2 wherever that is
// in range, and still in range for an image under ~161 pixels, whose padding (the field is at least 320) upstream's
// formula overruns (tools/gen_wgsl.mjs: repeatedMirror).
__device__ uint32_t mirror(uint32_t i, uint32_t n) {
  if (i < n) return i;
  if (n == 1u) return 0u;
  uint32_t period = 2u * (n - 1u), r = i % period;
  return r < n ? r : period - r;
}

// The GPU's approximate transcendentals, as a Vulkan driver evaluates GLSL's log2 / cos / sin / sqrt.
__device__ float approx_sqrt(float x) {
  float r;
  asm("sqrt.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

extern "C" __global__ void nr_preprocess(const float4* __restrict__ proxy, float* __restrict__ features,
                                         uint32_t fullWidth, uint32_t fullHeight, uint32_t validWidth,
                                         uint32_t validHeight, uint32_t sourceWidth, uint32_t sourceHeight,
                                         uint32_t seed, float autoMask, float localTone, float localStructure,
                                         float skinStructure, float style) {
  // One thread per padded pixel, consecutive threads on consecutive pixels, so the 64-byte feature rows are
  // written as whole float4s. The values do not depend on the mapping.
  uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= fullWidth * fullHeight) return;
  uint32_t x = index % fullWidth, y = index / fullWidth;
  uint32_t sourceX = mirror(x, validWidth), sourceY = mirror(y, validHeight);
  uint32_t imageX = ((2u * sourceX + 1u) * sourceWidth) / (2u * validWidth);
  uint32_t imageY = ((2u * sourceY + 1u) * sourceHeight) / (2u * validHeight);
  float4 rgba = proxy[imageY * sourceWidth + imageX];
  float r = round_f16(__fmul_rn(round_f16(__fsub_rn(round_f16(rgba.x), 0.5f)), 0.125f));
  float g = round_f16(__fmul_rn(round_f16(__fsub_rn(round_f16(rgba.y), 0.5f)), 0.125f));
  float b = round_f16(__fmul_rn(round_f16(__fsub_rn(round_f16(rgba.z), 0.5f)), 0.125f));

  uint32_t base = (x * 0x8da6b343u) ^ (seed * 0x9e3779b9u) ^ (y * 0xd8163841u) ^ 0x243f6a88u;
  base = (base >> ((base >> 28) + 4u)) ^ base;
  base *= 0x108ef2d9u;
  base = (base >> 22) ^ base;
  float u0 = hash_uniform(base * 0x2c9277b5u + 0xac564b05u);
  float u1 = hash_uniform(base * 0xfa6dc5f9u + 0x4712a88eu);
  float u2 = hash_uniform(base * 0xcaa5b80du + 0x21dd796bu);
  float u3 = hash_uniform(base * 0x83232c31u + 0x3463e0acu);
  float radius0 = approx_sqrt(__fmul_rn(__fmul_rn(__log2f(u0), __uint_as_float(0x3f317218u)), -2.0f));
  float radius1 = approx_sqrt(__fmul_rn(__fmul_rn(__log2f(u2), __uint_as_float(0x3f317218u)), -2.0f));
  float angle0 = __fmul_rn(u1, __uint_as_float(0x40c90fdbu));
  float angle1 = __fmul_rn(u3, __uint_as_float(0x40c90fdbu));

  float4* out = reinterpret_cast<float4*>(features) + (size_t)index * 4u;
  out[0] = make_float4(round_f16(__fmul_rn(radius0, __cosf(angle0))), round_f16(__fmul_rn(radius0, __sinf(angle0))),
                       round_f16(__fmul_rn(radius1, __cosf(angle1))), 1.0f);
  out[1] = make_float4(r, g, b, r);
  out[2] = make_float4(g, b, __fdiv_rn(style, 128.0f), round_f16(localTone));
  out[3] = make_float4(round_f16(autoMask > 0.0f ? 1.0f : localStructure),
                       round_f16(autoMask > 0.0f ? (skinStructure < 0.0f ? localStructure : skinStructure) : -1.0f),
                       round_f16(autoMask > 0.0f ? localStructure : -1.0f), 0.0f);
}

// The first `columns` f16 channels of each row of a wider tensor: the 64 -> 32 decoder transition runs on the
// PTX GEMM with its matrix padded to 64 zero columns, and keeps these.
extern "C" __global__ void nr_narrow_f16(const uint16_t* __restrict__ source, uint16_t* __restrict__ target, uint32_t rows,
                                         uint32_t sourceStride, uint32_t targetStride, uint32_t columns) {
  uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= rows * columns) return;
  uint32_t row = index / columns, column = index % columns;
  target[row * targetStride + column] = source[row * sourceStride + column];
}

// The 8-bit sRGB image -> the RGBA f32 display proxy the preprocess reads: code / 255, correctly rounded, as the
// host's network::proxy_of computes it.
extern "C" __global__ void nr_proxy_from_rgb8(const uint8_t* __restrict__ rgb, float4* __restrict__ proxy, uint32_t pixels) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= pixels) return;
  proxy[i] = make_float4(__fdiv_rn((float)rgb[3 * i], 255.0f), __fdiv_rn((float)rgb[3 * i + 1], 255.0f),
                         __fdiv_rn((float)rgb[3 * i + 2], 255.0f), 1.0f);
}

// Toward zero to the half grid, back as an f32 (network::truncate_to_half).
__device__ float truncate_to_half(float value) {
  uint32_t bits = __float_as_uint(value);
  uint32_t sign = (bits >> 16) & 0x8000u;
  int exponent = (int)((bits >> 23) & 0xffu);
  uint32_t mantissa = bits & 0x7fffffu;
  uint32_t half;
  if (exponent == 0xff) {
    half = sign | (mantissa ? 0x7e00u : 0x7c00u);
  } else {
    int e = exponent - 112;
    if (e >= 31) half = sign | 0x7c00u;
    else if (e <= 0) half = e < -10 ? sign : sign | ((mantissa | 0x800000u) >> (14 - e));
    else half = sign | ((uint32_t)e << 10) | (mantissa >> 13);
  }
  return __half2float(__ushort_as_half((unsigned short)half));
}

// network::compose on the device: the head's residual on the centred proxy, truncated to the half grid, then
// eight bits - every step the same operation as the host's (fma, f32 rounding, the f64 quantization).
extern "C" __global__ void nr_compose(const float* __restrict__ head, const uint8_t* __restrict__ rgb,
                                      uint8_t* __restrict__ out, uint32_t width, uint32_t height, uint32_t fullWidth) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= width * height * 3u) return;
  uint32_t pixel = i / 3u, c = i % 3u;
  uint32_t x = pixel % width, y = pixel / width;
  float code = __fdiv_rn((float)rgb[i], 255.0f);
  float centred = __fmaf_rn(code, 0.125f, -0.0625f);
  float inner = __fmaf_rn(head[((size_t)y * fullWidth + x) * 4u + c], 0.03125f, centred);
  float value = truncate_to_half(fminf(fmaxf(__fadd_rn(__fmul_rn(inner, 8.0f), 0.5f), 0.0f), 1.0f));
  float scaled = __double2float_rn(__dadd_rn(__dmul_rn((double)value, 255.0), 0.5));
  out[i] = (uint8_t)fminf(fmaxf(floorf(scaled), 0.0f), 255.0f);
}
