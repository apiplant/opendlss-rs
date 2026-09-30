// Stand-in for the reference's src/nr_model.h: the model types reference.cpp reads, without Vulkan.
#pragma once
#include <cstdint>
#include <cstring>
#include <string>

namespace nr {

struct Tensor {
  std::string name;
  uint32_t byteLength = 0;
  const uint8_t* bytes = nullptr;
};

uint32_t packedInputIndex(uint32_t k);
uint32_t inversePackedInputIndex(uint32_t k);
uint32_t packedWeightIndex(uint32_t k, uint32_t n, uint32_t outputChannels);

inline uint16_t auxHalf(const Tensor& tensor, uint32_t byteOffset, uint32_t column) {
  const uint8_t* p = tensor.bytes + byteOffset + column * 2;
  return (uint16_t)(p[0] | (p[1] << 8));
}
inline float auxF32(const Tensor& tensor, uint32_t byteOffset) {
  float value;
  memcpy(&value, tensor.bytes + byteOffset, 4);
  return value;
}

}  // namespace nr

// fma(x, x, highSquare) in half, rounded once, as the hardware's fma.rn.f16 (and upstream's GLSL and PTX) do.
// Upstream's reference.cpp spells it roundF16((float)((double)x * x + highSquare)), which rounds twice; run.sh
// patches its copy to call this instead (the same algorithm as nr_norm_fma in tools/gen_wgsl.mjs).
#include <cmath>
inline bool isHalfTie(float value) {
  uint32_t bits;
  memcpy(&bits, &value, 4);
  uint32_t magnitude = bits & 0x7fffffffu;
  if (magnitude >= 0x38800000u) return (magnitude & 0x1fffu) == 0x1000u;
  uint32_t exponent = magnitude >> 23;
  if (exponent < 102u) return false;
  uint32_t mantissa = (magnitude & 0x7fffffu) | 0x800000u, shift = 126u - exponent;
  return (mantissa & ((1u << shift) - 1u)) == (1u << (shift - 1u));
}
inline float halfFmaOnce(float x, float highSquare, float (*roundHalf)(float)) {
  volatile float product = x * x;   // exact: x is a half
  volatile float sum = product + highSquare;
  volatile float productPart = sum - highSquare;
  volatile float addendPart = sum - productPart;
  float error = (product - productPart) + (highSquare - addendPart);
  float value = sum;
  if (error != 0.0f && isHalfTie(value)) value = std::nextafter(value, error > 0 ? INFINITY : -INFINITY);
  return roundHalf(value);
}
