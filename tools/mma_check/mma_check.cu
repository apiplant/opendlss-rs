// Is the reference's model of the FP8 tensor-core step (ref::adaFp8Fdpa16, src/reference.cpp upstream) what the
// hardware does? Runs mma.sync.m16n8k32.e4m3.e4m3.f16 on random operands and compares every output with two
// chained adaFp8Fdpa16 calls (k 0..15, then 16..31).
//   tools/mma_check/run.sh [tiles]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <random>
#include <vector>
#include <cuda_fp16.h>

#include "numeric.h"
#include "reference.h"

// reference.cpp's GEMM helpers link against these; this check never calls them.
namespace nr {
uint32_t packedInputIndex(uint32_t k) { return k; }
uint32_t packedWeightIndex(uint32_t k, uint32_t n, uint32_t outputChannels) { return k * outputChannels + n; }
}  // namespace nr

// One warp per 16x8x32 tile. A row-major [16][32] bytes, B column-major [8][32] bytes, C/D [16][8] halves.
__global__ void mma_tiles(const uint8_t* A, const uint8_t* B, const uint16_t* C, uint16_t* D) {
  const int tile = blockIdx.x, lane = threadIdx.x, group = lane >> 2, t = lane & 3;
  const uint8_t* a = A + tile * 512;
  const uint8_t* b = B + tile * 256;
  const uint16_t* c = C + tile * 128;
  uint32_t ar[4], br[2], cr[2], dr[2];
  for (int r = 0; r < 4; ++r) {
    uint32_t word = 0;
    for (int j = 0; j < 4; ++j) {
      int i = r * 4 + j;
      int row = (i < 4 || (i >= 8 && i < 12)) ? group : group + 8;
      int col = t * 4 + (i & 3) + (i >= 8 ? 16 : 0);
      word |= (uint32_t)a[row * 32 + col] << (8 * j);
    }
    ar[r] = word;
  }
  for (int r = 0; r < 2; ++r) {
    uint32_t word = 0;
    for (int j = 0; j < 4; ++j) {
      int i = r * 4 + j;
      int k = t * 4 + (i & 3) + (i >= 4 ? 16 : 0);
      word |= (uint32_t)b[group * 32 + k] << (8 * j);
    }
    br[r] = word;
  }
  for (int r = 0; r < 2; ++r) {
    int row = r == 0 ? group : group + 8;
    cr[r] = (uint32_t)c[row * 8 + t * 2] | ((uint32_t)c[row * 8 + t * 2 + 1] << 16);
  }
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};"
               : "=r"(dr[0]), "=r"(dr[1])
               : "r"(ar[0]), "r"(ar[1]), "r"(ar[2]), "r"(ar[3]), "r"(br[0]), "r"(br[1]), "r"(cr[0]), "r"(cr[1]));
  uint16_t* d = D + tile * 128;
  for (int r = 0; r < 2; ++r) {
    int row = r == 0 ? group : group + 8;
    d[row * 8 + t * 2] = dr[r] & 0xffff;
    d[row * 8 + t * 2 + 1] = dr[r] >> 16;
  }
}

// Operand mode: the scores of one query against 64 keys (q . k + prior, two chained 16-product halves), on the
// hardware and in the model. File layout: see NR_DUMP_OPERANDS in tools/verify_block0/verify_expert_block.cpp.
static uint8_t code_of(float value) {
  for (int c = 0; c < 256; ++c) if ((c & 0x7f) != 0x7f && num::e4m3ToF32(c) == value && !(value == 0 && c == 0x80)) return c;
  printf("%g is not an E4M3 value\n", value);
  exit(1);
}

static int operands(const char* path) {
  FILE* f = fopen(path, "rb");
  std::vector<float> in(32 + 64 * 32 + 64 + 64 + 64 * 32 + 64 + 32);
  if (!f || fread(in.data(), 4, in.size(), f) != in.size()) { printf("cannot read %s\n", path); return 2; }
  const float* q = in.data(); const float* keys = q + 32; const float* prior = keys + 64 * 32; const float* model = prior + 64;
  // 8 tiles of n = 8 keys: row 0 of A is the query, column n of B is key 8 * tile + n, C is the prior.
  const int tiles = 8;
  std::vector<uint8_t> A(tiles * 512, 0), B(tiles * 256, 0);
  std::vector<uint16_t> C(tiles * 128, 0), D(tiles * 128);
  for (int tile = 0; tile < tiles; ++tile) {
    for (int k = 0; k < 32; ++k) A[tile * 512 + k] = code_of(q[k]);
    for (int n = 0; n < 8; ++n) {
      for (int k = 0; k < 32; ++k) B[tile * 256 + n * 32 + k] = code_of(keys[(tile * 8 + n) * 32 + k]);
      C[tile * 128 + n] = num::f16Bits(prior[tile * 8 + n]);
    }
  }
  uint8_t *dA, *dB; uint16_t *dC, *dD;
  cudaMalloc(&dA, A.size()); cudaMalloc(&dB, B.size()); cudaMalloc(&dC, C.size() * 2); cudaMalloc(&dD, D.size() * 2);
  cudaMemcpy(dA, A.data(), A.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B.data(), B.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dC, C.data(), C.size() * 2, cudaMemcpyHostToDevice);
  mma_tiles<<<tiles, 32>>>(dA, dB, dC, dD);
  cudaDeviceSynchronize();
  cudaMemcpy(D.data(), dD, D.size() * 2, cudaMemcpyDeviceToHost);
  int differ = 0;
  for (int key = 0; key < 64; ++key) {
    float hardware = num::f16ToF32(D[(key / 8) * 128 + key % 8]);
    if (hardware != model[key]) {
      ++differ;
      printf("key %2d: prior %g, hardware %g, model %g\n", key, prior[key], hardware, model[key]);
    }
  }
  printf("scores: %d of 64 differ between the hardware and the model\n", differ);

  // P V: row 0 of A is the weights, column n of B is value channel n over the keys; two k32 steps, chained.
  const float* values = model + 64;
  const float* weights = values + 64 * 32;
  const float* attended = weights + 64;
  std::vector<uint16_t> acc(4 * 128, 0);
  for (int step = 0; step < 2; ++step) {
    std::vector<uint8_t> PA(4 * 512, 0), PB(4 * 256, 0);
    for (int tile = 0; tile < 4; ++tile) {
      for (int k = 0; k < 32; ++k) PA[tile * 512 + k] = code_of(weights[step * 32 + k]);
      for (int n = 0; n < 8; ++n)
        for (int k = 0; k < 32; ++k) PB[tile * 256 + n * 32 + k] = code_of(values[(step * 32 + k) * 32 + tile * 8 + n]);
    }
    cudaMemcpy(dA, PA.data(), PA.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, PB.data(), PB.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dC, acc.data(), acc.size() * 2, cudaMemcpyHostToDevice);
    mma_tiles<<<4, 32>>>(dA, dB, dC, dD);
    cudaDeviceSynchronize();
    cudaMemcpy(acc.data(), dD, acc.size() * 2, cudaMemcpyDeviceToHost);
  }
  int pvDiffer = 0;
  for (int c = 0; c < 32; ++c) {
    float hardware = num::f16ToF32(acc[(c / 8) * 128 + c % 8]);
    if (hardware != attended[c]) {
      ++pvDiffer;
      printf("channel %2d: hardware %g (E4 %g), model %g (E4 %g)\n", c, hardware, ref::fp8Domain(hardware), attended[c],
             ref::fp8Domain(attended[c]));
    }
  }
  printf("P V: %d of 32 channels differ between the hardware and the model\n", pvDiffer);
  return differ + pvDiffer != 0;
}

// The f16 step (m16n8k16, f16 accumulate) where two products cancel to a value below half the smallest subnormal:
// does the hardware publish the sign of that zero? Row 0 of A holds a0, a1; column 0 of B holds b0, b1; C = 0.
__global__ void mma_f16_one(const uint16_t* in, uint16_t* out) {
  const int lane = threadIdx.x, group = lane >> 2, t = lane & 3;
  uint32_t a[4] = {0, 0, 0, 0}, b[2] = {0, 0}, c[2] = {0, 0}, d[2];
  if (group == 0 && t == 0) { a[0] = in[0] | ((uint32_t)in[1] << 16); b[0] = in[2] | ((uint32_t)in[3] << 16); }
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};"
               : "=r"(d[0]), "=r"(d[1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(c[0]), "r"(c[1]));
  if (lane == 0) out[0] = d[0] & 0xffff;
}

static int f16_zero_sign() {
  // (1 + 2^-10)^2 2^-10 - (1 + 2^-9) 2^-10 = 2^-30, below half the smallest subnormal (2^-25): both round to zero.
  const float p = 1.0f + 1.0f / 1024, q = 1.0f + 1.0f / 512, s = 1.0f / 32;
  int failures = 0;
  for (int sign = 0; sign < 2; ++sign) {
    float a0 = (sign ? -p : p) * s, b0 = p * s, a1 = (sign ? q : -q) * s, b1 = s;
    uint16_t in[4] = {num::f16Bits(a0), num::f16Bits(a1), num::f16Bits(b0), num::f16Bits(b1)}, *dIn, *dOut, hw;
    cudaMalloc(&dIn, 8); cudaMalloc(&dOut, 2);
    cudaMemcpy(dIn, in, 8, cudaMemcpyHostToDevice);
    mma_f16_one<<<1, 32>>>(dIn, dOut);
    cudaMemcpy(&hw, dOut, 2, cudaMemcpyDeviceToHost);
    float a[8] = {a0, a1}, b[8] = {b0, b1};
    uint16_t model = num::f16Bits(ref::adaF16Fdpa8(a, b, 8, 0.0f));
    printf("exact sum %s2^-30: hardware %04x, model %04x%s\n", sign ? "-" : "+", hw, model, hw == model ? "" : "  <- differ");
    failures += hw != model;
  }
  return failures;
}

int main(int argc, char** argv) {
  if (argc > 1 && !strcmp(argv[1], "--f16-zero")) return f16_zero_sign();
  if (argc > 2 && !strcmp(argv[1], "--operands")) return operands(argv[2]);
  const int tiles = argc > 1 ? atoi(argv[1]) : 65536;
  std::mt19937 rng(12345);
  auto e4 = [&]() { uint8_t v; do { v = rng() & 0xff; } while ((v & 0x7f) == 0x7f); return v; };
  std::vector<uint8_t> A(tiles * 512), B(tiles * 256);
  std::vector<uint16_t> C(tiles * 128), D(tiles * 128);
  for (auto& v : A) v = e4();
  for (auto& v : B) v = e4();
  // Accumulators spread over the whole finite half range, zero included, so every alignment case occurs.
  for (auto& v : C) { uint16_t h; do { h = rng() & 0xffff; } while ((h & 0x7c00) == 0x7c00); v = (rng() % 8 == 0) ? 0 : h; }
  uint8_t *dA, *dB; uint16_t *dC, *dD;
  cudaMalloc(&dA, A.size()); cudaMalloc(&dB, B.size()); cudaMalloc(&dC, C.size() * 2); cudaMalloc(&dD, D.size() * 2);
  cudaMemcpy(dA, A.data(), A.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B.data(), B.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dC, C.data(), C.size() * 2, cudaMemcpyHostToDevice);
  mma_tiles<<<tiles, 32>>>(dA, dB, dC, dD);
  if (cudaDeviceSynchronize() != cudaSuccess) { printf("kernel failed\n"); return 1; }
  cudaMemcpy(D.data(), dD, D.size() * 2, cudaMemcpyDeviceToHost);

  size_t total = 0, mismatches = 0, nanPairs = 0;
  for (int tile = 0; tile < tiles; ++tile) {
    for (int m = 0; m < 16; ++m) {
      for (int n = 0; n < 8; ++n) {
        float a[32], b[32];
        for (int k = 0; k < 32; ++k) {
          a[k] = num::e4m3ToF32(A[tile * 512 + m * 32 + k]);
          b[k] = num::e4m3ToF32(B[tile * 256 + n * 32 + k]);
        }
        float acc = num::f16ToF32(C[tile * 128 + m * 8 + n]);
        float expected = ref::adaFp8Fdpa16(a, b, 16, acc);
        expected = ref::adaFp8Fdpa16(a + 16, b + 16, 16, expected);
        uint16_t want = num::f16Bits(expected), got = D[tile * 128 + m * 8 + n];
        ++total;
        if (want == got) continue;
        if ((want & 0x7fff) > 0x7c00 && (got & 0x7fff) > 0x7c00) { ++nanPairs; continue; }
        if (mismatches++ < 10)
          printf("tile %d (%d, %d): C %04x -> hardware %04x (%g), model %04x (%g)\n", tile, m, n,
                 C[tile * 128 + m * 8 + n], got, num::f16ToF32(got), want, expected);
      }
    }
  }
  printf("%zu outputs: %zu differ from the model (%.6f%%), %zu NaN pairs\n", total, mismatches,
         100.0 * mismatches / total, nanPairs);
  return mismatches != 0;
}
