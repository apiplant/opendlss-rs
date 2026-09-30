// Recomputes one expert block (64 / 128 / 256 channels, a plain fused layout) with the CPU reference of the
// OpenDLSS-NR Vulkan implementation (src/reference.cpp there), for the windows where two routes' captured outputs
// disagree, and says which route - if either - equals the reference there.
//
//   verify_expert_block <model> <input E4 [rows][C]> <route A output> <route B output> <block> <channels>
//                       <width> <height> <shift x> <shift y>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "json.h"
#include "nr_model.h"
#include "numeric.h"
#include "reference.h"

namespace nr {
uint32_t packedInputIndex(uint32_t k) {
  uint32_t base = k & ~31u, within = k & 31u, half = within & 16u, quarter = within & 15u;
  return base + half + (quarter >> 2) * 2 + (quarter & 1u) + ((quarter & 2u) ? 8u : 0u);
}
uint32_t inversePackedInputIndex(uint32_t k) {
  uint32_t base = k & ~31u, within = k & 31u;
  return base + (within & 17u) + ((within & 2u) << 1) + ((within & 4u) << 1) + ((within & 8u) >> 2);
}
uint32_t packedWeightIndex(uint32_t k, uint32_t n, uint32_t outputChannels) {
  uint32_t kTile = k >> 5, kIn = k & 31, nTile = n >> 7, nIn = n & 127;
  uint32_t nHalf = nIn >> 6, nGroup = (nIn & 63) >> 4, nInGroup = nIn & 15;
  uint32_t lane = ((nInGroup & 7) << 2) | ((kIn & 15) >> 2);
  uint32_t byteInLane = ((nInGroup >> 3) << 3) | ((kIn >> 4) << 2) | (kIn & 3);
  return kTile * outputChannels * 32 + nTile * 4096 + nHalf * 2048 + nGroup * 512 + lane * 16 + byteInLane;
}
}  // namespace nr

namespace {
std::vector<uint8_t> readFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) throw std::runtime_error("cannot read " + path);
  std::streamsize size = file.tellg();
  file.seekg(0);
  std::vector<uint8_t> bytes((size_t)size);
  file.read(reinterpret_cast<char*>(bytes.data()), size);
  return bytes;
}
float e4(uint8_t code) { return (code & 0x7f) == 0x7f ? 0.0f : num::e4m3ToF32(code); }
}  // namespace

int main(int argc, char** argv) {
  if (argc < 11) {
    fprintf(stderr, "usage: verify_expert_block <model> <input> <route A> <route B> <block> <channels> <width> <height> <shift x> <shift y>\n");
    return 2;
  }
  const std::string modelDir = argv[1];
  const std::vector<uint8_t> input = readFile(argv[2]), routeA = readFile(argv[3]), routeB = readFile(argv[4]);
  const int block = atoi(argv[5]);
  const uint32_t C = atoi(argv[6]), width = atoi(argv[7]), height = atoi(argv[8]);
  const int shiftX = atoi(argv[9]), shiftY = atoi(argv[10]);
  const uint32_t tokens = width * height, heads = C / 32, experts = C / 32;
  if (input.size() != (size_t)tokens * C || routeA.size() != input.size() || routeB.size() != input.size())
    throw std::runtime_error("tensor sizes do not match the geometry");

  std::vector<uint8_t> manifestBytes = readFile(modelDir + "/manifest.json");
  json::Value manifest = json::parse(std::string(manifestBytes.begin(), manifestBytes.end()));
  std::vector<uint8_t> stage;
  nr::Tensor tensor;
  const std::string name = "block" + std::to_string(block) + ".layer0.layer";
  for (const json::Value& entry : manifest["tensors"].array) {
    if (entry["name"].str() != name) continue;
    for (const json::Value& s : manifest["stages"].array)
      if (s["id"].str() == entry["stage"].str()) stage = readFile(modelDir + "/model/" + s["file"].str());
    tensor.name = name;
    tensor.bytes = stage.data() + (size_t)entry["stageOffset"].integer();
    tensor.byteLength = (uint32_t)entry["byteLength"].integer();
  }
  if (!tensor.bytes) throw std::runtime_error(name + " not found");

  // fusedLayout(C) (src/nr_graph.cpp; BlockLayout::fused in src/geometry.rs).
  const uint32_t expandBytes = experts * C * 128, w2Base = expandBytes, w3Base = w2Base + experts * 128 * 32;
  const uint32_t ffnWeightBytes = expandBytes + experts * 128 * 32 + experts * 32 * C;
  const uint32_t ffnCosSkip = ffnWeightBytes + 16, qkvOffset = ffnCosSkip + C * 2 + 16;
  const uint32_t relative = qkvOffset + C * C * 3, scaleOffset = relative + heads * 8192;
  const uint32_t projection = scaleOffset + ((heads * 4 + 15) / 16) * 16, attnCosSkip = projection + C * C;

  auto windowOf = [&](uint32_t token) {
    int x = (int)(token % width), y = (int)(token / width);
    return std::make_pair((int)std::floor((x + shiftX) / 8.0) * 8 - shiftX, (int)std::floor((y + shiftY) / 8.0) * 8 - shiftY);
  };
  std::set<std::pair<int, int>> windows;
  for (uint32_t t = 0; t < tokens; ++t)
    for (uint32_t c = 0; c < C; ++c)
      if (routeA[(size_t)t * C + c] != routeB[(size_t)t * C + c]) windows.insert(windowOf(t));
  printf("block %d: the routes differ in %zu windows; recomputing them with the CPU reference\n", block, windows.size());

  // FFN and QKV + normalization for every token of those windows.
  std::vector<float> ffnQ((size_t)tokens * C, 0.0f), normalized((size_t)tokens * C * 3, 0.0f);
  std::vector<bool> done(tokens, false);
  const float* none = nullptr;
  (void)none;
  for (auto [wx, wy] : windows) {
    for (int q = 0; q < 64; ++q) {
      int x = wx + (q & 7), y = wy + (q >> 3);
      if (x < 0 || y < 0 || x >= (int)width || y >= (int)height) continue;
      uint32_t t = (uint32_t)y * width + (uint32_t)x;
      if (done[t]) continue;
      done[t] = true;
      std::vector<float> state(C), narrow(C);
      for (uint32_t c = 0; c < C; ++c) state[c] = e4(input[(size_t)t * C + c]);
      for (uint32_t e = 0; e < experts; ++e) {
        float hidden[128];
        ref::GemmRef w1{&tensor, e * C * 128, C, 128, 0, 0, true};
        for (uint32_t n = 0; n < 128; ++n) hidden[n] = ref::fp8Domain(ref::mpCubicSilu(ref::gemmFp8Element(w1, state.data(), n, 0.0f)));
        ref::GemmRef w2{&tensor, w2Base + e * 128 * 32, 128, 32, 0, 0, true};
        for (uint32_t n = 0; n < 32; ++n) narrow[e * 32 + n] = ref::fp8Domain(ref::gemmFp8Element(w2, hidden, n, 0.0f));
      }
      ref::GemmRef w3{&tensor, w3Base, C, C, 0, 0, true};
      for (uint32_t n = 0; n < C; ++n) {
        float initial = state[n] * num::f16ToF32(nr::auxHalf(tensor, ffnCosSkip, n));
        ffnQ[(size_t)t * C + n] = ref::fp8Domain(ref::gemmFp8Element(w3, narrow.data(), n, initial));
      }
      std::vector<float> qkv(C * 3);
      ref::GemmRef wq{&tensor, qkvOffset, C, C * 3, 0, 0, true};
      for (uint32_t n = 0; n < C * 3; ++n) qkv[n] = ref::gemmFp8Element(wq, &ffnQ[(size_t)t * C], n, 0.0f);
      for (uint32_t h = 0; h < heads; ++h)
        ref::windowNormalizeRef(qkv.data(), h, nr::auxF32(tensor, scaleOffset + h * 4), &normalized[(size_t)t * C * 3 + h * 96]);
    }
  }

  // Where the reference's correctly rounded 1/sqrt and the hardware's rsqrt.approx (about one f32 ulp) could
  // publish different halves: norms within a few f32 ulps of a half rounding boundary.
  if (getenv("NR_NORM_TIES")) {
    for (uint32_t t = 0; t < tokens; ++t) {
      if (!done[t]) continue;
      std::vector<float> qkv(C * 3);
      ref::GemmRef wq{&tensor, qkvOffset, C, C * 3, 0, 0, true};
      for (uint32_t n = 0; n < C * 3; ++n) qkv[n] = ref::gemmFp8Element(wq, &ffnQ[(size_t)t * C], n, 0.0f);
      for (uint32_t h = 0; h < heads; ++h) {
        for (int part = 0; part < 2; ++part) {
          const float* x = &qkv[h * 96 + part * 32];
          float r[16];
          for (int c = 0; c < 16; ++c) {
            float highSquare = ref::roundF16(x[c + 16] * x[c + 16]);
            r[c] = ref::roundF16((float)((double)x[c] * (double)x[c] + (double)highSquare));
          }
          for (int stride = 8; stride > 0; stride >>= 1)
            for (int c = 0; c < stride; ++c) r[c] = ref::roundF16(r[c] + r[c + stride]);
          float exact = 1.0f / std::sqrt(r[0]);
          float half = ref::roundF16(exact);
          // distance to the nearest half rounding boundary, in f32 ulps of `exact`
          uint32_t bits; memcpy(&bits, &exact, 4);
          uint32_t low = bits & 0x1fffu;
          int toTie = std::abs((int)low - 0x1000);
          static const char* focus = getenv("NR_NORM_TIES");
          bool focused = strchr(focus, ',') && (int)(t % width) == atoi(focus) && (int)(t / width) == atoi(strchr(focus, ',') + 1);
          if (toTie <= 4 || focused)
            printf("  token (%u, %u) head %u %s: 1/sqrt(%g) = %.9g lies %d f32 ulps from a half boundary (half %g); max |qkv| %g\n",
                   t % width, t / width, h, part ? "k" : "q", r[0], exact, toTie, half,
                   [&] { float m = 0; for (int c = 0; c < 32; ++c) m = std::max(m, std::fabs(x[c])); return m; }());
        }
      }
    }
  }

  // NR_DUMP_RAW=x,y,path: the reference's raw half QKV row [C * 3] of that token.
  if (const char* spec = getenv("NR_DUMP_RAW")) {
    int fx, fy; char path[512];
    if (sscanf(spec, "%d,%d,%511s", &fx, &fy, path) == 3) {
      uint32_t t = (uint32_t)fy * width + (uint32_t)fx;
      std::vector<float> qkv(C * 3);
      ref::GemmRef wq{&tensor, qkvOffset, C, C * 3, 0, 0, true};
      for (uint32_t n = 0; n < C * 3; ++n) qkv[n] = ref::gemmFp8Element(wq, &ffnQ[(size_t)t * C], n, 0.0f);
      FILE* f = fopen(path, "wb");
      fwrite(qkv.data(), 4, qkv.size(), f);
      fclose(f);
      printf("dumped the raw QKV of token (%d, %d)%s\n", fx, fy, done[t] ? "" : " (NOT in a recomputed window)");
    }
  }

  // NR_DUMP_OPERANDS=x,y,head,path: the reference's operands of that query's two tensor-core steps (scores and
  // P V) as floats: q[32], k[64][32] (physical key order), prior[64], weights[64], v[64][32].
  if (const char* spec = getenv("NR_DUMP_OPERANDS")) {
    int fx, fy, fh; char path[512];
    if (sscanf(spec, "%d,%d,%d,%511s", &fx, &fy, &fh, path) == 4) {
      auto [wx, wy] = windowOf((uint32_t)fy * width + (uint32_t)fx);
      int qLocal = (fx - wx) + (fy - wy) * 8;
      const uint32_t stride3 = C * 3, headBase = fh * 96;
      auto inverseTiled = [](uint32_t token) {
        uint32_t tile = token >> 4, within = token & 15;
        return ((tile >> 1) * 4 + (within >> 2)) * 8 + (tile & 1) * 4 + (within & 3);
      };
      std::vector<float> dump;
      const float* q = &normalized[((size_t)fy * width + fx) * stride3 + headBase];
      dump.insert(dump.end(), q, q + 32);
      float keys[64][32], values[64][32], prior[64], scores[64];
      for (uint32_t p = 0; p < 64; ++p) {
        uint32_t local = inverseTiled(p);
        int kx = wx + (int)(local & 7), ky = wy + (int)(local >> 3);
        for (int c = 0; c < 32; ++c) {
          keys[p][c] = normalized[((size_t)ky * width + kx) * stride3 + headBase + 32 + c];
          values[p][c] = normalized[((size_t)ky * width + kx) * stride3 + headBase + 64 + c];
        }
        // the prior of (query, key) and the reference's score -> exponential, as windowAttendRef does
        uint16_t bits = 0;
        {
          // Model::relativeBias key-physical layout, read straight from the tensor
          auto tiled = [](uint32_t token) { uint32_t x = token & 7, y = token >> 3; return (y >> 2) * 32 + (x >> 2) * 16 + (y & 3) * 4 + (x & 3); };
          uint32_t qq = tiled((uint32_t)qLocal), k = p, m = qq & 15, n = k & 15;
          uint32_t lane = ((m & 7) << 2) | ((n & 7) >> 1), fragment = (m >= 8 ? 2 : 0) + (n & 1);
          uint32_t half = (qq >> 4) * 1024 + (k >> 4) * 256 + lane * 8 + (n >> 3) * 4 + fragment;
          uint32_t at = relative + fh * 8192 + half * 2;
          bits = (uint16_t)(tensor.bytes[at] | (tensor.bytes[at + 1] << 8));
        }
        prior[p] = num::f16ToF32(bits);
        float s = ref::adaFp8Fdpa16(q, keys[p], 16, prior[p]);
        s = ref::adaFp8Fdpa16(q + 16, keys[p] + 16, 16, s);
        scores[p] = s;
      }
      for (auto& row : keys) dump.insert(dump.end(), row, row + 32);
      dump.insert(dump.end(), prior, prior + 64);
      dump.insert(dump.end(), scores, scores + 64);
      for (auto& row : values) dump.insert(dump.end(), row, row + 32);
      // the reference's softmax weights (windowAttendRef's order) and attended row
      {
        float e[64];
        for (uint32_t p = 0; p < 64; ++p) {
          float affine = ref::roundF16(std::fma(ref::roundF16(scores[p]), 0.044921875f, 1.30078125f));
          affine = std::min(std::max(affine, 1.03125f), 1.5693359375f);
          uint32_t b = num::f16Bits(affine);
          e[p] = num::f16ToF32((uint16_t)(((b << 5) + 0x8000u) & 0xffffu));
        }
        auto add = [](float a, float b) { return ref::roundF16((float)((double)a + (double)b)); };
        auto pair = [&](uint32_t pairIndex, uint32_t parity) {
          uint32_t key = pairIndex * 2 + parity;
          return add(add(add(add(e[key], e[key + 8]), add(e[key + 16], e[key + 24])), add(e[key + 32], e[key + 40])),
                     add(e[key + 48], e[key + 56]));
        };
        float even = add(add(add(pair(0, 0), pair(1, 0)), pair(2, 0)), pair(3, 0));
        float odd = add(add(add(pair(0, 1), pair(1, 1)), pair(2, 1)), pair(3, 1));
        float reciprocal = ref::roundF16(1.0f / add(even, odd));
        float w[64];
        for (int p = 0; p < 64; ++p) w[p] = ref::fp8Domain(ref::roundF16(e[p] * reciprocal));
        dump.insert(dump.end(), w, w + 64);
        for (int c = 0; c < 32; ++c) {
          float value = 0.0f;
          for (int group = 0; group < 4; ++group) {
            float vv[16];
            for (int i = 0; i < 16; ++i) vv[i] = values[group * 16 + i][c];
            value = ref::adaFp8Fdpa16(w + group * 16, vv, 16, value);
          }
          dump.push_back(value);
        }
      }
      FILE* f = fopen(path, "wb");
      fwrite(dump.data(), 4, dump.size(), f);
      fclose(f);
      printf("dumped the operands of query (%d, %d) head %d (window (%d, %d), local %d)\n", fx, fy, fh, wx, wy, qLocal);
    }
  }

  // Attention, projection, and the verdict per value.
  size_t checked = 0, equalA = 0, equalB = 0, equalNeither = 0, reported = 0;
  for (auto [wx, wy] : windows) {
    std::vector<std::vector<float>> attended(heads, std::vector<float>(64 * 32));
    for (uint32_t h = 0; h < heads; ++h)
      ref::windowAttendRef(normalized, width, height, C, h, wx, wy, tensor, relative, attended[h].data());
    for (int q = 0; q < 64; ++q) {
      int x = wx + (q & 7), y = wy + (q >> 3);
      if (x < 0 || y < 0 || x >= (int)width || y >= (int)height) continue;
      uint32_t t = (uint32_t)y * width + (uint32_t)x;
      std::vector<float> row(C);
      for (uint32_t h = 0; h < heads; ++h)
        for (int c = 0; c < 32; ++c) row[h * 32 + c] = attended[h][q * 32 + c];
      ref::GemmRef wp{&tensor, projection, C, C, 0, 0, true};
      for (uint32_t n = 0; n < C; ++n) {
        float initial = ffnQ[(size_t)t * C + n] * num::f16ToF32(nr::auxHalf(tensor, attnCosSkip, n));
        float expected = ref::fp8Domain(ref::gemmFp8Element(wp, row.data(), n, initial));
        float a = e4(routeA[(size_t)t * C + n]), b = e4(routeB[(size_t)t * C + n]);
        ++checked;
        bool matchA = a == expected, matchB = b == expected;
        equalA += matchA;
        equalB += matchB;
        if (!matchA && !matchB) {
          ++equalNeither;
          if (reported++ < 5) printf("  token (%d, %d) channel %u: reference %g, route A %g, route B %g\n", x, y, n, expected, a, b);
        }
      }
    }
  }
  // Optional: route B's own FFN publication and attention output for the block, to name the kernel.
  if (argc >= 13) {
    const std::vector<uint8_t> ffnB = readFile(argv[11]), attendedB = readFile(argv[12]);
    size_t ffnWrong = 0, ffnChecked = 0, attWrong = 0, attChecked = 0;
    for (uint32_t t = 0; t < tokens; ++t) {
      if (!done[t]) continue;
      for (uint32_t c = 0; c < C; ++c) { ++ffnChecked; ffnWrong += e4(ffnB[(size_t)t * C + c]) != ffnQ[(size_t)t * C + c]; }
    }
    for (auto [wx, wy] : windows) {
      for (uint32_t h = 0; h < heads; ++h) {
        std::vector<float> out(64 * 32);
        ref::windowAttendRef(normalized, width, height, C, h, wx, wy, tensor, relative, out.data());
        for (int q = 0; q < 64; ++q) {
          int x = wx + (q & 7), y = wy + (q >> 3);
          if (x < 0 || y < 0 || x >= (int)width || y >= (int)height) continue;
          uint32_t t = (uint32_t)y * width + (uint32_t)x;
          for (int c = 0; c < 32; ++c) {
            ++attChecked;
            float got = e4(attendedB[(size_t)t * C + h * 32 + c]);
            if (got != out[q * 32 + c]) {
              if (attWrong++ < 5) printf("  attention: token (%d, %d) head %u channel %d: reference %g, route B %g\n", x, y, h, c, out[q * 32 + c], got);
            }
          }
        }
      }
    }
    printf("route B's FFN publication: %zu of %zu differ from the reference; its attention output: %zu of %zu\n",
           ffnWrong, ffnChecked, attWrong, attChecked);
  }
  printf("%zu values in those windows: route A equals the reference on %zu, route B on %zu, neither on %zu\n",
         checked, equalA, equalB, equalNeither);
  return 0;
}
