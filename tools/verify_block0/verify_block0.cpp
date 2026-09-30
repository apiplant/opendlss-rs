// Checks every kernel of block 0, as `opendlss-nr dump-tensors` captured it, against the CPU reference of the
// OpenDLSS-NR Vulkan implementation (src/reference.cpp there). A port of that repository's src/verify.cpp to
// this crate's intermediates: each check is fed the GPU's own inputs, so a mismatch names one kernel.
//
//   tools/verify_block0/run.sh <image> [rows]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
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

enum class Format { E4, F16, F32 };

struct Tensor2D {
  std::vector<float> values;
  uint32_t channels = 0;
  const float* row(uint32_t r) const { return values.data() + (size_t)r * channels; }
};

Tensor2D load(const std::string& dir, const std::string& label, uint32_t rows, uint32_t channels, Format format) {
  std::vector<uint8_t> bytes = readFile(dir + "/" + label + ".bin");
  size_t count = (size_t)rows * channels;
  size_t width = format == Format::E4 ? 1 : format == Format::F16 ? 2 : 4;
  if (bytes.size() != count * width) throw std::runtime_error(label + ": unexpected size");
  Tensor2D t;
  t.channels = channels;
  t.values.resize(count);
  for (size_t i = 0; i < count; ++i) {
    if (format == Format::E4) {
      t.values[i] = (bytes[i] & 0x7f) == 0x7f ? 0.0f : num::e4m3ToF32(bytes[i]);
    } else if (format == Format::F16) {
      t.values[i] = num::f16ToF32((uint16_t)(bytes[2 * i] | (bytes[2 * i + 1] << 8)));
    } else {
      memcpy(&t.values[i], bytes.data() + 4 * i, 4);
    }
  }
  return t;
}

size_t g_failed = 0;

struct Stats {
  size_t count = 0, mismatches = 0, firstRow = 0, firstColumn = 0;
  float firstActual = 0, firstExpected = 0;
  void add(size_t row, size_t column, float actual, float expected) {
    ++count;
    if (actual == expected || (actual != actual && expected != expected)) return;
    if (!mismatches) { firstRow = row; firstColumn = column; firstActual = actual; firstExpected = expected; }
    ++mismatches;
  }
  void report(const char* label) const {
    if (!mismatches) printf("  %-28s equal (%zu values)\n", label, count);
    else printf("  %-28s MISMATCH %zu/%zu  first row %zu col %zu: gpu %.8g ref %.8g (gpu bits %04x ref bits %04x)\n",
                label, mismatches, count, firstRow, firstColumn, firstActual, firstExpected,
                num::f16Bits(firstActual), num::f16Bits(firstExpected));
    if (mismatches || !count) ++g_failed;
  }
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: verify_block0 <model dir> <dump dir> <field width> <field height> [rows]\n");
    return 2;
  }
  std::string modelDir = argv[1], dump = argv[2];
  uint32_t width = (uint32_t)atoi(argv[3]), height = (uint32_t)atoi(argv[4]);
  uint32_t fullRows = width * height;
  uint32_t sampleRows = std::min<uint32_t>(argc > 5 ? (uint32_t)atoi(argv[5]) : 4096, fullRows);

  std::vector<uint8_t> manifestBytes = readFile(modelDir + "/manifest.json");
  json::Value manifest = json::parse(std::string(manifestBytes.begin(), manifestBytes.end()));
  std::vector<std::vector<uint8_t>> stages;
  auto loadTensor = [&](const std::string& name) {
    nr::Tensor tensor;
    for (const json::Value& entry : manifest["tensors"].array) {
      if (entry["name"].str() != name) continue;
      std::string stageId = entry["stage"].str();
      for (const json::Value& s : manifest["stages"].array)
        if (s["id"].str() == stageId) stages.push_back(readFile(modelDir + "/model/" + s["file"].str()));
      tensor.name = name;
      tensor.bytes = stages.back().data() + (size_t)entry["stageOffset"].integer();
      tensor.byteLength = (uint32_t)entry["byteLength"].integer();
    }
    if (!tensor.bytes) throw std::runtime_error(name + " not found");
    return tensor;
  };
  stages.reserve(4);
  nr::Tensor tensor = loadTensor("block0.layer0.layer");

  // The block-0 layout (BlockLayout::pre in src/geometry.rs).
  const uint32_t expand = 0, contractWeights = 4096, inputAdapter = 8208, ffnCosSkip = 9232, qkvOffset = 9312,
                 relative = 12384, scaleOffset = 20576, projectionOffset = 20592, attnCosSkip = 21616;

  printf("checking block 0 against the CPU reference on %u of %u rows\n", sampleRows, fullRows);
  Tensor2D features = load(dump, "input features", fullRows, 16, Format::F32);
  Tensor2D adapterF16 = load(dump, "adapter f16", fullRows, 32, Format::F16);
  Tensor2D adapterE4 = load(dump, "adapter e4", fullRows, 32, Format::E4);
  {
    Stats raw, quantized;
    for (uint32_t r = 0; r < sampleRows; ++r)
      for (uint32_t n = 0; n < 32; ++n) {
        float expected = ref::gemmF16Element(tensor, inputAdapter, 16, 32, features.row(r), n);
        raw.add(r, n, adapterF16.values[(size_t)r * 32 + n], expected);
        quantized.add(r, n, adapterE4.values[(size_t)r * 32 + n], ref::fp8Domain(expected));
      }
    raw.report("input adapter f16 GEMM");
    quantized.report("input adapter quantize");
  }

  Tensor2D ffn = load(dump, "full ffn", fullRows, 128, Format::E4);
  {
    Stats s;
    ref::GemmRef gemm{&tensor, expand, 32, 128, 0, 0, true};
    for (uint32_t r = 0; r < sampleRows; ++r)
      for (uint32_t n = 0; n < 128; ++n) {
        float pre = ref::gemmFp8Element(gemm, adapterE4.row(r), n, 0.0f);
        s.add(r, n, ffn.values[(size_t)r * 128 + n], ref::fp8Domain(ref::mpCubicSilu(pre)));
      }
    s.report("FFN expand + SiLU");
  }

  Tensor2D ffnResidual = load(dump, "full ffn residual", fullRows, 32, Format::F16);
  Tensor2D ffnQuantized = load(dump, "full ffn quantized", fullRows, 32, Format::E4);
  {
    Stats raw, quantized;
    ref::GemmRef gemm{&tensor, contractWeights, 128, 32, 0, 0, true};
    for (uint32_t r = 0; r < sampleRows; ++r)
      for (uint32_t n = 0; n < 32; ++n) {
        float scale = num::f16ToF32(nr::auxHalf(tensor, ffnCosSkip, n));
        float initial = adapterF16.values[(size_t)r * 32 + n] * scale;
        float expected = ref::gemmFp8Element(gemm, ffn.row(r), n, initial);
        raw.add(r, n, ffnResidual.values[(size_t)r * 32 + n], expected);
        quantized.add(r, n, ffnQuantized.values[(size_t)r * 32 + n], ref::fp8Domain(expected));
      }
    raw.report("FFN contract (raw f16)");
    quantized.report("FFN contract (E4 dual)");
  }

  Tensor2D qkv = load(dump, "full qkv", fullRows, 96, Format::F16);
  {
    Stats s;
    ref::GemmRef gemm{&tensor, qkvOffset, 32, 96, 0, 0, true};
    for (uint32_t r = 0; r < sampleRows; ++r)
      for (uint32_t n = 0; n < 96; ++n)
        s.add(r, n, qkv.values[(size_t)r * 96 + n], ref::gemmFp8Element(gemm, ffnQuantized.row(r), n, 0.0f));
    s.report("QKV GEMM");
  }

  // The window kernel normalizes internally, so the reference normalizes our raw QKV for it.
  Tensor2D attended = load(dump, "full attended", fullRows, 32, Format::E4);
  {
    float scale = nr::auxF32(tensor, scaleOffset);
    std::vector<float> normalized((size_t)fullRows * 96);
    for (uint32_t r = 0; r < fullRows; ++r) ref::windowNormalizeRef(qkv.row(r), 0, scale, &normalized[(size_t)r * 96]);
    Stats s;
    std::vector<float> out(64 * 32);
    uint32_t windowsX = (width + 7) / 8;
    uint32_t sampleWindows = std::min((sampleRows + width * 8 - 1) / (width * 8) * windowsX,
                                      windowsX * ((height + 7) / 8));
    for (uint32_t w = 0; w < sampleWindows; ++w) {
      int windowX = (int)((w % windowsX) * 8), windowY = (int)((w / windowsX) * 8);
      ref::windowAttendRef(normalized, width, height, 32, 0, windowX, windowY, tensor, relative, out.data());
      for (uint32_t q = 0; q < 64; ++q) {
        int x = windowX + (int)(q & 7), y = windowY + (int)(q >> 3);
        if (x >= (int)width || y >= (int)height) continue;
        size_t token = (size_t)y * width + x;
        for (uint32_t c = 0; c < 32; ++c) s.add(token, c, attended.values[token * 32 + c], out[q * 32 + c]);
      }
    }
    s.report("window attention");
  }

  Tensor2D blockRaw = load(dump, "block 0 raw", fullRows, 32, Format::F16);
  Tensor2D blockOut = load(dump, "block 0 out", fullRows, 32, Format::E4);
  {
    Stats raw, quantized;
    ref::GemmRef gemm{&tensor, projectionOffset, 32, 32, 0, 0, true};
    for (uint32_t r = 0; r < sampleRows; ++r)
      for (uint32_t n = 0; n < 32; ++n) {
        float scale = num::f16ToF32(nr::auxHalf(tensor, attnCosSkip, n));
        float initial = ffnResidual.values[(size_t)r * 32 + n] * scale;
        float expected = ref::gemmFp8Element(gemm, attended.row(r), n, initial);
        raw.add(r, n, blockRaw.values[(size_t)r * 32 + n], expected);
        quantized.add(r, n, blockOut.values[(size_t)r * 32 + n], ref::fp8Domain(expected));
      }
    raw.report("projection (raw f16)");
    quantized.report("projection (E4 block-0)");
  }
  // Block 70's attention: full resolution again, but window phase 1, so the grid starts at (-4, -4) and the
  // edge windows are partial - the case block 0 never exercises.
  {
    nr::Tensor post = loadTensor("block70.layer0.layer");
    const uint32_t postRelative = 11472, postScale = 19664;
    Tensor2D postQkv = load(dump, "post qkv", fullRows, 96, Format::F16);
    Tensor2D postAttended = load(dump, "post attended", fullRows, 32, Format::E4);
    float scale = nr::auxF32(post, postScale);
    std::vector<float> normalized((size_t)fullRows * 96);
    for (uint32_t r = 0; r < fullRows; ++r)
      ref::windowNormalizeRef(postQkv.row(r), 0, scale, &normalized[(size_t)r * 96]);
    Stats s;
    std::vector<float> out(64 * 32);
    uint32_t windowsX = (width + 4 + 7) / 8;
    uint32_t sampledHeight = std::min(height, (sampleRows + width - 1) / width);
    uint32_t sampleWindows = std::min((sampledHeight + 4 + 7) / 8, (height + 4 + 7) / 8) * windowsX;
    for (uint32_t w = 0; w < sampleWindows; ++w) {
      int windowX = (int)((w % windowsX) * 8) - 4, windowY = (int)((w / windowsX) * 8) - 4;
      ref::windowAttendRef(normalized, width, height, 32, 0, windowX, windowY, post, postRelative, out.data());
      for (uint32_t q = 0; q < 64; ++q) {
        int x = windowX + (int)(q & 7), y = windowY + (int)(q >> 3);
        if (x < 0 || y < 0 || x >= (int)width || y >= (int)height) continue;
        size_t token = (size_t)y * width + x;
        for (uint32_t c = 0; c < 32; ++c) s.add(token, c, postAttended.values[token * 32 + c], out[q * 32 + c]);
      }
    }
    s.report("block 70 shifted attention");
  }
  printf("%s\n", g_failed ? "VERIFY: MISMATCH" : "VERIFY: every kernel of block 0, and block 70's shifted attention, equals the CPU reference");
  return g_failed ? 1 : 0;
}
