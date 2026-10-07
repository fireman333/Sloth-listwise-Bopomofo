// Personalized R weights (personal-finetune design D9): the merged and
// per-channel int8-quantized linear layers of a fine-tuned R model, bound to
// the factory model they were trained from by its SHA-256.
//
// File `weights.bin` (little-endian):
//   magic "AFMRPW01"
//   32 bytes  SHA-256 of the factory model_quantized.onnx
//   uint32    tensor count (84 for R)
//   per tensor:
//     uint32 name length, name bytes (ONNX initializer base name, e.g. val_22;
//            the IME overrides <name>_quantized and <name>_scale)
//     uint32 in, uint32 out          (MatMul B layout [in, out])
//     float32[out]                   per-output-channel scales
//     int8[in * out]                 row-major values, zero point 0
//   32 bytes  SHA-256 of every byte before it
//
// Readers never throw and never trust the input (lengths are bounds-checked
// before any allocation).

#ifndef SOURCE_ENGINE_PERSONALIZE_PERSONAL_WEIGHTS_H_
#define SOURCE_ENGINE_PERSONALIZE_PERSONAL_WEIGHTS_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace McBopomofo {
namespace Personalize {

using Sha256Digest = std::array<uint8_t, 32>;

Sha256Digest Sha256(const void* data, size_t size);
// False (with *error) when the file cannot be read.
bool Sha256File(const std::string& path, Sha256Digest* out, std::string* error);
std::string ToHex(const Sha256Digest& digest);

// Reads a whole file. False (with *error) on failure.
bool ReadFileBytes(const std::string& path, std::string* out, std::string* error);

constexpr char kWeightsMagic[] = "AFMRPW01";  // 8 bytes, no terminator in the file
constexpr size_t kRLinearCount = 84;

struct QuantizedLinear {
  std::string name;  // ONNX initializer base name (val_N)
  uint32_t in = 0;
  uint32_t out = 0;
  std::vector<float> scales;   // [out]
  std::vector<int8_t> values;  // [in * out], row-major [in, out]
};

struct PersonalWeights {
  Sha256Digest factorySha{};
  std::vector<QuantizedLinear> tensors;
};

struct WeightsReadOptions {
  // Required tensor count; 0 accepts any count >= 1.
  size_t expectedTensors = kRLinearCount;
  // When set, every tensor name must be a key and its (in, out) must match.
  const std::unordered_map<std::string, std::pair<uint32_t, uint32_t>>* expectedShapes = nullptr;
  // Upper bound for in and out (guards allocations on garbage input).
  uint32_t maxDim = 1u << 16;
};

// Serialized bytes of `weights` (including the trailer hash). Tensors with
// inconsistent buffer sizes make it return an empty string.
std::string SerializeWeights(const PersonalWeights& weights);

// Writes `weights` to `path` via `<path>.tmp` + rename, mode 0600.
bool WriteWeightsFile(const std::string& path, const PersonalWeights& weights, std::string* error);

// Parses and validates `size` bytes (magic, sizes, trailer hash, tensor count,
// shapes, finite non-negative scales, unique names, no trailing bytes).
// nullptr (with *error) on any problem. Never throws.
std::shared_ptr<const PersonalWeights> ParseWeights(const void* data, size_t size,
                                                     const WeightsReadOptions& options,
                                                     std::string* error);
std::shared_ptr<const PersonalWeights> ReadWeightsFile(const std::string& path,
                                                        const WeightsReadOptions& options,
                                                        std::string* error);

}  // namespace Personalize
}  // namespace McBopomofo

#endif  // SOURCE_ENGINE_PERSONALIZE_PERSONAL_WEIGHTS_H_
