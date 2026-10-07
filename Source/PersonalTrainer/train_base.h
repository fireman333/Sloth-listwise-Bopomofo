// Training base of the personalization pack (personal-finetune design D3):
// `train-base.bin`, generated at build time from the shipped
// model_quantized.onnx by train/personalize/pack/build_pack.py. It holds
// every tensor the MLX forward needs, in torch state-dict naming:
//
//   magic "AFMRTB01"
//   32 bytes  SHA-256 of the factory model_quantized.onnx (zeros in fixtures)
//   uint32 L, L bytes JSON config {"n_syl","n_char","dim","depth","heads","kv","ffn"}
//   uint32 tensor count
//   per tensor:
//     uint32 len + name           (e.g. blocks.3.attn.q.weight, embed.weight)
//     uint8  kind                 0 fp32, 1 int8 per-output-channel linear,
//                                 2 uint8 per-tensor with zero point
//     uint32 ndim, uint32 dims[]  torch shape (linears [out, in])
//     kind 0: float32[prod(dims)] torch layout
//     kind 1: uint32 len + ONNX base name (val_N), float32 scales[out],
//             int8 values[in * out] in ONNX layout [in, out]
//     kind 2: float32 scale, uint8 zero point, uint8 values[prod(dims)]
//   32 bytes  SHA-256 of every byte before it
//
// Linears dequantize as q * scale (zero point 0), embed / head as
// (q - zp) * scale (research 17, E2). Readers never throw.

#ifndef SOURCE_PERSONALTRAINER_TRAIN_BASE_H_
#define SOURCE_PERSONALTRAINER_TRAIN_BASE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "Personalize/personal_weights.h"

namespace McBopomofo {
namespace Trainer {

struct ModelConfig {
  int nSyl = 0, nChar = 0, dim = 0, depth = 0, heads = 0, kv = 0, ffn = 0;
};

enum class TensorKind : uint8_t { kFloat = 0, kLinearInt8 = 1, kUint8Zp = 2 };

struct BaseTensor {
  std::string name;
  TensorKind kind = TensorKind::kFloat;
  std::vector<uint32_t> dims;  // torch shape
  std::vector<float> data;     // kind 0: torch layout
  // kind 1: the factory linear (ONNX layout [in, out]) incl. its ONNX name.
  Personalize::QuantizedLinear linear;
  // kind 2
  float scale = 0;
  uint8_t zeroPoint = 0;
  std::vector<uint8_t> u8;

};

struct TrainBase {
  Personalize::Sha256Digest factorySha{};
  ModelConfig config;
  std::vector<BaseTensor> tensors;
  std::unordered_map<std::string, size_t> byName;

  const BaseTensor* find(const std::string& name) const;
};

std::shared_ptr<const TrainBase> ParseTrainBase(const std::string& bytes, std::string* error);
std::shared_ptr<const TrainBase> ReadTrainBase(const std::string& path, std::string* error);

// Serialization (fixtures / tests).
std::string SerializeTrainBase(const TrainBase& base);

// fp32 values of a tensor in torch layout ([out, in] for linears).
std::vector<float> DequantizedTorchLayout(const BaseTensor& t);

}  // namespace Trainer
}  // namespace McBopomofo

#endif  // SOURCE_PERSONALTRAINER_TRAIN_BASE_H_
