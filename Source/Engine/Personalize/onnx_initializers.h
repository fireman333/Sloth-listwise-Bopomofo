// Minimal ONNX (protobuf) reader for the R model: lists the graph's
// initializers (name, data type, dims, data) and the weight inputs of its
// MatMulInteger nodes, without a protobuf library. Used to check personalized
// weight names / shapes against the factory model before an ORT session is
// created (ORT silently ignores an AddInitializer name that is not in the
// graph, design D11 note), and to read the factory int8 linears.

#ifndef SOURCE_ENGINE_PERSONALIZE_ONNX_INITIALIZERS_H_
#define SOURCE_ENGINE_PERSONALIZE_ONNX_INITIALIZERS_H_

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "personal_weights.h"

namespace McBopomofo {
namespace Personalize {

// ONNX TensorProto.DataType values used here.
enum OnnxDataType : int32_t { kOnnxFloat = 1, kOnnxUint8 = 2, kOnnxInt8 = 3 };

struct OnnxInitializer {
  std::string name;
  int32_t dataType = 0;
  std::vector<int64_t> dims;
  // raw_data bytes (view into the model bytes) when present.
  std::string_view raw;
  // float_data when the tensor stores floats outside raw_data.
  std::vector<float> floatData;
};

struct OnnxModelInfo {
  std::vector<OnnxInitializer> initializers;
  std::unordered_map<std::string, size_t> byName;  // name -> index
  // input[1] (the weight) of every MatMulInteger node, in graph order.
  std::vector<std::string> matMulIntegerWeights;

  const OnnxInitializer* find(const std::string& name) const;
};

// Parses `bytes` (an ONNX ModelProto). Views in the result point into `bytes`,
// which must outlive it. False (with *error) on malformed input; never throws.
bool ScanOnnxModel(std::string_view bytes, OnnxModelInfo* out, std::string* error);

// The factory linears of an int8 R model: one entry per MatMulInteger weight
// <base>_quantized (int8 [in, out]) with <base>_scale (float [out]) and an
// all-zero <base>_zero_point, in graph order. False (with *error) when any of
// these is missing or has another type / shape.
// withData == false leaves scales / values empty (shape checks only).
bool FactoryLinears(const OnnxModelInfo& model, std::vector<QuantizedLinear>* out,
                    std::string* error, bool withData = true);

// (in, out) per base name, for WeightsReadOptions::expectedShapes.
std::unordered_map<std::string, std::pair<uint32_t, uint32_t>> LinearShapes(
    const std::vector<QuantizedLinear>& linears);

}  // namespace Personalize
}  // namespace McBopomofo

#endif  // SOURCE_ENGINE_PERSONALIZE_ONNX_INITIALIZERS_H_
