#include "quantize.h"

#include <algorithm>
#include <cmath>

namespace McBopomofo {
namespace Personalize {

void QuantizePerChannel(const float* w, uint32_t in, uint32_t out, QuantizedLinear* t) {
  t->in = in;
  t->out = out;
  t->scales.assign(out, 0.0f);
  t->values.assign(static_cast<size_t>(in) * out, 0);
  for (uint32_t i = 0; i < in; ++i) {
    const float* row = w + static_cast<size_t>(i) * out;
    for (uint32_t j = 0; j < out; ++j) t->scales[j] = std::max(t->scales[j], std::fabs(row[j]));
  }
  for (uint32_t j = 0; j < out; ++j) t->scales[j] = t->scales[j] / 127.0f;
  for (uint32_t i = 0; i < in; ++i) {
    const float* row = w + static_cast<size_t>(i) * out;
    int8_t* q = t->values.data() + static_cast<size_t>(i) * out;
    for (uint32_t j = 0; j < out; ++j) {
      const float s = t->scales[j];
      // Same operations as numpy: fp32 division, rint (ties to even), clip.
      const float v = s > 0 ? std::nearbyint(row[j] / s) : 0.0f;
      q[j] = static_cast<int8_t>(std::min(127.0f, std::max(-127.0f, v)));
    }
  }
}

std::vector<float> Dequantize(const QuantizedLinear& t) {
  std::vector<float> w(static_cast<size_t>(t.in) * t.out);
  for (uint32_t i = 0; i < t.in; ++i) {
    for (uint32_t j = 0; j < t.out; ++j) {
      const size_t k = static_cast<size_t>(i) * t.out + j;
      w[k] = static_cast<float>(t.values[k]) * t.scales[j];
    }
  }
  return w;
}

void MergeLoRA(std::vector<float>* w, uint32_t in, uint32_t out, const float* A, const float* B,
               uint32_t r, float scale) {
  for (uint32_t i = 0; i < in; ++i) {
    for (uint32_t j = 0; j < out; ++j) {
      double d = 0;
      for (uint32_t k = 0; k < r; ++k)
        d += static_cast<double>(B[static_cast<size_t>(j) * r + k]) *
             static_cast<double>(A[static_cast<size_t>(k) * in + i]);
      float& x = (*w)[static_cast<size_t>(i) * out + j];
      x = x + static_cast<float>(static_cast<double>(scale) * d);
    }
  }
}

}  // namespace Personalize
}  // namespace McBopomofo
