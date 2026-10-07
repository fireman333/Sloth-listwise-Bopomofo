// Merge of LoRA deltas and the per-output-channel int8 quantizer of the
// shipped R model (personal-finetune design D9):
//   W' = W + scale * (B A)^T      (W in MatMul layout [in, out])
//   s_j = max_i |W'[i, j]| / 127  (fp32), q = clip(rint(W' / s_j), -127, 127)
// Requantizing the dequantized factory weights reproduces the shipped bytes.

#ifndef SOURCE_ENGINE_PERSONALIZE_QUANTIZE_H_
#define SOURCE_ENGINE_PERSONALIZE_QUANTIZE_H_

#include <cstdint>
#include <vector>

#include "personal_weights.h"

namespace McBopomofo {
namespace Personalize {

// w: [in * out] row-major fp32. Fills t->in, t->out, t->scales, t->values
// (t->name is left unchanged). An all-zero channel gets scale 0 and values 0.
void QuantizePerChannel(const float* w, uint32_t in, uint32_t out, QuantizedLinear* t);

// q * scale per element, [in * out] row-major.
std::vector<float> Dequantize(const QuantizedLinear& t);

// w[i * out + j] += scale * sum_k B[j * r + k] * A[k * in + i]
// (A: [r, in], B: [out, r], the torch LoRA layout of a [out, in] weight).
// The dot product is accumulated in double and added in fp32.
void MergeLoRA(std::vector<float>* w, uint32_t in, uint32_t out, const float* A, const float* B,
               uint32_t r, float scale);

}  // namespace Personalize
}  // namespace McBopomofo

#endif  // SOURCE_ENGINE_PERSONALIZE_QUANTIZE_H_
