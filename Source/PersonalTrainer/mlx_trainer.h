// MLX (CPU) LoRA fine-tune of R (personal-finetune design D8): C++ port of
// research/spike_runtime/mlx_probe.py — R forward (RMSNorm, QK-norm, RoPE,
// GQA, SwiGLU), LoRA r=8 scale 2 on the 84 block linears, loss = top-50
// listwise CE with the n-gram residual + 0.5 * full-vocabulary MLE,
// MLX-Python-equivalent AdamW (no bias correction, no weight decay), global
// norm clipping and a cosine schedule with warm-up and floor.

#ifndef SOURCE_PERSONALTRAINER_MLX_TRAINER_H_
#define SOURCE_PERSONALTRAINER_MLX_TRAINER_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "Personalize/personal_weights.h"
#include "train_base.h"
#include "train_data.h"

namespace McBopomofo {
namespace Trainer {

struct Recipe {
  int rank = 8;
  float loraScale = 2.0f;
  double lr = 5e-4;
  double beta1 = 0.9, beta2 = 0.95, eps = 1e-8;
  double warmupFrac = 0.05, lrFloor = 0.1;
  double epochs = 3;
  int minSteps = 60;
  int batchUser = 16, batchReplay = 16;
  double clip = 1.0;
  double mleWeight = 0.5;
  double ngramWeight = 1.0;
  float missingLogP = -30.0f;
  size_t cacheMB = 64;
};

// Learning rate at step s (0-based) of `steps` (mlx_probe.py sched).
double ScheduledLr(const Recipe& r, int step, int steps);
// max(minSteps, ceil(nTrain * epochs / batchUser)).
int StepsFor(const Recipe& r, size_t nTrain);

struct LossValues {
  double loss = 0, listwise = 0, mle = 0;
  double gradNormB = 0;  // global L2 norm of the LoRA-B gradients (before clipping)
  double gradNorm = 0;   // of all LoRA gradients
};

struct TrainProgress {
  int step = 0, steps = 0;
  double loss = 0;
};

class LoraTrainer {
 public:
  // Sets the MLX default device to CPU and the buffer-cache limit.
  LoraTrainer(std::shared_ptr<const TrainBase> base, const ModelVocab& vocab, Recipe recipe);
  ~LoraTrainer();

  bool ok() const;
  const std::string& error() const;

  // Random A ~ N(0, 1/in) from `seed`, B = 0.
  void InitLora(uint64_t seed);
  // A in the torch order of add_lora (per block q, k, v, o, w1, w3, w2), each
  // [r, in] row-major; B = 0.
  bool SetLoraA(const std::vector<std::vector<float>>& a);

  // Loss and gradients on one batch without updating (numeric checks).
  LossValues Evaluate(const std::vector<const EncodedRow*>& batch);

  // Full run: `steps` steps, each with batchUser user rows (shuffled epochs,
  // python order) + batchReplay distinct replay rows. `cont` is polled
  // between steps (false = stop, returns false with error "cancelled").
  // Non-finite loss -> false with error "non-finite loss".
  bool Train(const std::vector<EncodedRow>& user, const std::vector<EncodedRow>& replay, int steps,
             uint64_t seed, const std::function<bool(const TrainProgress&)>& cont,
             double* firstLoss = nullptr);

  // W + scale * (B A)^T for the 84 linears, per-channel int8 (D9), in the
  // order of the training base; names are the ONNX base names.
  bool Merge(Personalize::PersonalWeights* out, std::string* error) const;

  size_t trainableParameters() const;
  // LoRA A / B host copies (tests).
  std::vector<float> loraA(size_t i) const;
  std::vector<float> loraB(size_t i) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace Trainer
}  // namespace McBopomofo

#endif  // SOURCE_PERSONALTRAINER_MLX_TRAINER_H_
