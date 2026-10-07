// SlothRescorer: SlothLM-E 12M (non-autoregressive zhuyin -> char encoder,
// ONNX int8, onnxruntime, CPU) as a candidate-sentence Rescorer.
//
// One forward pass over the syllable sequence yields per-position logits;
// every candidate is then scored as sum_t log p(char_t | all syllables), with
// the softmax at position t restricted to the characters legal for syllable t
// (plus the characters the candidates use there). Mirrors
// bench/src/rescore_sloth.py (Sloth.score_cands).
//
// Model directory layout (as in bench/out/_models/sloth):
//   onnx/model_quantized.onnx, syl_vocab.json, phonetic_table.tsv,
//   tokenizer/tokenizer.json

#ifndef SOURCE_ENGINE_RESCORER_SLOTH_RESCORER_H_
#define SOURCE_ENGINE_RESCORER_SLOTH_RESCORER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "rescorer.h"

namespace McBopomofo {

class SlothRescorer : public Rescorer {
 public:
  struct Options {
    int intraOpThreads = 1;
    // File name under <modelDir>/onnx/.
    std::string onnxFile = "model_quantized.onnx";
    // Initializers that replace same-name initializers of the ONNX file at
    // session creation (OrtApi::AddInitializer); the file is not modified.
    // Used to load personalized int8 weights (val_N_quantized / val_N_scale).
    // Buffers are shared with the rescorer, which keeps them alive for the
    // session's lifetime. nullptr = the file's own weights.
    struct InitializerOverride {
      std::string name;
      std::vector<int64_t> shape;
      std::vector<int8_t> int8Data;  // used when non-empty
      std::vector<float> floatData;  // used otherwise
    };
    std::shared_ptr<const std::vector<InitializerOverride>> overrides;
  };

  // Never throws. On any load failure ready() is false and score() returns {}.
  explicit SlothRescorer(const std::string& modelDir);
  SlothRescorer(const std::string& modelDir, const Options& options);
  ~SlothRescorer() override;

  SlothRescorer(const SlothRescorer&) = delete;
  SlothRescorer& operator=(const SlothRescorer&) = delete;

  bool ready() const override;
  std::vector<double> score(const RescoreRequest& request) override;

  // Per-position character log-probabilities for in-walk scoring (R-walk).
  // result[t] maps each character legal for readings[t] (phonetic_table.tsv),
  // plus every in-vocabulary character of extraChars[t] when given, to
  //   log p(char | all readings) = logit[t, char] - logsumexp over that set.
  // For a syllable with no legal set the normaliser is the full vocabulary and
  // only extraChars[t] are listed. Characters outside the model vocabulary are
  // never listed (callers apply their own missing-character penalty, -30 in
  // the reference). One forward pass, hints = 0. Same normaliser as score()
  // when extraChars carries the candidate characters. Returns an empty vector
  // on failure or when not ready (callers fall back to the plain walk).
  using CharLogProbTable = std::vector<std::unordered_map<std::string, double>>;
  CharLogProbTable charLogProbs(
      const std::vector<std::string>& readings,
      const std::vector<std::vector<std::string>>* extraChars = nullptr);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace McBopomofo

#endif  // SOURCE_ENGINE_RESCORER_SLOTH_RESCORER_H_
