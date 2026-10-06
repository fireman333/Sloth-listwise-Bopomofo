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

#include <memory>
#include <string>
#include <vector>

#include "rescorer.h"

namespace McBopomofo {

class SlothRescorer : public Rescorer {
 public:
  struct Options {
    int intraOpThreads = 1;
    // File name under <modelDir>/onnx/.
    std::string onnxFile = "model_quantized.onnx";
  };

  // Never throws. On any load failure ready() is false and score() returns {}.
  explicit SlothRescorer(const std::string& modelDir);
  SlothRescorer(const std::string& modelDir, const Options& options);
  ~SlothRescorer() override;

  SlothRescorer(const SlothRescorer&) = delete;
  SlothRescorer& operator=(const SlothRescorer&) = delete;

  bool ready() const override;
  std::vector<double> score(const RescoreRequest& request) override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace McBopomofo

#endif  // SOURCE_ENGINE_RESCORER_SLOTH_RESCORER_H_
