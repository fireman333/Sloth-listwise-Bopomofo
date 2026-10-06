// ElectraRescorer: hfl/chinese-electra-180g-small-discriminator (replaced-token
// detection head, ONNX int8, onnxruntime, CPU) as a candidate-sentence Rescorer.
//
// Each candidate is tokenized character by character ("[CLS] c1 ... cn [SEP]";
// Chinese WordPiece degenerates to a per-character vocab.txt lookup, unknown
// characters become [UNK]) and all candidates run in one batch. The score is
//   sum_i log(1 - p_replaced_i) = -sum_i softplus(logit_i)
// over the character positions (CLS/SEP excluded); higher = more natural.
// The readings of the request are not used by this model.
// Mirrors bench/src/rescore_electra.py.
//
// Model directory layout (bench/src/export_electra_onnx.py, models/electra):
//   model_int8.onnx (or Options::onnxFile), vocab.txt

#ifndef SOURCE_ENGINE_RESCORER_ELECTRA_RESCORER_H_
#define SOURCE_ENGINE_RESCORER_ELECTRA_RESCORER_H_

#include <memory>
#include <string>
#include <vector>

#include "rescorer.h"

namespace McBopomofo {

class ElectraRescorer : public Rescorer {
 public:
  struct Options {
    int intraOpThreads = 1;
    std::string onnxFile = "model_int8.onnx";
  };

  // Never throws. On any load failure ready() is false and score() returns {}.
  explicit ElectraRescorer(const std::string& modelDir);
  ElectraRescorer(const std::string& modelDir, const Options& options);
  ~ElectraRescorer() override;

  ElectraRescorer(const ElectraRescorer&) = delete;
  ElectraRescorer& operator=(const ElectraRescorer&) = delete;

  bool ready() const override;
  std::vector<double> score(const RescoreRequest& request) override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace McBopomofo

#endif  // SOURCE_ENGINE_RESCORER_ELECTRA_RESCORER_H_
