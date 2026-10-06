// QwenRescorer: scores candidate sentences with Qwen3-0.6B-Base (GGUF) via
// llama.cpp (Metal). score(c) = sum of token log-probs of the candidate given
// the fixed prefix "以下是一句台灣繁體中文：" (prefix tokens are not scored).
// Mirrors bench/src/rescore_lm.py with lambda = 0.
//
// The prefix KV state is computed once at load and copied (llama_memory_seq_cp)
// to one sequence per candidate; all candidates are decoded in a single batch.
//
// Threading: NOT thread-safe. The caller must serialize score() calls
// (the IME runs it on a single background queue).
// Failure policy: never throws. If loading fails, ready() == false and
// score() returns an empty vector (as it does on any decode error).

#ifndef SOURCE_ENGINE_RESCORER_QWEN_RESCORER_H_
#define SOURCE_ENGINE_RESCORER_QWEN_RESCORER_H_

#include <string>
#include <vector>

#include "rescorer.h"

struct llama_model;
struct llama_context;

namespace McBopomofo {

class QwenRescorer : public Rescorer {
 public:
  // n_gpu_layers: 99 = all layers on Metal; 0 = CPU only.
  explicit QwenRescorer(const std::string& gguf_path, int n_gpu_layers = 99);
  ~QwenRescorer() override;
  QwenRescorer(const QwenRescorer&) = delete;
  QwenRescorer& operator=(const QwenRescorer&) = delete;

  bool ready() const override { return ready_; }
  std::vector<double> score(const RescoreRequest& request) override;

 private:
  static constexpr int kMaxSeq = 16;  // seq 0 = prefix, 1..15 = candidates
  bool ready_ = false;
  llama_model* model_ = nullptr;
  llama_context* ctx_ = nullptr;
  int n_vocab_ = 0;
  int prefix_len_ = 0;
  std::vector<float> prefix_last_logp_;  // log-softmax at last prefix token
};

}  // namespace McBopomofo

#endif  // SOURCE_ENGINE_RESCORER_QWEN_RESCORER_H_
