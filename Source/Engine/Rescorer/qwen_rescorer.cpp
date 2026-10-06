#include "qwen_rescorer.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <cstdlib>

#include "llama.h"

namespace McBopomofo {

namespace {

constexpr int kScoreThreads = 4;
const char kPrefix[] = "以下是一句台灣繁體中文：";

void QuietLog(ggml_log_level level, const char* text, void*) {
  if (level >= GGML_LOG_LEVEL_ERROR) fputs(text, stderr);
}

std::vector<llama_token> Tokenize(const llama_vocab* vocab,
                                  const std::string& s) {
  std::vector<llama_token> toks(s.size() + 8);
  int n = llama_tokenize(vocab, s.data(), (int32_t)s.size(), toks.data(),
                         (int32_t)toks.size(), /*add_special=*/false,
                         /*parse_special=*/false);
  if (n < 0) {
    toks.resize(-n);
    n = llama_tokenize(vocab, s.data(), (int32_t)s.size(), toks.data(),
                       (int32_t)toks.size(), false, false);
  }
  if (n < 0) return {};
  toks.resize(n);
  return toks;
}

// Vectorizable float exp for x <= 0 (rel. error ~1e-7); the vocab-wide
// log-sum-exp dominates CPU time, so std::exp in double is too slow.
inline float FastExpNeg(float x) {
  x = std::max(x, -87.0f);
  float t = x * 1.44269504f;
  float n = std::nearbyint(t);
  float r = x - n * 0.693145752f - n * 1.42860677e-6f;
  float p = 1.0f + r * (1.0f + r * (0.5f + r * (0.166666672f + r * (0.0416666679f + r * (0.00833333377f + r * 0.00138888892f)))));
  int32_t bits = ((int32_t)n + 127) << 23;
  float scale;
  std::memcpy(&scale, &bits, sizeof(scale));
  return p * scale;
}

inline double LogSumExp(const float* logits, int n) {
  float mx = logits[0];
  for (int i = 1; i < n; ++i) mx = std::max(mx, logits[i]);
  float sum = 0;
  for (int i = 0; i < n; ++i) sum += FastExpNeg(logits[i] - mx);
  return (double)mx + std::log((double)sum);
}

// log-softmax value of `target` given raw logits.
inline double LogProb(const float* logits, int n, int target) {
  return (double)logits[target] - LogSumExp(logits, n);
}

}  // namespace

QwenRescorer::QwenRescorer(const std::string& gguf_path, int n_gpu_layers) {
  llama_log_set(QuietLog, nullptr);
  static bool backend_inited = false;
  if (!backend_inited) {
    llama_backend_init();
    backend_inited = true;
  }
  llama_model_params mp = llama_model_default_params();
  mp.n_gpu_layers = n_gpu_layers;
  model_ = llama_model_load_from_file(gguf_path.c_str(), mp);
  if (!model_) return;

  llama_context_params cp = llama_context_default_params();
  cp.n_ctx = 1024;
  cp.n_batch = 512;
  cp.n_ubatch = 512;
  cp.n_seq_max = kMaxSeq;
  cp.kv_unified = true;
  cp.n_threads = 4;
  cp.n_threads_batch = 4;
  ctx_ = llama_init_from_model(model_, cp);
  if (!ctx_) return;

  const llama_vocab* vocab = llama_model_get_vocab(model_);
  n_vocab_ = llama_vocab_n_tokens(vocab);
  std::vector<llama_token> pre = Tokenize(vocab, kPrefix);
  if (pre.empty()) return;
  prefix_len_ = (int)pre.size();

  llama_batch b = llama_batch_init((int32_t)pre.size(), 0, 1);
  for (int i = 0; i < prefix_len_; ++i) {
    b.token[i] = pre[i];
    b.pos[i] = i;
    b.n_seq_id[i] = 1;
    b.seq_id[i][0] = 0;
    b.logits[i] = (i == prefix_len_ - 1);
  }
  b.n_tokens = prefix_len_;
  int rc = llama_decode(ctx_, b);
  if (rc == 0) {
    const float* lg = llama_get_logits_ith(ctx_, prefix_len_ - 1);
    if (lg) {
      double lse = LogSumExp(lg, n_vocab_);
      prefix_last_logp_.resize(n_vocab_);
      for (int i = 0; i < n_vocab_; ++i)
        prefix_last_logp_[i] = (float)(lg[i] - lse);
      ready_ = true;
    }
  }
  llama_batch_free(b);
}

QwenRescorer::~QwenRescorer() {
  if (ctx_) llama_free(ctx_);
  if (model_) llama_model_free(model_);
}

std::vector<double> QwenRescorer::score(const RescoreRequest& request) {
  if (!ready_) return {};
  const size_t n = request.candidates.size();
  std::vector<double> out(n, 0.0);
  if (n == 0) return out;

  const llama_vocab* vocab = llama_model_get_vocab(model_);
  llama_memory_t mem = llama_get_memory(ctx_);
  std::vector<std::vector<llama_token>> toks(n);
  for (size_t i = 0; i < n; ++i) {
    toks[i] = Tokenize(vocab, request.candidates[i]);
    if (toks[i].empty()) return {};
  }

  // Process in chunks of (kMaxSeq - 1) candidates, each a single batch.
  for (size_t start = 0; start < n; start += kMaxSeq - 1) {
    size_t end = std::min(n, start + (size_t)(kMaxSeq - 1));
    int total = 0;
    for (size_t i = start; i < end; ++i) total += (int)toks[i].size();
    if (total > 512) return {};  // exceeds n_batch; not expected for IME input

    llama_batch b = llama_batch_init(total, 0, 1);
    std::vector<int> offset(end - start);
    int k = 0;
    for (size_t i = start; i < end; ++i) {
      llama_seq_id sid = (llama_seq_id)(i - start + 1);
      llama_memory_seq_cp(mem, 0, sid, 0, prefix_len_);
      offset[i - start] = k;
      for (size_t j = 0; j < toks[i].size(); ++j, ++k) {
        b.token[k] = toks[i][j];
        b.pos[k] = prefix_len_ + (int)j;
        b.n_seq_id[k] = 1;
        b.seq_id[k][0] = sid;
        // Logits at position j predict token j+1; the last token's output is
        // never scored.
        b.logits[k] = (j + 1 < toks[i].size());
      }
    }
    b.n_tokens = total;
    int rc = llama_decode(ctx_, b);
    if (rc == 0) {
      // Rows needing a vocab-wide log-sum-exp: (row index in batch, target).
      struct Row { int idx; llama_token target; size_t cand; };
      std::vector<Row> rows;
      for (size_t i = start; i < end; ++i) {
        out[i] = prefix_last_logp_[toks[i][0]];
        for (size_t j = 0; j + 1 < toks[i].size(); ++j)
          rows.push_back({offset[i - start] + (int)j, toks[i][j + 1], i});
      }
      std::vector<const float*> ptr(rows.size());
      for (size_t r = 0; r < rows.size(); ++r) {
        ptr[r] = llama_get_logits_ith(ctx_, rows[r].idx);  // syncs on first call
        if (!ptr[r]) rc = -1;
      }
      if (rc == 0) {
        std::vector<double> lp(rows.size());
        const int nt = (int)std::min<size_t>(kScoreThreads, rows.size());
        auto work = [&](int tid) {
          for (size_t r = tid; r < rows.size(); r += nt)
            lp[r] = LogProb(ptr[r], n_vocab_, rows[r].target);
        };
        std::vector<std::thread> th;
        for (int t = 1; t < nt; ++t) th.emplace_back(work, t);
        if (nt > 0) work(0);
        for (auto& t : th) t.join();
        for (size_t r = 0; r < rows.size(); ++r) out[rows[r].cand] += lp[r];
      }
    }
    llama_batch_free(b);
    // Drop candidate sequences; keep prefix (seq 0, positions [0, prefix_len)).
    for (size_t i = start; i < end; ++i)
      llama_memory_seq_rm(mem, (llama_seq_id)(i - start + 1), -1, -1);
    if (rc != 0) return {};
  }
  return out;
}

}  // namespace McBopomofo
