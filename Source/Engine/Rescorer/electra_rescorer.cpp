#include "electra_rescorer.h"

#include <onnxruntime_cxx_api.h>

#include "ort_env.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <unordered_map>

namespace McBopomofo {

namespace {

// Splits UTF-8 into per-code-point strings (invalid bytes -> 1-byte pieces).
std::vector<std::string> SplitCodePoints(const std::string& s) {
  std::vector<std::string> out;
  size_t i = 0;
  while (i < s.size()) {
    unsigned char c = static_cast<unsigned char>(s[i]);
    size_t n = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 1;
    if (i + n > s.size()) n = 1;
    out.push_back(s.substr(i, n));
    i += n;
  }
  return out;
}

inline double Softplus(double x) { return std::max(x, 0.0) + std::log1p(std::exp(-std::fabs(x))); }

}  // namespace

struct ElectraRescorer::Impl {
  Ort::Env env = MakeOfflineOrtEnv("ElectraRescorer");
  std::unique_ptr<Ort::Session> session;
  std::unordered_map<std::string, int64_t> vocab;
  int64_t pad = 0, unk = 100, cls = 101, sep = 102;
  bool ready = false;

  void Load(const std::string& dir, const ElectraRescorer::Options& opt) {
    {
      std::ifstream in(dir + "/vocab.txt");
      if (!in) throw std::runtime_error("cannot open vocab.txt");
      std::string line;
      int64_t id = 0;
      while (std::getline(in, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        vocab.emplace(line, id++);
      }
      if (vocab.empty()) throw std::runtime_error("empty vocab");
      auto get = [&](const char* k, int64_t& dst) {
        auto it = vocab.find(k);
        if (it != vocab.end()) dst = it->second;
      };
      get("[PAD]", pad);
      get("[UNK]", unk);
      get("[CLS]", cls);
      get("[SEP]", sep);
    }
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(opt.intraOpThreads > 0 ? opt.intraOpThreads : 1);
    so.SetInterOpNumThreads(1);
    std::string onnxPath = dir + "/" + opt.onnxFile;
    if (!std::ifstream(onnxPath)) throw std::runtime_error("cannot open " + onnxPath);
    session = std::make_unique<Ort::Session>(env, onnxPath.c_str(), so);
    ready = true;
  }

  std::vector<double> Score(const RescoreRequest& req) {
    const size_t N = req.candidates.size();
    if (N == 0) return {};
    std::vector<std::vector<int64_t>> toks(N);
    size_t maxLen = 0;
    for (size_t i = 0; i < N; ++i) {
      auto& t = toks[i];
      t.push_back(cls);
      for (const auto& c : SplitCodePoints(req.candidates[i])) {
        auto it = vocab.find(c);
        t.push_back(it == vocab.end() ? unk : it->second);
      }
      t.push_back(sep);
      maxLen = std::max(maxLen, t.size());
    }
    const size_t T = maxLen;
    std::vector<int64_t> ids(N * T, pad), mask(N * T, 0);
    for (size_t i = 0; i < N; ++i) {
      for (size_t t = 0; t < toks[i].size(); ++t) {
        ids[i * T + t] = toks[i][t];
        mask[i * T + t] = 1;
      }
    }
    const int64_t shape[2] = {static_cast<int64_t>(N), static_cast<int64_t>(T)};
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputs[2] = {Ort::Value::CreateTensor<int64_t>(mem, ids.data(), ids.size(), shape, 2),
                            Ort::Value::CreateTensor<int64_t>(mem, mask.data(), mask.size(), shape, 2)};
    const char* inNames[2] = {"input_ids", "attention_mask"};
    const char* outNames[1] = {"logits"};
    auto outs = session->Run(Ort::RunOptions{nullptr}, inNames, inputs, 2, outNames, 1);
    auto oshape = outs[0].GetTensorTypeAndShapeInfo().GetShape();
    if (oshape.size() != 2 || oshape[0] != static_cast<int64_t>(N) || oshape[1] != static_cast<int64_t>(T))
      return {};
    const float* lg = outs[0].GetTensorData<float>();
    std::vector<double> scores(N, 0.0);
    for (size_t i = 0; i < N; ++i) {
      const size_t L = toks[i].size();  // positions 1 .. L-2 are the characters
      double s = 0;
      for (size_t t = 1; t + 1 < L; ++t) s -= Softplus(static_cast<double>(lg[i * T + t]));
      scores[i] = s;
    }
    return scores;
  }
};

ElectraRescorer::ElectraRescorer(const std::string& modelDir)
    : ElectraRescorer(modelDir, Options()) {}

ElectraRescorer::ElectraRescorer(const std::string& modelDir, const Options& options)
    : impl_(nullptr) {
  try {
    impl_ = std::make_unique<Impl>();
    impl_->Load(modelDir, options);
  } catch (...) {
    if (impl_) impl_->ready = false;
  }
}

ElectraRescorer::~ElectraRescorer() = default;

bool ElectraRescorer::ready() const { return impl_ && impl_->ready; }

std::vector<double> ElectraRescorer::score(const RescoreRequest& request) {
  if (!ready()) return {};
  try {
    return impl_->Score(request);
  } catch (...) {
    return {};
  }
}

}  // namespace McBopomofo
