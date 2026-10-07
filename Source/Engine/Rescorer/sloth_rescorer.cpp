#include "sloth_rescorer.h"

#include <onnxruntime_cxx_api.h>

#include "ort_env.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace McBopomofo {

namespace {

// ---------------------------------------------------------------------------
// Minimal JSON reader (enough for syl_vocab.json and tokenizer.json).
struct Json {
  enum Type { kNull, kBool, kNumber, kString, kArray, kObject } type = kNull;
  double number = 0;
  std::string str;
  std::vector<Json> arr;
  std::vector<std::pair<std::string, Json>> obj;

  const Json* find(const std::string& key) const {
    for (const auto& kv : obj) {
      if (kv.first == key) return &kv.second;
    }
    return nullptr;
  }
};

class JsonParser {
 public:
  explicit JsonParser(const std::string& s) : s_(s) {}
  Json parse() {
    Json j = value();
    ws();
    return j;
  }

 private:
  const std::string& s_;
  size_t p_ = 0;

  [[noreturn]] void fail(const char* m) { throw std::runtime_error(m); }
  void ws() {
    while (p_ < s_.size() &&
           (s_[p_] == ' ' || s_[p_] == '\n' || s_[p_] == '\t' || s_[p_] == '\r'))
      ++p_;
  }
  static void appendUtf8(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
      out += static_cast<char>(cp);
    } else if (cp < 0x800) {
      out += static_cast<char>(0xC0 | (cp >> 6));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
      out += static_cast<char>(0xE0 | (cp >> 12));
      out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
      out += static_cast<char>(0xF0 | (cp >> 18));
      out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
      out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    }
  }
  uint32_t hex4() {
    if (p_ + 4 > s_.size()) fail("bad \\u escape");
    uint32_t v = static_cast<uint32_t>(std::strtoul(s_.substr(p_, 4).c_str(), nullptr, 16));
    p_ += 4;
    return v;
  }
  std::string string() {
    if (s_[p_] != '"') fail("expected string");
    ++p_;
    std::string out;
    while (p_ < s_.size() && s_[p_] != '"') {
      char c = s_[p_++];
      if (c != '\\') {
        out += c;
        continue;
      }
      if (p_ >= s_.size()) fail("bad escape");
      char e = s_[p_++];
      switch (e) {
        case 'n': out += '\n'; break;
        case 't': out += '\t'; break;
        case 'r': out += '\r'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'u': {
          uint32_t cp = hex4();
          if (cp >= 0xD800 && cp < 0xDC00 && p_ + 1 < s_.size() && s_[p_] == '\\' &&
              s_[p_ + 1] == 'u') {
            p_ += 2;
            uint32_t lo = hex4();
            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
          }
          appendUtf8(out, cp);
          break;
        }
        default: out += e;
      }
    }
    if (p_ >= s_.size()) fail("unterminated string");
    ++p_;
    return out;
  }
  Json value() {
    ws();
    if (p_ >= s_.size()) fail("unexpected end");
    Json j;
    char c = s_[p_];
    if (c == '{') {
      ++p_;
      j.type = Json::kObject;
      ws();
      if (s_[p_] == '}') { ++p_; return j; }
      while (true) {
        ws();
        std::string k = string();
        ws();
        if (s_[p_++] != ':') fail("expected ':'");
        j.obj.emplace_back(std::move(k), value());
        ws();
        char d = s_[p_++];
        if (d == '}') break;
        if (d != ',') fail("expected ',' or '}'");
      }
    } else if (c == '[') {
      ++p_;
      j.type = Json::kArray;
      ws();
      if (s_[p_] == ']') { ++p_; return j; }
      while (true) {
        j.arr.push_back(value());
        ws();
        char d = s_[p_++];
        if (d == ']') break;
        if (d != ',') fail("expected ',' or ']'");
      }
    } else if (c == '"') {
      j.type = Json::kString;
      j.str = string();
    } else if (s_.compare(p_, 4, "true") == 0) {
      j.type = Json::kBool; j.number = 1; p_ += 4;
    } else if (s_.compare(p_, 5, "false") == 0) {
      j.type = Json::kBool; p_ += 5;
    } else if (s_.compare(p_, 4, "null") == 0) {
      p_ += 4;
    } else {
      char* end = nullptr;
      j.type = Json::kNumber;
      j.number = std::strtod(s_.c_str() + p_, &end);
      if (end == s_.c_str() + p_) fail("bad number");
      p_ = static_cast<size_t>(end - s_.c_str());
    }
    return j;
  }
};

std::string ReadFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

// Splits UTF-8 into per-code-point strings. Invalid bytes become one-byte
// pieces (which will not be found in the vocabulary).
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

constexpr double kMissingCharLogp = -30.0;  // same as the Python reference.

}  // namespace

struct SlothRescorer::Impl {
  Ort::Env env = MakeOfflineOrtEnv("SlothRescorer");
  // Override buffers and their OrtValues must outlive the session (declared
  // before it, so destroyed after it).
  std::shared_ptr<const std::vector<SlothRescorer::Options::InitializerOverride>> overrides;
  std::vector<Ort::Value> overrideValues;
  std::unique_ptr<Ort::Session> session;
  std::unordered_map<std::string, int64_t> sylId;   // syllable -> input id
  std::unordered_map<std::string, int64_t> charId;  // char -> output id
  std::vector<std::string> idChar;                   // output id -> char
  std::unordered_map<std::string, std::vector<int64_t>> legal;  // sorted unique
  bool ready = false;

  void Load(const std::string& dir, const SlothRescorer::Options& opt) {
    {
      Json j = JsonParser(ReadFile(dir + "/syl_vocab.json")).parse();
      for (const auto& kv : j.obj) sylId[kv.first] = static_cast<int64_t>(kv.second.number);
    }
    {
      Json j = JsonParser(ReadFile(dir + "/tokenizer/tokenizer.json")).parse();
      const Json* model = j.find("model");
      const Json* vocab = model ? model->find("vocab") : nullptr;
      if (vocab) {
        for (const auto& kv : vocab->obj) charId[kv.first] = static_cast<int64_t>(kv.second.number);
      }
      const Json* added = j.find("added_tokens");
      if (added) {
        for (const auto& t : added->arr) {
          const Json* id = t.find("id");
          const Json* content = t.find("content");
          if (id && content) charId.emplace(content->str, static_cast<int64_t>(id->number));
        }
      }
      if (charId.empty()) throw std::runtime_error("empty char vocab");
      for (const auto& kv : charId) {
        if (kv.second < 0) continue;
        size_t id = static_cast<size_t>(kv.second);
        if (id >= idChar.size()) idChar.resize(id + 1);
        idChar[id] = kv.first;
      }
    }
    {
      std::ifstream in(dir + "/phonetic_table.tsv");
      if (!in) throw std::runtime_error("cannot open phonetic_table.tsv");
      std::string line;
      while (std::getline(in, line)) {
        size_t tab = line.find('\t');
        if (tab == std::string::npos) continue;
        std::string syl = line.substr(0, tab);
        std::string chars = line.substr(tab + 1);
        while (!chars.empty() && (chars.back() == '\r' || chars.back() == '\n')) chars.pop_back();
        std::vector<int64_t> ids;
        for (const auto& c : SplitCodePoints(chars)) {
          auto it = charId.find(c);
          if (it != charId.end()) ids.push_back(it->second);
        }
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        legal[syl] = std::move(ids);
      }
    }
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(opt.intraOpThreads > 0 ? opt.intraOpThreads : 1);
    so.SetInterOpNumThreads(1);
    // Keep resident memory low in the long-lived IME process: no memory
    // pattern planning and no CPU arena (tensors use plain malloc).
    so.DisableMemPattern();
    so.DisableCpuMemArena();
    std::string onnxPath = dir + "/onnx/" + opt.onnxFile;
    if (!std::ifstream(onnxPath)) throw std::runtime_error("cannot open " + onnxPath);
    if (opt.overrides) {
      overrides = opt.overrides;
      Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
      overrideValues.reserve(overrides->size());
      for (const auto& o : *overrides) {
        if (!o.int8Data.empty()) {
          overrideValues.push_back(Ort::Value::CreateTensor<int8_t>(
              mem, const_cast<int8_t*>(o.int8Data.data()), o.int8Data.size(), o.shape.data(),
              o.shape.size()));
        } else {
          overrideValues.push_back(Ort::Value::CreateTensor<float>(
              mem, const_cast<float*>(o.floatData.data()), o.floatData.size(), o.shape.data(),
              o.shape.size()));
        }
        so.AddInitializer(o.name.c_str(), overrideValues.back());
      }
    }
    session = std::make_unique<Ort::Session>(env, onnxPath.c_str(), so);
    ready = true;
  }

  // One forward pass (batch of 1, hints = 0). Returns false on shape error.
  bool Forward(const std::vector<std::string>& readings, std::vector<Ort::Value>* outs,
               size_t* V) {
    const size_t T = readings.size();
    std::vector<int64_t> syl(T);
    for (size_t t = 0; t < T; ++t) {
      auto it = sylId.find(readings[t]);
      syl[t] = it == sylId.end() ? 1 : it->second;  // 1 == <unk>
    }
    // std::vector<bool> has no data(); use a plain array.
    std::unique_ptr<bool[]> amask(new bool[T]);
    std::fill(amask.get(), amask.get() + T, true);
    std::vector<int64_t> hints(T, 0);
    const int64_t shape[2] = {1, static_cast<int64_t>(T)};
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputs[3] = {
        Ort::Value::CreateTensor<int64_t>(mem, syl.data(), T, shape, 2),
        Ort::Value::CreateTensor<bool>(mem, amask.get(), T, shape, 2),
        Ort::Value::CreateTensor<int64_t>(mem, hints.data(), T, shape, 2)};
    const char* inNames[3] = {"syl", "amask", "hints"};
    const char* outNames[1] = {"logits"};
    *outs = session->Run(Ort::RunOptions{nullptr}, inNames, inputs, 3, outNames, 1);
    auto oshape = (*outs)[0].GetTensorTypeAndShapeInfo().GetShape();
    if (oshape.size() != 3 || oshape[1] != static_cast<int64_t>(T)) return false;
    *V = static_cast<size_t>(oshape[2]);
    return true;
  }

  SlothRescorer::CharLogProbTable CharLogProbs(
      const std::vector<std::string>& readings,
      const std::vector<std::vector<std::string>>* extraChars) {
    const size_t T = readings.size();
    if (T == 0) return {};
    if (extraChars != nullptr && extraChars->size() != T) return {};
    std::vector<Ort::Value> outs;
    size_t V = 0;
    if (!Forward(readings, &outs, &V)) return {};
    const float* lg = outs[0].GetTensorData<float>();

    SlothRescorer::CharLogProbTable table(T);
    std::vector<int64_t> ids;
    for (size_t t = 0; t < T; ++t) {
      const float* row = lg + t * V;
      auto lit = legal.find(readings[t]);
      const bool full = lit == legal.end();
      ids.clear();
      if (!full) ids = lit->second;
      if (extraChars != nullptr) {
        for (const auto& c : (*extraChars)[t]) {
          auto it = charId.find(c);
          if (it != charId.end()) ids.push_back(it->second);
        }
      }
      std::sort(ids.begin(), ids.end());
      ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
      double m = -INFINITY;
      auto valid = [&](int64_t id) { return id >= 0 && static_cast<size_t>(id) < V; };
      if (full) {
        for (size_t id = 0; id < V; ++id) m = std::max(m, static_cast<double>(row[id]));
      } else {
        for (int64_t id : ids) if (valid(id)) m = std::max(m, static_cast<double>(row[id]));
      }
      if (!std::isfinite(m)) continue;
      double sum = 0;
      if (full) {
        for (size_t id = 0; id < V; ++id) sum += std::exp(static_cast<double>(row[id]) - m);
      } else {
        for (int64_t id : ids) if (valid(id)) sum += std::exp(static_cast<double>(row[id]) - m);
      }
      const double lse = m + std::log(sum);
      auto& out = table[t];
      out.reserve(ids.size());
      for (int64_t id : ids) {
        if (!valid(id) || static_cast<size_t>(id) >= idChar.size() || idChar[id].empty()) continue;
        out.emplace(idChar[id], static_cast<double>(row[id]) - lse);
      }
    }
    return table;
  }

  std::vector<double> Score(const RescoreRequest& req) {
    const size_t T = req.readings.size();
    const size_t N = req.candidates.size();
    if (T == 0) return {};
    if (N == 0) return {};

    std::vector<Ort::Value> outs;
    size_t V = 0;
    if (!Forward(req.readings, &outs, &V)) return {};
    const float* lg = outs[0].GetTensorData<float>();

    // Candidate char ids; -1 == not in vocab.
    std::vector<std::vector<int64_t>> cid(N);
    for (size_t i = 0; i < N; ++i) {
      for (const auto& c : SplitCodePoints(req.candidates[i])) {
        auto it = charId.find(c);
        cid[i].push_back(it == charId.end() ? -1 : it->second);
      }
    }

    // Per-position restricted log-normalizer: logsumexp over legal ∪ extra.
    std::vector<double> maxv(T), lse(T);
    std::vector<int64_t> extra;
    for (size_t t = 0; t < T; ++t) {
      const float* row = lg + t * V;
      extra.clear();
      for (size_t i = 0; i < N; ++i) {
        if (t < cid[i].size() && cid[i][t] >= 0) extra.push_back(cid[i][t]);
      }
      std::sort(extra.begin(), extra.end());
      extra.erase(std::unique(extra.begin(), extra.end()), extra.end());

      auto lit = legal.find(req.readings[t]);
      const std::vector<int64_t>* L = lit == legal.end() ? nullptr : &lit->second;
      auto inLegal = [&](int64_t id) {
        return L == nullptr || std::binary_search(L->begin(), L->end(), id);
      };
      double m = -INFINITY;
      auto visit = [&](auto&& fn) {
        if (L == nullptr) {
          for (size_t id = 0; id < V; ++id) fn(static_cast<int64_t>(id));
        } else {
          for (int64_t id : *L) fn(id);
        }
        for (int64_t id : extra) {
          if (!inLegal(id)) fn(id);
        }
      };
      visit([&](int64_t id) {
        if (id >= 0 && static_cast<size_t>(id) < V) m = std::max(m, static_cast<double>(row[id]));
      });
      double sum = 0;
      visit([&](int64_t id) {
        if (id >= 0 && static_cast<size_t>(id) < V) sum += std::exp(static_cast<double>(row[id]) - m);
      });
      maxv[t] = m;
      lse[t] = m + std::log(sum);
    }

    std::vector<double> scores(N, 0.0);
    for (size_t i = 0; i < N; ++i) {
      double s = 0;
      for (size_t t = 0; t < cid[i].size(); ++t) {
        int64_t c = cid[i][t];
        if (c < 0 || t >= T || static_cast<size_t>(c) >= V) {
          s += kMissingCharLogp;
        } else {
          s += static_cast<double>(lg[t * V + c]) - lse[t];
        }
      }
      scores[i] = s;
    }
    return scores;
  }
};

SlothRescorer::SlothRescorer(const std::string& modelDir)
    : SlothRescorer(modelDir, Options()) {}

SlothRescorer::SlothRescorer(const std::string& modelDir, const Options& options)
    : impl_(nullptr) {
  try {
    impl_ = std::make_unique<Impl>();
    impl_->Load(modelDir, options);
  } catch (...) {
    if (impl_) impl_->ready = false;
  }
}

SlothRescorer::~SlothRescorer() = default;

bool SlothRescorer::ready() const { return impl_ && impl_->ready; }

std::vector<double> SlothRescorer::score(const RescoreRequest& request) {
  if (!ready()) return {};
  try {
    return impl_->Score(request);
  } catch (...) {
    return {};
  }
}

SlothRescorer::CharLogProbTable SlothRescorer::charLogProbs(
    const std::vector<std::string>& readings,
    const std::vector<std::vector<std::string>>* extraChars) {
  if (!ready()) return {};
  try {
    return impl_->CharLogProbs(readings, extraChars);
  } catch (...) {
    return {};
  }
}

}  // namespace McBopomofo
