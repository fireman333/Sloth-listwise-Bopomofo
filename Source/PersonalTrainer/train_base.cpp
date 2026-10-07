#include "train_base.h"

#include <cmath>
#include <cstring>

#include "nlohmann/json.hpp"

namespace McBopomofo {
namespace Trainer {

namespace {

constexpr char kMagic[] = "AFMRTB01";
constexpr uint32_t kMaxDim = 1u << 20;

class Cursor {
 public:
  Cursor(const std::string& b, size_t end) : b_(b), end_(end) {}
  bool take(void* dst, size_t n) {
    if (n > end_ - pos_) return false;
    std::memcpy(dst, b_.data() + pos_, n);
    pos_ += n;
    return true;
  }
  bool u32(uint32_t* v) { return take(v, 4); }
  bool str(std::string* s, uint32_t maxLen) {
    uint32_t n;
    if (!u32(&n) || n > maxLen || n > end_ - pos_) return false;
    s->assign(b_.data() + pos_, n);
    pos_ += n;
    return true;
  }
  size_t pos() const { return pos_; }

 private:
  const std::string& b_;
  size_t end_;
  size_t pos_ = 0;
};

void PutU32(std::string* s, uint32_t v) { s->append(reinterpret_cast<const char*>(&v), 4); }
void PutStr(std::string* s, const std::string& v) {
  PutU32(s, static_cast<uint32_t>(v.size()));
  s->append(v);
}

}  // namespace

const BaseTensor* TrainBase::find(const std::string& name) const {
  auto it = byName.find(name);
  return it == byName.end() ? nullptr : &tensors[it->second];
}

std::shared_ptr<const TrainBase> ParseTrainBase(const std::string& bytes, std::string* error) {
  auto fail = [&](const std::string& m) {
    if (error) *error = "train-base.bin: " + m;
    return nullptr;
  };
  if (bytes.size() < 8 + 32 + 4 + 4 + 32) return fail("too short");
  const size_t body = bytes.size() - 32;
  auto trailer = Personalize::Sha256(bytes.data(), body);
  if (std::memcmp(trailer.data(), bytes.data() + body, 32) != 0) return fail("trailer hash mismatch");
  if (std::memcmp(bytes.data(), kMagic, 8) != 0) return fail("bad magic");
  Cursor c(bytes, body);
  char magic[8];
  c.take(magic, 8);
  auto base = std::make_shared<TrainBase>();
  if (!c.take(base->factorySha.data(), 32)) return fail("truncated");
  std::string cfgText;
  if (!c.str(&cfgText, 1u << 16)) return fail("bad config");
  auto cfg = nlohmann::json::parse(cfgText, nullptr, false);
  if (cfg.is_discarded() || !cfg.is_object()) return fail("bad config json");
  auto geti = [&](const char* k, int* v) {
    if (!cfg.contains(k) || !cfg[k].is_number_integer()) return false;
    *v = cfg[k].get<int>();
    return *v > 0 && *v < (1 << 20);
  };
  ModelConfig& m = base->config;
  if (!geti("n_syl", &m.nSyl) || !geti("n_char", &m.nChar) || !geti("dim", &m.dim) ||
      !geti("depth", &m.depth) || !geti("heads", &m.heads) || !geti("kv", &m.kv) ||
      !geti("ffn", &m.ffn))
    return fail("incomplete config");
  if (m.dim % m.heads != 0 || m.heads % m.kv != 0) return fail("inconsistent config");
  uint32_t count;
  if (!c.u32(&count) || count == 0 || count > 10000) return fail("bad tensor count");
  for (uint32_t i = 0; i < count; ++i) {
    BaseTensor t;
    uint8_t kind;
    uint32_t ndim;
    if (!c.str(&t.name, 256) || t.name.empty() || !c.take(&kind, 1) || !c.u32(&ndim) || ndim == 0 ||
        ndim > 4)
      return fail("bad tensor header");
    if (kind > 2) return fail("bad kind for " + t.name);
    t.kind = static_cast<TensorKind>(kind);
    size_t n = 1;
    for (uint32_t d = 0; d < ndim; ++d) {
      uint32_t v;
      if (!c.u32(&v) || v == 0 || v > kMaxDim) return fail("bad dims for " + t.name);
      t.dims.push_back(v);
      n *= v;
      if (n > (1ull << 31)) return fail("tensor too large");
    }
    if (t.kind == TensorKind::kFloat) {
      t.data.resize(n);
      if (!c.take(t.data.data(), n * 4)) return fail("truncated " + t.name);
      for (float f : t.data)
        if (!std::isfinite(f)) return fail("non-finite value in " + t.name);
    } else if (t.kind == TensorKind::kLinearInt8) {
      if (ndim != 2) return fail("linear must be 2-D: " + t.name);
      auto& q = t.linear;
      if (!c.str(&q.name, 256) || q.name.empty()) return fail("bad onnx name for " + t.name);
      q.out = t.dims[0];
      q.in = t.dims[1];
      q.scales.resize(q.out);
      q.values.resize(static_cast<size_t>(q.in) * q.out);
      if (!c.take(q.scales.data(), q.out * 4) || !c.take(q.values.data(), q.values.size()))
        return fail("truncated " + t.name);
      for (float f : q.scales)
        if (!std::isfinite(f) || f < 0) return fail("bad scale in " + t.name);
    } else {
      if (!c.take(&t.scale, 4) || !c.take(&t.zeroPoint, 1)) return fail("truncated " + t.name);
      if (!std::isfinite(t.scale) || t.scale <= 0) return fail("bad scale in " + t.name);
      t.u8.resize(n);
      if (!c.take(t.u8.data(), n)) return fail("truncated " + t.name);
    }
    if (base->byName.count(t.name)) return fail("duplicate tensor " + t.name);
    base->byName[t.name] = base->tensors.size();
    base->tensors.push_back(std::move(t));
  }
  if (c.pos() != body) return fail("trailing bytes");
  return base;
}

std::shared_ptr<const TrainBase> ReadTrainBase(const std::string& path, std::string* error) {
  std::string bytes;
  if (!Personalize::ReadFileBytes(path, &bytes, error)) return nullptr;
  return ParseTrainBase(bytes, error);
}

std::string SerializeTrainBase(const TrainBase& base) {
  std::string s(kMagic, 8);
  s.append(reinterpret_cast<const char*>(base.factorySha.data()), 32);
  const ModelConfig& m = base.config;
  nlohmann::json cfg = {{"n_syl", m.nSyl}, {"n_char", m.nChar}, {"dim", m.dim}, {"depth", m.depth},
                        {"heads", m.heads}, {"kv", m.kv},       {"ffn", m.ffn}};
  PutStr(&s, cfg.dump());
  PutU32(&s, static_cast<uint32_t>(base.tensors.size()));
  for (const auto& t : base.tensors) {
    PutStr(&s, t.name);
    s.push_back(static_cast<char>(t.kind));
    PutU32(&s, static_cast<uint32_t>(t.dims.size()));
    for (uint32_t d : t.dims) PutU32(&s, d);
    if (t.kind == TensorKind::kFloat) {
      s.append(reinterpret_cast<const char*>(t.data.data()), t.data.size() * 4);
    } else if (t.kind == TensorKind::kLinearInt8) {
      PutStr(&s, t.linear.name);
      s.append(reinterpret_cast<const char*>(t.linear.scales.data()), t.linear.scales.size() * 4);
      s.append(reinterpret_cast<const char*>(t.linear.values.data()), t.linear.values.size());
    } else {
      s.append(reinterpret_cast<const char*>(&t.scale), 4);
      s.push_back(static_cast<char>(t.zeroPoint));
      s.append(reinterpret_cast<const char*>(t.u8.data()), t.u8.size());
    }
  }
  auto h = Personalize::Sha256(s.data(), s.size());
  s.append(reinterpret_cast<const char*>(h.data()), 32);
  return s;
}

std::vector<float> DequantizedTorchLayout(const BaseTensor& t) {
  if (t.kind == TensorKind::kFloat) return t.data;
  if (t.kind == TensorKind::kUint8Zp) {
    std::vector<float> v(t.u8.size());
    for (size_t i = 0; i < v.size(); ++i)
      v[i] = (static_cast<float>(t.u8[i]) - static_cast<float>(t.zeroPoint)) * t.scale;
    return v;
  }
  const auto& q = t.linear;  // ONNX [in, out] -> torch [out, in]
  std::vector<float> v(static_cast<size_t>(q.in) * q.out);
  for (uint32_t i = 0; i < q.in; ++i)
    for (uint32_t j = 0; j < q.out; ++j)
      v[static_cast<size_t>(j) * q.in + i] =
          static_cast<float>(q.values[static_cast<size_t>(i) * q.out + j]) * q.scales[j];
  return v;
}

}  // namespace Trainer
}  // namespace McBopomofo
