#include "onnx_initializers.h"

#include <cstring>

namespace McBopomofo {
namespace Personalize {

namespace {

// Protobuf wire reader over a byte range.
class Pb {
 public:
  explicit Pb(std::string_view s) : s_(s) {}
  bool done() const { return pos_ >= s_.size(); }

  bool Varint(uint64_t* v) {
    uint64_t r = 0;
    for (int shift = 0; shift < 64; shift += 7) {
      if (pos_ >= s_.size()) return false;
      uint8_t b = static_cast<uint8_t>(s_[pos_++]);
      r |= static_cast<uint64_t>(b & 0x7F) << shift;
      if ((b & 0x80) == 0) {
        *v = r;
        return true;
      }
    }
    return false;
  }
  // Reads a tag; false at end or on error (check ok()).
  bool Tag(uint32_t* field, uint32_t* wire) {
    if (done()) return false;
    uint64_t t = 0;
    if (!Varint(&t)) {
      bad_ = true;
      return false;
    }
    *field = static_cast<uint32_t>(t >> 3);
    *wire = static_cast<uint32_t>(t & 7);
    if (*field == 0) {
      bad_ = true;
      return false;
    }
    return true;
  }
  bool Len(std::string_view* out) {
    uint64_t n = 0;
    if (!Varint(&n) || n > s_.size() - pos_) return false;
    *out = s_.substr(pos_, static_cast<size_t>(n));
    pos_ += static_cast<size_t>(n);
    return true;
  }
  bool Fixed32(uint32_t* v) {
    if (s_.size() - pos_ < 4) return false;
    std::memcpy(v, s_.data() + pos_, 4);
    pos_ += 4;
    return true;
  }
  bool Skip(uint32_t wire) {
    uint64_t v;
    std::string_view sv;
    switch (wire) {
      case 0: return Varint(&v);
      case 1:
        if (s_.size() - pos_ < 8) return false;
        pos_ += 8;
        return true;
      case 2: return Len(&sv);
      case 5:
        if (s_.size() - pos_ < 4) return false;
        pos_ += 4;
        return true;
      default: return false;
    }
  }
  bool ok() const { return !bad_; }
  void setBad() { bad_ = true; }

 private:
  std::string_view s_;
  size_t pos_ = 0;
  bool bad_ = false;
};

bool ParseTensor(std::string_view b, OnnxInitializer* t) {
  Pb pb(b);
  uint32_t f, w;
  while (pb.Tag(&f, &w)) {
    if (f == 1) {  // dims (int64, packed or not)
      if (w == 2) {
        std::string_view sv;
        if (!pb.Len(&sv)) return false;
        Pb inner(sv);
        while (!inner.done()) {
          uint64_t v;
          if (!inner.Varint(&v)) return false;
          t->dims.push_back(static_cast<int64_t>(v));
        }
      } else if (w == 0) {
        uint64_t v;
        if (!pb.Varint(&v)) return false;
        t->dims.push_back(static_cast<int64_t>(v));
      } else {
        return false;
      }
    } else if (f == 2 && w == 0) {
      uint64_t v;
      if (!pb.Varint(&v)) return false;
      t->dataType = static_cast<int32_t>(v);
    } else if (f == 4) {  // float_data
      if (w == 2) {
        std::string_view sv;
        if (!pb.Len(&sv) || sv.size() % 4 != 0) return false;
        const size_t n = sv.size() / 4;
        const size_t base = t->floatData.size();
        t->floatData.resize(base + n);
        std::memcpy(t->floatData.data() + base, sv.data(), sv.size());
      } else if (w == 5) {
        uint32_t v;
        if (!pb.Fixed32(&v)) return false;
        float fv;
        std::memcpy(&fv, &v, 4);
        t->floatData.push_back(fv);
      } else {
        return false;
      }
    } else if (f == 8 && w == 2) {
      std::string_view sv;
      if (!pb.Len(&sv)) return false;
      t->name.assign(sv.data(), sv.size());
    } else if (f == 9 && w == 2) {
      if (!pb.Len(&t->raw)) return false;
    } else {
      if (!pb.Skip(w)) return false;
    }
  }
  return pb.ok();
}

bool ParseNode(std::string_view b, std::vector<std::string>* inputs, std::string* opType) {
  Pb pb(b);
  uint32_t f, w;
  while (pb.Tag(&f, &w)) {
    if (f == 1 && w == 2) {
      std::string_view sv;
      if (!pb.Len(&sv)) return false;
      inputs->emplace_back(sv.data(), sv.size());
    } else if (f == 4 && w == 2) {
      std::string_view sv;
      if (!pb.Len(&sv)) return false;
      opType->assign(sv.data(), sv.size());
    } else {
      if (!pb.Skip(w)) return false;
    }
  }
  return pb.ok();
}

bool ParseGraph(std::string_view b, OnnxModelInfo* m) {
  Pb pb(b);
  uint32_t f, w;
  while (pb.Tag(&f, &w)) {
    if (f == 1 && w == 2) {
      std::string_view sv;
      if (!pb.Len(&sv)) return false;
      std::vector<std::string> inputs;
      std::string op;
      if (!ParseNode(sv, &inputs, &op)) return false;
      if (op == "MatMulInteger" && inputs.size() >= 2) m->matMulIntegerWeights.push_back(inputs[1]);
    } else if (f == 5 && w == 2) {
      std::string_view sv;
      if (!pb.Len(&sv)) return false;
      OnnxInitializer t;
      if (!ParseTensor(sv, &t)) return false;
      m->initializers.push_back(std::move(t));
    } else {
      if (!pb.Skip(w)) return false;
    }
  }
  return pb.ok();
}

}  // namespace

const OnnxInitializer* OnnxModelInfo::find(const std::string& name) const {
  auto it = byName.find(name);
  return it == byName.end() ? nullptr : &initializers[it->second];
}

bool ScanOnnxModel(std::string_view bytes, OnnxModelInfo* out, std::string* error) {
  try {
    *out = OnnxModelInfo();
    Pb pb(bytes);
    uint32_t f, w;
    bool sawGraph = false;
    while (pb.Tag(&f, &w)) {
      if (f == 7 && w == 2) {
        std::string_view sv;
        if (!pb.Len(&sv) || !ParseGraph(sv, out)) {
          if (error) *error = "malformed graph";
          return false;
        }
        sawGraph = true;
      } else if (!pb.Skip(w)) {
        if (error) *error = "malformed model";
        return false;
      }
    }
    if (!pb.ok() || !sawGraph) {
      if (error) *error = sawGraph ? "malformed model" : "no graph";
      return false;
    }
    for (size_t i = 0; i < out->initializers.size(); ++i)
      out->byName.emplace(out->initializers[i].name, i);
    return true;
  } catch (...) {
    if (error) *error = "exception";
    return false;
  }
}

bool FactoryLinears(const OnnxModelInfo& m, std::vector<QuantizedLinear>* out, std::string* error,
                    bool withData) {
  static const std::string kQ = "_quantized";
  out->clear();
  auto fail = [&](const std::string& s) {
    if (error) *error = s;
    out->clear();
    return false;
  };
  for (const auto& wname : m.matMulIntegerWeights) {
    if (wname.size() <= kQ.size() || wname.compare(wname.size() - kQ.size(), kQ.size(), kQ) != 0)
      return fail("unexpected MatMulInteger weight " + wname);
    const std::string base = wname.substr(0, wname.size() - kQ.size());
    const OnnxInitializer* q = m.find(wname);
    const OnnxInitializer* sc = m.find(base + "_scale");
    const OnnxInitializer* zp = m.find(base + "_zero_point");
    if (!q || !sc) return fail("missing initializer for " + base);
    if (q->dataType != kOnnxInt8 || q->dims.size() != 2 || q->dims[0] <= 0 || q->dims[1] <= 0)
      return fail("bad quantized tensor " + wname);
    QuantizedLinear t;
    t.name = base;
    t.in = static_cast<uint32_t>(q->dims[0]);
    t.out = static_cast<uint32_t>(q->dims[1]);
    if (q->raw.size() != static_cast<size_t>(t.in) * t.out) return fail("bad data size " + wname);
    if (sc->dataType != kOnnxFloat || sc->dims.size() != 1 || sc->dims[0] != t.out)
      return fail("bad scale tensor for " + base);
    if (!sc->floatData.empty()) {
      if (sc->floatData.size() != t.out) return fail("bad scale size for " + base);
      if (withData) t.scales = sc->floatData;
    } else {
      if (sc->raw.size() != 4u * t.out) return fail("bad scale size for " + base);
      if (withData) {
        t.scales.resize(t.out);
        std::memcpy(t.scales.data(), sc->raw.data(), sc->raw.size());
      }
    }
    if (zp) {
      for (char c : zp->raw)
        if (c != 0) return fail("non-zero zero point for " + base);
    }
    if (withData) {
      t.values.resize(q->raw.size());
      std::memcpy(t.values.data(), q->raw.data(), q->raw.size());
    }
    out->push_back(std::move(t));
  }
  if (out->empty()) return fail("no MatMulInteger weights");
  return true;
}

std::unordered_map<std::string, std::pair<uint32_t, uint32_t>> LinearShapes(
    const std::vector<QuantizedLinear>& linears) {
  std::unordered_map<std::string, std::pair<uint32_t, uint32_t>> m;
  for (const auto& t : linears) m.emplace(t.name, std::make_pair(t.in, t.out));
  return m;
}

}  // namespace Personalize
}  // namespace McBopomofo
