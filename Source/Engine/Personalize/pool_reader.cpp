#include "pool_reader.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <unordered_map>

#include "nlohmann/json.hpp"
#include "personal_weights.h"

namespace McBopomofo {
namespace Personalize {

using nlohmann::json;

namespace {

bool ReadAll(const std::string& path, uint64_t offset, std::string* out, std::string* error) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    if (error) *error = "cannot open " + path;
    return false;
  }
  in.seekg(0, std::ios::end);
  const std::streamoff size = in.tellg();
  if (size < 0) {
    if (error) *error = "cannot stat " + path;
    return false;
  }
  if (offset >= static_cast<uint64_t>(size)) {
    out->clear();
    return true;
  }
  in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
  out->resize(static_cast<size_t>(size - static_cast<std::streamoff>(offset)));
  in.read(out->data(), static_cast<std::streamsize>(out->size()));
  out->resize(static_cast<size_t>(in.gcount()));
  return true;
}

int64_t ToMs(double t) { return static_cast<int64_t>(std::llround(t * 1000.0)); }

uint32_t DecodeCodePoint(std::string_view cp) {
  const auto* p = reinterpret_cast<const unsigned char*>(cp.data());
  switch (cp.size()) {
    case 1: return p[0];
    case 2: return ((p[0] & 0x1Fu) << 6) | (p[1] & 0x3Fu);
    case 3: return ((p[0] & 0x0Fu) << 12) | ((p[1] & 0x3Fu) << 6) | (p[2] & 0x3Fu);
    case 4:
      return ((p[0] & 0x07u) << 18) | ((p[1] & 0x3Fu) << 12) | ((p[2] & 0x3Fu) << 6) |
             (p[3] & 0x3Fu);
    default: return 0;
  }
}

std::vector<std::string> SplitDash(const std::string& s) {
  std::vector<std::string> out;
  size_t start = 0;
  while (true) {
    size_t d = s.find('-', start);
    out.push_back(s.substr(start, d == std::string::npos ? std::string::npos : d - start));
    if (d == std::string::npos) break;
    start = d + 1;
  }
  return out;
}

// Calls fn(line) for every complete line of bytes; returns the offset after
// the last '\n'.
template <typename F>
uint64_t ForEachLine(std::string_view bytes, F&& fn) {
  size_t start = 0;
  while (start < bytes.size()) {
    size_t nl = bytes.find('\n', start);
    if (nl == std::string_view::npos) break;
    fn(bytes.substr(start, nl - start));
    start = nl + 1;
  }
  return start;
}

}  // namespace

std::vector<std::string> CodePoints(std::string_view s) {
  std::vector<std::string> out;
  size_t i = 0;
  while (i < s.size()) {
    unsigned char c = static_cast<unsigned char>(s[i]);
    size_t n = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 1;
    if (i + n > s.size()) n = 1;
    out.emplace_back(s.substr(i, n));
    i += n;
  }
  return out;
}

bool IsBopomofoSyllable(std::string_view s) {
  if (s.empty()) return false;
  bool letter = false;
  for (const auto& cp : CodePoints(s)) {
    uint32_t c = DecodeCodePoint(cp);
    if ((c >= 0x3105 && c <= 0x312F) || (c >= 0x31A0 && c <= 0x31BF)) {
      letter = true;
    } else if (c == 0x02CA || c == 0x02C7 || c == 0x02CB || c == 0x02D9) {
      // tone marks ˊ ˇ ˋ ˙
    } else {
      return false;
    }
  }
  return letter;
}

bool RecordSegments(const std::vector<PoolNode>& nodes, std::vector<Segment>* out) {
  std::vector<Segment> segs;
  Segment cur;
  auto flush = [&]() {
    if (!cur.readings.empty()) segs.push_back(std::move(cur));
    cur = Segment();
  };
  for (const auto& n : nodes) {
    std::vector<std::string> syl = SplitDash(n.reading);
    bool syllableNode = !n.reading.empty();
    for (const auto& s : syl) syllableNode = syllableNode && IsBopomofoSyllable(s);
    if (!syllableNode) {
      flush();  // punctuation, symbols, spaces: segment boundary
      continue;
    }
    std::vector<std::string> chars = CodePoints(n.raw);
    if (chars.size() != syl.size()) return false;  // whole record ineligible
    for (size_t i = 0; i < syl.size(); ++i) {
      cur.readings.push_back(syl[i]);
      cur.target += chars[i];
    }
  }
  flush();
  for (auto& s : segs) out->push_back(std::move(s));
  return true;
}

void ScanPoolBytes(std::string_view bytes, const Vocab& vocab, const SegmentOptions& opt,
                   PoolScan* out) {
  *out = PoolScan();
  struct Rec {
    int64_t tMs;
    std::vector<PoolNode> nodes;
  };
  std::vector<Rec> recs;
  out->scanOffset = ForEachLine(bytes, [&](std::string_view line) {
    ++out->lines;
    if (line.empty()) return;
    json j = json::parse(line.begin(), line.end(), nullptr, /*allow_exceptions=*/false);
    if (j.is_discarded() || !j.is_object()) {
      ++out->malformedLines;
      return;
    }
    auto v = j.find("v");
    if (v == j.end() || !v->is_number_integer() || v->get<int64_t>() != 1) {
      ++out->malformedLines;
      return;
    }
    auto r = j.find("retract");
    if (r != j.end()) {
      if (r->is_number()) {
        out->retracted.insert(ToMs(r->get<double>()));
      } else {
        ++out->malformedLines;
      }
      return;
    }
    auto t = j.find("t");
    auto manual = j.find("manual");
    auto nodes = j.find("nodes");
    if (t == j.end() || !t->is_number() || manual == j.end() || !manual->is_boolean() ||
        nodes == j.end() || !nodes->is_array()) {
      ++out->malformedLines;
      return;
    }
    Rec rec;
    rec.tMs = ToMs(t->get<double>());
    for (const auto& n : *nodes) {
      if (!n.is_array() || n.size() < 2 || !n[0].is_string() || !n[1].is_string()) {
        ++out->malformedLines;
        return;
      }
      rec.nodes.push_back({n[0].get<std::string>(), n[1].get<std::string>()});
    }
    ++out->records;
    recs.push_back(std::move(rec));
  });

  // Eligible records -> filtered segments (file order).
  std::vector<Segment> all;
  for (auto& rec : recs) {
    if (out->retracted.count(rec.tMs)) {
      ++out->retractedRecords;
      continue;
    }
    std::vector<Segment> segs;
    if (!RecordSegments(rec.nodes, &segs)) {
      ++out->ineligibleRecords;
      continue;
    }
    for (auto& s : segs) {
      bool keep = s.readings.size() >= opt.minSyllables && s.readings.size() <= opt.maxSyllables;
      for (size_t i = 0; keep && i < s.readings.size(); ++i)
        keep = vocab.syllables.count(s.readings[i]) > 0;
      if (keep) {
        for (const auto& c : CodePoints(s.target)) {
          if (!vocab.chars.count(c)) {
            keep = false;
            break;
          }
        }
      }
      if (!keep) {
        ++out->droppedSegments;
        continue;
      }
      s.tMs = rec.tMs;
      all.push_back(std::move(s));
    }
  }
  // Oldest first (stable: file order among equal times), then keep the newest
  // of identical (readings, target) pairs.
  std::stable_sort(all.begin(), all.end(),
                   [](const Segment& a, const Segment& b) { return a.tMs < b.tMs; });
  std::unordered_map<std::string, size_t> last;
  std::vector<std::string> keys(all.size());
  for (size_t i = 0; i < all.size(); ++i) {
    std::string k;
    for (const auto& r : all[i].readings) {
      k += r;
      k += ' ';
    }
    k += '\t';
    k += all[i].target;
    last[k] = i;
    keys[i] = std::move(k);
  }
  for (size_t i = 0; i < all.size(); ++i) {
    if (last[keys[i]] == i) {
      out->segments.push_back(std::move(all[i]));
    } else {
      ++out->duplicateSegments;
    }
  }
}

bool ScanPool(const std::string& path, const Vocab& vocab, const SegmentOptions& options,
              PoolScan* out, std::string* error) {
  std::string bytes;
  if (!ReadAll(path, 0, &bytes, error)) return false;
  ScanPoolBytes(bytes, vocab, options, out);
  return true;
}

size_t HoldoutSize(size_t n, const SplitOptions& o) {
  size_t h = static_cast<size_t>(std::llround(o.holdoutFraction * static_cast<double>(n)));
  h = std::max(h, o.holdoutMin);
  if (o.holdoutMax > 0) h = std::min(h, std::max(o.holdoutMax, o.holdoutMin));
  return std::min(n, h);
}

Split SplitSegments(const std::vector<Segment>& segs, const SplitOptions& o) {
  Split s;
  const size_t nTest = HoldoutSize(segs.size(), o);
  const size_t nTrainAll = segs.size() - nTest;
  const size_t nTrain = o.trainCap > 0 ? std::min(nTrainAll, o.trainCap) : nTrainAll;
  s.train.assign(segs.begin() + static_cast<std::ptrdiff_t>(nTrainAll - nTrain),
                 segs.begin() + static_cast<std::ptrdiff_t>(nTrainAll));
  s.test.assign(segs.begin() + static_cast<std::ptrdiff_t>(nTrainAll), segs.end());
  return s;
}

void ScanRetractionBytes(std::string_view bytes, std::unordered_set<int64_t>* out) {
  ForEachLine(bytes, [&](std::string_view line) {
    if (line.find("\"retract\"") == std::string_view::npos) return;
    json j = json::parse(line.begin(), line.end(), nullptr, false);
    if (j.is_discarded() || !j.is_object()) return;
    auto v = j.find("v");
    auto r = j.find("retract");
    if (v == j.end() || !v->is_number_integer() || v->get<int64_t>() != 1) return;
    if (r == j.end() || !r->is_number()) return;
    out->insert(ToMs(r->get<double>()));
  });
}

bool ScanRetractions(const std::string& path, uint64_t offset, std::unordered_set<int64_t>* out,
                     std::string* error) {
  std::string bytes;
  if (!ReadAll(path, offset, &bytes, error)) return false;
  ScanRetractionBytes(bytes, out);
  return true;
}

bool UsesRetracted(const Split& split, const std::unordered_set<int64_t>& retracted) {
  if (retracted.empty()) return false;
  for (const auto* v : {&split.train, &split.test})
    for (const auto& s : *v)
      if (retracted.count(s.tMs)) return true;
  return false;
}

bool LoadModelVocab(const std::string& dir, Vocab* out, std::string* error) {
  auto load = [&](const std::string& p, json* j) {
    std::string text;
    if (!ReadFileBytes(p, &text, error)) return false;
    *j = json::parse(text, nullptr, false);
    if (j->is_discarded()) {
      if (error) *error = "bad json " + p;
      return false;
    }
    return true;
  };
  *out = Vocab();
  json syl, tok;
  if (!load(dir + "/syl_vocab.json", &syl) || !load(dir + "/tokenizer/tokenizer.json", &tok))
    return false;
  if (!syl.is_object()) {
    if (error) *error = "bad syl_vocab.json";
    return false;
  }
  for (auto it = syl.begin(); it != syl.end(); ++it) out->syllables.insert(it.key());
  if (tok.contains("model") && tok["model"].contains("vocab") && tok["model"]["vocab"].is_object()) {
    const auto& v = tok["model"]["vocab"];
    for (auto it = v.begin(); it != v.end(); ++it) out->chars.insert(it.key());
  }
  if (tok.contains("added_tokens") && tok["added_tokens"].is_array()) {
    for (const auto& t : tok["added_tokens"])
      if (t.contains("content") && t["content"].is_string())
        out->chars.insert(t["content"].get<std::string>());
  }
  if (out->syllables.empty() || out->chars.empty()) {
    if (error) *error = "empty vocabulary";
    return false;
  }
  return true;
}

}  // namespace Personalize
}  // namespace McBopomofo
