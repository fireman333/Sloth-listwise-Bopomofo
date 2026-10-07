#include "pool_reader.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fstream>  // nlohmann/json needs std::streambuf (Clang modules)
#include <unordered_map>

#include "nlohmann/json.hpp"
#include "personal_weights.h"

namespace McBopomofo {
namespace Personalize {

using nlohmann::json;

namespace {

// Streams the complete lines of [begin, end) of the file to fn (64 KiB reads,
// never the whole file in memory). end = read.endOffset or EOF; when the range
// is longer than read.maxPoolBytes (> 0) only its last maxPoolBytes are read and
// the partial line at the window start is skipped. Lines longer than
// read.maxLineBytes are skipped and counted in *overlong. *afterLast = offset
// just after the last complete line (or the window start), *windowStart = the
// first byte considered.
template <typename F>
bool StreamLines(const std::string& path, uint64_t begin, const ReadOptions& read,
                 uint64_t* afterLast, uint64_t* windowStart, size_t* overlong,
                 std::string* error, F&& fn) {
  int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    if (error) *error = "cannot open " + path;
    return false;
  }
  struct stat st;
  if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) {
    close(fd);
    if (error) *error = "cannot stat " + path;
    return false;
  }
  uint64_t end = static_cast<uint64_t>(st.st_size);
  if (read.endOffset > 0 && read.endOffset < end) end = read.endOffset;
  uint64_t start = begin;
  if (start < end && read.maxPoolBytes > 0 && end - start > read.maxPoolBytes)
    start = end - read.maxPoolBytes;
  *windowStart = start;
  *afterLast = start;
  if (start >= end) {
    close(fd);
    return true;
  }
  // Skip the partial line at a window start that is not a line start.
  bool dropping = false;
  if (start > begin) {
    char prev = 0;
    dropping = pread(fd, &prev, 1, static_cast<off_t>(start - 1)) != 1 || prev != '\n';
  }
  bool tooLong = false;
  std::string line;
  std::vector<char> buf(64 * 1024);
  uint64_t pos = start;
  while (pos < end) {
    size_t want = static_cast<size_t>(std::min<uint64_t>(buf.size(), end - pos));
    ssize_t n = pread(fd, buf.data(), want, static_cast<off_t>(pos));
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) break;
    size_t i = 0;
    while (i < static_cast<size_t>(n)) {
      const char* nl = static_cast<const char*>(memchr(buf.data() + i, '\n', static_cast<size_t>(n) - i));
      size_t stop = nl ? static_cast<size_t>(nl - buf.data()) : static_cast<size_t>(n);
      if (!dropping) {
        if (line.size() + (stop - i) > read.maxLineBytes) {
          dropping = tooLong = true;
          line.clear();
          line.shrink_to_fit();
        } else {
          line.append(buf.data() + i, stop - i);
        }
      }
      if (nl == nullptr) break;
      if (!dropping) fn(std::string_view(line));
      if (tooLong) ++*overlong;
      dropping = tooLong = false;
      line.clear();
      *afterLast = pos + stop + 1;
      i = stop + 1;
    }
    pos += static_cast<uint64_t>(n);
  }
  close(fd);
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

namespace {

struct PoolRecord {
  int64_t tMs;
  std::vector<PoolNode> nodes;
};

// Parses one pool line into out (counts, retractions) and recs.
void ParsePoolLine(std::string_view line, PoolScan* out, std::vector<PoolRecord>* recs) {
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
  PoolRecord rec;
  rec.tMs = ToMs(t->get<double>());
  for (const auto& n : *nodes) {
    if (!n.is_array() || n.size() < 2 || !n[0].is_string() || !n[1].is_string()) {
      ++out->malformedLines;
      return;
    }
    rec.nodes.push_back({n[0].get<std::string>(), n[1].get<std::string>()});
  }
  ++out->records;
  recs->push_back(std::move(rec));
}

void FinishPoolScan(std::vector<PoolRecord>& recs, const Vocab& vocab, const SegmentOptions& opt,
                    PoolScan* out) {
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

}  // namespace

void ScanPoolBytes(std::string_view bytes, const Vocab& vocab, const SegmentOptions& opt,
                   PoolScan* out) {
  *out = PoolScan();
  std::vector<PoolRecord> recs;
  out->scanOffset = ForEachLine(bytes, [&](std::string_view line) { ParsePoolLine(line, out, &recs); });
  FinishPoolScan(recs, vocab, opt, out);
}

bool ScanPool(const std::string& path, const Vocab& vocab, const SegmentOptions& options,
              PoolScan* out, std::string* error, const ReadOptions& read) {
  *out = PoolScan();
  std::vector<PoolRecord> recs;
  if (!StreamLines(path, 0, read, &out->scanOffset, &out->windowStart, &out->overlongLines, error,
                   [&](std::string_view line) { ParsePoolLine(line, out, &recs); }))
    return false;
  FinishPoolScan(recs, vocab, options, out);
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

namespace {

void RetractionLine(std::string_view line, std::unordered_set<int64_t>* out) {
  if (line.find("\"retract\"") == std::string_view::npos) return;
  json j = json::parse(line.begin(), line.end(), nullptr, false);
  if (j.is_discarded() || !j.is_object()) return;
  auto v = j.find("v");
  auto r = j.find("retract");
  if (v == j.end() || !v->is_number_integer() || v->get<int64_t>() != 1) return;
  if (r == j.end() || !r->is_number()) return;
  out->insert(ToMs(r->get<double>()));
}

}  // namespace

void ScanRetractionBytes(std::string_view bytes, std::unordered_set<int64_t>* out) {
  ForEachLine(bytes, [&](std::string_view line) { RetractionLine(line, out); });
}

bool ScanRetractions(const std::string& path, uint64_t offset, std::unordered_set<int64_t>* out,
                     std::string* error, const ReadOptions& read) {
  ReadOptions r = read;
  r.maxPoolBytes = 0;
  r.endOffset = 0;
  uint64_t after = 0, window = 0;
  size_t overlong = 0;
  return StreamLines(path, offset, r, &after, &window, &overlong, error, [&](std::string_view line) { RetractionLine(line, out); });
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
