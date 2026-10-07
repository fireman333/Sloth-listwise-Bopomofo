#include "precise_walk.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>

#include "sloth_rescorer.h"

namespace McBopomofo {

using Formosa::Gramambular2::ReadingGrid;
using Clock = std::chrono::steady_clock;

namespace {

double MsSince(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

std::string Trim(const std::string& s) {
  size_t b = s.find_first_not_of(" \t\r\n");
  if (b == std::string::npos) return "";
  size_t e = s.find_last_not_of(" \t\r\n");
  return s.substr(b, e - b + 1);
}

template <typename T>
bool ParseList(const std::string& v, std::vector<T>* out) {
  out->clear();
  std::istringstream ss(v);
  std::string tok;
  while (ss >> tok) {
    std::istringstream ts(tok);
    T x;
    if (!(ts >> x) || !ts.eof()) return false;
    out->push_back(x);
  }
  return true;
}

template <typename T>
bool ParseOne(const std::string& v, T* out) {
  std::vector<T> l;
  if (!ParseList(v, &l) || l.size() != 1) return false;
  *out = l[0];
  return true;
}

std::vector<double> LogSoftmax(const std::vector<double>& x) {
  std::vector<double> out(x.size());
  if (x.empty()) return out;
  double m = *std::max_element(x.begin(), x.end());
  double s = 0;
  for (double v : x) s += std::exp(v - m);
  double lse = m + std::log(s);
  for (size_t i = 0; i < x.size(); ++i) out[i] = x[i] - lse;
  return out;
}

std::string Join(const ReadingGrid::WalkResult& r) {
  std::string s;
  for (const auto& n : r.nodes) s += n->value();
  return s;
}

// UTF-8 code points of s (invalid lead bytes are taken as single bytes).
void SplitCodePoints(const std::string& s, std::vector<std::string>* out) {
  out->clear();
  for (size_t i = 0; i < s.size();) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    size_t n = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 1;
    if (i + n > s.size()) n = s.size() - i;
    out->push_back(s.substr(i, n));
    i += n;
  }
}

}  // namespace

bool PreciseConfig::valid() const {
  if (temperatures.size() != bucketUpper.size() + 1) return false;
  for (double t : temperatures) {
    if (!(t > 0) || !std::isfinite(t)) return false;
  }
  for (size_t i = 1; i < bucketUpper.size(); ++i) {
    if (bucketUpper[i] <= bucketUpper[i - 1]) return false;
  }
  return k > 0 && std::isfinite(alpha) && std::isfinite(tau) &&
         std::isfinite(rWeight) && std::isfinite(beta);
}

double PreciseConfig::temperatureFor(size_t nSyllables) const {
  if (temperatures.empty()) return 1.0;
  for (size_t i = 0; i < bucketUpper.size() && i < temperatures.size(); ++i) {
    if (static_cast<long>(nSyllables) <= bucketUpper[i]) return temperatures[i];
  }
  return temperatures.back();
}

bool PreciseConfig::LoadFile(const std::string& path, PreciseConfig* out,
                             std::string* error) {
  auto fail = [&](const std::string& msg) {
    if (error != nullptr) *error = msg;
    return false;
  };
  std::ifstream f(path);
  if (!f) return fail("cannot open " + path);
  PreciseConfig c;
  std::string line;
  int lineNo = 0;
  while (std::getline(f, line)) {
    ++lineNo;
    size_t hash = line.find('#');
    if (hash != std::string::npos) line = line.substr(0, hash);
    line = Trim(line);
    if (line.empty()) continue;
    size_t eq = line.find('=');
    if (eq == std::string::npos) return fail("line " + std::to_string(lineNo) + ": no '='");
    std::string key = Trim(line.substr(0, eq));
    std::string val = Trim(line.substr(eq + 1));
    bool ok = true;
    if (key == "alpha") ok = ParseOne(val, &c.alpha);
    else if (key == "k") ok = ParseOne(val, &c.k);
    else if (key == "bucket_upper") ok = ParseList(val, &c.bucketUpper);
    else if (key == "temperatures") ok = ParseList(val, &c.temperatures);
    else if (key == "tau") ok = ParseOne(val, &c.tau);
    else if (key == "r_weight") ok = ParseOne(val, &c.rWeight);
    else if (key == "beta") ok = ParseOne(val, &c.beta);
    else if (key == "qwen_timeout_ms") ok = ParseOne(val, &c.qwenTimeoutMs);
    else return fail("line " + std::to_string(lineNo) + ": unknown key " + key);
    if (!ok) return fail("line " + std::to_string(lineNo) + ": bad value for " + key);
  }
  if (!c.valid()) return fail("inconsistent config (bucket/temperature sizes, k, values)");
  *out = c;
  return true;
}

std::vector<double> CalibratedLogProbs(const std::vector<double>& scores,
                                       double temperature) {
  if (!(temperature > 0) || !std::isfinite(temperature)) temperature = 1.0;
  std::vector<double> x(scores.size());
  for (size_t i = 0; i < scores.size(); ++i) x[i] = scores[i] / temperature;
  return LogSoftmax(x);
}

bool ShouldCallHeavy(double pTop1, double tau) {
  if (tau >= 1.0) return true;
  return pTop1 < tau;
}

PreciseDecision PreciseSelect(const std::vector<std::string>& readings,
                              const std::vector<std::string>& candidates,
                              const std::vector<double>& rScores,
                              Rescorer* heavy, const PreciseConfig& config) {
  PreciseDecision d;
  if (rScores.empty() || candidates.size() != rScores.size()) return d;
  d.logPR = CalibratedLogProbs(rScores, config.temperatureFor(readings.size()));
  d.pTop1 = std::exp(d.logPR[0]);
  d.gateOpen = ShouldCallHeavy(d.pTop1, config.tau);
  if (!d.gateOpen || heavy == nullptr || !heavy->ready() || candidates.size() < 2) {
    return d;
  }
  RescoreRequest req;
  req.readings = readings;
  req.candidates = candidates;
  d.heavyCalled = true;
  auto t0 = Clock::now();
  std::vector<double> q = heavy->score(req);
  d.heavyMs = MsSince(t0);
  if (d.heavyMs > config.qwenTimeoutMs) {
    d.heavyTimedOut = true;
    return d;
  }
  if (q.size() != candidates.size()) return d;
  for (double v : q) {
    if (!std::isfinite(v)) return d;
  }
  d.heavyScores = q;
  std::vector<double> logPQ = LogSoftmax(q);
  d.fused.resize(q.size());
  size_t best = 0;
  for (size_t i = 0; i < q.size(); ++i) {
    d.fused[i] = config.rWeight * d.logPR[i] + config.beta * logPQ[i];
    if (d.fused[i] > d.fused[best]) best = i;  // ties keep the lower R rank
  }
  d.heavyOk = true;
  d.chosen = best;
  return d;
}

PreciseCandidateSet PreciseCandidates(
    ReadingGrid& grid, const ReadingGrid::PositionalCharScores& scores,
    const PreciseConfig& config) {
  PreciseCandidateSet set;
  if (grid.length() == 0 || config.k == 0 || scores.table.size() != grid.length()) {
    return set;
  }
  set.paths = grid.walkNBestWithCharBonus(config.k, config.alpha, scores);
  for (const auto& w : set.paths) {
    set.texts.push_back(Join(w));
    set.rScores.push_back(w.score);
  }
  return set;
}

std::vector<FixedPosition> FixedPositionsOf(const ReadingGrid& grid) {
  std::vector<FixedPosition> out;
  const auto& spans = grid.spans();
  for (size_t loc = 0; loc < spans.size(); ++loc) {
    const auto& span = spans[loc];
    for (size_t len = 1; len <= span.maxLength(); ++len) {
      const auto& node = span.nodeOf(len);
      if (node != nullptr && node->isOverridden()) {
        out.push_back({loc, node->spanningLength(), node->value()});
      }
    }
  }
  return out;
}

bool KeepsFixedPositions(const ReadingGrid::WalkResult& path,
                         const std::vector<FixedPosition>& fixed) {
  if (fixed.empty()) return true;
  // Per reading position: the code point the path places there, or nullopt
  // if the covering node cannot be split one code point per reading.
  // Also remember node boundaries for the exact-span check.
  std::vector<std::optional<std::string>> perPos;
  std::vector<std::pair<size_t, const ReadingGrid::NodePtr*>> starts;  // (pos, node)
  std::vector<std::string> cps;
  for (const auto& node : path.nodes) {
    const size_t len = node->spanningLength();
    starts.emplace_back(perPos.size(), &node);
    SplitCodePoints(node->value(), &cps);
    const bool aligned = cps.size() == len;
    for (size_t j = 0; j < len; ++j) {
      perPos.push_back(aligned ? std::optional<std::string>(cps[j]) : std::nullopt);
    }
  }
  for (const auto& f : fixed) {
    if (f.length == 0 || f.location + f.length > perPos.size()) return false;
    bool exact = false;
    for (const auto& [pos, nodePtr] : starts) {
      if (pos == f.location && (*nodePtr)->spanningLength() == f.length) {
        exact = (*nodePtr)->value() == f.value;
        if (!exact) return false;
        break;
      }
    }
    if (exact) continue;
    SplitCodePoints(f.value, &cps);
    if (cps.size() != f.length) return false;
    for (size_t j = 0; j < f.length; ++j) {
      const auto& c = perPos[f.location + j];
      if (!c.has_value() || *c != cps[j]) return false;
    }
  }
  return true;
}

void FilterFixedPositions(PreciseCandidateSet* set,
                          const std::vector<FixedPosition>& fixed) {
  if (set == nullptr || fixed.empty()) return;
  PreciseCandidateSet kept;
  for (size_t i = 0; i < set->size(); ++i) {
    if (!KeepsFixedPositions(set->paths[i], fixed)) continue;
    kept.paths.push_back(std::move(set->paths[i]));
    kept.texts.push_back(std::move(set->texts[i]));
    kept.rScores.push_back(set->rScores[i]);
  }
  *set = std::move(kept);
}

size_t MoveDisplayedToFront(PreciseCandidateSet* set, const std::string& displayed) {
  if (set == nullptr) return std::string::npos;
  auto it = std::find(set->texts.begin(), set->texts.end(), displayed);
  if (it == set->texts.end()) return std::string::npos;
  const size_t idx = static_cast<size_t>(std::distance(set->texts.begin(), it));
  if (idx > 0) {
    std::rotate(set->paths.begin(), set->paths.begin() + idx, set->paths.begin() + idx + 1);
    std::rotate(set->texts.begin(), set->texts.begin() + idx, set->texts.begin() + idx + 1);
    std::rotate(set->rScores.begin(), set->rScores.begin() + idx, set->rScores.begin() + idx + 1);
  }
  return idx;
}

std::optional<PreciseDecision> PreciseGate(const std::vector<std::string>& readings,
                                           const PreciseCandidateSet& set,
                                           const PreciseConfig& config) {
  if (set.size() < 2) return std::nullopt;
  return PreciseSelect(readings, set.texts, set.rScores, nullptr, config);
}

PreciseWalkResult PreciseWalk(ReadingGrid& grid, SlothRescorer* rModel,
                              Rescorer* heavy, const PreciseConfig& config) {
  PreciseWalkResult res;
  auto tStart = Clock::now();
  auto plain = [&]() {
    res.walk = grid.walk();
    res.candidates = {Join(res.walk)};
    res.totalMs = MsSince(tStart);
    return res;
  };
  if (rModel == nullptr || !rModel->ready() || grid.length() == 0 || config.k == 0) {
    return plain();
  }
  auto t0 = Clock::now();
  ReadingGrid::PositionalCharScores scores;
  std::vector<std::vector<std::string>> extra = grid.charsAtPositions();
  scores.table = rModel->charLogProbs(grid.readings(), &extra);
  res.forwardMs = MsSince(t0);
  if (scores.table.size() != grid.length()) return plain();

  t0 = Clock::now();
  PreciseCandidateSet nb = PreciseCandidates(grid, scores, config);
  res.nbestMs = MsSince(t0);
  if (nb.empty()) return plain();
  res.usedNeural = true;
  res.candidates = std::move(nb.texts);
  res.rScores = std::move(nb.rScores);
  res.decision = PreciseSelect(grid.readings(), res.candidates, res.rScores, heavy, config);
  res.walk = nb.paths[std::min(res.decision.chosen, nb.paths.size() - 1)];
  res.totalMs = MsSince(tStart);
  return res;
}

}  // namespace McBopomofo
