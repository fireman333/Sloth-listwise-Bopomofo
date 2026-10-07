#include "gate.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "nlohmann/json.hpp"

namespace McBopomofo {
namespace Personalize {

namespace {

// P(X <= k) for X ~ binomial(n, 1/2), k <= n / 2 not required. The largest
// term C(n, k) 2^-n is built as a product with a separately tracked binary
// exponent (no overflow / underflow before the final ldexp), then the smaller
// terms follow from the ratio C(n, i-1) / C(n, i) = i / (n - i + 1) and are
// summed smallest first. Relative error ~ (3k) ulp.
double BinomialLowerTail(uint64_t k, uint64_t n) {
  if (k >= n) return 1.0;
  double m = 1.0;
  long e = 0;
  for (uint64_t j = 0; j < k; ++j) {
    m *= static_cast<double>(n - j) / static_cast<double>(k - j);
    int ex = 0;
    m = std::frexp(m, &ex);
    e += ex;
  }
  // terms relative to the top term: r_k = 1, r_{i-1} = r_i * i / (n - i + 1)
  std::vector<double> r;
  r.reserve(static_cast<size_t>(k) + 1);
  double cur = 1.0;
  r.push_back(cur);
  for (uint64_t i = k; i >= 1; --i) {
    cur *= static_cast<double>(i) / static_cast<double>(n - i + 1);
    if (cur == 0) break;
    r.push_back(cur);
  }
  double sum = 0;
  for (auto it = r.rbegin(); it != r.rend(); ++it) sum += *it;
  // top term = m * 2^(e - n)
  double top = std::ldexp(m, static_cast<int>(e - static_cast<long>(n)));
  return top * sum;
}

bool Pct(const std::vector<bool>& v, double* out) {
  if (v.empty()) return false;
  size_t c = 0;
  for (bool b : v) c += b;
  *out = 100.0 * static_cast<double>(c) / static_cast<double>(v.size());
  return true;
}

}  // namespace

double McNemarExactTwoSided(uint64_t b, uint64_t c) {
  const uint64_t n = b + c;
  if (n == 0) return 1.0;
  const double p = BinomialLowerTail(std::min(b, c), n);
  return std::min(1.0, 2.0 * p);
}

double McNemarExactOneSided(uint64_t wins, uint64_t losses) {
  const uint64_t n = wins + losses;
  if (n == 0) return 1.0;
  // P(X >= wins) = P(n - X <= losses), and n - X has the same distribution.
  return std::min(1.0, BinomialLowerTail(losses, n));
}

bool ParsePackThresholds(const std::string& text, GateThresholds* gate, DataThresholds* data,
                         std::string* error) {
  using nlohmann::json;
  auto fail = [&](const std::string& m) {
    if (error) *error = m;
    return false;
  };
  json j = json::parse(text, nullptr, false);
  if (j.is_discarded() || !j.is_object()) return fail("pack.json is not a JSON object");
  GateThresholds g = gate ? *gate : GateThresholds();
  DataThresholds d = data ? *data : DataThresholds();
  auto num = [&](const json& o, const char* key, double* v, double lo, double hi) {
    auto it = o.find(key);
    if (it == o.end()) return true;
    if (!it->is_number()) return fail(std::string("not a number: ") + key);
    double x = it->get<double>();
    if (!std::isfinite(x) || x < lo || x > hi) return fail(std::string("out of range: ") + key);
    *v = x;
    return true;
  };
  auto count = [&](const json& o, const char* key, size_t* v, size_t lo, size_t hi) {
    auto it = o.find(key);
    if (it == o.end()) return true;
    if (!it->is_number_unsigned() && !(it->is_number_integer() && it->get<int64_t>() >= 0))
      return fail(std::string("not a non-negative integer: ") + key);
    uint64_t x = it->get<uint64_t>();
    if (x < lo || x > hi) return fail(std::string("out of range: ") + key);
    *v = static_cast<size_t>(x);
    return true;
  };
  auto flag = [&](const json& o, const char* key, bool* v) {
    auto it = o.find(key);
    if (it == o.end()) return true;
    if (!it->is_boolean()) return fail(std::string("not a boolean: ") + key);
    *v = it->get<bool>();
    return true;
  };
  if (auto it = j.find("gate"); it != j.end()) {
    if (!it->is_object()) return fail("gate is not an object");
    const json& o = *it;
    if (!num(o, "min_personal_gain_pt", &g.minPersonalGainPt, -100, 100)) return false;
    if (!num(o, "p_max", &g.pMax, 0, 1)) return false;
    if (!num(o, "max_general_drop_pt", &g.maxGeneralDropPt, 0, 100)) return false;
    if (!flag(o, "not_worse_than_active", &g.notWorseThanActive)) return false;
    if (auto m = o.find("mcnemar"); m != o.end()) {
      if (!m->is_string()) return fail("mcnemar is not a string");
      const std::string s = m->get<std::string>();
      if (s == "two-sided") g.oneSided = false;
      else if (s == "one-sided") g.oneSided = true;
      else return fail("mcnemar must be two-sided or one-sided");
    }
  }
  if (auto it = j.find("data"); it != j.end()) {
    if (!it->is_object()) return fail("data is not an object");
    const json& o = *it;
    if (!count(o, "min_eligible", &d.minEligible, 1, 10000000)) return false;
    if (!num(o, "holdout_fraction", &d.holdoutFraction, 0, 0.9)) return false;
    if (!count(o, "holdout_min", &d.holdoutMin, 1, 1000000)) return false;
    if (!count(o, "holdout_max", &d.holdoutMax, 0, 1000000)) return false;
    if (d.holdoutMax != 0 && d.holdoutMax < d.holdoutMin) return fail("holdout_max < holdout_min");
    if (!count(o, "train_cap", &d.trainCap, 0, 10000000)) return false;
    if (!count(o, "min_syllables", &d.minSyllables, 1, 1000)) return false;
    if (!count(o, "max_syllables", &d.maxSyllables, 1, 1000)) return false;
    if (d.minSyllables > d.maxSyllables) return fail("min_syllables > max_syllables");
  }
  if (gate) *gate = g;
  if (data) *data = d;
  return true;
}

GateResult EvaluateGate(const GateInput& in, const GateThresholds& th) {
  GateResult r;
  r.reason = "failed:gate";
  const size_t np = in.factoryPersonal.size(), ng = in.factoryGeneral.size();
  if (np == 0 || ng == 0 || in.candidatePersonal.size() != np ||
      in.candidateGeneral.size() != ng || (!in.activePersonal.empty() && in.activePersonal.size() != np))
    return r;
  r.valid = true;
  r.nPersonal = np;
  r.nGeneral = ng;
  Pct(in.factoryPersonal, &r.factoryPersonalPct);
  Pct(in.candidatePersonal, &r.candidatePersonalPct);
  if (!in.activePersonal.empty()) Pct(in.activePersonal, &r.activePersonalPct);
  Pct(in.factoryGeneral, &r.factoryGeneralPct);
  Pct(in.candidateGeneral, &r.candidateGeneralPct);
  r.deltaPersonalPt = r.candidatePersonalPct - r.factoryPersonalPct;
  r.deltaGeneralPt = r.candidateGeneralPct - r.factoryGeneralPct;
  for (size_t i = 0; i < np; ++i) {
    r.wins += in.candidatePersonal[i] && !in.factoryPersonal[i];
    r.losses += !in.candidatePersonal[i] && in.factoryPersonal[i];
  }
  r.p = th.oneSided ? McNemarExactOneSided(r.wins, r.losses)
                    : McNemarExactTwoSided(r.wins, r.losses);
  // Small epsilon so that e.g. exactly -1.0 pt (computed as a float difference)
  // still counts as "no more than 1.0 pt below".
  constexpr double kEps = 1e-9;
  if (r.deltaGeneralPt < -th.maxGeneralDropPt - kEps) r.failedRules.push_back("general_regression");
  if (r.deltaPersonalPt < th.minPersonalGainPt - kEps) r.failedRules.push_back("too_small_gain");
  if (!(r.p < th.pMax)) r.failedRules.push_back("not_significant");
  if (th.notWorseThanActive && !in.activePersonal.empty() &&
      r.candidatePersonalPct < r.activePersonalPct - kEps)
    r.failedRules.push_back("worse_than_active");
  r.adopted = r.failedRules.empty();
  r.reason = r.adopted ? "adopted" : r.failedRules.front();
  return r;
}

}  // namespace Personalize
}  // namespace McBopomofo
