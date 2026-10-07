#include "temperature.h"

#include <algorithm>
#include <cmath>

namespace McBopomofo {
namespace Personalize {

const std::vector<double>& LogTGrid() {
  static const std::vector<double> grid = [] {
    const double a = std::log(0.02), b = std::log(50.0);
    const int n = 1201;
    const double step = (b - a) / (n - 1);
    std::vector<double> g(n);
    for (int i = 0; i < n; ++i) g[i] = i * step + a;  // numpy: arange * step + start
    g[n - 1] = b;
    return g;
  }();
  return grid;
}

double PTop1(const std::vector<double>& r, double T, size_t K) {
  const size_t k = std::min(K, r.size());
  if (k == 0) return 1.0;
  double m = -INFINITY;
  for (size_t j = 0; j < k; ++j) m = std::max(m, r[j] / T);
  double s = 0;
  for (size_t j = 0; j < k; ++j) s += std::exp(r[j] / T - m);
  return std::exp(r[0] / T - (m + std::log(s)));
}

double LogLoss(const std::vector<double>& ps, const std::vector<bool>& ys) {
  if (ps.empty()) return 0;
  double t = 0;
  for (size_t i = 0; i < ps.size(); ++i) {
    const double p = std::min(1 - 1e-12, std::max(1e-12, ps[i]));
    t += ys[i] ? -std::log(p) : -std::log(1 - p);
  }
  return t / static_cast<double>(ps.size());
}

double Ece(const std::vector<double>& ps, const std::vector<bool>& ys, int bins) {
  if (ps.empty()) return 0;
  double tot = 0;
  for (int b = 0; b < bins; ++b) {
    const double lo = static_cast<double>(b) / bins, hi = static_cast<double>(b + 1) / bins;
    double conf = 0, acc = 0;
    size_t n = 0;
    for (size_t i = 0; i < ps.size(); ++i) {
      const bool in = ps[i] >= lo && (b < bins - 1 ? ps[i] < hi : ps[i] <= hi);
      if (!in) continue;
      ++n;
      conf += ps[i];
      acc += ys[i] ? 1.0 : 0.0;
    }
    if (n == 0) continue;
    tot += static_cast<double>(n) / static_cast<double>(ps.size()) *
           std::fabs(conf / static_cast<double>(n) - acc / static_cast<double>(n));
  }
  return tot;
}

double FitTemperature(const std::vector<CalibRow>& rows, size_t K) {
  if (rows.empty()) return 1.0;
  const auto& grid = LogTGrid();
  double bestLoss = INFINITY;
  size_t best = 0;
  for (size_t g = 0; g < grid.size(); ++g) {
    const double T = std::exp(grid[g]);
    double loss = 0;
    for (const auto& r : rows) {
      const size_t k = std::min(K, r.rScores.size());
      double s = 0;
      for (size_t j = 0; j < k; ++j) s += std::exp((r.rScores[j] - r.rScores[0]) / T);
      double p = s > 0 ? 1.0 / s : 1.0;
      p = std::min(1 - 1e-12, std::max(1e-12, p));
      loss += r.top1Correct ? -std::log(p) : -std::log(1 - p);
    }
    loss /= static_cast<double>(rows.size());
    if (loss < bestLoss) {  // first minimum, as np.argmin
      bestLoss = loss;
      best = g;
    }
  }
  return std::exp(grid[best]);
}

size_t BucketOf(size_t n, const std::vector<size_t>& upper) {
  for (size_t i = 0; i < upper.size(); ++i)
    if (n <= upper[i]) return i;
  return upper.size();
}

std::vector<double> FitTemperatures(const std::vector<CalibRow>& rows, size_t K,
                                    const std::vector<size_t>& upper, bool perBucket) {
  const size_t nb = upper.size() + 1;
  if (!perBucket) return std::vector<double>(nb, FitTemperature(rows, K));
  std::vector<double> t(nb);
  for (size_t b = 0; b < nb; ++b) {
    std::vector<CalibRow> sub;
    for (const auto& r : rows)
      if (BucketOf(r.nSyl, upper) == b) sub.push_back(r);
    t[b] = FitTemperature(sub, K);
  }
  return t;
}

CalibQuality EvaluateCalibration(const std::vector<CalibRow>& rows, size_t K,
                                 const std::vector<size_t>& upper, const std::vector<double>& temps) {
  CalibQuality q;
  std::vector<double> before, after;
  std::vector<bool> ys;
  for (const auto& r : rows) {
    before.push_back(PTop1(r.rScores, 1.0, K));
    const size_t b = BucketOf(r.nSyl, upper);
    after.push_back(PTop1(r.rScores, b < temps.size() ? temps[b] : 1.0, K));
    ys.push_back(r.top1Correct);
  }
  q.eceBefore = Ece(before, ys);
  q.eceAfter = Ece(after, ys);
  q.logLossBefore = LogLoss(before, ys);
  q.logLossAfter = LogLoss(after, ys);
  return q;
}

}  // namespace Personalize
}  // namespace McBopomofo
