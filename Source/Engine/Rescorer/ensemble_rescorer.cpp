#include "ensemble_rescorer.h"

#include <cmath>
#include <future>

namespace McBopomofo {

EnsembleRescorer::EnsembleRescorer(std::vector<Member> members)
    : EnsembleRescorer(std::move(members), Options()) {}

EnsembleRescorer::EnsembleRescorer(std::vector<Member> members, const Options& options)
    : members_(std::move(members)), options_(options) {}

bool EnsembleRescorer::ready() const {
  for (const auto& m : members_) {
    if (m.rescorer && m.rescorer->ready()) return true;
  }
  return false;
}

std::vector<double> EnsembleRescorer::ZScore(const std::vector<double>& v) {
  std::vector<double> z(v.size(), 0.0);
  if (v.empty()) return z;
  double mean = 0;
  for (double x : v) mean += x;
  mean /= static_cast<double>(v.size());
  double var = 0;
  for (double x : v) var += (x - mean) * (x - mean);
  double sd = std::sqrt(var / static_cast<double>(v.size()));
  if (sd < 1e-9) return z;
  for (size_t i = 0; i < v.size(); ++i) z[i] = (v[i] - mean) / sd;
  return z;
}

std::vector<double> EnsembleRescorer::score(const RescoreRequest& request) {
  lastActive_ = 0;
  lastMask_ = 0;
  const size_t n = request.candidates.size();
  if (n == 0) return {};

  std::vector<std::vector<double>> raw(members_.size());
  auto run = [&](size_t i) {
    const auto& m = members_[i];
    if (!m.rescorer || !m.rescorer->ready()) return;
    try {
      raw[i] = m.rescorer->score(request);
    } catch (...) {
      raw[i].clear();
    }
  };
  if (options_.parallel && members_.size() > 1) {
    std::vector<std::future<void>> fs;
    for (size_t i = 1; i < members_.size(); ++i)
      fs.push_back(std::async(std::launch::async, [&, i] { run(i); }));
    run(0);
    for (auto& f : fs) f.get();
  } else {
    for (size_t i = 0; i < members_.size(); ++i) run(i);
  }

  std::vector<double> total(n, 0.0);
  for (size_t i = 0; i < members_.size(); ++i) {
    if (raw[i].size() != n) continue;
    bool finite = true;
    for (double x : raw[i]) finite = finite && std::isfinite(x);
    if (!finite) continue;
    std::vector<double> z = ZScore(raw[i]);
    for (size_t j = 0; j < n; ++j) total[j] += members_[i].weight * z[j];
    ++lastActive_;
    lastMask_ |= 1u << i;
  }
  if (lastActive_ == 0) return {};
  return total;
}

}  // namespace McBopomofo
