// EnsembleRescorer: combines several Rescorers by within-sentence z-score
// (over the request's candidates) and a weighted sum:
//   score_i = sum_m w_m * z_m(i),  z_m = (x - mean) / population-std  (std < 1e-9 -> 0)
// Mirrors bench/src/ensemble_DE.py (w*zD + (1-w)*zE).
//
// Fallback: a member that is not ready, returns the wrong number of scores or
// non-finite values is skipped for that call; the remaining members are still
// combined. If every member fails, score() returns {} so callers fall back to
// the original ranking. Scores are only comparable within one call.

#ifndef SOURCE_ENGINE_RESCORER_ENSEMBLE_RESCORER_H_
#define SOURCE_ENGINE_RESCORER_ENSEMBLE_RESCORER_H_

#include <memory>
#include <string>
#include <vector>

#include "rescorer.h"

namespace McBopomofo {

class EnsembleRescorer : public Rescorer {
 public:
  struct Member {
    std::shared_ptr<Rescorer> rescorer;
    double weight = 1.0;
    std::string name;
  };
  struct Options {
    // Run members on separate threads (latency = max instead of sum).
    bool parallel = false;
  };

  explicit EnsembleRescorer(std::vector<Member> members);
  EnsembleRescorer(std::vector<Member> members, const Options& options);

  // True if at least one member is ready.
  bool ready() const override;
  std::vector<double> score(const RescoreRequest& request) override;

  // Number of members that contributed to the most recent score() call
  // (0 after a total failure). Not thread-safe with concurrent score() calls.
  size_t lastActiveMembers() const { return lastActive_; }
  // Bit i set = member i contributed to the most recent score() call.
  unsigned lastActiveMask() const { return lastMask_; }

  static std::vector<double> ZScore(const std::vector<double>& v);

 private:
  std::vector<Member> members_;
  Options options_;
  size_t lastActive_ = 0;
  unsigned lastMask_ = 0;
};

}  // namespace McBopomofo

#endif  // SOURCE_ENGINE_RESCORER_ENSEMBLE_RESCORER_H_
