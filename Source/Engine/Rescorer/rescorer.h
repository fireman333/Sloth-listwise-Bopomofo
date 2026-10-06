// Candidate-sentence rescoring interface ("Jev-style" judging).
// Given the Bopomofo readings of the composing buffer and N candidate
// sentences (all sharing the same readings), return one score per candidate;
// higher is better. Scores are only comparable within a single call.

#ifndef SOURCE_ENGINE_RESCORER_RESCORER_H_
#define SOURCE_ENGINE_RESCORER_RESCORER_H_

#include <string>
#include <vector>

namespace McBopomofo {

struct RescoreRequest {
  // One syllable per character, McBopomofo notation (first tone unmarked,
  // neutral tone as suffix "˙"), e.g. {"ㄧ", "ㄕㄥ"}.
  std::vector<std::string> readings;
  // UTF-8 candidate sentences; each has exactly readings.size() code points.
  std::vector<std::string> candidates;
};

class Rescorer {
 public:
  virtual ~Rescorer() = default;
  // True once the model is loaded and ready.
  virtual bool ready() const = 0;
  // Returns candidates.size() scores (log-prob-like). On failure returns an
  // empty vector; callers must fall back to the original ranking.
  virtual std::vector<double> score(const RescoreRequest& request) = 0;
};

}  // namespace McBopomofo

#endif  // SOURCE_ENGINE_RESCORER_RESCORER_H_
