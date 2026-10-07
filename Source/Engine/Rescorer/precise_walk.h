// PreciseWalk ("precise mode"): R-walk N-best + calibrated confidence gate +
// R/Qwen score fusion.
//
//   1. R-walk N-best: one SlothRescorer::charLogProbs() forward, then
//      ReadingGrid::walkNBestWithCharBonus(k, alpha) -> K distinct sentences
//      with bonus-inclusive path scores s_1 >= ... >= s_K (s_1 = R-walk pick).
//   2. Calibration: p_R = softmax(s / T_b) over the K candidates, T_b chosen
//      by the syllable-count bucket b of the buffer (fitted offline,
//      bench/src/precise_calib.py). p_R[0] = calibrated P(R-walk pick correct).
//   3. Gate: call the heavy rescorer (Qwen) only if p_R[0] < tau
//      (tau >= 1 always calls, tau <= 0 never calls).
//   4. Fusion: final_i = rWeight * log p_R[i] + beta * log p_Q[i], where
//      p_Q = softmax(qwen scores) over the same K; argmax (ties -> lower R
//      rank). rWeight = 0, beta = 1 is "Qwen replaces R" on the R N-best.
//
// Fallbacks (never worse-defined than R-walk): no/failed R model -> plain
// walk(); Qwen not ready, empty/wrong-size/non-finite scores, or slower than
// qwenTimeoutMs -> the R-walk pick (candidate 0). The timeout is checked after
// the (synchronous) call returns: a late result is discarded, it does not
// interrupt the call. Asynchronous cancellation belongs to the IME layer.

#ifndef SOURCE_ENGINE_RESCORER_PRECISE_WALK_H_
#define SOURCE_ENGINE_RESCORER_PRECISE_WALK_H_

#include <optional>
#include <string>
#include <vector>

#include "gramambular2/reading_grid.h"
#include "rescorer.h"

namespace McBopomofo {

class SlothRescorer;

struct PreciseConfig {
  double alpha = 0.5;    // R-walk char-bonus weight
  size_t k = 10;         // N-best size
  // Syllable-count buckets: bucket i holds n <= bucketUpper[i] (first match);
  // the last temperature covers everything above the last bound.
  std::vector<int> bucketUpper;
  std::vector<double> temperatures{1.0};  // size == bucketUpper.size() + 1
  double tau = 0.5;        // gate threshold on calibrated p_R[0]
  double rWeight = 1.0;    // weight of log p_R in the fusion
  double beta = 1.0;       // weight of log p_Q in the fusion
  double qwenTimeoutMs = 1000.0;

  // Parses "key = value" lines (# comments). Keys: alpha, k, bucket_upper
  // (space-separated ints), temperatures (space-separated doubles), tau,
  // r_weight, beta, qwen_timeout_ms. Returns false (and leaves *out
  // untouched) on a missing file, unknown key, bad number or inconsistent
  // bucket/temperature sizes.
  static bool LoadFile(const std::string& path, PreciseConfig* out,
                       std::string* error = nullptr);
  bool valid() const;
  double temperatureFor(size_t nSyllables) const;
};

// log softmax(scores / T). Empty in -> empty out. T <= 0 is treated as 1.
std::vector<double> CalibratedLogProbs(const std::vector<double>& scores,
                                       double temperature);

// Gate rule: tau >= 1 -> true; else pTop1 < tau.
bool ShouldCallHeavy(double pTop1, double tau);

struct PreciseDecision {
  size_t chosen = 0;
  double pTop1 = 1.0;          // calibrated p_R[0]
  bool gateOpen = false;       // p_R[0] < tau
  bool heavyCalled = false;
  bool heavyOk = false;        // heavy scores used in the fusion
  bool heavyTimedOut = false;
  double heavyMs = 0;
  std::vector<double> logPR;   // calibrated R log-probs
  std::vector<double> heavyScores;  // raw heavy scores (empty if unused)
  std::vector<double> fused;   // final scores (empty if heavy unused)
};

// Steps 2-4 on precomputed R N-best scores (candidates aligned with
// rScores, best R first). heavy may be nullptr. Pure apart from the heavy
// call; used by PreciseWalk and by tests / offline parity checks.
PreciseDecision PreciseSelect(const std::vector<std::string>& readings,
                              const std::vector<std::string>& candidates,
                              const std::vector<double>& rScores,
                              Rescorer* heavy, const PreciseConfig& config);

// R N-best on a precomputed score table: paths[i] / texts[i] / rScores[i]
// are aligned, best R first (paths[0] has the walkWithCharBonus values, up
// to ties).
struct PreciseCandidateSet {
  std::vector<Formosa::Gramambular2::ReadingGrid::WalkResult> paths;
  std::vector<std::string> texts;     // joined node values
  std::vector<double> rScores;        // bonus-inclusive path scores
  size_t size() const { return texts.size(); }
  bool empty() const { return texts.empty(); }
};

// Step 1 without the forward pass: walkNBestWithCharBonus(config.k,
// config.alpha, scores) on the grid. Returns an empty set for an empty grid,
// k == 0, or scores.table.size() != grid.length() (no plain-walk fallback;
// that is the caller's decision).
PreciseCandidateSet PreciseCandidates(
    Formosa::Gramambular2::ReadingGrid& grid,
    const Formosa::Gramambular2::ReadingGrid::PositionalCharScores& scores,
    const PreciseConfig& config);

// A position range fixed by an override (manual pick or user-override-model
// suggestion): reading positions [location, location + length) show value.
struct FixedPosition {
  size_t location = 0;
  size_t length = 0;
  std::string value;
};

// Every overridden node of the grid (any override type), by location.
std::vector<FixedPosition> FixedPositionsOf(
    const Formosa::Gramambular2::ReadingGrid& grid);

// True if the path shows each fixed value at its positions. A path node that
// spans exactly the fixed range must carry the fixed value; otherwise the
// characters placed at the fixed positions (nodes split one code point per
// reading) must equal the value's code points. A path whose node at a fixed
// position cannot be split per reading (code points != spanning length)
// does not keep that position.
bool KeepsFixedPositions(
    const Formosa::Gramambular2::ReadingGrid::WalkResult& path,
    const std::vector<FixedPosition>& fixed);

// Drops candidates that change a fixed position; relative order is kept.
void FilterFixedPositions(PreciseCandidateSet* set,
                          const std::vector<FixedPosition>& fixed);

// Moves the candidate whose text equals displayed to rank 0, keeping the
// relative order of the others. Returns its original rank, or npos if the
// text is not among the candidates (set unchanged).
size_t MoveDisplayedToFront(PreciseCandidateSet* set,
                            const std::string& displayed);

// Steps 2-3 without the heavy model: the calibrated gate on a (filtered,
// displayed-first) candidate set. std::nullopt when fewer than 2 candidates
// remain (gate not evaluated, no request).
std::optional<PreciseDecision> PreciseGate(
    const std::vector<std::string>& readings, const PreciseCandidateSet& set,
    const PreciseConfig& config);

struct PreciseWalkResult {
  Formosa::Gramambular2::ReadingGrid::WalkResult walk;  // chosen path
  std::vector<std::string> candidates;                 // R N-best texts
  std::vector<double> rScores;                         // R path scores
  PreciseDecision decision;
  bool usedNeural = false;     // false = plain walk() fallback
  double forwardMs = 0;
  double nbestMs = 0;
  double totalMs = 0;
};

PreciseWalkResult PreciseWalk(Formosa::Gramambular2::ReadingGrid& grid,
                              SlothRescorer* rModel, Rescorer* heavy,
                              const PreciseConfig& config);

}  // namespace McBopomofo

#endif  // SOURCE_ENGINE_RESCORER_PRECISE_WALK_H_
