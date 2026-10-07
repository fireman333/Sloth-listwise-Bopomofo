// NeuralWalk ("R-walk"): McBopomofo walk with SlothLM-E per-character scores
// embedded in the node scores, instead of re-ranking a top-k list.
//
//   node score = max over its unigrams u of
//                ngram(u) + alpha * sum_j log p(u_j | all readings, position i+j)
//
// log p comes from one SlothRescorer::charLogProbs() forward pass, normalised
// over the characters legal for the syllable plus every character the grid
// can place at that position (port of train/ranker/eval_ranker.py
// Walker.best + inwalk_tables). Characters the model cannot score get
// missingLogProb (-30, as in the Python reference). Nodes overridden by the
// user keep their stock score (ReadingGrid::walkWithCharBonus). Any failure
// (no model, load failure, empty table, alpha == 0) falls back to the plain
// walk(), so the result is never worse-defined than stock McBopomofo.

#ifndef SOURCE_ENGINE_RESCORER_NEURAL_WALK_H_
#define SOURCE_ENGINE_RESCORER_NEURAL_WALK_H_

#include <chrono>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "gramambular2/reading_grid.h"
#include "sloth_rescorer.h"

namespace McBopomofo {

struct NeuralWalkOptions {
  double alpha = 1.0;
  double missingLogProb = -30.0;
  // Add the grid's characters to each position's normaliser (Python parity).
  // false = normalise over the syllable's legal characters only.
  bool normaliseOverGridChars = true;
};

struct NeuralWalkStats {
  bool usedNeural = false;
  double forwardMs = 0;  // charsAtPositions + forward + table
  double walkMs = 0;
};

// Injectable scorer: (readings, optional per-position extra chars) -> table.
// Production wraps SlothRescorer::charLogProbs; tests pass a fake.
using NeuralScorer = std::function<SlothRescorer::CharLogProbTable(
    const std::vector<std::string>&, const std::vector<std::vector<std::string>>*)>;

// Characters the grid can place at each position (ReadingGrid::charsAtPositions).
using CharsAtPositions = std::vector<std::vector<std::string>>;

// Returns an empty NeuralScorer when model is null or not ready.
inline NeuralScorer MakeNeuralScorer(SlothRescorer* model) {
  if (model == nullptr || !model->ready()) return NeuralScorer();
  return [model](const std::vector<std::string>& readings, const CharsAtPositions* extra) {
    return model->charLogProbs(readings, extra);
  };
}

// Step 1 of R-walk: one forward pass producing the per-position score table.
// chars must be grid.charsAtPositions(); it is taken as an argument so a
// caller that also keys a cache on it computes it only once.
// Returns an empty table (table.empty()) on any failure: no scorer,
// alpha == 0, empty grid, or a table whose length != grid.length().
inline Formosa::Gramambular2::ReadingGrid::PositionalCharScores ComputeCharScores(
    Formosa::Gramambular2::ReadingGrid& grid, const NeuralScorer& scorer,
    const NeuralWalkOptions& options, const CharsAtPositions& chars) {
  Formosa::Gramambular2::ReadingGrid::PositionalCharScores scores;
  scores.missingScore = options.missingLogProb;
  if (!scorer || options.alpha == 0 || grid.length() == 0) return scores;
  scores.table = scorer(grid.readings(), options.normaliseOverGridChars ? &chars : nullptr);
  if (scores.table.size() != grid.length()) scores.table.clear();
  return scores;
}

inline Formosa::Gramambular2::ReadingGrid::PositionalCharScores ComputeCharScores(
    Formosa::Gramambular2::ReadingGrid& grid, const NeuralScorer& scorer,
    const NeuralWalkOptions& options = NeuralWalkOptions()) {
  return ComputeCharScores(grid, scorer, options,
                           options.normaliseOverGridChars ? grid.charsAtPositions()
                                                          : CharsAtPositions());
}

// Step 2: walk with a precomputed table. Falls back to grid.walk() when the
// table is empty or its length != grid.length(). *usedNeural reports which
// path ran.
inline Formosa::Gramambular2::ReadingGrid::WalkResult WalkWithScores(
    Formosa::Gramambular2::ReadingGrid& grid,
    const Formosa::Gramambular2::ReadingGrid::PositionalCharScores& scores, double alpha,
    bool* usedNeural = nullptr) {
  bool neural = !scores.table.empty() && scores.table.size() == grid.length();
  if (usedNeural != nullptr) *usedNeural = neural;
  if (!neural) return grid.walk();
  return grid.walkWithCharBonus(alpha, scores);
}

inline Formosa::Gramambular2::ReadingGrid::WalkResult NeuralWalk(
    Formosa::Gramambular2::ReadingGrid& grid, SlothRescorer* model,
    const NeuralWalkOptions& options = NeuralWalkOptions(),
    NeuralWalkStats* stats = nullptr) {
  using Clock = std::chrono::steady_clock;
  auto ms = [](Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
  };
  NeuralWalkStats local;
  NeuralWalkStats& st = stats != nullptr ? *stats : local;
  st = NeuralWalkStats();
  NeuralScorer scorer = MakeNeuralScorer(model);
  // forwardMs stays 0 when nothing would be scored (no model, alpha 0, empty grid).
  auto t0 = Clock::now();
  auto scores = ComputeCharScores(grid, scorer, options);
  if (scorer && options.alpha != 0 && grid.length() != 0) st.forwardMs = ms(t0);
  auto t1 = Clock::now();
  auto r = WalkWithScores(grid, scores, options.alpha, &st.usedNeural);
  st.walkMs = ms(t1);
  return r;
}

// One-entry cache of the score table. The table is a pure function of
// (readings, characters each position can take), so that pair is the key;
// user overrides / UOM only change node scores, not the table. Empty tables
// (failed forward) are never stored, so the next keystroke retries.
class NeuralScoreCache {
 public:
  using Scores = Formosa::Gramambular2::ReadingGrid::PositionalCharScores;

  // Returns the cached table if it was stored for this key, else nullptr.
  // An empty cached table means nothing is cached.
  const Scores* lookup(const std::vector<std::string>& readings,
                       const CharsAtPositions& chars) const {
    if (scores_.table.empty()) return nullptr;
    if (readings != readings_ || chars != chars_) return nullptr;
    return &scores_;
  }

  const Scores* lookup(const Formosa::Gramambular2::ReadingGrid& grid) const {
    if (scores_.table.empty() || grid.readings() != readings_) return nullptr;
    return lookup(grid.readings(), grid.charsAtPositions());
  }

  // Takes ownership of scores and returns the cached copy, or nullptr (cache
  // untouched) when the table is empty.
  const Scores* store(std::vector<std::string> readings, CharsAtPositions chars, Scores&& scores) {
    if (scores.table.empty()) return nullptr;
    readings_ = std::move(readings);
    chars_ = std::move(chars);
    scores_ = std::move(scores);
    return &scores_;
  }

  void clear() { *this = NeuralScoreCache(); }

 private:
  std::vector<std::string> readings_;
  CharsAtPositions chars_;
  Scores scores_;
};

// Trips (sticky) after kConsecutive consecutive keystrokes slower than
// kThresholdMs. Record only keystrokes that ran inference; a fast one resets
// the streak. Time is fed in, so tests need no clock.
class NeuralCircuitBreaker {
 public:
  static constexpr double kThresholdMs = 10.0;
  static constexpr int kConsecutive = 3;

  void record(double ms) {
    if (tripped_) return;
    if (ms > kThresholdMs) {
      if (++slowStreak_ >= kConsecutive) tripped_ = true;
    } else {
      slowStreak_ = 0;
    }
  }
  bool tripped() const { return tripped_; }

 private:
  int slowStreak_ = 0;
  bool tripped_ = false;
};

}  // namespace McBopomofo

#endif  // SOURCE_ENGINE_RESCORER_NEURAL_WALK_H_
