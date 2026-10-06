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
  if (model == nullptr || !model->ready() || options.alpha == 0 || grid.length() == 0) {
    auto t = Clock::now();
    auto r = grid.walk();
    st.walkMs = ms(t);
    return r;
  }
  auto t0 = Clock::now();
  Formosa::Gramambular2::ReadingGrid::PositionalCharScores scores;
  scores.missingScore = options.missingLogProb;
  if (options.normaliseOverGridChars) {
    std::vector<std::vector<std::string>> extra = grid.charsAtPositions();
    scores.table = model->charLogProbs(grid.readings(), &extra);
  } else {
    scores.table = model->charLogProbs(grid.readings());
  }
  st.forwardMs = ms(t0);
  auto t1 = Clock::now();
  if (scores.table.size() != grid.length()) {
    auto r = grid.walk();
    st.walkMs = ms(t1);
    return r;
  }
  auto r = grid.walkWithCharBonus(options.alpha, scores);
  st.walkMs = ms(t1);
  st.usedNeural = true;
  return r;
}

}  // namespace McBopomofo

#endif  // SOURCE_ENGINE_RESCORER_NEURAL_WALK_H_
