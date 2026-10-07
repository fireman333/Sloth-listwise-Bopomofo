// Exactness test for the optimized ReadingGrid::walkNBestImpl() on the real
// McBopomofo LM (Source/Data/data.txt): walkNBest() and
// walkNBestWithCharBonus() must equal the original algorithm
// (gramambular2/nbest_reference_for_test.h) on grids built from real
// sentences (bench/data/general/test.jsonl readings, rolled into buffers of
// up to 40 syllables as the IME composes them), with and without user
// overrides and with real-like or random positional char tables.

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "McBopomofoLM.h"
#include "gramambular2/nbest_reference_for_test.h"
#include "gramambular2/reading_grid.h"
#include "gtest/gtest.h"

namespace McBopomofo {
namespace {

using Formosa::Gramambular2::ReadingGrid;
namespace Ref = Formosa::Gramambular2::NBestReference;

struct Sentence {
  std::string text;
  std::vector<std::string> readings;
};

// Minimal extraction of "text" and "reading" from one JSONL line (the bench
// file has no escapes inside these fields).
bool ParseLine(const std::string& line, Sentence* out) {
  auto field = [&](const char* name) -> size_t {
    std::string key = std::string("\"") + name + "\":";
    size_t p = line.find(key);
    return p == std::string::npos ? p : p + key.size();
  };
  size_t t = field("text");
  size_t r = field("reading");
  if (t == std::string::npos || r == std::string::npos) return false;
  size_t q1 = line.find('"', t);
  size_t q2 = line.find('"', q1 + 1);
  out->text = line.substr(q1 + 1, q2 - q1 - 1);
  size_t lb = line.find('[', r);
  size_t rb = line.find(']', lb);
  out->readings.clear();
  size_t p = lb;
  while (true) {
    size_t a = line.find('"', p + 1);
    if (a == std::string::npos || a > rb) break;
    size_t b = line.find('"', a + 1);
    out->readings.push_back(line.substr(a + 1, b - a - 1));
    p = b;
  }
  return !out->readings.empty();
}

std::vector<std::string> SplitChars(const std::string& s) {
  std::vector<std::string> out;
  Ref::SplitCodePointsRef(s, &out);
  return out;
}

// Real-like table: the sentence's own character at each position scores
// near 0, every other candidate character gets a log-prob in [-14, -2];
// some characters are missing (missingScore).
ReadingGrid::PositionalCharScores RealLikeScores(
    const ReadingGrid& grid, const std::vector<std::string>& gold,
    std::mt19937* rng) {
  ReadingGrid::PositionalCharScores s;
  s.table.resize(grid.length());
  const auto chars = grid.charsAtPositions();
  std::uniform_real_distribution<double> other(-14.0, -2.0);
  std::uniform_real_distribution<double> top(-1.0, 0.0);
  std::bernoulli_distribution keep(0.9);
  for (size_t t = 0; t < chars.size(); ++t) {
    for (const auto& c : chars[t]) {
      if (t < gold.size() && c == gold[t]) {
        s.table[t][c] = top(*rng);
      } else if (keep(*rng)) {
        s.table[t][c] = other(*rng);
      }
    }
  }
  return s;
}

ReadingGrid::PositionalCharScores RandomScores(const ReadingGrid& grid,
                                               std::mt19937* rng) {
  ReadingGrid::PositionalCharScores s;
  s.table.resize(grid.length());
  const auto chars = grid.charsAtPositions();
  std::uniform_real_distribution<double> v(-12.0, 0.0);
  for (size_t t = 0; t < chars.size(); ++t) {
    for (const auto& c : chars[t]) s.table[t][c] = v(*rng);
  }
  return s;
}

TEST(NBestEquivalenceTest, RealLMMatchesReference) {
  const std::string jsonl = NBEST_BENCH_JSONL;
  std::ifstream f(jsonl);
  if (!f.good()) {
    GTEST_SKIP() << "bench file not found: " << jsonl;
  }
  std::vector<Sentence> sentences;
  std::string line;
  while (std::getline(f, line) && sentences.size() < 300) {
    Sentence s;
    if (ParseLine(line, &s)) sentences.push_back(std::move(s));
  }
  ASSERT_EQ(sentences.size(), 300u);

  auto lm = std::make_shared<McBopomofoLM>();
  const std::string data = std::string(READING_POLICY_DATA_DIR) + "/data.txt";
  lm->loadLanguageModel(data.c_str());
  ASSERT_TRUE(lm->isDataModelLoaded());

  std::mt19937 rng(7);
  std::bernoulli_distribution doOverride(0.35);
  size_t comparisons = 0;
  size_t overridden = 0;
  size_t maxLen = 0;
  double refUs = 0, newUs = 0;  // k = 10, alpha = 0.5, 31-40 syllables
  size_t timed = 0;

  for (size_t si = 0; si < sentences.size(); ++si) {
    // Rolling buffer: this sentence followed by the next ones, up to 40.
    std::vector<std::string> readings, gold;
    for (size_t sj = si; sj < sentences.size() && readings.size() < 40; ++sj) {
      auto chars = SplitChars(sentences[sj].text);
      const bool aligned = chars.size() == sentences[sj].readings.size();
      for (size_t r = 0; r < sentences[sj].readings.size() && readings.size() < 40; ++r) {
        readings.push_back(sentences[sj].readings[r]);
        gold.push_back(aligned ? chars[r] : "");
      }
    }
    // Two buffer lengths per sentence: the sentence alone, and the full
    // rolled buffer (up to 40 syllables).
    for (size_t want : {sentences[si].readings.size(), readings.size()}) {
      ReadingGrid grid(lm);
      std::vector<std::string> g;
      for (size_t r = 0; r < want && r < readings.size(); ++r) {
        if (grid.insertReading(readings[r])) g.push_back(gold[r]);
      }
      if (grid.length() == 0) continue;
      maxLen = std::max(maxLen, grid.length());
      if (doOverride(rng)) {
        std::uniform_int_distribution<size_t> loc(0, grid.length() - 1);
        const size_t at = loc(rng);
        auto cands = grid.candidatesAt(at);
        if (!cands.empty()) {
          std::uniform_int_distribution<size_t> pc(0, std::min<size_t>(cands.size() - 1, 15));
          const auto type = (si % 2)
                                ? ReadingGrid::Node::OverrideType::kOverrideValueWithHighScore
                                : ReadingGrid::Node::OverrideType::kOverrideValueWithScoreFromTopUnigram;
          if (grid.overrideCandidate(at, cands[pc(rng)], type)) ++overridden;
        }
      }
      const std::string label = "sentence " + std::to_string(si) + " len " +
                                std::to_string(grid.length());
      for (size_t k : {1, 3, 10}) {
        {
          auto got = grid.walkNBest(k);
          auto want = Ref::WalkNBestReference(grid, k, 0.0, nullptr);
          ASSERT_EQ(Ref::DiffNBest(got, want, &grid), "") << label << " k=" << k << " plain";
          ++comparisons;
        }
        for (int table = 0; table < 2; ++table) {
          auto scores = table == 0 ? RealLikeScores(grid, g, &rng) : RandomScores(grid, &rng);
          for (double alpha : {0.0, 0.5}) {
            auto t0 = std::chrono::steady_clock::now();
            auto got = grid.walkNBestWithCharBonus(k, alpha, scores);
            auto t1 = std::chrono::steady_clock::now();
            auto want = Ref::WalkNBestReference(grid, k, alpha, &scores);
            auto t2 = std::chrono::steady_clock::now();
            ASSERT_EQ(Ref::DiffNBest(got, want, &grid), "")
                << label << " k=" << k << " alpha=" << alpha << " table=" << table;
            ++comparisons;
            if (k == 10 && alpha == 0.5 && table == 0 && grid.length() > 30) {
              newUs += std::chrono::duration<double, std::micro>(t1 - t0).count();
              refUs += std::chrono::duration<double, std::micro>(t2 - t1).count();
              ++timed;
            }
          }
        }
      }
    }
  }
  std::cout << "[ nbest-equivalence ] comparisons=" << comparisons
            << " overridden_grids=" << overridden << " max_len=" << maxLen
            << " | k=10 alpha=0.5 31-40 syl: mean new " << (timed ? newUs / timed : 0)
            << " us vs reference " << (timed ? refUs / timed : 0) << " us (n=" << timed
            << ")\n";
  EXPECT_EQ(maxLen, 40u);
  EXPECT_GT(overridden, 50u);
  EXPECT_GT(comparisons, 8000u);
}

}  // namespace
}  // namespace McBopomofo
