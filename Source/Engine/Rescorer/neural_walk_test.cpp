// Tests for NeuralWalk (R-walk) on the real McBopomofo LM + SlothLM-E ranker.
// Needs: DATA_TXT (McBopomofo data.txt), RWALK_MODEL_DIR (export dir with
// onnx/model_quantized.onnx), RWALK_FIXTURE (bench/src/make_rwalk_fixture.py).

#include "neural_walk.h"

#include <cmath>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "McBopomofoLM.h"
#include "gtest/gtest.h"

namespace McBopomofo {
namespace {

using Formosa::Gramambular2::ReadingGrid;

struct FixtureRow {
  std::string id;
  std::vector<std::string> readings;
  std::string gold, pyFp32, pyInt8;
};

std::vector<FixtureRow> LoadFixture() {
  std::vector<FixtureRow> rows;
  std::ifstream f(RWALK_FIXTURE);
  std::string line;
  while (std::getline(f, line)) {
    std::vector<std::string> cols;
    std::stringstream ss(line);
    std::string c;
    while (std::getline(ss, c, '\t')) cols.push_back(c);
    if (cols.size() < 5) continue;
    FixtureRow r;
    r.id = cols[0];
    std::stringstream rs(cols[1]);
    std::string syl;
    while (rs >> syl) r.readings.push_back(syl);
    r.gold = cols[2];
    r.pyFp32 = cols[3];
    r.pyInt8 = cols[4];
    rows.push_back(std::move(r));
  }
  return rows;
}

std::string Join(const ReadingGrid::WalkResult& r) {
  std::string s;
  for (const auto& n : r.nodes) s += n->value();
  return s;
}

class NeuralWalkTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    lm_ = std::make_shared<McBopomofoLM>();
    lm_->loadLanguageModel(DATA_TXT);
    model_ = std::make_unique<SlothRescorer>(RWALK_MODEL_DIR);
    rows_ = LoadFixture();
  }
  static void TearDownTestSuite() {
    model_.reset();
    lm_.reset();
  }
  static std::unique_ptr<ReadingGrid> Grid(const std::vector<std::string>& readings) {
    auto g = std::make_unique<ReadingGrid>(lm_);
    for (const auto& r : readings) {
      if (!g->insertReading(r)) return nullptr;
    }
    return g;
  }
  static void ExpectSameWalk(const ReadingGrid::WalkResult& a, const ReadingGrid::WalkResult& b) {
    ASSERT_EQ(a.nodes.size(), b.nodes.size());
    for (size_t i = 0; i < a.nodes.size(); ++i) EXPECT_EQ(a.nodes[i], b.nodes[i]);
    EXPECT_EQ(Join(a), Join(b));
  }

  static std::shared_ptr<McBopomofoLM> lm_;
  static std::unique_ptr<SlothRescorer> model_;
  static std::vector<FixtureRow> rows_;
};

std::shared_ptr<McBopomofoLM> NeuralWalkTest::lm_;
std::unique_ptr<SlothRescorer> NeuralWalkTest::model_;
std::vector<FixtureRow> NeuralWalkTest::rows_;

TEST_F(NeuralWalkTest, ResourcesLoaded) {
  ASSERT_TRUE(lm_->isDataModelLoaded());
  ASSERT_TRUE(model_->ready());
  ASSERT_EQ(rows_.size(), 50u);
}

TEST_F(NeuralWalkTest, ModelLoadFailureFallsBackToWalk) {
  SlothRescorer bad("/nonexistent/model/dir");
  ASSERT_FALSE(bad.ready());
  EXPECT_TRUE(bad.charLogProbs({"ㄧ"}).empty());
  for (size_t i = 0; i < 10; ++i) {
    auto g = Grid(rows_[i].readings);
    ASSERT_NE(g, nullptr);
    NeuralWalkStats st;
    auto r = NeuralWalk(*g, &bad, NeuralWalkOptions(), &st);
    EXPECT_FALSE(st.usedNeural);
    ExpectSameWalk(r, g->walk());
    ExpectSameWalk(NeuralWalk(*g, nullptr), g->walk());
  }
}

TEST_F(NeuralWalkTest, AlphaZeroEqualsWalk) {
  for (const auto& row : rows_) {
    auto g = Grid(row.readings);
    ASSERT_NE(g, nullptr) << row.id;
    auto extra = g->charsAtPositions();
    ReadingGrid::PositionalCharScores sc;
    sc.table = model_->charLogProbs(g->readings(), &extra);
    ASSERT_EQ(sc.table.size(), g->length());
    // Real table, alpha 0, no short-circuit: identical nodes to walk().
    ExpectSameWalk(g->walkWithCharBonus(0.0, sc), g->walk());
    NeuralWalkOptions o;
    o.alpha = 0;
    ExpectSameWalk(NeuralWalk(*g, model_.get(), o), g->walk());
  }
}

TEST_F(NeuralWalkTest, CharLogProbsNormalisedAndConsistentWithScore) {
  const auto& row = rows_[1];
  auto legalOnly = model_->charLogProbs(row.readings);
  ASSERT_EQ(legalOnly.size(), row.readings.size());
  for (const auto& m : legalOnly) {
    ASSERT_FALSE(m.empty());
    double z = 0;
    for (const auto& kv : m) z += std::exp(kv.second);
    EXPECT_NEAR(z, 1.0, 1e-6);
  }
  // With the candidate's chars as extras, the per-char sum equals score().
  std::vector<std::string> chars;
  for (size_t i = 0; i < row.gold.size();) {
    unsigned char c = static_cast<unsigned char>(row.gold[i]);
    size_t n = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : 4;
    chars.push_back(row.gold.substr(i, n));
    i += n;
  }
  ASSERT_EQ(chars.size(), row.readings.size());
  std::vector<std::vector<std::string>> extra;
  for (const auto& c : chars) extra.push_back({c});
  auto t = model_->charLogProbs(row.readings, &extra);
  double sum = 0;
  for (size_t i = 0; i < chars.size(); ++i) {
    auto it = t[i].find(chars[i]);
    sum += it == t[i].end() ? -30.0 : it->second;
  }
  RescoreRequest req;
  req.readings = row.readings;
  req.candidates = {row.gold};
  auto s = model_->score(req);
  ASSERT_EQ(s.size(), 1u);
  EXPECT_NEAR(sum, s[0], 1e-6);
}

TEST_F(NeuralWalkTest, OverriddenNodeKeepsUserPick) {
  // Force an unlikely single-char pick at every position of a sentence in
  // turn; the neural walk must keep it, like the stock walk does.
  const auto& row = rows_[0];
  for (size_t loc = 0; loc < row.readings.size(); ++loc) {
    auto g = Grid(row.readings);
    ASSERT_NE(g, nullptr);
    auto node = g->spans()[loc].nodeOf(1);
    ASSERT_NE(node, nullptr);
    const auto& u = node->unigrams();
    const std::string pick = u.back().value();  // lowest-scored unigram
    ASSERT_TRUE(g->overrideCandidate(loc, ReadingGrid::Candidate(node->reading(), pick)));
    NeuralWalkOptions o;
    o.alpha = 3.0;
    auto r = NeuralWalk(*g, model_.get(), o);
    ReadingGrid::NodePtr at;
    size_t pos = 0;
    for (const auto& n : r.nodes) {
      if (loc >= pos && loc < pos + n->spanningLength()) at = n;
      pos += n->spanningLength();
    }
    ASSERT_NE(at, nullptr);
    EXPECT_TRUE(at->isOverridden());
    EXPECT_EQ(at->value(), pick) << row.id << " loc " << loc;
  }
}

TEST_F(NeuralWalkTest, MatchesPythonRWalkOnGeneralDev50) {
  size_t sameFp32 = 0, sameInt8 = 0, n = 0;
  std::string diffs;
  for (const auto& row : rows_) {
    auto g = Grid(row.readings);
    ASSERT_NE(g, nullptr) << row.id;
    NeuralWalkStats st;
    std::string c = Join(NeuralWalk(*g, model_.get(), NeuralWalkOptions(), &st));
    EXPECT_TRUE(st.usedNeural);
    ++n;
    sameFp32 += c == row.pyFp32;
    sameInt8 += c == row.pyInt8;
    if (c != row.pyFp32 || c != row.pyInt8) {
      diffs += row.id + " cpp=" + c + " py_fp32=" + row.pyFp32 + " py_int8=" + row.pyInt8 +
               " gold=" + row.gold + "\n";
    }
  }
  std::printf("[R-walk parity] C++ int8 vs Python fp32: %zu/%zu, vs Python int8: %zu/%zu\n%s",
              sameFp32, n, sameInt8, n, diffs.c_str());
  EXPECT_GE(sameFp32 * 100, n * 98);
  EXPECT_GE(sameInt8 * 100, n * 98);
}

}  // namespace
}  // namespace McBopomofo
