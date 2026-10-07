// Model-free tests for NeuralScoreCache, NeuralCircuitBreaker and the split
// ComputeCharScores / WalkWithScores. Needs only DATA_TXT (no onnx model).

#include "neural_walk.h"

#include <memory>
#include <utility>
#include <string>
#include <vector>

#include "McBopomofoLM.h"
#include "gramambular2/language_model.h"
#include "gtest/gtest.h"

namespace McBopomofo {
namespace {

using Formosa::Gramambular2::ReadingGrid;

class NeuralCacheTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    lm_ = std::make_shared<McBopomofoLM>();
    lm_->loadLanguageModel(DATA_TXT);
  }
  static void TearDownTestSuite() { lm_.reset(); }
  static std::unique_ptr<ReadingGrid> Grid(const std::vector<std::string>& readings) {
    auto g = std::make_unique<ReadingGrid>(lm_);
    for (const auto& r : readings) {
      if (!g->insertReading(r)) return nullptr;
    }
    return g;
  }
  // Counting fake: uniform table of the right length (or empty when failing).
  struct Fake {
    int calls = 0;
    bool fail = false;
    NeuralScorer fn() {
      return [this](const std::vector<std::string>& readings,
                    const std::vector<std::vector<std::string>>*) {
        ++calls;
        SlothRescorer::CharLogProbTable t;
        if (!fail) t.assign(readings.size(), {});
        return t;
      };
    }
  };
  // Mimics the selector's usage: lookup, else compute + store, then walk
  // from the cached table (empty table when nothing was stored).
  static ReadingGrid::PositionalCharScores GetOrCompute(NeuralScoreCache& cache, ReadingGrid& g,
                                                        const NeuralScorer& s) {
    auto chars = g.charsAtPositions();
    const auto* hit = cache.lookup(g.readings(), chars);
    if (hit == nullptr) {
      auto sc = ComputeCharScores(g, s, NeuralWalkOptions(), chars);
      hit = cache.store(g.readings(), std::move(chars), std::move(sc));
    }
    return hit != nullptr ? *hit : ReadingGrid::PositionalCharScores();
  }
  static std::shared_ptr<McBopomofoLM> lm_;
};
std::shared_ptr<McBopomofoLM> NeuralCacheTest::lm_;

TEST_F(NeuralCacheTest, DataLoaded) { ASSERT_TRUE(lm_->isDataModelLoaded()); }

TEST_F(NeuralCacheTest, IdenticalKeyHits) {
  Fake fake;
  NeuralScoreCache cache;
  auto g = Grid({"ㄋㄧˇ", "ㄏㄠˇ"});
  ASSERT_NE(g, nullptr);
  EXPECT_FALSE(GetOrCompute(cache, *g, fake.fn()).table.empty());
  EXPECT_EQ(fake.calls, 1);
  auto g2 = Grid({"ㄋㄧˇ", "ㄏㄠˇ"});
  EXPECT_NE(cache.lookup(*g2), nullptr);
  GetOrCompute(cache, *g2, fake.fn());
  EXPECT_EQ(fake.calls, 1);
}

TEST_F(NeuralCacheTest, SameLengthDifferentReadingsMisses) {
  Fake fake;
  NeuralScoreCache cache;
  auto a = Grid({"ㄋㄧˇ", "ㄏㄠˇ"});
  auto b = Grid({"ㄨㄛˇ", "ㄏㄠˇ"});
  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);
  ASSERT_EQ(a->length(), b->length());
  GetOrCompute(cache, *a, fake.fn());
  EXPECT_EQ(cache.lookup(*b), nullptr);
  GetOrCompute(cache, *b, fake.fn());
  EXPECT_EQ(fake.calls, 2);
}

// Tiny LM whose per-reading characters can differ between instances.
class TinyLM : public Formosa::Gramambular2::LanguageModel {
 public:
  explicit TinyLM(std::vector<std::string> ni) : ni_(std::move(ni)) {}
  std::vector<Unigram> getUnigrams(const std::string& key) override {
    std::vector<Unigram> v;
    if (key == "ㄋㄧˇ") {
      for (const auto& c : ni_) v.emplace_back(c, -3.0);
    } else if (key == "ㄏㄠˇ") {
      v.emplace_back("好", -3.0);
    }
    return v;
  }
  bool hasUnigrams(const std::string& key) override { return key == "ㄋㄧˇ" || key == "ㄏㄠˇ"; }

 private:
  std::vector<std::string> ni_;
};

TEST_F(NeuralCacheTest, ChangedPerPositionCharsMisses) {
  Fake fake;
  NeuralScoreCache cache;
  auto build = [](std::vector<std::string> ni) {
    auto g = std::make_unique<ReadingGrid>(std::make_shared<TinyLM>(std::move(ni)));
    g->insertReading("ㄋㄧˇ");
    g->insertReading("ㄏㄠˇ");
    return g;
  };
  auto a = build({"你"});
  auto b = build({"你", "妳"});  // same readings, extra char at position 0
  ASSERT_EQ(a->readings(), b->readings());
  ASSERT_NE(a->charsAtPositions(), b->charsAtPositions());
  GetOrCompute(cache, *a, fake.fn());
  EXPECT_NE(cache.lookup(*a), nullptr);
  EXPECT_EQ(cache.lookup(*b), nullptr);
  GetOrCompute(cache, *b, fake.fn());
  EXPECT_EQ(fake.calls, 2);
  // One-entry cache: the first key is gone now.
  EXPECT_EQ(cache.lookup(*a), nullptr);
}

TEST_F(NeuralCacheTest, FailedComputeNotCached) {
  Fake fake;
  fake.fail = true;
  NeuralScoreCache cache;
  auto g = Grid({"ㄋㄧˇ", "ㄏㄠˇ"});
  ASSERT_NE(g, nullptr);
  EXPECT_TRUE(GetOrCompute(cache, *g, fake.fn()).table.empty());
  EXPECT_EQ(cache.lookup(*g), nullptr);
  GetOrCompute(cache, *g, fake.fn());
  EXPECT_EQ(fake.calls, 2);
  fake.fail = false;
  GetOrCompute(cache, *g, fake.fn());
  EXPECT_EQ(fake.calls, 3);
  EXPECT_NE(cache.lookup(*g), nullptr);
}

TEST_F(NeuralCacheTest, ComputeAndWalkFallbacks) {
  auto g = Grid({"ㄋㄧˇ", "ㄏㄠˇ"});
  ASSERT_NE(g, nullptr);
  Fake fake;
  NeuralWalkOptions zero;
  zero.alpha = 0;
  EXPECT_TRUE(ComputeCharScores(*g, fake.fn(), zero).table.empty());
  EXPECT_EQ(fake.calls, 0);
  EXPECT_TRUE(ComputeCharScores(*g, NeuralScorer()).table.empty());
  EXPECT_FALSE(MakeNeuralScorer(nullptr));
  EXPECT_TRUE(ComputeCharScores(*g, MakeNeuralScorer(nullptr)).table.empty());
  // Length mismatch -> empty.
  NeuralScorer wrong = [](const std::vector<std::string>&,
                          const std::vector<std::vector<std::string>>*) {
    return SlothRescorer::CharLogProbTable(1);
  };
  EXPECT_TRUE(ComputeCharScores(*g, wrong).table.empty());
  // Empty table -> stock walk.
  bool used = true;
  auto r = WalkWithScores(*g, ReadingGrid::PositionalCharScores(), 0.5, &used);
  EXPECT_FALSE(used);
  EXPECT_EQ(r.nodes.size(), g->walk().nodes.size());
  // Table of wrong size -> stock walk.
  ReadingGrid::PositionalCharScores bad;
  bad.table.resize(5);
  WalkWithScores(*g, bad, 0.5, &used);
  EXPECT_FALSE(used);
  // Right-sized table -> neural path.
  WalkWithScores(*g, ComputeCharScores(*g, fake.fn()), 0.5, &used);
  EXPECT_TRUE(used);
}

TEST(NeuralCircuitBreakerTest, ThresholdIs30Ms) {
  EXPECT_DOUBLE_EQ(NeuralCircuitBreaker::kThresholdMs, 30.0);
  EXPECT_EQ(NeuralCircuitBreaker::kConsecutive, 3);
}

TEST(NeuralCircuitBreakerTest, ThreeSlowTrips) {
  NeuralCircuitBreaker b;
  b.record(31);
  b.record(35);
  EXPECT_FALSE(b.tripped());
  b.record(60);
  EXPECT_TRUE(b.tripped());
}

TEST(NeuralCircuitBreakerTest, FastResetsStreak) {
  NeuralCircuitBreaker b;
  b.record(31);
  b.record(2);
  b.record(31);
  b.record(31);
  EXPECT_FALSE(b.tripped());
  b.record(30.0);  // exactly the threshold is not slow
  b.record(31);
  b.record(31);
  EXPECT_FALSE(b.tripped());
}

TEST(NeuralCircuitBreakerTest, IdleGapLatencyDoesNotTrip) {
  // Keystrokes after idle gaps (design D9: ~10-25 ms) never trip.
  NeuralCircuitBreaker b;
  for (int i = 0; i < 100; ++i) b.record(i % 2 ? 25.0 : 29.9);
  EXPECT_FALSE(b.tripped());
}

TEST(NeuralCircuitBreakerTest, TrippedIsSticky) {
  NeuralCircuitBreaker b;
  for (int i = 0; i < 3; ++i) b.record(50);
  ASSERT_TRUE(b.tripped());
  for (int i = 0; i < 10; ++i) b.record(1);
  EXPECT_TRUE(b.tripped());
}

}  // namespace
}  // namespace McBopomofo
