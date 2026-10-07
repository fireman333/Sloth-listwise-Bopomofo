// Tests for PreciseWalk / PreciseSelect (precise mode).
// Model-free: calibration math, buckets, gate, fusion, heavy-model fallbacks,
// config parsing. With the real R export (RWALK_MODEL_DIR) + data.txt:
// candidate 0 == R-walk, tau = 0 == R-walk. With PRECISE_FIXTURE
// (bench/src/precise_calib.py fixture): C++ fusion == Python decision on
// recorded R/Qwen scores, and the real R N-best reproduces the recorded one.

#include "precise_walk.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "McBopomofoLM.h"
#include "gtest/gtest.h"
#include "neural_walk.h"
#include "nlohmann/json.hpp"
#include "sloth_rescorer.h"

namespace McBopomofo {
namespace {

using Formosa::Gramambular2::ReadingGrid;
using json = nlohmann::json;

class FakeHeavy : public Rescorer {
 public:
  explicit FakeHeavy(std::vector<double> out, bool ready = true, int sleepMs = 0)
      : out_(std::move(out)), ready_(ready), sleepMs_(sleepMs) {}
  bool ready() const override { return ready_; }
  std::vector<double> score(const RescoreRequest& req) override {
    ++calls;
    lastCandidates = req.candidates.size();
    if (sleepMs_ > 0) std::this_thread::sleep_for(std::chrono::milliseconds(sleepMs_));
    return out_;
  }
  int calls = 0;
  size_t lastCandidates = 0;

 private:
  std::vector<double> out_;
  bool ready_;
  int sleepMs_;
};

const std::vector<std::string> kReadings{"a", "b", "c"};
const std::vector<std::string> kCands{"x", "y", "z"};

TEST(PreciseWalkTest, CalibratedLogProbsMath) {
  auto lp = CalibratedLogProbs({2.0, 1.0, 0.0}, 2.0);
  ASSERT_EQ(lp.size(), 3u);
  double z = std::exp(1.0) + std::exp(0.5) + 1.0;
  EXPECT_NEAR(std::exp(lp[0]), std::exp(1.0) / z, 1e-12);
  EXPECT_NEAR(std::exp(lp[1]), std::exp(0.5) / z, 1e-12);
  EXPECT_NEAR(std::exp(lp[0]) + std::exp(lp[1]) + std::exp(lp[2]), 1.0, 1e-12);
  // Higher temperature -> flatter; argmax unchanged.
  auto cold = CalibratedLogProbs({-10.0, -12.0, -15.0}, 0.5);
  auto hot = CalibratedLogProbs({-10.0, -12.0, -15.0}, 5.0);
  EXPECT_GT(cold[0], hot[0]);
  EXPECT_GT(hot[0], hot[1]);
  // Large magnitudes are stable; T <= 0 -> 1.
  auto big = CalibratedLogProbs({-1e4, -1e4 - 1}, 0.0);
  EXPECT_NEAR(std::exp(big[0]), 1.0 / (1.0 + std::exp(-1.0)), 1e-12);
  EXPECT_TRUE(CalibratedLogProbs({}, 1.0).empty());
  EXPECT_DOUBLE_EQ(CalibratedLogProbs({3.0}, 1.0)[0], 0.0);
}

TEST(PreciseWalkTest, BucketTemperatures) {
  PreciseConfig c;
  c.bucketUpper = {9, 14};
  c.temperatures = {1.5, 2.0, 3.0};
  ASSERT_TRUE(c.valid());
  EXPECT_DOUBLE_EQ(c.temperatureFor(1), 1.5);
  EXPECT_DOUBLE_EQ(c.temperatureFor(9), 1.5);
  EXPECT_DOUBLE_EQ(c.temperatureFor(10), 2.0);
  EXPECT_DOUBLE_EQ(c.temperatureFor(14), 2.0);
  EXPECT_DOUBLE_EQ(c.temperatureFor(15), 3.0);
  EXPECT_DOUBLE_EQ(c.temperatureFor(100), 3.0);
  c.temperatures = {1.0, 2.0};
  EXPECT_FALSE(c.valid());
  c.temperatures = {1.0, -2.0, 1.0};
  EXPECT_FALSE(c.valid());
}

TEST(PreciseWalkTest, GateRule) {
  EXPECT_FALSE(ShouldCallHeavy(0.0, 0.0));  // tau = 0: never
  EXPECT_FALSE(ShouldCallHeavy(0.3, 0.0));
  EXPECT_TRUE(ShouldCallHeavy(1.0, 1.0));   // tau = 1: always
  EXPECT_TRUE(ShouldCallHeavy(0.999, 1.0));
  EXPECT_TRUE(ShouldCallHeavy(0.49, 0.5));
  EXPECT_FALSE(ShouldCallHeavy(0.5, 0.5));
  EXPECT_FALSE(ShouldCallHeavy(0.9, 0.5));
}

TEST(PreciseWalkTest, GateClosedNeverCallsHeavy) {
  PreciseConfig c;
  c.tau = 0.0;
  FakeHeavy h({0.0, 100.0, 0.0});
  auto d = PreciseSelect(kReadings, kCands, {-1.0, -2.0, -3.0}, &h, c);
  EXPECT_EQ(h.calls, 0);
  EXPECT_FALSE(d.gateOpen);
  EXPECT_FALSE(d.heavyCalled);
  EXPECT_EQ(d.chosen, 0u);  // == R-walk
  // Confident R (p_top1 >= tau) also skips the heavy model.
  c.tau = 0.5;
  d = PreciseSelect(kReadings, kCands, {0.0, -20.0, -30.0}, &h, c);
  EXPECT_GT(d.pTop1, 0.99);
  EXPECT_EQ(h.calls, 0);
  EXPECT_EQ(d.chosen, 0u);
}

TEST(PreciseWalkTest, TauOneReplaceEqualsHeavyArgmax) {
  PreciseConfig c;
  c.tau = 1.0;
  c.rWeight = 0.0;
  c.beta = 1.0;
  FakeHeavy h({-30.0, -10.0, -20.0});
  auto d = PreciseSelect(kReadings, kCands, {0.0, -20.0, -30.0}, &h, c);
  EXPECT_EQ(h.calls, 1);
  EXPECT_EQ(h.lastCandidates, 3u);
  EXPECT_TRUE(d.heavyOk);
  EXPECT_EQ(d.chosen, 1u);  // heavy alone picks y even though R is sure of x
}

TEST(PreciseWalkTest, FusionWeighsBoth) {
  PreciseConfig c;
  c.tau = 1.0;
  c.temperatures = {1.0};
  // R: x slightly ahead of y; heavy: y slightly ahead of x.
  std::vector<double> r{0.0, -0.5, -9.0};
  FakeHeavy h({-1.0, 0.0, -9.0});
  c.beta = 0.1;  // R dominates
  EXPECT_EQ(PreciseSelect(kReadings, kCands, r, &h, c).chosen, 0u);
  c.beta = 2.0;  // heavy dominates
  auto d = PreciseSelect(kReadings, kCands, r, &h, c);
  EXPECT_EQ(d.chosen, 1u);
  // fused_i = log pR_i + beta * log pQ_i
  auto lpr = CalibratedLogProbs(r, 1.0);
  auto lpq = CalibratedLogProbs({-1.0, 0.0, -9.0}, 1.0);
  for (size_t i = 0; i < 3; ++i) EXPECT_NEAR(d.fused[i], lpr[i] + 2.0 * lpq[i], 1e-12);
  // Exact tie keeps the lower R rank.
  c.beta = 1.0;
  FakeHeavy tie({-0.5, -0.0, -50.0});  // 0 + -0.5 vs -0.5 + 0 (up to log-normalisers)
  auto dt = PreciseSelect(kReadings, kCands, {0.0, -0.5, -50.0}, &tie, c);
  EXPECT_EQ(dt.chosen, 0u);
}

TEST(PreciseWalkTest, HeavyFailureFallsBackToR) {
  PreciseConfig c;
  c.tau = 1.0;
  c.rWeight = 0.0;
  std::vector<double> r{0.0, -1.0, -2.0};
  struct Case {
    std::unique_ptr<FakeHeavy> h;
    const char* name;
  };
  std::vector<Case> cases;
  cases.push_back({std::make_unique<FakeHeavy>(std::vector<double>{}), "empty"});
  cases.push_back({std::make_unique<FakeHeavy>(std::vector<double>{0.0, 5.0}), "wrong size"});
  cases.push_back({std::make_unique<FakeHeavy>(std::vector<double>{0.0, NAN, 1.0}), "nan"});
  cases.push_back({std::make_unique<FakeHeavy>(std::vector<double>{0.0, 5.0, 1.0}, false), "not ready"});
  for (auto& cs : cases) {
    auto d = PreciseSelect(kReadings, kCands, r, cs.h.get(), c);
    EXPECT_EQ(d.chosen, 0u) << cs.name;
    EXPECT_FALSE(d.heavyOk) << cs.name;
    EXPECT_TRUE(d.gateOpen) << cs.name;
  }
  // nullptr heavy.
  EXPECT_EQ(PreciseSelect(kReadings, kCands, r, nullptr, c).chosen, 0u);
  // Too slow -> result discarded.
  c.qwenTimeoutMs = 5;
  FakeHeavy slow({0.0, 5.0, 1.0}, true, 30);
  auto d = PreciseSelect(kReadings, kCands, r, &slow, c);
  EXPECT_EQ(slow.calls, 1);
  EXPECT_TRUE(d.heavyTimedOut);
  EXPECT_FALSE(d.heavyOk);
  EXPECT_EQ(d.chosen, 0u);
  // Same heavy within the deadline is used.
  c.qwenTimeoutMs = 5000;
  d = PreciseSelect(kReadings, kCands, r, &slow, c);
  EXPECT_TRUE(d.heavyOk);
  EXPECT_EQ(d.chosen, 1u);
}

TEST(PreciseWalkTest, ConfigFile) {
  const std::string path = ::testing::TempDir() + "precise_test.conf";
  {
    std::ofstream f(path);
    f << "# comment\nalpha = 0.5\nk = 8\nbucket_upper = 9 14\n"
         "temperatures = 1.5 2 3.25  # trailing\ntau = 0.6\nr_weight = 1\nbeta = 0.75\n"
         "qwen_timeout_ms = 250\n";
  }
  PreciseConfig c;
  std::string err;
  ASSERT_TRUE(PreciseConfig::LoadFile(path, &c, &err)) << err;
  EXPECT_DOUBLE_EQ(c.alpha, 0.5);
  EXPECT_EQ(c.k, 8u);
  EXPECT_EQ(c.bucketUpper, (std::vector<int>{9, 14}));
  EXPECT_EQ(c.temperatures, (std::vector<double>{1.5, 2.0, 3.25}));
  EXPECT_DOUBLE_EQ(c.tau, 0.6);
  EXPECT_DOUBLE_EQ(c.beta, 0.75);
  EXPECT_DOUBLE_EQ(c.qwenTimeoutMs, 250);
  {
    std::ofstream f(path);
    f << "bucket_upper = 9 14\ntemperatures = 1 2\n";
  }
  PreciseConfig c2;
  EXPECT_FALSE(PreciseConfig::LoadFile(path, &c2, &err));
  {
    std::ofstream f(path);
    f << "gamma = 1\n";
  }
  EXPECT_FALSE(PreciseConfig::LoadFile(path, &c2, &err));
  {
    std::ofstream f(path);
    f << "tau = abc\n";
  }
  EXPECT_FALSE(PreciseConfig::LoadFile(path, &c2, &err));
  EXPECT_FALSE(PreciseConfig::LoadFile(path + ".missing", &c2, &err));
  std::remove(path.c_str());
}

// ---- PreciseCandidates / fixed positions / displayed-first / gate (model-free).

class TinyLM : public Formosa::Gramambular2::LanguageModel {
 public:
  explicit TinyLM(std::vector<std::tuple<std::string, std::string, double>> rows) {
    for (auto& [k, v, s] : rows) db_[k].emplace_back(v, s);
  }
  std::vector<Unigram> getUnigrams(const std::string& key) override {
    auto f = db_.find(key);
    return f == db_.end() ? std::vector<Unigram>() : f->second;
  }
  bool hasUnigrams(const std::string& key) override { return db_.count(key) > 0; }

 private:
  std::map<std::string, std::vector<Unigram>> db_;
};

// Readings a b c. Single syllables: a {甲, 乙}, b {丙, 丁}, c {戊}.
// Multi-syllable phrases that can bypass a single-syllable node:
// a-b {己庚 (bypasses a and b), 甲辛 (keeps 甲 at a, changes b)}, b-c {壬癸}.
std::shared_ptr<TinyLM> MakeTinyLM() {
  return std::make_shared<TinyLM>(std::vector<std::tuple<std::string, std::string, double>>{
      {"a", "甲", -1.0}, {"a", "乙", -2.0}, {"b", "丙", -1.0}, {"b", "丁", -3.0},
      {"c", "戊", -1.0}, {"a-b", "己庚", -1.5}, {"a-b", "甲辛", -2.5}, {"b-c", "壬癸", -1.8}});
}

std::unique_ptr<ReadingGrid> TinyGrid(const std::shared_ptr<TinyLM>& lm) {
  auto g = std::make_unique<ReadingGrid>(lm);
  for (const char* r : {"a", "b", "c"}) EXPECT_TRUE(g->insertReading(r));
  return g;
}

ReadingGrid::PositionalCharScores TinyScores() {
  ReadingGrid::PositionalCharScores s;
  s.table.resize(3);
  s.table[0] = {{"甲", -0.5}, {"乙", -2.0}};
  s.table[1] = {{"丙", -0.2}, {"庚", -0.1}};
  s.table[2] = {{"戊", -0.3}, {"癸", -1.0}};
  return s;
}

bool Contains(const std::vector<std::string>& v, const std::string& x) {
  return std::find(v.begin(), v.end(), x) != v.end();
}

TEST(PreciseCandidatesTest, MatchesNBestWithCharBonusAndPreciseWalkContract) {
  auto lm = MakeTinyLM();
  auto g = TinyGrid(lm);
  PreciseConfig c;
  c.alpha = 0.5;
  c.k = 10;
  auto scores = TinyScores();
  auto set = PreciseCandidates(*g, scores, c);
  auto ref = g->walkNBestWithCharBonus(c.k, c.alpha, scores);
  ASSERT_EQ(set.size(), ref.size());
  ASSERT_EQ(set.paths.size(), set.size());
  ASSERT_EQ(set.rScores.size(), set.size());
  for (size_t i = 0; i < ref.size(); ++i) {
    std::string t;
    for (const auto& n : ref[i].nodes) t += n->value();
    EXPECT_EQ(set.texts[i], t);
    EXPECT_DOUBLE_EQ(set.rScores[i], ref[i].score);
    if (i > 0) EXPECT_LE(set.rScores[i], set.rScores[i - 1]);
  }
  std::string top;
  for (const auto& n : g->walkWithCharBonus(c.alpha, scores).nodes) top += n->value();
  EXPECT_EQ(set.texts[0], top);
  // No fallback inside PreciseCandidates: mismatched table / k == 0 -> empty.
  ReadingGrid::PositionalCharScores bad;
  bad.table.resize(2);
  EXPECT_TRUE(PreciseCandidates(*g, bad, c).empty());
  PreciseConfig k0 = c;
  k0.k = 0;
  EXPECT_TRUE(PreciseCandidates(*g, scores, k0).empty());
  ReadingGrid empty(lm);
  EXPECT_TRUE(PreciseCandidates(empty, ReadingGrid::PositionalCharScores{}, c).empty());
}

TEST(PreciseCandidatesTest, FixedPositionFilterDropsBypassingPhrases) {
  auto lm = MakeTinyLM();
  auto g = TinyGrid(lm);
  PreciseConfig c;
  c.alpha = 0.5;
  c.k = 10;
  // Manual pick of 乙 at position 0.
  ASSERT_TRUE(g->overrideCandidate(0, "乙"));
  auto fixed = FixedPositionsOf(*g);
  ASSERT_EQ(fixed.size(), 1u);
  EXPECT_EQ(fixed[0].location, 0u);
  EXPECT_EQ(fixed[0].length, 1u);
  EXPECT_EQ(fixed[0].value, "乙");

  auto set = PreciseCandidates(*g, TinyScores(), c);
  // Unfiltered N-best contains paths through the a-b phrases that bypass the
  // overridden node (and thus change position 0).
  EXPECT_TRUE(Contains(set.texts, "己庚戊"));
  EXPECT_TRUE(Contains(set.texts, "甲辛戊"));
  size_t violating = 0;
  for (const auto& p : set.paths) violating += !KeepsFixedPositions(p, fixed);
  EXPECT_GT(violating, 0u);

  const size_t before = set.size();
  FilterFixedPositions(&set, fixed);
  EXPECT_LT(set.size(), before);
  ASSERT_EQ(set.paths.size(), set.size());
  ASSERT_EQ(set.rScores.size(), set.size());
  ASSERT_FALSE(set.empty());
  for (size_t i = 0; i < set.size(); ++i) {
    EXPECT_TRUE(KeepsFixedPositions(set.paths[i], fixed)) << set.texts[i];
    EXPECT_EQ(set.texts[i].substr(0, std::string("乙").size()), "乙") << set.texts[i];
    if (i > 0) EXPECT_LE(set.rScores[i], set.rScores[i - 1]);
  }
  EXPECT_FALSE(Contains(set.texts, "己庚戊"));
  EXPECT_FALSE(Contains(set.texts, "甲辛戊"));
  EXPECT_TRUE(Contains(set.texts, "乙壬癸"));  // different segmentation, 乙 kept
}

TEST(PreciseCandidatesTest, FixedPositionCheckIsPerCharacter) {
  auto lm = MakeTinyLM();
  auto g = TinyGrid(lm);
  // Fixed 甲 at position 0: the a-b phrase 甲辛 keeps it, 己庚 does not.
  ASSERT_TRUE(g->overrideCandidate(0, "甲"));
  auto fixed = FixedPositionsOf(*g);
  PreciseConfig c;
  c.k = 10;
  auto set = PreciseCandidates(*g, TinyScores(), c);
  FilterFixedPositions(&set, fixed);
  EXPECT_TRUE(Contains(set.texts, "甲辛戊"));
  EXPECT_FALSE(Contains(set.texts, "己庚戊"));
  // A fixed two-syllable phrase.
  auto g2 = TinyGrid(lm);
  ASSERT_TRUE(g2->overrideCandidate(1, "壬癸"));
  auto fixed2 = FixedPositionsOf(*g2);
  ASSERT_EQ(fixed2.size(), 1u);
  EXPECT_EQ(fixed2[0].location, 1u);
  EXPECT_EQ(fixed2[0].length, 2u);
  auto set2 = PreciseCandidates(*g2, TinyScores(), c);
  FilterFixedPositions(&set2, fixed2);
  ASSERT_FALSE(set2.empty());
  for (const auto& t : set2.texts) EXPECT_EQ(t.substr(t.size() - std::string("壬癸").size()), "壬癸") << t;
  // A node that cannot be split per reading never satisfies a fixed position
  // it covers unless it spans exactly that range with the same value.
  ReadingGrid::WalkResult odd;
  odd.nodes.push_back(std::make_shared<ReadingGrid::Node>(
      "a-b", 2, std::vector<Formosa::Gramambular2::LanguageModel::Unigram>{
                    Formosa::Gramambular2::LanguageModel::Unigram("甲", -1)}));
  EXPECT_FALSE(KeepsFixedPositions(odd, {{0, 1, "甲"}}));
  EXPECT_TRUE(KeepsFixedPositions(odd, {{0, 2, "甲"}}));
  EXPECT_FALSE(KeepsFixedPositions(odd, {{1, 2, "甲乙"}}));  // out of range
  EXPECT_TRUE(KeepsFixedPositions(odd, {}));
}

TEST(PreciseCandidatesTest, MoveDisplayedToFront) {
  auto lm = MakeTinyLM();
  auto g = TinyGrid(lm);
  PreciseConfig c;
  c.k = 10;
  auto set = PreciseCandidates(*g, TinyScores(), c);
  ASSERT_GE(set.size(), 4u);
  const auto orig = set;
  EXPECT_EQ(MoveDisplayedToFront(&set, orig.texts[0]), 0u);
  EXPECT_EQ(set.texts, orig.texts);
  EXPECT_EQ(MoveDisplayedToFront(&set, orig.texts[2]), 2u);
  EXPECT_EQ(set.texts[0], orig.texts[2]);
  EXPECT_EQ(set.texts[1], orig.texts[0]);
  EXPECT_EQ(set.texts[2], orig.texts[1]);
  EXPECT_EQ(set.texts[3], orig.texts[3]);
  EXPECT_DOUBLE_EQ(set.rScores[0], orig.rScores[2]);
  EXPECT_DOUBLE_EQ(set.rScores[1], orig.rScores[0]);
  std::string p0;
  for (const auto& n : set.paths[0].nodes) p0 += n->value();
  EXPECT_EQ(p0, orig.texts[2]);
  // Absent text: npos, set unchanged.
  auto before = set.texts;
  EXPECT_EQ(MoveDisplayedToFront(&set, "不存在"), std::string::npos);
  EXPECT_EQ(set.texts, before);
}

TEST(PreciseCandidatesTest, GateNotEvaluatedBelowTwoCandidates) {
  auto lm = MakeTinyLM();
  auto g = TinyGrid(lm);
  PreciseConfig c;
  c.k = 10;
  c.tau = 1.0;  // gate would always open if evaluated
  ASSERT_TRUE(g->overrideCandidate(0, "乙"));
  ASSERT_TRUE(g->overrideCandidate(1, "壬癸"));
  auto set = PreciseCandidates(*g, TinyScores(), c);
  EXPECT_GE(set.size(), 2u);
  FilterFixedPositions(&set, FixedPositionsOf(*g));
  ASSERT_EQ(set.size(), 1u);
  EXPECT_EQ(set.texts[0], "乙壬癸");
  EXPECT_FALSE(PreciseGate(g->readings(), set, c).has_value());
  EXPECT_FALSE(PreciseGate(g->readings(), PreciseCandidateSet{}, c).has_value());
  // >= 2 candidates: gate evaluated with no heavy call.
  auto g2 = TinyGrid(lm);
  auto set2 = PreciseCandidates(*g2, TinyScores(), c);
  auto d = PreciseGate(g2->readings(), set2, c);
  ASSERT_TRUE(d.has_value());
  EXPECT_TRUE(d->gateOpen);
  EXPECT_FALSE(d->heavyCalled);
  EXPECT_EQ(d->chosen, 0u);
  EXPECT_EQ(d->logPR.size(), set2.size());
}

// QwenRescorer-shaped fake: like QwenRescorer::score(), returns {} when the
// candidates' token total exceeds the 512-token batch (worst case 2 tokens
// per Han character here).
class TokenBudgetHeavy : public Rescorer {
 public:
  bool ready() const override { return true; }
  std::vector<double> score(const RescoreRequest& req) override {
    ++calls;
    size_t total = 0;
    for (const auto& c : req.candidates) {
      size_t n = 0;
      for (unsigned char ch : c) n += (ch & 0xC0) != 0x80;
      total += 2 * n;
    }
    if (total > 512) return {};
    return std::vector<double>(req.candidates.size(), 0.0);
  }
  int calls = 0;
};

TEST(PreciseWalkTest, HeavyCannotScoreLongInputKeepsCandidateZero) {
  std::vector<std::string> readings(40, "ㄧ");
  std::vector<std::string> cands;
  std::vector<double> r;
  for (int i = 0; i < 10; ++i) {
    std::string s;
    for (int j = 0; j < 40; ++j) s += (j == i) ? "乙" : "一";
    cands.push_back(s);
    r.push_back(-1.0 - 0.01 * i);
  }
  PreciseConfig c;
  c.tau = 1.0;
  c.rWeight = 0.0;  // heavy alone would decide if it could score
  TokenBudgetHeavy h;
  auto d = PreciseSelect(readings, cands, r, &h, c);
  EXPECT_EQ(h.calls, 1);
  EXPECT_TRUE(d.gateOpen);
  EXPECT_TRUE(d.heavyCalled);
  EXPECT_FALSE(d.heavyOk);
  EXPECT_FALSE(d.heavyTimedOut);  // a failure, not a timeout
  EXPECT_TRUE(d.heavyScores.empty());
  EXPECT_TRUE(d.fused.empty());
  EXPECT_EQ(d.chosen, 0u);
  // The same fake scores a short request (sanity: the fake is not always empty).
  auto ds = PreciseSelect({"a", "b"}, {"一一", "乙一"}, {-1.0, -1.5}, &h, c);
  EXPECT_TRUE(ds.heavyOk);
}

// ---- fixture: recorded R N-best + Qwen scores + Python decision.
struct FixtureRow {
  std::string id;
  std::vector<std::string> readings, cands;
  std::vector<double> rScores, qScores;
  int pyChosen = 0;
  bool pyCalled = false;
};

bool LoadFixture(PreciseConfig* cfg, std::vector<FixtureRow>* rows) {
  std::ifstream f(PRECISE_FIXTURE);
  if (!f) return false;
  std::string line;
  bool first = true;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    json j = json::parse(line);
    if (first) {
      first = false;
      cfg->alpha = j["alpha"];
      cfg->k = j["k"];
      cfg->bucketUpper = j["bucket_upper"].get<std::vector<int>>();
      cfg->temperatures = j["temperatures"].get<std::vector<double>>();
      cfg->tau = j["tau"];
      cfg->rWeight = j["r_weight"];
      cfg->beta = j["beta"];
      continue;
    }
    FixtureRow r;
    r.id = j["id"];
    r.readings = j["reading"].get<std::vector<std::string>>();
    r.cands = j["cands"].get<std::vector<std::string>>();
    r.rScores = j["r_scores"].get<std::vector<double>>();
    r.qScores = j["q_scores"].get<std::vector<double>>();
    r.pyChosen = j["py_chosen"];
    r.pyCalled = j["py_called"];
    rows->push_back(std::move(r));
  }
  return !rows->empty();
}

TEST(PreciseWalkTest, FixtureFusionMatchesPython) {
  PreciseConfig cfg;
  std::vector<FixtureRow> rows;
  if (!LoadFixture(&cfg, &rows)) GTEST_SKIP() << "no fixture " << PRECISE_FIXTURE;
  ASSERT_TRUE(cfg.valid());
  size_t agree = 0;
  for (const auto& r : rows) {
    FakeHeavy h(r.qScores);
    auto d = PreciseSelect(r.readings, r.cands, r.rScores, &h, cfg);
    EXPECT_EQ(d.heavyCalled, r.pyCalled) << r.id << " p=" << d.pTop1;
    EXPECT_EQ(static_cast<int>(d.chosen), r.pyChosen) << r.id;
    agree += static_cast<int>(d.chosen) == r.pyChosen && d.heavyCalled == r.pyCalled;
  }
  EXPECT_EQ(agree, rows.size());
}

class PreciseWalkModelTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    lm_ = std::make_shared<McBopomofoLM>();
    lm_->loadLanguageModel(DATA_TXT);
    model_ = std::make_unique<SlothRescorer>(RWALK_MODEL_DIR);
  }
  static void TearDownTestSuite() {
    model_.reset();
    lm_.reset();
  }
  bool ready() const { return lm_->isDataModelLoaded() && model_->ready(); }
  std::unique_ptr<ReadingGrid> Grid(const std::vector<std::string>& rs) {
    auto g = std::make_unique<ReadingGrid>(lm_);
    for (const auto& r : rs) {
      if (!g->insertReading(r)) return nullptr;
    }
    return g;
  }
  static std::string Join(const ReadingGrid::WalkResult& r) {
    std::string s;
    for (const auto& n : r.nodes) s += n->value();
    return s;
  }
  static std::shared_ptr<McBopomofoLM> lm_;
  static std::unique_ptr<SlothRescorer> model_;
};
std::shared_ptr<McBopomofoLM> PreciseWalkModelTest::lm_;
std::unique_ptr<SlothRescorer> PreciseWalkModelTest::model_;

TEST_F(PreciseWalkModelTest, TauZeroEqualsRWalkAndNoModelEqualsWalk) {
  if (!ready()) GTEST_SKIP() << "model or data.txt missing";
  const std::vector<std::vector<std::string>> sents{
      {"ㄧ", "ㄉㄧㄥˋ", "ㄏㄞˊ", "ㄗㄞˋ", "ㄇㄡˇ", "ㄍㄜ˙", "ㄉㄧˋ", "ㄈㄤ"},
      {"ㄐㄧㄣˇ", "ㄍㄨㄢˇ", "ㄨㄛˇ", "ㄊㄞˊ", "ㄩˇ", "ㄕㄨㄛ", "ㄉㄜ˙", "ㄅㄨˋ", "ㄏㄠˇ"},
      {"ㄍㄠ", "ㄎㄜ", "ㄐㄧˋ", "ㄍㄨㄥ", "ㄙ"}};
  PreciseConfig c;
  c.alpha = 0.5;
  c.k = 10;
  c.tau = 0.0;
  NeuralWalkOptions o;
  o.alpha = 0.5;
  for (const auto& rs : sents) {
    auto g = Grid(rs);
    ASSERT_NE(g, nullptr);
    FakeHeavy h(std::vector<double>(10, 0.0));
    auto r = PreciseWalk(*g, model_.get(), &h, c);
    const std::string rw = Join(NeuralWalk(*g, model_.get(), o));
    EXPECT_TRUE(r.usedNeural);
    EXPECT_EQ(h.calls, 0);
    ASSERT_FALSE(r.candidates.empty());
    EXPECT_EQ(r.candidates[0], rw);
    EXPECT_EQ(Join(r.walk), rw);
    for (size_t i = 1; i < r.rScores.size(); ++i) EXPECT_LE(r.rScores[i], r.rScores[i - 1]);
    // No R model -> plain walk.
    auto p = PreciseWalk(*g, nullptr, &h, c);
    EXPECT_FALSE(p.usedNeural);
    EXPECT_EQ(Join(p.walk), Join(g->walk()));
  }
}

TEST_F(PreciseWalkModelTest, FixtureNBestReproduced) {
  if (!ready()) GTEST_SKIP() << "model or data.txt missing";
  PreciseConfig cfg;
  std::vector<FixtureRow> rows;
  if (!LoadFixture(&cfg, &rows)) GTEST_SKIP() << "no fixture";
  cfg.tau = 0.0;
  size_t n = 0;
  for (const auto& row : rows) {
    if (n++ >= 40) break;
    auto g = Grid(row.readings);
    ASSERT_NE(g, nullptr) << row.id;
    auto r = PreciseWalk(*g, model_.get(), nullptr, cfg);
    ASSERT_EQ(r.candidates.size(), row.cands.size()) << row.id;
    for (size_t i = 0; i < r.candidates.size(); ++i) {
      EXPECT_EQ(r.candidates[i], row.cands[i]) << row.id << " #" << i;
      EXPECT_NEAR(r.rScores[i], row.rScores[i], 1e-3) << row.id << " #" << i;
    }
  }
}

}  // namespace
}  // namespace McBopomofo
