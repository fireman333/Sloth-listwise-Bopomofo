#include "ensemble_rescorer.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#ifdef ENSEMBLE_WITH_REAL_MODELS
#include "electra_rescorer.h"
#include "sloth_rescorer.h"
#endif

namespace McBopomofo {
namespace {

class Fake : public Rescorer {
 public:
  Fake(bool ready, std::vector<double> scores) : ready_(ready), scores_(std::move(scores)) {}
  bool ready() const override { return ready_; }
  std::vector<double> score(const RescoreRequest&) override {
    ++calls;
    return scores_;
  }
  int calls = 0;

 private:
  bool ready_;
  std::vector<double> scores_;
};

RescoreRequest Req(size_t n) {
  RescoreRequest r;
  r.readings = {"ㄧ", "ㄕㄥ"};
  for (size_t i = 0; i < n; ++i) r.candidates.push_back("c" + std::to_string(i));
  return r;
}

size_t ArgMax(const std::vector<double>& v) {
  return static_cast<size_t>(std::max_element(v.begin(), v.end()) - v.begin());
}

TEST(EnsembleRescorerTest, ZScoreBasics) {
  auto z = EnsembleRescorer::ZScore({1, 2, 3});
  ASSERT_EQ(z.size(), 3u);
  EXPECT_NEAR(z[0], -std::sqrt(1.5), 1e-12);  // population std = sqrt(2/3)
  EXPECT_NEAR(z[1], 0.0, 1e-12);
  EXPECT_NEAR(z[2], std::sqrt(1.5), 1e-12);
  for (double x : EnsembleRescorer::ZScore({5, 5, 5})) EXPECT_EQ(x, 0.0);  // constant -> zeros
}

TEST(EnsembleRescorerTest, WeightedSumOfZScores) {
  auto a = std::make_shared<Fake>(true, std::vector<double>{0, 10, 20});
  auto b = std::make_shared<Fake>(true, std::vector<double>{-3, -1, -2});
  EnsembleRescorer e({{a, 0.7, "a"}, {b, 0.3, "b"}});
  ASSERT_TRUE(e.ready());
  auto s = e.score(Req(3));
  ASSERT_EQ(s.size(), 3u);
  auto za = EnsembleRescorer::ZScore({0, 10, 20}), zb = EnsembleRescorer::ZScore({-3, -1, -2});
  for (size_t i = 0; i < 3; ++i) EXPECT_NEAR(s[i], 0.7 * za[i] + 0.3 * zb[i], 1e-12);
  EXPECT_EQ(e.lastActiveMembers(), 2u);
  EXPECT_EQ(e.lastActiveMask(), 3u);
}

TEST(EnsembleRescorerTest, ScaleInvariantPerMember) {
  // A member whose raw scale is 1000x larger must not dominate after z-scoring.
  auto a = std::make_shared<Fake>(true, std::vector<double>{1000, 3000, 2000});
  auto b = std::make_shared<Fake>(true, std::vector<double>{0.002, 0.001, 0.003});
  EnsembleRescorer e({{a, 0.5, "a"}, {b, 0.5, "b"}});
  auto s = e.score(Req(3));
  ASSERT_EQ(s.size(), 3u);
  EXPECT_EQ(ArgMax(s), 2u);  // a ranks 1>2>0, b ranks 2>0>1 -> index 2 has best sum
}

TEST(EnsembleRescorerTest, FallsBackWhenMemberNotReady) {
  auto a = std::make_shared<Fake>(false, std::vector<double>{});
  auto b = std::make_shared<Fake>(true, std::vector<double>{1, 3, 2});
  EnsembleRescorer e({{a, 0.5, "a"}, {b, 0.5, "b"}});
  EXPECT_TRUE(e.ready());
  auto s = e.score(Req(3));
  ASSERT_EQ(s.size(), 3u);
  EXPECT_EQ(ArgMax(s), 1u);
  EXPECT_EQ(e.lastActiveMembers(), 1u);
  EXPECT_EQ(e.lastActiveMask(), 2u);
  EXPECT_EQ(a->calls, 0);
}

TEST(EnsembleRescorerTest, FallsBackWhenMemberReturnsEmptyOrWrongSizeOrNaN) {
  auto good = std::make_shared<Fake>(true, std::vector<double>{3, 1, 2});
  for (auto bad : {std::vector<double>{}, std::vector<double>{1, 2},
                   std::vector<double>{1, std::numeric_limits<double>::quiet_NaN(), 2},
                   std::vector<double>{1, std::numeric_limits<double>::infinity(), 2}}) {
    auto b = std::make_shared<Fake>(true, bad);
    EnsembleRescorer e({{b, 0.5, "bad"}, {good, 0.5, "good"}});
    auto s = e.score(Req(3));
    ASSERT_EQ(s.size(), 3u);
    EXPECT_EQ(ArgMax(s), 0u);
    EXPECT_EQ(e.lastActiveMembers(), 1u);
    EXPECT_EQ(e.lastActiveMask(), 2u);
  }
}

TEST(EnsembleRescorerTest, AllMembersFailReturnsEmpty) {
  auto a = std::make_shared<Fake>(false, std::vector<double>{});
  auto b = std::make_shared<Fake>(true, std::vector<double>{});
  EnsembleRescorer e({{a, 0.5, "a"}, {b, 0.5, "b"}});
  EXPECT_TRUE(e.score(Req(3)).empty());
  EXPECT_EQ(e.lastActiveMembers(), 0u);
  EnsembleRescorer none({});
  EXPECT_FALSE(none.ready());
  EXPECT_TRUE(none.score(Req(3)).empty());
  EnsembleRescorer nullMember({{nullptr, 1.0, "null"}});
  EXPECT_FALSE(nullMember.ready());
  EXPECT_TRUE(nullMember.score(Req(3)).empty());
  EXPECT_TRUE(EnsembleRescorer({{a, 1, "a"}}).score(Req(0)).empty());
}

TEST(EnsembleRescorerTest, ParallelMatchesSequential) {
  auto a = std::make_shared<Fake>(true, std::vector<double>{1, 5, 2, 9});
  auto b = std::make_shared<Fake>(true, std::vector<double>{4, 1, 8, 2});
  EnsembleRescorer seq({{a, 0.4, "a"}, {b, 0.6, "b"}});
  EnsembleRescorer::Options o;
  o.parallel = true;
  EnsembleRescorer par({{a, 0.4, "a"}, {b, 0.6, "b"}}, o);
  auto s1 = seq.score(Req(4)), s2 = par.score(Req(4));
  ASSERT_EQ(s1.size(), 4u);
  ASSERT_EQ(s2.size(), 4u);
  for (size_t i = 0; i < 4; ++i) EXPECT_NEAR(s1[i], s2[i], 1e-12);
}

#ifdef ENSEMBLE_WITH_REAL_MODELS
std::vector<std::string> Split(const std::string& s, char d) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, d)) out.push_back(tok);
  return out;
}

// de_fixture.tsv: id, readings, then (candidate, python ensemble score at w=0.5
// using D = Sloth int8, E = ELECTRA int8) pairs.
TEST(EnsembleRescorerTest, RealModelsMatchPythonEnsemble) {
  auto d = std::make_shared<SlothRescorer>(SLOTH_MODEL_DIR);
  auto e = std::make_shared<ElectraRescorer>(ELECTRA_MODEL_DIR);
  ASSERT_TRUE(d->ready());
  ASSERT_TRUE(e->ready());
  EnsembleRescorer de({{d, 0.5, "D"}, {e, 0.5, "E"}});
  std::ifstream in(DE_FIXTURE);
  ASSERT_TRUE(in.good());
  std::string line;
  int n = 0;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    auto f = Split(line, '\t');
    RescoreRequest r;
    r.readings = Split(f[1], ' ');
    std::vector<double> exp;
    for (size_t i = 2; i + 1 < f.size(); i += 2) {
      r.candidates.push_back(f[i]);
      exp.push_back(std::stod(f[i + 1]));
    }
    auto s = de.score(r);
    ASSERT_EQ(s.size(), exp.size()) << f[0];
    for (size_t i = 0; i < s.size(); ++i) EXPECT_NEAR(s[i], exp[i], 2e-3) << f[0] << " cand " << i;
    EXPECT_EQ(ArgMax(s), ArgMax(exp)) << f[0];
    EXPECT_EQ(de.lastActiveMembers(), 2u);
    ++n;
  }
  EXPECT_GE(n, 5);
}

TEST(EnsembleRescorerTest, RealModelsFallBackWhenOneModelMissing) {
  auto d = std::make_shared<SlothRescorer>(SLOTH_MODEL_DIR);
  auto bad = std::make_shared<ElectraRescorer>("/nonexistent/electra");
  ASSERT_TRUE(d->ready());
  ASSERT_FALSE(bad->ready());
  EnsembleRescorer de({{d, 0.5, "D"}, {bad, 0.5, "E"}});
  EXPECT_TRUE(de.ready());
  RescoreRequest r;
  r.readings = {"ㄧ", "ㄕㄥ"};
  r.candidates = {"醫生", "一聲", "異聲"};
  auto s = de.score(r);
  auto direct = d->score(r);
  ASSERT_EQ(s.size(), 3u);
  EXPECT_EQ(ArgMax(s), ArgMax(direct));
  EXPECT_EQ(de.lastActiveMask(), 1u);
  // Both missing -> empty.
  auto bad2 = std::make_shared<SlothRescorer>("/nonexistent/sloth");
  EnsembleRescorer none({{bad2, 0.5, "D"}, {bad, 0.5, "E"}});
  EXPECT_FALSE(none.ready());
  EXPECT_TRUE(none.score(r).empty());
}
#endif

}  // namespace
}  // namespace McBopomofo
