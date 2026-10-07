// Task 2.5: exact McNemar p-values against ranker_lib.mcnemar_exact / scipy
// (testdata/mcnemar_fixture.tsv, tools/make_fixtures.py), every reason code,
// thresholds from pack.json.

#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "gate.h"

namespace McBopomofo {
namespace Personalize {
namespace {

TEST(GateTest, McNemarMatchesPythonFixture) {
  std::ifstream in(std::string(PERSONALIZE_TESTDATA) + "/mcnemar_fixture.tsv");
  ASSERT_TRUE(in);
  std::string line;
  int n = 0;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    uint64_t b, c;
    double two, one;
    ss >> b >> c >> two >> one;
    EXPECT_NEAR(McNemarExactTwoSided(b, c), two, 1e-12) << b << " " << c;
    EXPECT_NEAR(McNemarExactOneSided(b, c), one, 1e-12) << b << " " << c;
    ++n;
  }
  EXPECT_GE(n, 10);
}

TEST(GateTest, McNemarLargeTablesStayFinite) {
  // Beyond double range of 2^-n (n > 1074): still finite and in [0, 1].
  double p = McNemarExactTwoSided(1500, 1400);
  EXPECT_TRUE(std::isfinite(p));
  EXPECT_GT(p, 0.0);
  EXPECT_LT(p, 1.0);
  EXPECT_EQ(McNemarExactTwoSided(3000, 0), 0.0);  // underflows to 0, never NaN
  EXPECT_EQ(McNemarExactTwoSided(700, 700), 1.0);
}

// n sentences; the first `ok` are correct.
std::vector<bool> Vec(size_t n, size_t ok) {
  std::vector<bool> v(n, false);
  for (size_t i = 0; i < ok; ++i) v[i] = true;
  return v;
}

// Personal test of 200: factory correct on [0, fOk), candidate on
// [0, fOk - losses) U [fOk, fOk + wins).
GateInput Input(size_t fOk, size_t wins, size_t losses, size_t genF, size_t genC) {
  GateInput in;
  const size_t n = 200;
  in.factoryPersonal = Vec(n, fOk);
  in.candidatePersonal = Vec(n, 0);
  for (size_t i = 0; i < fOk - losses; ++i) in.candidatePersonal[i] = true;
  for (size_t i = fOk; i < fOk + wins; ++i) in.candidatePersonal[i] = true;
  in.factoryGeneral = Vec(1000, genF);
  in.candidateGeneral = Vec(1000, genC);
  return in;
}

TEST(GateTest, ReasonCodes) {
  GateThresholds th;
  th.oneSided = false;  // two-sided here; one-sided (the default) below
  // adopted: +5 pt (12 wins, 2 losses -> p = 0.013), general -0.4 pt.
  GateResult r = EvaluateGate(Input(100, 12, 2, 800, 796), th);
  EXPECT_TRUE(r.valid);
  EXPECT_TRUE(r.adopted);
  EXPECT_EQ(r.reason, "adopted");
  EXPECT_NEAR(r.deltaPersonalPt, 5.0, 1e-9);
  EXPECT_NEAR(r.deltaGeneralPt, -0.4, 1e-9);
  EXPECT_EQ(r.wins, 12u);
  EXPECT_EQ(r.losses, 2u);
  EXPECT_NEAR(r.p, McNemarExactTwoSided(12, 2), 1e-15);
  EXPECT_LT(r.p, 0.05);

  // general_regression: +6 pt but general -1.5 pt (spec scenario).
  r = EvaluateGate(Input(100, 13, 1, 800, 785), th);
  EXPECT_FALSE(r.adopted);
  EXPECT_EQ(r.reason, "general_regression");
  EXPECT_EQ(r.failedRules, std::vector<std::string>{"general_regression"});

  // exactly -1.0 pt general is allowed.
  r = EvaluateGate(Input(100, 13, 1, 800, 790), th);
  EXPECT_TRUE(r.adopted) << r.reason;

  // not_significant: +2.5 pt with p ~ 0.2 (spec scenario).
  r = EvaluateGate(Input(100, 13, 8, 800, 800), th);
  EXPECT_NEAR(r.deltaPersonalPt, 2.5, 1e-9);
  EXPECT_GT(r.p, 0.05);
  EXPECT_EQ(r.reason, "not_significant");

  // too_small_gain: +1.5 pt even though discordant pairs are lopsided.
  r = EvaluateGate(Input(100, 3, 0, 800, 800), th);
  EXPECT_EQ(r.reason, "too_small_gain");
  EXPECT_EQ(r.failedRules, (std::vector<std::string>{"too_small_gain", "not_significant"}));

  // worse_than_active: candidate +5 pt but the active version had +6 pt.
  GateInput in = Input(100, 12, 2, 800, 796);
  in.activePersonal = Vec(200, 112);
  r = EvaluateGate(in, th);
  EXPECT_EQ(r.reason, "worse_than_active");
  EXPECT_NEAR(r.activePersonalPct, 56.0, 1e-9);
  // ... unless rule 3 is off in the pack.
  GateThresholds noActive = th;
  noActive.notWorseThanActive = false;
  EXPECT_TRUE(EvaluateGate(in, noActive).adopted);
  // equal to the active version passes.
  in.activePersonal = Vec(200, 110);
  EXPECT_TRUE(EvaluateGate(in, th).adopted);

  // failed:gate on inconsistent input.
  in = Input(100, 12, 2, 800, 796);
  in.candidateGeneral.pop_back();
  r = EvaluateGate(in, th);
  EXPECT_FALSE(r.valid);
  EXPECT_FALSE(r.adopted);
  EXPECT_EQ(r.reason, "failed:gate");
  EXPECT_FALSE(EvaluateGate(GateInput(), th).valid);
}

TEST(GateTest, OneSidedChangesTheDecision) {
  // 10 wins, 3 losses: two-sided p = 0.092, one-sided p = 0.046 (+3.5 pt).
  GateInput in = Input(100, 10, 3, 800, 800);
  GateThresholds th;
  EXPECT_TRUE(th.oneSided);  // research 17 default
  th.oneSided = false;
  EXPECT_EQ(EvaluateGate(in, th).reason, "not_significant");
  th.oneSided = true;
  GateResult r = EvaluateGate(in, th);
  EXPECT_TRUE(r.adopted);
  EXPECT_NEAR(r.p, 0.046142578125, 1e-12);
  // One-sided never adopts a candidate that is worse on the discordant pairs.
  r = EvaluateGate(Input(100, 3, 10, 800, 800), th);
  EXPECT_GT(r.p, 0.5);
  EXPECT_FALSE(r.adopted);
}

TEST(GateTest, ThresholdsFromPackJson) {
  GateThresholds g;
  DataThresholds d;
  std::string err;
  ASSERT_TRUE(ParsePackThresholds(R"({"factory_sha256":"ab","gate":{"min_personal_gain_pt":3.0,
      "p_max":0.01,"mcnemar":"one-sided","max_general_drop_pt":0.5,"not_worse_than_active":false},
      "data":{"min_eligible":1500,"holdout_fraction":0.15,"holdout_min":250,"holdout_max":500,"train_cap":3000}})",
                                  &g, &d, &err))
      << err;
  EXPECT_EQ(g.minPersonalGainPt, 3.0);
  EXPECT_EQ(g.pMax, 0.01);
  EXPECT_TRUE(g.oneSided);
  EXPECT_EQ(g.maxGeneralDropPt, 0.5);
  EXPECT_FALSE(g.notWorseThanActive);
  EXPECT_EQ(d.minEligible, 1500u);
  EXPECT_EQ(d.holdoutFraction, 0.15);
  EXPECT_EQ(d.holdoutMin, 250u);
  EXPECT_EQ(d.holdoutMax, 500u);
  EXPECT_EQ(d.trainCap, 3000u);
  EXPECT_EQ(d.minSyllables, 2u);   // default kept
  EXPECT_EQ(d.maxSyllables, 40u);

  // The pack thresholds drive the decision: one-sided p = 0.0065 < 0.01 and
  // +5 pt >= 3 pass, but -0.8 pt general exceeds the 0.5 pt allowance.
  GateResult r = EvaluateGate(Input(100, 12, 2, 800, 792), g);
  EXPECT_EQ(r.reason, "general_regression");
  EXPECT_NEAR(r.p, McNemarExactOneSided(12, 2), 1e-15);
  EXPECT_TRUE(EvaluateGate(Input(100, 12, 2, 800, 796), g).adopted);

  // Missing objects keep defaults.
  GateThresholds g2;
  DataThresholds d2;
  ASSERT_TRUE(ParsePackThresholds("{}", &g2, &d2, &err));
  EXPECT_EQ(g2.minPersonalGainPt, 2.0);
  EXPECT_EQ(g2.pMax, 0.05);
  EXPECT_TRUE(g2.oneSided);
  EXPECT_EQ(d2.minEligible, 1000u);
  EXPECT_EQ(d2.holdoutFraction, 0.30);
  EXPECT_EQ(d2.holdoutMin, 300u);
  EXPECT_EQ(d2.holdoutMax, 400u);

  // Errors leave the outputs untouched.
  for (const char* bad : {"", "[]", R"({"gate":{"p_max":"0.05"}})", R"({"gate":{"p_max":2}})",
                          R"({"gate":{"mcnemar":"three-sided"}})", R"({"data":{"holdout_min":-1}})",
                          R"({"data":{"min_syllables":5,"max_syllables":3}})", R"({"gate":[]})",
                          R"({"data":{"holdout_min":300,"holdout_max":200}})"}) {
    GateThresholds g3;
    g3.pMax = 0.123;
    EXPECT_FALSE(ParsePackThresholds(bad, &g3, nullptr, &err)) << bad;
    EXPECT_EQ(g3.pMax, 0.123);
  }
}

}  // namespace
}  // namespace Personalize
}  // namespace McBopomofo
