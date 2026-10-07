// Task 2.6: temperature refit (C++ port of precise_calib.py step 1).
// testdata/calib_fit_fixture.jsonl = the factory fit set (general dev 200 +
// calib 378): syllable count, R-walk top-10 path scores and whether the R-walk
// pick was correct; header = factory T (models/qwen/precise.conf) and the
// Python values (tools/make_fixtures.py).

#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "temperature.h"

namespace McBopomofo {
namespace Personalize {
namespace {

using nlohmann::json;

struct Fixture {
  json head;
  std::vector<CalibRow> rows;
};

Fixture Load() {
  Fixture f;
  std::ifstream in(std::string(PERSONALIZE_TESTDATA) + "/calib_fit_fixture.jsonl");
  std::string line;
  bool first = true;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    json j = json::parse(line);
    if (first) {
      f.head = j;
      first = false;
      continue;
    }
    CalibRow r;
    r.nSyl = j["n"].get<size_t>();
    r.rScores = j["r"].get<std::vector<double>>();
    r.top1Correct = j["y"].get<bool>();
    f.rows.push_back(std::move(r));
  }
  return f;
}

double GridStep() {
  const auto& g = LogTGrid();
  return g[1] - g[0];
}

TEST(TemperatureTest, GridMatchesNumPyLinspace) {
  const auto& g = LogTGrid();
  ASSERT_EQ(g.size(), 1201u);
  EXPECT_DOUBLE_EQ(g.front(), std::log(0.02));
  EXPECT_DOUBLE_EQ(g.back(), std::log(50.0));
  EXPECT_NEAR(GridStep(), (std::log(50.0) - std::log(0.02)) / 1200, 1e-15);
}

TEST(TemperatureTest, GlobalRefitReproducesTheFactoryTemperature) {
  Fixture f = Load();
  ASSERT_EQ(f.rows.size(), 578u);
  const size_t K = f.head["k"].get<size_t>();
  const double factoryT = f.head["factory_T"].get<double>();
  const double T = FitTemperature(f.rows, K);
  EXPECT_LE(std::fabs(std::log(T) - std::log(factoryT)), GridStep() + 1e-12)
      << "T " << T << " factory " << factoryT;
  EXPECT_NEAR(T, f.head["py_T_global"].get<double>(), 1e-12);  // same grid point as Python
}

TEST(TemperatureTest, BucketRefitMatchesPython) {
  Fixture f = Load();
  const size_t K = f.head["k"].get<size_t>();
  std::vector<size_t> upper = f.head["bucket_upper"].get<std::vector<size_t>>();
  std::vector<double> py = f.head["py_T_bucket"].get<std::vector<double>>();
  std::vector<double> t = FitTemperatures(f.rows, K, upper, true);
  ASSERT_EQ(t.size(), py.size());
  for (size_t b = 0; b < t.size(); ++b)
    EXPECT_LE(std::fabs(std::log(t[b]) - std::log(py[b])), GridStep() + 1e-12) << b;
  std::vector<double> g = FitTemperatures(f.rows, K, upper, false);
  ASSERT_EQ(g.size(), 3u);
  EXPECT_EQ(g[0], g[2]);
  EXPECT_EQ(BucketOf(7, upper), 0u);
  EXPECT_EQ(BucketOf(8, upper), 1u);
  EXPECT_EQ(BucketOf(10, upper), 1u);
  EXPECT_EQ(BucketOf(11, upper), 2u);
}

TEST(TemperatureTest, EceAndLogLossAsInPython) {
  Fixture f = Load();
  const size_t K = f.head["k"].get<size_t>();
  std::vector<size_t> upper = f.head["bucket_upper"].get<std::vector<size_t>>();
  const double Tg = f.head["py_T_global"].get<double>();
  CalibQuality q = EvaluateCalibration(f.rows, K, upper, {Tg, Tg, Tg});
  EXPECT_NEAR(q.eceBefore, f.head["ece_T1"].get<double>(), 1e-12);
  EXPECT_NEAR(q.eceAfter, f.head["ece_global"].get<double>(), 1e-12);
  EXPECT_NEAR(q.logLossBefore, f.head["logloss_T1"].get<double>(), 1e-12);
  EXPECT_NEAR(q.logLossAfter, f.head["logloss_global"].get<double>(), 1e-12);
  std::vector<double> tb = f.head["py_T_bucket"].get<std::vector<double>>();
  CalibQuality qb = EvaluateCalibration(f.rows, K, upper, tb);
  EXPECT_NEAR(qb.eceAfter, f.head["ece_bucket"].get<double>(), 1e-12);
  EXPECT_LT(q.eceAfter, q.eceBefore);
}

TEST(TemperatureTest, EdgeCases) {
  EXPECT_EQ(FitTemperature({}, 10), 1.0);
  CalibRow one;
  one.rScores = {-1.0};
  one.top1Correct = true;
  EXPECT_DOUBLE_EQ(PTop1(one.rScores, 1.0, 10), 1.0);
  EXPECT_TRUE(std::isfinite(FitTemperature({one}, 10)));
  EXPECT_EQ(Ece({}, {}), 0.0);
  // A sharper R needs a higher temperature: scaling scores by 2 doubles T.
  Fixture f = Load();
  std::vector<CalibRow> sharp = f.rows;
  for (auto& r : sharp)
    for (auto& s : r.rScores) s *= 2;
  const double t1 = FitTemperature(f.rows, 10), t2 = FitTemperature(sharp, 10);
  EXPECT_LE(std::fabs(std::log(t2 / t1) - std::log(2.0)), GridStep() + 1e-12);
}

}  // namespace
}  // namespace Personalize
}  // namespace McBopomofo
