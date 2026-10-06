#include "electra_rescorer.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace McBopomofo {
namespace {

std::string ModelDir() {
  const char* e = std::getenv("ELECTRA_MODEL_DIR");
  return e ? e : ELECTRA_MODEL_DIR;
}

std::vector<std::string> Split(const std::string& s, char d) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, d)) out.push_back(tok);
  return out;
}

struct Case {
  std::string id;
  RescoreRequest req;
  std::vector<double> expected;
};

// Fixture lines (tab separated): id, readings (space separated), then repeated
// (candidate, python score) pairs. Python score = int8 ONNX model through
// bench/src/ensemble_DE.py (E8), HF tokenizer ids, run with onnxruntime 1.30.0
// (same version as the vendored C++ one: int8 kernels differ numerically across
// ORT versions, ~0.1-1 score units, so fixtures from another version do not match).
std::vector<Case> LoadFixture() {
  std::vector<Case> cases;
  std::ifstream in(ELECTRA_FIXTURE);
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    auto f = Split(line, '\t');
    Case c;
    c.id = f[0];
    c.req.readings = Split(f[1], ' ');
    for (size_t i = 2; i + 1 < f.size(); i += 2) {
      c.req.candidates.push_back(f[i]);
      c.expected.push_back(std::stod(f[i + 1]));
    }
    cases.push_back(std::move(c));
  }
  return cases;
}

TEST(ElectraRescorerTest, MatchesPythonScores) {
  ElectraRescorer r(ModelDir());
  ASSERT_TRUE(r.ready());
  auto cases = LoadFixture();
  ASSERT_GE(cases.size(), 5u);
  double maxDiff = 0;
  for (const auto& c : cases) {
    auto s = r.score(c.req);
    ASSERT_EQ(s.size(), c.expected.size()) << c.id;
    for (size_t i = 0; i < s.size(); ++i) {
      double d = std::fabs(s[i] - c.expected[i]);
      maxDiff = std::max(maxDiff, d);
      EXPECT_LT(d, 1e-3) << c.id << " cand " << i << " cpp=" << s[i] << " py=" << c.expected[i];
    }
    auto am = [](const std::vector<double>& v) {
      return std::max_element(v.begin(), v.end()) - v.begin();
    };
    EXPECT_EQ(am(s), am(c.expected)) << c.id;
  }
  std::printf("[ INFO ] max |cpp - python| over fixture = %.3g\n", maxDiff);
}

TEST(ElectraRescorerTest, BadModelPathNotReady) {
  ElectraRescorer r("/nonexistent/path/to/electra");
  EXPECT_FALSE(r.ready());
  RescoreRequest req;
  req.readings = {"ㄧ", "ㄕㄥ"};
  req.candidates = {"一生", "醫生"};
  EXPECT_TRUE(r.score(req).empty());
}

TEST(ElectraRescorerTest, MissingOnnxFileNotReady) {
  ElectraRescorer::Options o;
  o.onnxFile = "does_not_exist.onnx";
  ElectraRescorer r(ModelDir(), o);
  EXPECT_FALSE(r.ready());
  EXPECT_TRUE(r.score(RescoreRequest{{"ㄧ"}, {"一"}}).empty());
}

TEST(ElectraRescorerTest, EdgeCases) {
  ElectraRescorer r(ModelDir());
  ASSERT_TRUE(r.ready());
  RescoreRequest req;
  EXPECT_TRUE(r.score(req).empty());  // no candidates
  req.readings = {"ㄧ", "ㄕㄥ"};
  req.candidates = {"醫生", "一聲", "ab", "醫生醫生"};  // OOV chars -> [UNK]; mixed lengths are padded
  auto s = r.score(req);
  ASSERT_EQ(s.size(), 4u);
  for (double x : s) EXPECT_TRUE(std::isfinite(x));
  // Dynamic int8 quantization picks activation scales per batch, so scoring one
  // candidate alone is only approximately equal to scoring it inside a batch.
  RescoreRequest one;
  one.readings = req.readings;
  one.candidates = {"醫生"};
  auto s1 = r.score(one);
  ASSERT_EQ(s1.size(), 1u);
  EXPECT_NEAR(s1[0], s[0], 0.1);
}

TEST(ElectraRescorerTest, LatencyTop10) {
  ElectraRescorer r(ModelDir());
  ASSERT_TRUE(r.ready());
  auto cases = LoadFixture();
  ASSERT_FALSE(cases.empty());
  for (int i = 0; i < 10; ++i) r.score(cases[0].req);
  std::vector<double> ms;
  for (int rep = 0; rep < 10; ++rep) {
    for (const auto& c : cases) {
      auto t0 = std::chrono::steady_clock::now();
      auto s = r.score(c.req);
      auto t1 = std::chrono::steady_clock::now();
      ASSERT_EQ(s.size(), c.req.candidates.size());
      ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
  }
  std::sort(ms.begin(), ms.end());
  std::printf("[ INFO ] E 10-candidate score(): n=%zu p50=%.3f ms p95=%.3f ms max=%.3f ms\n",
              ms.size(), ms[ms.size() / 2], ms[static_cast<size_t>(ms.size() * 0.95)], ms.back());
  // Loose bound, skipped when the machine is saturated by other jobs.
  double la[1] = {0};
  getloadavg(la, 1);
  if (la[0] > 2.0 * std::thread::hardware_concurrency()) {
    GTEST_SKIP() << "load average " << la[0] << " too high for a latency assertion";
  }
  EXPECT_LT(ms[ms.size() / 2], 100.0);
}

}  // namespace
}  // namespace McBopomofo
