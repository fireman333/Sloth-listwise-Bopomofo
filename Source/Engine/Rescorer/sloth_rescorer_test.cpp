#include "sloth_rescorer.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace McBopomofo {
namespace {

std::string ModelDir() {
  const char* e = std::getenv("SLOTH_MODEL_DIR");
  return e ? e : SLOTH_MODEL_DIR;
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

// Fixture lines (tab separated): id, readings (space separated), then
// repeated (candidate, python score) pairs. Produced by bench/src/rescore_sloth.py.
std::vector<Case> LoadFixture() {
  std::vector<Case> cases;
  std::ifstream in(SLOTH_FIXTURE);
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

TEST(SlothRescorerTest, MatchesPythonScores) {
  SlothRescorer r(ModelDir());
  ASSERT_TRUE(r.ready());
  auto cases = LoadFixture();
  ASSERT_EQ(cases.size(), 5u);
  double maxDiff = 0;
  for (const auto& c : cases) {
    auto s = r.score(c.req);
    ASSERT_EQ(s.size(), c.expected.size()) << c.id;
    for (size_t i = 0; i < s.size(); ++i) {
      double d = std::fabs(s[i] - c.expected[i]);
      maxDiff = std::max(maxDiff, d);
      EXPECT_LT(d, 1e-3) << c.id << " cand " << i << " cpp=" << s[i] << " py=" << c.expected[i];
    }
    // Same argmax as Python.
    auto am = [](const std::vector<double>& v) {
      return std::max_element(v.begin(), v.end()) - v.begin();
    };
    EXPECT_EQ(am(s), am(c.expected)) << c.id;
  }
  std::printf("[ INFO ] max |cpp - python| over fixture = %.3g\n", maxDiff);
}

TEST(SlothRescorerTest, BadModelPathNotReady) {
  SlothRescorer r("/nonexistent/path/to/sloth");
  EXPECT_FALSE(r.ready());
  RescoreRequest req;
  req.readings = {"ㄧ", "ㄕㄥ"};
  req.candidates = {"一生", "醫生"};
  EXPECT_TRUE(r.score(req).empty());
}

TEST(SlothRescorerTest, EdgeCases) {
  SlothRescorer r(ModelDir());
  ASSERT_TRUE(r.ready());
  RescoreRequest req;
  EXPECT_TRUE(r.score(req).empty());
  req.readings = {"ㄧ", "ㄕㄥ"};
  EXPECT_TRUE(r.score(req).empty());  // no candidates
  req.candidates = {"醫生", "一聲", "ab"};  // third has non-vocab chars
  auto s = r.score(req);
  ASSERT_EQ(s.size(), 3u);
  EXPECT_LT(s[2], s[0] - 20);  // out-of-vocab chars get -30 each
}

TEST(SlothRescorerTest, LatencyTop10) {
  SlothRescorer r(ModelDir());
  ASSERT_TRUE(r.ready());
  auto cases = LoadFixture();
  ASSERT_FALSE(cases.empty());
  for (int i = 0; i < 20; ++i) r.score(cases[0].req);  // warm up
  std::vector<double> ms;
  for (int rep = 0; rep < 20; ++rep) {
    for (const auto& c : cases) {
      auto t0 = std::chrono::steady_clock::now();
      auto s = r.score(c.req);
      auto t1 = std::chrono::steady_clock::now();
      ASSERT_EQ(s.size(), c.req.candidates.size());
      ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
  }
  std::sort(ms.begin(), ms.end());
  std::printf("[ INFO ] 10-candidate score(): n=%zu p50=%.3f ms p95=%.3f ms max=%.3f ms\n",
              ms.size(), ms[ms.size() / 2], ms[static_cast<size_t>(ms.size() * 0.95)], ms.back());
  EXPECT_LT(ms[ms.size() / 2], 10.0);
}

}  // namespace
}  // namespace McBopomofo
