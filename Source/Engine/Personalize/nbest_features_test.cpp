// Task 2.7: N-best feature builder on the shipped data.txt against 20 E5 rows
// (testdata/nbest_fixture.jsonl: 10 simulated-pool + 10 replay rows; texts
// and scores from walkNBest(50), identical to bench/src/decoder.py, E5).

#include <gtest/gtest.h>

#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "McBopomofoLM.h"
#include "nbest_features.h"
#include "nlohmann/json.hpp"

namespace McBopomofo {
namespace Personalize {
namespace {

using nlohmann::json;

std::shared_ptr<McBopomofoLM> SharedLM() {
  static std::shared_ptr<McBopomofoLM> lm = [] {
    auto m = std::make_shared<McBopomofoLM>();
    m->loadLanguageModel(DATA_TXT);
    return m;
  }();
  return lm;
}

TEST(NBestFeaturesTest, MatchesE5Fixture) {
  auto lm = SharedLM();
  ASSERT_TRUE(lm->isDataModelLoaded());
  ApplyReadingPolicy(lm.get(), true);  // E5 ran with an empty reading policy
  std::ifstream in(std::string(PERSONALIZE_TESTDATA) + "/nbest_fixture.jsonl");
  std::string line;
  size_t rows = 0, golds = 0;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    json j = json::parse(line);
    NBestFeatures f;
    ASSERT_TRUE(BuildNBest(lm, j["reading"].get<std::vector<std::string>>(), j["text"], 50, &f));
    const auto texts = j["nbest"].get<std::vector<std::string>>();
    const auto scores = j["ngram"].get<std::vector<double>>();
    EXPECT_EQ(f.texts, texts) << j["id"];
    ASSERT_EQ(f.scores.size(), scores.size());
    for (size_t i = 0; i < scores.size(); ++i) EXPECT_NEAR(f.scores[i], scores[i], 1e-9) << j["id"];
    EXPECT_EQ(f.gold, j["gold"].get<int>()) << j["id"];
    golds += f.gold >= 0;
    ++rows;
  }
  EXPECT_EQ(rows, 20u);
  EXPECT_GT(golds, 10u);
}

TEST(NBestFeaturesTest, ReadingPolicyFlag) {
  auto lm = SharedLM();
  const std::vector<std::string> readings = {"ㄎㄤˋ", "ㄧㄢˊ"};
  NBestFeatures accept, reject;
  ApplyReadingPolicy(lm.get(), true);
  ASSERT_TRUE(BuildNBest(lm, readings, "抗癌", 50, &accept));
  ApplyReadingPolicy(lm.get(), false);
  ASSERT_TRUE(BuildNBest(lm, readings, "抗癌", 50, &reject));
  ApplyReadingPolicy(lm.get(), true);
  EXPECT_GE(accept.gold, 0);
  EXPECT_EQ(reject.gold, -1);  // 癌 is not a ㄧㄢˊ candidate under the default policy
  for (const auto& t : reject.texts) EXPECT_EQ(t.find("癌"), std::string::npos) << t;
}

TEST(NBestFeaturesTest, UnknownReadingFails) {
  NBestFeatures f;
  EXPECT_FALSE(BuildNBest(SharedLM(), {"ㄓㄨㄥ", "xyz"}, "中", 50, &f));
  EXPECT_TRUE(f.texts.empty());
  EXPECT_FALSE(BuildNBest(SharedLM(), {}, "", 50, &f));
  EXPECT_FALSE(BuildNBest(nullptr, {"ㄓㄨㄥ"}, "中", 50, &f));
}

}  // namespace
}  // namespace Personalize
}  // namespace McBopomofo
