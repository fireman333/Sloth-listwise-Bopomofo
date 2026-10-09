// Tests for the citation / sandhi tone lookup of 一 and 不 in McBopomofoLM
// (change citation-tone-yi-bu): a multi-syllable key also returns built-in
// phrases listed under the other tone forms of a 一 / 不 syllable when the
// phrase has 一 / 不 at that position, with the listed score, deduplicated and
// ordered by score. The data test runs on the shipped Source/Data/data.txt.

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "AssociatedPhrasesV2.h"
#include "McBopomofoLM.h"
#include "UTF8Helper.h"
#include "gramambular2/reading_grid.h"
#include "gtest/gtest.h"

namespace McBopomofo {
namespace {

using Formosa::Gramambular2::ReadingGrid;
using Unigram = Formosa::Gramambular2::LanguageModel::Unigram;

// Byte-sorted by key; within a key, the order is what the LM returns.
constexpr char kToneLMData[] = R"(
# format org.openvanilla.mcbopomofo.sorted
ㄅㄧㄢˋ 變 -3.2
ㄅㄧㄢˋ 便 -3.4
ㄅㄨˊ-ㄅㄧㄢˋ 不變 -4.67595511
ㄅㄨˊ-ㄅㄧㄢˋ 不便 -4.69728319
ㄅㄨˊ-ㄍㄨㄛˋ 不過 -3.2
ㄅㄨˊ-ㄧㄠˋ 不要 -3.48468449
ㄅㄨˊ-ㄧㄢˊ 不癌 -5.0
ㄅㄨˊ-ㄧㄢˊ 不言 -5.5
ㄅㄨˋ 不 -2.0
ㄅㄨˋ 步 -3.0
ㄅㄨˋ-ㄍㄨㄛˋ 不過 -3.5
ㄅㄨˋ-ㄍㄨㄛˋ 布過 -6.2
ㄅㄨˋ-ㄧㄠˋ 布要 -6.0
ㄉㄧㄥˋ 定 -3.0
ㄓㄨㄥ 中 -2.8
ㄧ 一 -2.1
ㄧ 衣 -3.5
ㄧ-ㄉㄧㄥˋ 一定 -3.41937414
ㄧ-ㄉㄧㄥˋ 宜定 -7.0
ㄧ-ㄓㄨㄥ 一中 -5.0
ㄧˊ-ㄉㄧㄥˋ 一定 -3.41937414
ㄧˋ 意 -3.0
ㄧˋ-ㄓㄨㄥ 意中 -6.5
ㄧㄠˋ 要 -2.5
ㄧㄢˊ 言 -3.6
ㄧㄢˊ 癌 -4.3
)";

std::shared_ptr<McBopomofoLM> ToneLM() {
  auto lm = std::make_shared<McBopomofoLM>();
  lm->loadLanguageModel(std::make_unique<ParselessPhraseDB>(kToneLMData, sizeof(kToneLMData)));
  return lm;
}

std::vector<std::pair<std::string, double>> Pairs(const std::vector<Unigram>& us) {
  std::vector<std::pair<std::string, double>> out;
  for (const auto& u : us) out.emplace_back(u.value(), u.score());
  return out;
}

std::string Walk(const std::shared_ptr<McBopomofoLM>& lm, const std::vector<std::string>& readings) {
  ReadingGrid grid(lm);
  for (const auto& r : readings) EXPECT_TRUE(grid.insertReading(r)) << r;
  std::string text;
  for (const auto& n : grid.walk().nodes) text += n->value();
  return text;
}

TEST(CitationToneTest, CitationFindsSandhiListedPhrase) {
  auto lm = ToneLM();
  std::vector<std::pair<std::string, double>> expected{{"不要", -3.48468449}, {"布要", -6.0}};
  EXPECT_EQ(Pairs(lm->getUnigrams("ㄅㄨˋ-ㄧㄠˋ")), expected);
}

TEST(CitationToneTest, VariantOnlyKeyHasUnigramsInScoreOrder) {
  auto lm = ToneLM();
  EXPECT_TRUE(lm->hasUnigrams("ㄅㄨˋ-ㄅㄧㄢˋ"));
  std::vector<std::pair<std::string, double>> expected{{"不變", -4.67595511}, {"不便", -4.69728319}};
  EXPECT_EQ(Pairs(lm->getUnigrams("ㄅㄨˋ-ㄅㄧㄢˋ")), expected);
}

TEST(CitationToneTest, SandhiFindsCitationListedPhrase) {
  auto lm = ToneLM();
  std::vector<std::pair<std::string, double>> expected{{"一中", -5.0}, {"意中", -6.5}};
  EXPECT_EQ(Pairs(lm->getUnigrams("ㄧˋ-ㄓㄨㄥ")), expected);
  EXPECT_TRUE(lm->hasUnigrams("ㄧˊ-ㄓㄨㄥ"));
  std::vector<std::pair<std::string, double>> onlyYi{{"一中", -5.0}};
  EXPECT_EQ(Pairs(lm->getUnigrams("ㄧˊ-ㄓㄨㄥ")), onlyYi);
}

TEST(CitationToneTest, OtherCharactersNotAddedAndListedTwiceAppearsOnce) {
  auto lm = ToneLM();
  // 宜定 is listed under ㄧ-ㄉㄧㄥˋ but 宜 is not 一; 一定 is listed under both readings.
  std::vector<std::pair<std::string, double>> expected{{"一定", -3.41937414}};
  EXPECT_EQ(Pairs(lm->getUnigrams("ㄧˊ-ㄉㄧㄥˋ")), expected);
  std::vector<std::pair<std::string, double>> citation{{"一定", -3.41937414}, {"宜定", -7.0}};
  EXPECT_EQ(Pairs(lm->getUnigrams("ㄧ-ㄉㄧㄥˋ")), citation);
}

TEST(CitationToneTest, SingleSyllableUnchanged) {
  auto lm = ToneLM();
  std::vector<std::pair<std::string, double>> expected{{"不", -2.0}, {"步", -3.0}};
  EXPECT_EQ(Pairs(lm->getUnigrams("ㄅㄨˋ")), expected);
  EXPECT_FALSE(lm->hasUnigrams("ㄅㄨˊ"));
}

TEST(CitationToneTest, ReadingPolicyAppliesToAddedEntries) {
  auto lm = ToneLM();
  lm->setReadingPolicyExclusions({{"ㄧㄢˊ", "癌"}});
  std::vector<std::pair<std::string, double>> expected{{"不言", -5.5}};
  EXPECT_EQ(Pairs(lm->getUnigrams("ㄅㄨˋ-ㄧㄢˊ")), expected);
  lm->setReadingPolicyExclusions({});
  std::vector<std::pair<std::string, double>> accepted{{"不癌", -5.0}, {"不言", -5.5}};
  EXPECT_EQ(Pairs(lm->getUnigrams("ㄅㄨˋ-ㄧㄢˊ")), accepted);
}

TEST(CitationToneTest, ExcludedPhraseStaysExcluded) {
  auto lm = ToneLM();
  constexpr char kExcluded[] = "不要 ㄅㄨˋ-ㄧㄠˋ\n";
  lm->loadExcludedPhrases(kExcluded, sizeof(kExcluded));
  std::vector<std::pair<std::string, double>> expected{{"布要", -6.0}};
  EXPECT_EQ(Pairs(lm->getUnigrams("ㄅㄨˋ-ㄧㄠˋ")), expected);
}

TEST(CitationToneTest, HigherScoreFromOtherReadingWins) {
  auto lm = ToneLM();
  // 不過 is listed under ㄅㄨˋ-ㄍㄨㄛˋ (-3.5) and under ㄅㄨˊ-ㄍㄨㄛˋ (-3.2): offered once, with -3.2.
  std::vector<std::pair<std::string, double>> expected{{"不過", -3.2}, {"布過", -6.2}};
  EXPECT_EQ(Pairs(lm->getUnigrams("ㄅㄨˋ-ㄍㄨㄛˋ")), expected);
}

TEST(CitationToneTest, UserPhrasesUnchanged) {
  auto lm = ToneLM();
  // A user phrase is offered only under the reading the user wrote, and keeps its user-phrase flag when
  // the same value also comes from a tone variant.
  constexpr char kUser[] = "步要 ㄅㄨˊ-ㄧㄠˋ\n不要 ㄅㄨˋ-ㄧㄠˋ\n";
  lm->loadUserPhrases(kUser, sizeof(kUser));
  auto got = lm->getUnigrams("ㄅㄨˋ-ㄧㄠˋ");
  EXPECT_FALSE(std::any_of(got.begin(), got.end(), [](const Unigram& u) { return u.value() == "步要"; }));
  ASSERT_FALSE(got.empty());
  EXPECT_EQ(got[0].value(), "不要");
  EXPECT_TRUE(got[0].isUserPhrase());
  EXPECT_EQ(std::count_if(got.begin(), got.end(), [](const Unigram& u) { return u.value() == "不要"; }), 1);
}

TEST(CitationToneTest, StockWalkComposesPhraseWithoutModel) {
  auto lm = ToneLM();
  EXPECT_EQ(Walk(lm, {"ㄅㄨˋ", "ㄧㄠˋ"}), "不要");
  EXPECT_EQ(Walk(lm, {"ㄅㄨˋ", "ㄅㄧㄢˋ"}), "不變");
}

// --- shipped data.txt

std::string DataPath() { return std::string(READING_POLICY_DATA_DIR) + "/data.txt"; }

TEST(CitationToneDataTest, EverySandhiListedYiBuPhraseFoundUnderCitationTone) {
  std::set<std::pair<std::string, std::string>> listed;   // (reading, value) in the file
  std::vector<std::pair<std::string, std::string>> lines;
  std::ifstream f(DataPath());
  ASSERT_TRUE(f.good()) << DataPath();
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#' || line[0] == '_') continue;
    std::istringstream ss(line);
    std::string key, value;
    ss >> key >> value;
    listed.emplace(key, value);
    lines.emplace_back(key, value);
  }
  McBopomofoLM lm;
  lm.loadLanguageModel(DataPath().c_str());
  ASSERT_TRUE(lm.isDataModelLoaded());
  size_t checked = 0;
  std::set<std::pair<std::string, std::string>> added;   // (citation key, value) not listed under that key
  for (const auto& [key, value] : lines) {
    auto syllables = AssociatedPhrasesV2::SplitReadings(key);
    auto chars = Split(value);
    if (syllables.size() < 2 || chars.size() != syllables.size()) continue;
    std::vector<std::string> cite = syllables;
    for (size_t i = 0; i < cite.size(); ++i) {
      if (chars[i] == "不" && cite[i] == "ㄅㄨˊ") cite[i] = "ㄅㄨˋ";
      if (chars[i] == "一" && (cite[i] == "ㄧˊ" || cite[i] == "ㄧˋ")) cite[i] = "ㄧ";
    }
    if (cite == syllables) continue;
    std::string citeKey = AssociatedPhrasesV2::CombineReadings(cite);
    ++checked;
    if (!listed.count({citeKey, value})) added.emplace(citeKey, value);
    auto got = lm.getUnigrams(citeKey);
    EXPECT_TRUE(std::any_of(got.begin(), got.end(), [&](const Unigram& u) { return u.value() == value; }))
        << key << " " << value << " not found under " << citeKey;
    EXPECT_TRUE(std::is_sorted(got.begin(), got.end(),
                               [](const Unigram& a, const Unigram& b) { return a.score() > b.score(); }))
        << citeKey;
  }
  // Keys without a 一 / 不 syllable return exactly the values listed under them.
  std::map<std::string, std::set<std::string>> byKey;
  for (const auto& [key, value] : lines) byKey[key].insert(value);
  const std::set<std::string> toneSyllables{"ㄧ", "ㄧˊ", "ㄧˋ", "ㄅㄨˋ", "ㄅㄨˊ"};
  size_t plainKeys = 0;
  for (const auto& [key, values] : byKey) {
    auto syllables = AssociatedPhrasesV2::SplitReadings(key);
    if (std::any_of(syllables.begin(), syllables.end(), [&](const auto& y) { return toneSyllables.count(y); }) ||
        std::any_of(values.begin(), values.end(), [](const auto& v) { return v.rfind("MACRO@", 0) == 0; })) {
      continue;
    }
    ++plainKeys;
    std::set<std::string> got;
    for (const auto& u : lm.getUnigrams(key)) got.insert(u.value());
    EXPECT_EQ(got, values) << key;
  }
  EXPECT_GT(plainKeys, 100000u);
  std::cout << "[ citation-tone ] sandhi-listed 一/不 phrases " << checked << ", newly reachable by citation tone "
            << added.size() << std::endl;
  EXPECT_EQ(added.size(), 976u);  // ../bench/src/decoder.py add_citation_tone_variants on the same data
}

}  // namespace
}  // namespace McBopomofo
