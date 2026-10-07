// Tests for the reading policy filter in McBopomofoLM
// (setReadingPolicyExclusions): built-in entries and associated phrases that
// align a character with an excluded syllable are dropped; user phrases are
// untouched; an empty policy is upstream behavior. The exhaustive tests run on
// the shipped Source/Data files (READING_POLICY_DATA_DIR).

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
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

constexpr char kAi[] = "癌";
constexpr char kYan[] = "ㄧㄢˊ";
constexpr char kAiReading[] = "ㄞˊ";

// Byte-sorted by key; within a key, the order is what the LM returns.
constexpr char kPolicyLMData[] = R"(
# format org.openvanilla.mcbopomofo.sorted
ㄈㄟˋ 肺 -3.6
ㄈㄟˋ-ㄞˊ 肺癌 -5.32640141
ㄎㄤˋ 抗 -3.2
ㄎㄤˋ-ㄧㄢˊ 抗癌 -5.12617647
ㄎㄤˋ-ㄧㄢˊ 炕沿 -6.69437819
ㄒㄩㄝˇ 血 -3.3
ㄒㄩㄝˇ-ㄧㄢˊ 血癌 -6.1
ㄞˊ 挨 -4.1
ㄞˊ 癌 -5.042188
ㄧㄢˊ 研 -3.41489151
ㄧㄢˊ 言 -3.61636020
ㄧㄢˊ 顏 -4.34518464
ㄧㄢˊ 癌 -4.34904077
)";

// Only the excluded entry exists for ㄧㄢˊ.
constexpr char kOnlyAiLMData[] = R"(
# format org.openvanilla.mcbopomofo.sorted
ㄎㄤˋ 抗 -3.2
ㄧㄢˊ 癌 -4.34904077
)";

constexpr char kPolicyAssocData[] = R"(
# format org.openvanilla.mcbopomofo.sorted
乳-ㄖㄨˇ-汁-ㄓ -5.9
乳-ㄖㄨˇ-癌-ㄞˊ -5.1715
乳-ㄖㄨˇ-癌-ㄧㄢˊ -5.1715
乳-ㄖㄨˇ-癌-ㄧㄢˊ-症-ㄓㄥˋ -6.2
)";

const std::vector<std::string> kKeys{"ㄈㄟˋ",       "ㄈㄟˋ-ㄞˊ",   "ㄎㄤˋ",
                                     "ㄎㄤˋ-ㄧㄢˊ", "ㄒㄩㄝˇ",     "ㄒㄩㄝˇ-ㄧㄢˊ",
                                     "ㄞˊ",          "ㄧㄢˊ",        "ㄇㄚ"};

std::vector<McBopomofoLM::ReadingPolicyExclusion> AiYan() {
  return {{kYan, kAi}};
}

std::unique_ptr<ParselessPhraseDB> Db(const char* data, size_t size) {
  return std::make_unique<ParselessPhraseDB>(data, size);
}

std::shared_ptr<McBopomofoLM> PolicyLM(bool withPolicy) {
  auto lm = std::make_shared<McBopomofoLM>();
  lm->loadLanguageModel(Db(kPolicyLMData, sizeof(kPolicyLMData)));
  lm->loadAssociatedPhrasesV2(Db(kPolicyAssocData, sizeof(kPolicyAssocData)));
  if (withPolicy) lm->setReadingPolicyExclusions(AiYan());
  return lm;
}

std::vector<std::pair<std::string, double>> Pairs(const std::vector<Unigram>& us) {
  std::vector<std::pair<std::string, double>> out;
  for (const auto& u : us) out.emplace_back(u.value(), u.score());
  return out;
}

bool HasValue(const std::vector<Unigram>& us, const std::string& v) {
  return std::any_of(us.begin(), us.end(), [&](const Unigram& u) { return u.value() == v; });
}

bool ValueContains(const std::vector<Unigram>& us, const std::string& ch) {
  return std::any_of(us.begin(), us.end(),
                     [&](const Unigram& u) { return u.value().find(ch) != std::string::npos; });
}

using PhraseList = std::vector<std::pair<std::string, std::vector<std::string>>>;
PhraseList Phrases(const std::vector<AssociatedPhrasesV2::Phrase>& ps) {
  PhraseList out;
  for (const auto& p : ps) out.emplace_back(p.value, p.readings);
  return out;
}

bool AlignsAiYan(const std::string& value, const std::vector<std::string>& readings) {
  auto chars = Split(value);
  if (chars.size() != readings.size()) return false;
  for (size_t i = 0; i < chars.size(); ++i) {
    if (chars[i] == kAi && readings[i] == kYan) return true;
  }
  return false;
}

// ---- 2.1: unigram filter.

TEST(ReadingPolicyTest, DefaultIsUpstream) {
  auto lm = PolicyLM(false);
  EXPECT_TRUE(lm->readingPolicyExclusions().empty());
  EXPECT_TRUE(HasValue(lm->getUnigrams(kYan), kAi));
  EXPECT_TRUE(HasValue(lm->getUnigrams("ㄎㄤˋ-ㄧㄢˊ"), "抗癌"));
}

TEST(ReadingPolicyTest, YanReadingOfAiRemoved) {
  auto lm = PolicyLM(true);
  auto yan = lm->getUnigrams(kYan);
  EXPECT_FALSE(ValueContains(yan, kAi));
  auto kangYan = lm->getUnigrams("ㄎㄤˋ-ㄧㄢˊ");
  EXPECT_FALSE(ValueContains(kangYan, kAi));
  EXPECT_FALSE(HasValue(kangYan, "抗癌"));
  EXPECT_TRUE(HasValue(kangYan, "炕沿"));
}

TEST(ReadingPolicyTest, OtherValuesKeepOrderAndScores) {
  auto upstream = PolicyLM(false);
  auto policy = PolicyLM(true);
  for (const auto& key : kKeys) {
    auto expected = Pairs(upstream->getUnigrams(key));
    expected.erase(std::remove_if(expected.begin(), expected.end(),
                                  [&](const auto& p) {
                                    return AlignsAiYan(p.first,
                                                       AssociatedPhrasesV2::SplitReadings(key));
                                  }),
                   expected.end());
    EXPECT_EQ(Pairs(policy->getUnigrams(key)), expected) << key;
  }
  // Concretely: ㄧㄢˊ keeps 研 言 顏 in that order with their scores.
  EXPECT_EQ(Pairs(policy->getUnigrams(kYan)),
            (std::vector<std::pair<std::string, double>>{
                {"研", -3.41489151}, {"言", -3.61636020}, {"顏", -4.34518464}}));
}

TEST(ReadingPolicyTest, AiReadingUnchanged) {
  auto upstream = PolicyLM(false);
  auto policy = PolicyLM(true);
  for (const char* key : {kAiReading, "ㄈㄟˋ-ㄞˊ"}) {
    EXPECT_EQ(Pairs(policy->getUnigrams(key)), Pairs(upstream->getUnigrams(key))) << key;
  }
  EXPECT_TRUE(HasValue(policy->getUnigrams(kAiReading), kAi));
  EXPECT_TRUE(HasValue(policy->getUnigrams("ㄈㄟˋ-ㄞˊ"), "肺癌"));
}

TEST(ReadingPolicyTest, ClearedPolicyIsIdenticalToUpstream) {
  auto upstream = PolicyLM(false);
  auto lm = PolicyLM(true);
  lm->setReadingPolicyExclusions({});
  for (const auto& key : kKeys) {
    EXPECT_EQ(Pairs(lm->getUnigrams(key)), Pairs(upstream->getUnigrams(key))) << key;
    EXPECT_EQ(lm->hasUnigrams(key), upstream->hasUnigrams(key)) << key;
  }
  EXPECT_EQ(Phrases(lm->findAssociatedPhrasesV2("乳", {"ㄖㄨˇ"})),
            Phrases(upstream->findAssociatedPhrasesV2("乳", {"ㄖㄨˇ"})));
}

TEST(ReadingPolicyTest, UserPhraseWithYanReadingKept) {
  constexpr char kUser[] = "癌 ㄧㄢˊ\n";
  auto lm = PolicyLM(true);
  lm->loadUserPhrases(kUser, sizeof(kUser));
  auto yan = lm->getUnigrams(kYan);
  ASSERT_FALSE(yan.empty());
  EXPECT_EQ(yan[0].value(), kAi);  // user phrase, boosted to the top
  EXPECT_EQ(std::count_if(yan.begin(), yan.end(),
                          [](const Unigram& u) { return u.value() == kAi; }),
            1);
  // The other built-in values are still there.
  EXPECT_TRUE(HasValue(yan, "研"));
}

TEST(ReadingPolicyTest, FilteredOutOnlyPhraseKeyLeavesGridWorking) {
  auto lm = PolicyLM(true);
  // ㄒㄩㄝˇ-ㄧㄢˊ only has 血癌 in the base dictionary.
  EXPECT_TRUE(lm->getUnigrams("ㄒㄩㄝˇ-ㄧㄢˊ").empty());
  ReadingGrid grid(lm);
  ASSERT_TRUE(grid.insertReading("ㄒㄩㄝˇ"));
  ASSERT_TRUE(grid.insertReading(kYan));
  auto walk = grid.walk();
  std::string text;
  for (const auto& n : walk.nodes) text += n->value();
  EXPECT_EQ(walk.totalReadings, 2u);
  EXPECT_EQ(text, "血研");
  for (size_t loc = 0; loc <= grid.length(); ++loc) {
    for (const auto& c : grid.candidatesAt(loc)) {
      EXPECT_EQ(c.value.find(kAi), std::string::npos) << loc << " " << c.value;
    }
  }
  // Same with the policy cleared: 血癌 comes back.
  lm->setReadingPolicyExclusions({});
  ReadingGrid grid2(lm);
  ASSERT_TRUE(grid2.insertReading("ㄒㄩㄝˇ"));
  ASSERT_TRUE(grid2.insertReading(kYan));
  bool sawXueAi = false;
  for (const auto& c : grid2.candidatesAt(0)) sawXueAi |= c.value == "血癌";
  EXPECT_TRUE(sawXueAi);
}

TEST(ReadingPolicyTest, FilteredOutOnlySingleSyllableKeyDoesNotCrashGrid) {
  auto lm = std::make_shared<McBopomofoLM>();
  lm->loadLanguageModel(Db(kOnlyAiLMData, sizeof(kOnlyAiLMData)));
  lm->setReadingPolicyExclusions(AiYan());
  EXPECT_TRUE(lm->getUnigrams(kYan).empty());
  // hasUnigrams may still report true for a key whose base entries are all
  // filtered; the grid then skips the span (reading_grid.cpp).
  ReadingGrid grid(lm);
  ASSERT_TRUE(grid.insertReading("ㄎㄤˋ"));
  grid.insertReading(kYan);  // accepted or rejected; either must be safe
  auto walk = grid.walk();
  std::string text;
  for (const auto& n : walk.nodes) text += n->value();
  EXPECT_EQ(text.find(kAi), std::string::npos);
  EXPECT_EQ(text.substr(0, std::string("抗").size()), "抗");
  for (size_t loc = 0; loc <= grid.length(); ++loc) {
    for (const auto& c : grid.candidatesAt(loc)) {
      EXPECT_EQ(c.value.find(kAi), std::string::npos);
    }
  }
  // Deleting and inserting around the empty span stays safe.
  grid.setCursor(grid.length());
  grid.deleteReadingBeforeCursor();
  EXPECT_TRUE(grid.insertReading("ㄎㄤˋ"));
  grid.walk();
}

// ---- 2.3: associated phrases.

TEST(ReadingPolicyTest, AssociatedPhrasesFiltered) {
  auto upstream = PolicyLM(false);
  auto lm = PolicyLM(true);
  auto all = Phrases(upstream->findAssociatedPhrasesV2("乳", {"ㄖㄨˇ"}));
  auto got = Phrases(lm->findAssociatedPhrasesV2("乳", {"ㄖㄨˇ"}));
  const std::pair<std::string, std::vector<std::string>> yan{"乳癌", {"ㄖㄨˇ", kYan}};
  const std::pair<std::string, std::vector<std::string>> ai{"乳癌", {"ㄖㄨˇ", kAiReading}};
  const std::pair<std::string, std::vector<std::string>> yanOnly{"乳癌症",
                                                                 {"ㄖㄨˇ", kYan, "ㄓㄥˋ"}};
  // Upstream dedups equal values; on a score tie the byte-earlier row (ㄞˊ)
  // wins, so upstream already shows 乳癌 as ㄞˊ. The ㄧㄢˊ-only phrase is
  // offered upstream and must be gone with the policy.
  EXPECT_NE(std::find(all.begin(), all.end(), ai), all.end());
  EXPECT_EQ(std::find(all.begin(), all.end(), yan), all.end());
  EXPECT_NE(std::find(all.begin(), all.end(), yanOnly), all.end());
  EXPECT_EQ(std::find(got.begin(), got.end(), yan), got.end());
  EXPECT_EQ(std::find(got.begin(), got.end(), yanOnly), got.end());
  EXPECT_NE(std::find(got.begin(), got.end(), ai), got.end());
  EXPECT_LT(got.size(), all.size());
  // Everything else unchanged, in order.
  PhraseList expected;
  for (const auto& p : all) {
    if (!AlignsAiYan(p.first, p.second)) expected.push_back(p);
  }
  EXPECT_EQ(got, expected);
  // Value-only prefix (no readings) is filtered too.
  for (const auto& p : Phrases(lm->findAssociatedPhrasesV2("乳", {}))) {
    EXPECT_FALSE(AlignsAiYan(p.first, p.second)) << p.first;
  }
}

// ---- Real data (Source/Data).

std::string DataPath(const char* name) {
  return std::string(READING_POLICY_DATA_DIR) + "/" + name;
}

struct AiLine {
  int lineNo;
  std::string key, value;
  double score;
};

std::vector<AiLine> LinesWithAi(const std::string& path) {
  std::vector<AiLine> out;
  std::ifstream f(path);
  std::string line;
  int n = 0;
  while (std::getline(f, line)) {
    ++n;
    if (line.empty() || line[0] == '#' || line.find(kAi) == std::string::npos) continue;
    std::istringstream ss(line);
    AiLine l{n, "", "", 0};
    ss >> l.key >> l.value >> l.score;
    out.push_back(l);
  }
  return out;
}

void ExhaustiveUnigramCheck(const char* file, size_t* yanCount, size_t* aiCount,
                            std::vector<std::string>* mismatched) {
  const std::string path = DataPath(file);
  auto lines = LinesWithAi(path);
  ASSERT_FALSE(lines.empty()) << path;
  McBopomofoLM upstream;
  upstream.loadLanguageModel(path.c_str());
  McBopomofoLM lm;
  lm.loadLanguageModel(path.c_str());
  lm.setReadingPolicyExclusions(AiYan());
  ASSERT_TRUE(lm.isDataModelLoaded());
  for (const auto& l : lines) {
    auto syllables = AssociatedPhrasesV2::SplitReadings(l.key);
    auto chars = Split(l.value);
    if (chars.size() != syllables.size()) {
      mismatched->push_back(std::string(file) + ":" + std::to_string(l.lineNo) + " " + l.key +
                            " " + l.value);
      continue;
    }
    auto got = lm.getUnigrams(l.key);
    if (AlignsAiYan(l.value, syllables)) {
      ++*yanCount;
      EXPECT_FALSE(HasValue(got, l.value)) << file << ":" << l.lineNo << " " << l.key;
    } else {
      bool ai = false;
      for (size_t i = 0; i < chars.size(); ++i) ai |= chars[i] == kAi && syllables[i] == kAiReading;
      *aiCount += ai;
      auto it = std::find_if(got.begin(), got.end(),
                             [&](const Unigram& u) { return u.value() == l.value; });
      ASSERT_NE(it, got.end()) << file << ":" << l.lineNo << " " << l.key << " " << l.value;
      auto up = upstream.getUnigrams(l.key);
      auto ut = std::find_if(up.begin(), up.end(),
                             [&](const Unigram& u) { return u.value() == l.value; });
      ASSERT_NE(ut, up.end());
      EXPECT_DOUBLE_EQ(it->score(), ut->score()) << l.key << " " << l.value;
    }
    // Every other value of the key is untouched, in order.
    auto expected = Pairs(upstream.getUnigrams(l.key));
    expected.erase(std::remove_if(expected.begin(), expected.end(),
                                  [&](const auto& p) { return AlignsAiYan(p.first, syllables); }),
                   expected.end());
    EXPECT_EQ(Pairs(got), expected) << l.key;
  }
}

TEST(ReadingPolicyDataTest, DataTxtExhaustive) {
  size_t yan = 0, ai = 0;
  std::vector<std::string> mismatched;
  ExhaustiveUnigramCheck("data.txt", &yan, &ai, &mismatched);
  EXPECT_GT(yan, 0u);
  EXPECT_GT(ai, 0u);
  for (const auto& m : mismatched) ADD_FAILURE() << "length-mismatched 癌 line: " << m;
  EXPECT_TRUE(mismatched.empty());
  std::cout << "[ reading-policy ] data.txt: ㄧㄢˊ-aligned filtered " << yan
            << ", ㄞˊ-aligned kept " << ai << ", length-mismatched " << mismatched.size()
            << std::endl;
}

TEST(ReadingPolicyDataTest, DataPlainBpmfExhaustive) {
  size_t yan = 0, ai = 0;
  std::vector<std::string> mismatched;
  ExhaustiveUnigramCheck("data-plain-bpmf.txt", &yan, &ai, &mismatched);
  EXPECT_GT(yan, 0u);
  EXPECT_GT(ai, 0u);
  for (const auto& m : mismatched) ADD_FAILURE() << "length-mismatched 癌 line: " << m;
  EXPECT_TRUE(mismatched.empty());
  std::cout << "[ reading-policy ] data-plain-bpmf.txt: ㄧㄢˊ-aligned filtered " << yan
            << ", ㄞˊ-aligned kept " << ai << ", length-mismatched " << mismatched.size()
            << std::endl;
}

TEST(ReadingPolicyDataTest, AssociatedPhrasesExhaustive) {
  const std::string path = DataPath("associated-phrases-v2.txt");
  McBopomofoLM upstream;
  upstream.loadAssociatedPhrasesV2(path.c_str());
  McBopomofoLM lm;
  lm.loadAssociatedPhrasesV2(path.c_str());
  lm.setReadingPolicyExclusions(AiYan());
  ASSERT_TRUE(lm.isAssociatedPhrasesV2Loaded());

  // 乳: ㄧㄢˊ version absent, ㄞˊ version present.
  auto ru = Phrases(lm.findAssociatedPhrasesV2("乳", {"ㄖㄨˇ"}));
  const std::pair<std::string, std::vector<std::string>> yan{"乳癌", {"ㄖㄨˇ", kYan}};
  const std::pair<std::string, std::vector<std::string>> ai{"乳癌", {"ㄖㄨˇ", kAiReading}};
  EXPECT_EQ(std::find(ru.begin(), ru.end(), yan), ru.end());
  EXPECT_NE(std::find(ru.begin(), ru.end(), ai), ru.end());
  auto ruUp = Phrases(upstream.findAssociatedPhrasesV2("乳", {"ㄖㄨˇ"}));
  EXPECT_NE(std::find(ruUp.begin(), ruUp.end(), ai), ruUp.end());

  // Every entry containing 癌: query with its first character + reading.
  std::ifstream f(path);
  std::string line;
  size_t yanEntries = 0, aiEntries = 0, queries = 0;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#' || line.find(kAi) == std::string::npos) continue;
    std::string combined = line.substr(0, line.find(' '));
    auto parts = AssociatedPhrasesV2::SplitReadings(combined);
    if (parts.size() < 2) continue;
    const std::string first = parts[0], firstReading = parts[1];
    std::string value;
    std::vector<std::string> readings;
    for (size_t i = 0; i + 1 < parts.size(); i += 2) {
      value += parts[i];
      readings.push_back(parts[i + 1]);
    }
    const bool isYan = AlignsAiYan(value, readings);
    yanEntries += isYan;
    aiEntries += !isYan;
    ++queries;
    auto got = Phrases(lm.findAssociatedPhrasesV2(first, {firstReading}));
    for (const auto& p : got) EXPECT_FALSE(AlignsAiYan(p.first, p.second)) << combined;
    auto up = Phrases(upstream.findAssociatedPhrasesV2(first, {firstReading}));
    PhraseList expected;
    for (const auto& p : up) {
      if (!AlignsAiYan(p.first, p.second)) expected.push_back(p);
    }
    EXPECT_EQ(got, expected) << combined;
    const std::pair<std::string, std::vector<std::string>> self{value, readings};
    const bool present = std::find(got.begin(), got.end(), self) != got.end();
    const bool presentUp = std::find(up.begin(), up.end(), self) != up.end();
    if (isYan) {
      EXPECT_FALSE(present) << combined;
    } else {
      EXPECT_EQ(present, presentUp) << combined;
      // The filter runs after upstream's same-value dedup; in the shipped data
      // no non-ㄧㄢˊ 癌 entry may be hidden behind a ㄧㄢˊ duplicate.
      EXPECT_TRUE(present) << combined;
    }
  }
  EXPECT_GT(yanEntries, 0u);
  EXPECT_GT(aiEntries, 0u);
  std::cout << "[ reading-policy ] associated-phrases-v2: ㄧㄢˊ entries " << yanEntries
            << ", other 癌 entries " << aiEntries << ", queries " << queries << std::endl;

  // Cleared policy == upstream for the same queries.
  lm.setReadingPolicyExclusions({});
  EXPECT_EQ(Phrases(lm.findAssociatedPhrasesV2("乳", {"ㄖㄨˇ"})), ruUp);
}

}  // namespace
}  // namespace McBopomofo
