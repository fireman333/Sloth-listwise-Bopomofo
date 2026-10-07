// Design D8 (precise-mode-and-preferences): user phrases are flagged by
// McBopomofoLM and receive the best per-position score under the R-walk char
// bonus, so whenever the stock walk picks a user phrase, R-walk does too, and
// a mono-syllable user phrase competes no more strongly than in stock (the
// 丼 / 動作 case from McBopomofoLM::getUnigrams()).

#include <memory>
#include <random>
#include <string>
#include <vector>

#include "McBopomofoLM.h"
#include "gramambular2/reading_grid.h"
#include "gtest/gtest.h"

namespace McBopomofo {
namespace {

using Formosa::Gramambular2::LanguageModel;
using Formosa::Gramambular2::ReadingGrid;

constexpr char kData[] = R"(
# format org.openvanilla.mcbopomofo.sorted
ㄉㄨㄥˋ 動 -2.83459585
ㄉㄨㄥˋ 洞 -4.31757780
ㄉㄨㄥˋ-ㄗㄨㄛˋ 動作 -4.17449149
ㄗㄨㄛˋ 作 -3.0
ㄗㄨㄛˋ 做 -3.2
ㄚ 啊 -3.0
ㄚ 阿 -3.2
ㄧㄚˋ 亞 -3.5
ㄧㄚˋ 訝 -5.0
)";

constexpr char kUserPhrases[] = R"(
丼 ㄉㄨㄥˋ
錏錒 ㄧㄚˋ-ㄚ
)";

constexpr char kReplacement[] = R"(
錏錒 錏錒X
)";

std::shared_ptr<McBopomofoLM> MakeLM(bool withUserPhrases) {
  auto lm = std::make_shared<McBopomofoLM>();
  lm->loadLanguageModel(
      std::make_unique<ParselessPhraseDB>(kData, sizeof(kData)));
  if (withUserPhrases) {
    lm->loadUserPhrases(kUserPhrases, sizeof(kUserPhrases));
  }
  return lm;
}

// Strips the user-phrase flag (pre-D8 behavior) for comparison.
class UnflaggedLM : public LanguageModel {
 public:
  explicit UnflaggedLM(std::shared_ptr<LanguageModel> lm) : lm_(std::move(lm)) {}
  std::vector<Unigram> getUnigrams(const std::string& key) override {
    std::vector<Unigram> out;
    for (const auto& u : lm_->getUnigrams(key)) {
      out.emplace_back(u.value(), u.score(), u.rawValue());
    }
    return out;
  }
  bool hasUnigrams(const std::string& key) override {
    return lm_->hasUnigrams(key);
  }

 private:
  std::shared_ptr<LanguageModel> lm_;
};

std::string Joined(const ReadingGrid::WalkResult& r) {
  std::string j;
  for (const auto& v : r.valuesAsStrings()) j += v;
  return j;
}

ReadingGrid MakeGrid(std::shared_ptr<LanguageModel> lm,
                     const std::vector<std::string>& readings) {
  ReadingGrid grid(std::move(lm));
  for (const auto& r : readings) {
    EXPECT_TRUE(grid.insertReading(r)) << r;
  }
  return grid;
}

TEST(UserPhraseCharBonusTest, McBopomofoLMFlagsOnlyUserPhrases) {
  auto lm = MakeLM(/*withUserPhrases=*/false);
  for (const char* key : {"ㄉㄨㄥˋ", "ㄉㄨㄥˋ-ㄗㄨㄛˋ", "ㄧㄚˋ", "ㄚ"}) {
    for (const auto& u : lm->getUnigrams(key)) {
      EXPECT_FALSE(u.isUserPhrase()) << key << " " << u.value();
    }
  }

  lm = MakeLM(/*withUserPhrases=*/true);
  // Multi-syllable user phrase: inserted on top, flagged, stock score 0.
  auto multi = lm->getUnigrams("ㄧㄚˋ-ㄚ");
  ASSERT_EQ(multi.size(), 1u);
  EXPECT_EQ(multi[0].value(), "錏錒");
  EXPECT_TRUE(multi[0].isUserPhrase());
  EXPECT_EQ(multi[0].score(), UserPhrasesLM::kUserUnigramScore);

  // Mono-syllable user phrase: score rewritten as upstream, flag kept.
  auto mono = lm->getUnigrams("ㄉㄨㄥˋ");
  ASSERT_EQ(mono.size(), 3u);
  EXPECT_EQ(mono[0].value(), "丼");
  EXPECT_TRUE(mono[0].isUserPhrase());
  EXPECT_LT(mono[0].score(), UserPhrasesLM::kUserUnigramScore);
  EXPECT_LT(mono[0].score() - mono[1].score(), 0.000001);
  EXPECT_EQ(mono[0].rawValue(), "");  // unchanged from upstream
  EXPECT_FALSE(mono[1].isUserPhrase());
  EXPECT_FALSE(mono[2].isUserPhrase());
  for (const auto& u : lm->getUnigrams("ㄉㄨㄥˋ-ㄗㄨㄛˋ")) {
    EXPECT_FALSE(u.isUserPhrase()) << u.value();
  }

  // Phrase replacement keeps the flag (the replaced value is still the
  // user's phrase).
  lm->loadPhraseReplacementMap(kReplacement, sizeof(kReplacement));
  lm->setPhraseReplacementEnabled(true);
  multi = lm->getUnigrams("ㄧㄚˋ-ㄚ");
  ASSERT_EQ(multi.size(), 1u);
  EXPECT_EQ(multi[0].value(), "錏錒X");
  EXPECT_EQ(multi[0].rawValue(), "錏錒");
  EXPECT_TRUE(multi[0].isUserPhrase());
}

TEST(UserPhraseCharBonusTest, RareCharUserPhraseWinsUnderRWalk) {
  // R thinks 亞啊 and finds the rare characters very unlikely.
  ReadingGrid::PositionalCharScores scores;
  scores.table.resize(2);
  scores.table[0] = {{"亞", -0.05}, {"訝", -6.0}, {"錏", -14.0}};
  scores.table[1] = {{"啊", -0.1}, {"阿", -2.5}, {"錒", -16.0}};
  const std::vector<std::string> readings{"ㄧㄚˋ", "ㄚ"};

  auto lm = MakeLM(/*withUserPhrases=*/true);
  ReadingGrid grid = MakeGrid(lm, readings);
  ASSERT_EQ(Joined(grid.walk()), "錏錒");  // stock picks the user phrase
  EXPECT_EQ(Joined(grid.walkWithCharBonus(0.5, scores)), "錏錒");
  auto nb = grid.walkNBestWithCharBonus(10, 0.5, scores);
  ASSERT_FALSE(nb.empty());
  EXPECT_EQ(Joined(nb[0]), "錏錒");

  // Pre-D8 (flag stripped): R-walk overrides the user phrase.
  ReadingGrid old = MakeGrid(std::make_shared<UnflaggedLM>(lm), readings);
  ASSERT_EQ(Joined(old.walk()), "錏錒");
  EXPECT_EQ(Joined(old.walkWithCharBonus(0.5, scores)), "亞啊");

  // Without user phrases, the flag never appears and R-walk is unchanged.
  ReadingGrid none = MakeGrid(MakeLM(false), readings);
  ReadingGrid noneOld =
      MakeGrid(std::make_shared<UnflaggedLM>(MakeLM(false)), readings);
  EXPECT_EQ(Joined(none.walkWithCharBonus(0.5, scores)), "亞啊");
  EXPECT_EQ(none.walkWithCharBonus(0.5, scores).score,
            noneOld.walkWithCharBonus(0.5, scores).score);
}

// 丼 (mono-syllable user phrase, score rewritten to top + epsilon) vs 動作.
// Stock picks 動作 (the reason for the upstream rewrite). Under R-walk, 丼
// gets the row max at position 0, which equals what 動 (stock's top single
// char) gets whenever R ranks 動 first there. So 丼 then competes exactly
// like 動 does without the user phrase: the R-walk result with the user
// phrase equals the result without it, with a standalone 動 replaced by 丼,
// however unlikely R finds the character 丼 itself.
TEST(UserPhraseCharBonusTest, MonoSyllableUserPhraseDoesNotDominate) {
  const std::vector<std::string> readings{"ㄉㄨㄥˋ", "ㄗㄨㄛˋ"};
  ReadingGrid grid = MakeGrid(MakeLM(/*withUserPhrases=*/true), readings);
  ReadingGrid base = MakeGrid(MakeLM(/*withUserPhrases=*/false), readings);
  ASSERT_EQ(Joined(grid.walk()), "動作");
  ASSERT_EQ(Joined(base.walk()), "動作");

  auto substituted = [](const ReadingGrid::WalkResult& r) {
    std::string j;
    for (const auto& v : r.valuesAsStrings()) j += v == "動" ? "丼" : v;
    return j;
  };

  std::mt19937 rng(20261007);
  std::uniform_real_distribution<double> v(-12.0, 0.0);
  std::uniform_real_distribution<double> gap(1e-6, 10.0);
  int trials = 0;
  int phrasePicked = 0;
  for (int i = 0; i < 2000; ++i) {
    ReadingGrid::PositionalCharScores s;
    s.table.resize(2);
    const double top = v(rng);
    s.table[0] = {{"動", top}};
    for (const char* c : {"洞", "丼"}) {
      s.table[0][c] = top - gap(rng);  // 動 is R's first choice at 0
    }
    s.table[1] = {{"作", v(rng)}, {"做", v(rng)}};
    for (double alpha : {0.25, 0.5, 1.0}) {
      auto withUser = grid.walkWithCharBonus(alpha, s);
      auto without = base.walkWithCharBonus(alpha, s);
      EXPECT_EQ(Joined(withUser), substituted(without)) << i << " " << alpha;
      auto nb = grid.walkNBestWithCharBonus(3, alpha, s);
      ASSERT_FALSE(nb.empty());
      EXPECT_EQ(Joined(nb[0]), Joined(withUser)) << i;
      phrasePicked += Joined(withUser) == "動作" ? 1 : 0;
      ++trials;
    }
  }
  EXPECT_EQ(trials, 6000);
  EXPECT_GT(phrasePicked, 3000);  // 動作 still wins most tables

  // When R also prefers 作 at position 1, 動作 wins as in stock.
  ReadingGrid::PositionalCharScores likes;
  likes.table.resize(2);
  likes.table[0] = {{"動", -0.2}, {"洞", -5.0}, {"丼", -15.0}};
  likes.table[1] = {{"作", -0.1}, {"做", -3.0}};
  for (double alpha : {0.25, 0.5, 1.0, 3.0}) {
    EXPECT_EQ(Joined(grid.walkWithCharBonus(alpha, likes)), "動作") << alpha;
  }

  // A flat table (R has no opinion) leaves the stock decision in place.
  ReadingGrid::PositionalCharScores flat;
  flat.table.resize(2);
  flat.table[0] = {{"動", -2.0}, {"洞", -2.0}, {"丼", -2.0}};
  flat.table[1] = {{"作", -2.0}, {"做", -2.0}};
  EXPECT_EQ(Joined(grid.walkWithCharBonus(0.5, flat)), "動作");
}

}  // namespace
}  // namespace McBopomofo
