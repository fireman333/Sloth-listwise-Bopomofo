// Copyright (c) 2022 and onwards The McBopomofo Authors.
//
// Permission is hereby granted, free of charge, to any person
// obtaining a copy of this software and associated documentation
// files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use,
// copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following
// conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.

#include <bit>
#include <charconv>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include "UserOverrideModel.h"
#include "gramambular2/reading_grid.h"
#include "gtest/gtest.h"

namespace McBopomofo {

namespace {
constexpr double kFakeNow = 1657772432;
constexpr int kCapacity = 5;
constexpr double kHalflife = 5400.0;  // 1.5 hr.
}  // namespace

TEST(UserOverrideModelTest, BasicOperation) {
  UserOverrideModel uom(kCapacity, kHalflife);
  std::string key = "abc";
  std::string candidate = "v";
  uom.observe(key, candidate, kFakeNow);

  auto v = uom.suggest(key, kFakeNow);
  ASSERT_EQ(v.candidate, candidate);

  v = uom.suggest(key, kFakeNow + kHalflife * 1);
  ASSERT_EQ(v.candidate, candidate);
  v = uom.suggest(key, kFakeNow + kHalflife * 5);
  ASSERT_EQ(v.candidate, candidate);
  v = uom.suggest(key, kFakeNow + kHalflife * 10);
  ASSERT_EQ(v.candidate, candidate);
  v = uom.suggest(key, kFakeNow + kHalflife * 20);
  ASSERT_EQ(v.candidate, candidate);

  // The suggestion is no longer valid after ~30 hours.
  v = uom.suggest(key, kFakeNow + kHalflife * 21);
  ASSERT_TRUE(v.empty());
}

TEST(UserOverrideModelTest, FreshVsFrequent) {
  UserOverrideModel uom(kCapacity, kHalflife);
  std::string key = "abc";
  std::string olderValue = "older";
  std::string newerValue = "newer";

  uom.observe(key, olderValue, kFakeNow);
  uom.observe(key, olderValue, kFakeNow + kHalflife * 1);
  uom.observe(key, olderValue, kFakeNow + kHalflife * 2);
  uom.observe(key, olderValue, kFakeNow + kHalflife * 3);
  uom.observe(key, olderValue, kFakeNow + kHalflife * 4);
  uom.observe(key, newerValue, kFakeNow + kHalflife * 5);
  uom.observe(key, newerValue, kFakeNow + kHalflife * 5.25);

  // Even if newerValue is more recent, olderValue is used more frequently,
  // and so initially olderValue is still suggested.
  auto v = uom.suggest(key, kFakeNow + kHalflife * 7);
  ASSERT_EQ(v.candidate, olderValue);
  v = uom.suggest(key, kFakeNow + kHalflife * 20);
  ASSERT_EQ(v.candidate, olderValue);
  v = uom.suggest(key, kFakeNow + kHalflife * 22);
  ASSERT_EQ(v.candidate, olderValue);

  // At this point, even if olderValue hasn't expired yet, but the
  // less-frequently observed newerValue is fresher.
  uom.observe(key, newerValue, kFakeNow + kHalflife * 23);
  v = uom.suggest(key, kFakeNow + kHalflife * 23.5);
  ASSERT_EQ(v.candidate, newerValue);

  v = uom.suggest(key, kFakeNow + kHalflife * 25);
  ASSERT_EQ(v.candidate, newerValue);

  v = uom.suggest(key, kFakeNow + kHalflife * 45);
  ASSERT_TRUE(v.empty());
}

TEST(UserOverrideModelTest, LRUBehavior) {
  UserOverrideModel uom(2, kHalflife);
  uom.observe("abc", "x", kFakeNow);
  uom.observe("def", "y", kFakeNow + kHalflife);
  uom.observe("ghi", "z", kFakeNow + kHalflife * 2);

  auto v = uom.suggest("ghi", kFakeNow + kHalflife * 3);
  ASSERT_EQ(v.candidate, "z");

  v = uom.suggest("def", kFakeNow + kHalflife * 4);
  ASSERT_EQ(v.candidate, "y");

  // abc evicted.
  v = uom.suggest("abc", kFakeNow + kHalflife * 5);
  ASSERT_TRUE(v.empty());

  uom.observe("jkl", "p", kFakeNow + kHalflife * 6);

  v = uom.suggest("ghi", kFakeNow + kHalflife * 7);
  ASSERT_EQ(v.candidate, "z");

  // def evicted.
  v = uom.suggest("def", kFakeNow + kHalflife * 7);
  ASSERT_TRUE(v.empty());
}

constexpr char kSampleData[] = R"(
ㄐㄧ 機 -3.02367199
ㄐㄧ 積 -3.72854036
ㄐㄧ-ㄧㄡˊ 機油 -6.03662914
ㄧㄡˊ 由 -3.00970678
ㄧㄡˊ 油 -3.75900671
)";

class SimpleLM : public Formosa::Gramambular2::LanguageModel {
 public:
  explicit SimpleLM(const char* input, bool readingIsFirstColumn = true) {
    std::stringstream sstream(input);
    while (sstream.good()) {
      std::string line;
      getline(sstream, line);
      if (line.empty() || line[0] == '#') {
        continue;
      }
      std::stringstream linestream(line);
      std::string col0;
      std::string col1;
      std::string col2;
      linestream >> col0;
      linestream >> col1;
      linestream >> col2;
      db_[readingIsFirstColumn ? col0 : col1].emplace_back(
          readingIsFirstColumn ? col1 : col0, std::stod(col2));
    }
  }

  std::vector<Unigram> getUnigrams(const std::string& key) override {
    const auto f = db_.find(key);
    return f == db_.end() ? std::vector<Unigram>() : (*f).second;
  }

  bool hasUnigrams(const std::string& key) override {
    return db_.find(key) != db_.end();
  }

 protected:
  std::map<std::string, std::vector<Unigram>> db_;
};

TEST(UserOverrideModelTest, WalkResultSnapshotTest) {
  // See https://github.com/openvanilla/McBopomofo/issues/885.
  // This is a kind of integration test for the following steps:
  // 1. Type ㄐㄧ ㄧㄡˊ -> 機由; this assumes P(機）P(由) > P(機油)
  // 2. Override with 機油 and tell UOM to observe
  // 3. Reset, type ㄐㄧ ㄧㄡˊ -> 機油 -- UOM should suggest the expected
  // override
  // 4. Reset, type ㄐㄧ -> 機
  // 5. Override with 積 and tell UOM to observe
  // 6. Reset, type ㄐㄧ ㄧㄡˊ -> 積由 -- this is expected due to UOM suggestion
  // 7. Override with 機油 and tell UOM to observe
  // 8. Reset, type ㄐㄧ -> 積 due to UOM suggestion
  // 9. Continue to type 由 -> walk result should be 積由, and the UOM should
  //    suggest 機油 as the override candidate.

  std::string sampleData(kSampleData);
  Formosa::Gramambular2::ReadingGrid grid(
      std::make_shared<SimpleLM>(sampleData.c_str()));
  UserOverrideModel uom(kCapacity, kHalflife);
  Formosa::Gramambular2::ReadingGrid::WalkResult walkBefore;
  Formosa::Gramambular2::ReadingGrid::WalkResult walkLatest;
  UserOverrideModel::Suggestion suggestion;
  std::vector<Formosa::Gramambular2::ReadingGrid::NodePtr>::const_iterator
      nodeIter;
  double timestamp = kFakeNow;

  grid.insertReading("ㄐㄧ");
  grid.insertReading("ㄧㄡˊ");
  walkBefore = grid.walk();
  ASSERT_EQ(walkBefore.valuesAsStrings(),
            (std::vector<std::string>{"機", "由"}));

  grid.overrideCandidate(1, "機油");
  walkLatest = grid.walk();
  ASSERT_EQ(walkLatest.valuesAsStrings(), (std::vector<std::string>{"機油"}));

  nodeIter = walkLatest.findNodeAt(1, /*outCursorPastNode=*/nullptr);
  ASSERT_NE(nodeIter, walkLatest.nodes.cend());
  uom.observe(walkBefore, walkLatest, 1, timestamp);
  timestamp += 1.0;

  grid.clear();
  grid.insertReading("ㄐㄧ");
  grid.insertReading("ㄧㄡˊ");
  walkLatest = grid.walk();
  suggestion = uom.suggest(walkLatest, 1, timestamp);
  timestamp += 1.0;
  ASSERT_EQ(suggestion.candidate, "機油");

  grid.clear();
  grid.insertReading("ㄐㄧ");
  walkBefore = grid.walk();
  ASSERT_EQ(walkBefore.valuesAsStrings(), (std::vector<std::string>{"機"}));

  grid.overrideCandidate(0, "積");
  walkLatest = grid.walk();
  ASSERT_EQ(walkLatest.valuesAsStrings(), (std::vector<std::string>{"積"}));

  nodeIter = walkLatest.findNodeAt(0, /*outCursorPastNode=*/nullptr);
  ASSERT_NE(nodeIter, walkLatest.nodes.cend());
  uom.observe(walkBefore, walkLatest, 0, timestamp);
  timestamp += 1.0;

  grid.clear();
  grid.insertReading("ㄐㄧ");
  walkLatest = grid.walk();
  suggestion = uom.suggest(walkLatest, 0, timestamp);
  timestamp += 1.0;
  ASSERT_EQ(suggestion.candidate, "積");

  grid.overrideCandidate(0, "積");
  grid.insertReading("ㄧㄡˊ");

  // THIS MUST BE A COPY OF NODES, otherwise the observation below will not
  // capture the correct state of the grid.
  // See https://github.com/openvanilla/McBopomofo/issues/885.
  //
  // This is wrong:
  //   walkBefore = grid.walk();
  //
  // The following is correct:
  walkBefore = grid.walk().copyWithFixedNodes();

  ASSERT_EQ(walkBefore.valuesAsStrings(),
            (std::vector<std::string>{"積", "由"}));

  grid.overrideCandidate(1, "機油");
  walkLatest = grid.walk();
  ASSERT_EQ(walkLatest.valuesAsStrings(), (std::vector<std::string>{"機油"}));

  nodeIter = walkLatest.findNodeAt(1, /*outCursorPastNode=*/nullptr);
  ASSERT_NE(nodeIter, walkLatest.nodes.cend());
  uom.observe(walkBefore, walkLatest, 1, timestamp);
  timestamp += 1.0;

  grid.clear();
  grid.insertReading("ㄐㄧ");
  walkLatest = grid.walk();
  suggestion = uom.suggest(walkLatest, 0, timestamp);
  timestamp += 1.0;
  ASSERT_EQ(suggestion.candidate, "積");
  grid.overrideCandidate(0, "積");
  grid.insertReading("ㄧㄡˊ");
  walkLatest = grid.walk();
  ASSERT_EQ(walkLatest.valuesAsStrings(),
            (std::vector<std::string>{"積", "由"}));

  suggestion = uom.suggest(walkLatest, 1, timestamp);
  timestamp += 1.0;
  ASSERT_EQ(suggestion.candidate, "機油");
}

namespace {
constexpr double kDay = 86400.0;
constexpr double kLongTermHalflife = 7 * kDay;
constexpr size_t kLongTermCapacity = 5000;
constexpr UserOverrideModel::Policy kLongTermPolicy{/*latestPickWins=*/true,
                                                    /*observeRefresh=*/true};

UserOverrideModel MakeLongTermModel(size_t capacity = kLongTermCapacity) {
  return {capacity, kLongTermHalflife, kLongTermPolicy};
}

std::string Serialize(const UserOverrideModel& uom) {
  std::ostringstream os;
  uom.serialize(os);
  return os.str();
}

bool Deserialize(UserOverrideModel* uom, const std::string& text) {
  std::istringstream is(text);
  return uom->deserialize(is);
}
}  // namespace

TEST(UserOverrideModelTest, StockConstructorHasStockPolicy) {
  UserOverrideModel uom(kCapacity, kHalflife);
  ASSERT_FALSE(uom.policy().latestPickWins);
  ASSERT_FALSE(uom.policy().observeRefresh);
  ASSERT_EQ(uom.capacity(), static_cast<size_t>(kCapacity));

  // refresh() is a no-op without the policy.
  uom.observe("abc", "x", kFakeNow);
  ASSERT_FALSE(uom.refresh("abc", "x", kFakeNow + kHalflife * 19));
  ASSERT_TRUE(uom.suggest("abc", kFakeNow + kHalflife * 21).empty());
}

TEST(UserOverrideModelTest, LongTermLatestPickWins) {
  UserOverrideModel longTerm = MakeLongTermModel();
  // Stock policy with the same half-life.
  UserOverrideModel stock(kLongTermCapacity, kLongTermHalflife);
  for (UserOverrideModel* uom : {&longTerm, &stock}) {
    uom->observe("ctx", "鍾", kFakeNow);
    uom->observe("ctx", "鍾", kFakeNow + 60);
    uom->observe("ctx", "鍾", kFakeNow + 120);
    uom->observe("ctx", "中", kFakeNow + kDay);
  }
  ASSERT_EQ(longTerm.suggest("ctx", kFakeNow + kDay + 1).candidate, "中");
  ASSERT_EQ(stock.suggest("ctx", kFakeNow + kDay + 1).candidate, "鍾");

  // Picking 鍾 again makes it win again right away.
  longTerm.observe("ctx", "鍾", kFakeNow + 2 * kDay);
  ASSERT_EQ(longTerm.suggest("ctx", kFakeNow + 2 * kDay + 1).candidate, "鍾");
}

TEST(UserOverrideModelTest, LongTermSuggestedAfterThreeDays) {
  UserOverrideModel longTerm = MakeLongTermModel();
  UserOverrideModel stock(500, kHalflife);
  longTerm.observe("ctx", "鍾", kFakeNow);
  stock.observe("ctx", "鍾", kFakeNow);
  ASSERT_EQ(longTerm.suggest("ctx", kFakeNow + 3 * kDay).candidate, "鍾");
  ASSERT_TRUE(stock.suggest("ctx", kFakeNow + 3 * kDay).empty());
}

TEST(UserOverrideModelTest, LongTermFadesAfter140Days) {
  UserOverrideModel uom = MakeLongTermModel();
  uom.observe("ctx", "鍾", kFakeNow);
  ASSERT_EQ(uom.suggest("ctx", kFakeNow + 139 * kDay).candidate, "鍾");
  ASSERT_TRUE(uom.suggest("ctx", kFakeNow + 141 * kDay).empty());
}

TEST(UserOverrideModelTest, RefreshKeepsMemoryFresh) {
  UserOverrideModel uom = MakeLongTermModel();
  UserOverrideModel notRefreshed = MakeLongTermModel();
  uom.observe("ctx", "鍾", kFakeNow);
  notRefreshed.observe("ctx", "鍾", kFakeNow);

  ASSERT_EQ(uom.suggest("ctx", kFakeNow + 100 * kDay).candidate, "鍾");
  ASSERT_TRUE(uom.refresh("ctx", "鍾", kFakeNow + 100 * kDay));

  ASSERT_EQ(uom.suggest("ctx", kFakeNow + 200 * kDay).candidate, "鍾");
  ASSERT_TRUE(notRefreshed.suggest("ctx", kFakeNow + 200 * kDay).empty());
}

TEST(UserOverrideModelTest, RefreshOfNonBestCandidateIsNoOp) {
  // Refresh on, latest-pick-wins off, so that a key can hold more than one
  // candidate.
  UserOverrideModel uom(2, kLongTermHalflife,
                        UserOverrideModel::Policy{/*latestPickWins=*/false,
                                                  /*observeRefresh=*/true});
  uom.observe("k1", "a", kFakeNow);
  uom.observe("k1", "a", kFakeNow + 1);
  uom.observe("k1", "b", kFakeNow + 2);
  uom.observe("k2", "x", kFakeNow + 3);
  ASSERT_EQ(uom.suggest("k1", kFakeNow + 4).candidate, "a");
  std::string before = Serialize(uom);

  ASSERT_FALSE(uom.refresh("k1", "b", kFakeNow + 100 * kDay));
  ASSERT_FALSE(uom.refresh("k1", "missing", kFakeNow + 5));
  ASSERT_FALSE(uom.refresh("missing", "a", kFakeNow + 5));
  // Expired: no best candidate at that time.
  ASSERT_FALSE(uom.refresh("k1", "a", kFakeNow + 200 * kDay));
  ASSERT_EQ(Serialize(uom), before);

  // k1 was not moved to the LRU front, so it is evicted first.
  uom.observe("k3", "y", kFakeNow + 6);
  ASSERT_TRUE(uom.suggest("k1", kFakeNow + 7).empty());
  ASSERT_EQ(uom.suggest("k2", kFakeNow + 7).candidate, "x");
  ASSERT_EQ(uom.suggest("k3", kFakeNow + 7).candidate, "y");
}

TEST(UserOverrideModelTest, CapacityEvictionWithRefresh) {
  UserOverrideModel uom = MakeLongTermModel(2);
  uom.observe("k1", "a", kFakeNow);
  uom.observe("k2", "b", kFakeNow + 1);
  ASSERT_TRUE(uom.refresh("k1", "a", kFakeNow + 2));
  uom.observe("k3", "c", kFakeNow + 3);

  // k2 is now the least recently used.
  ASSERT_EQ(uom.suggest("k1", kFakeNow + 4).candidate, "a");
  ASSERT_TRUE(uom.suggest("k2", kFakeNow + 4).empty());
  ASSERT_EQ(uom.suggest("k3", kFakeNow + 4).candidate, "c");
  ASSERT_EQ(uom.size(), 2u);

  // Without the refresh, k1 would have been evicted.
  UserOverrideModel other = MakeLongTermModel(2);
  other.observe("k1", "a", kFakeNow);
  other.observe("k2", "b", kFakeNow + 1);
  other.observe("k3", "c", kFakeNow + 3);
  ASSERT_TRUE(other.suggest("k1", kFakeNow + 4).empty());
  ASSERT_EQ(other.suggest("k2", kFakeNow + 4).candidate, "b");
}

TEST(UserOverrideModelTest, SuggestWithKey) {
  std::string sampleData(kSampleData);
  Formosa::Gramambular2::ReadingGrid grid(
      std::make_shared<SimpleLM>(sampleData.c_str()));
  UserOverrideModel uom = MakeLongTermModel();

  grid.insertReading("ㄐㄧ");
  grid.insertReading("ㄧㄡˊ");
  auto walkBefore = grid.walk();
  grid.overrideCandidate(1, "機油");
  auto walkAfter = grid.walk();
  uom.observe(walkBefore, walkAfter, 1, kFakeNow);

  grid.clear();
  grid.insertReading("ㄐㄧ");
  grid.insertReading("ㄧㄡˊ");
  auto walk = grid.walk();
  auto result = uom.suggestWithKey(walk, 1, kFakeNow + 1);
  ASSERT_EQ(result.suggestion.candidate, "機油");
  ASSERT_EQ(result.suggestion.forceHighScoreOverride,
            uom.suggest(walk, 1, kFakeNow + 1).forceHighScoreOverride);
  ASSERT_FALSE(result.key.empty());
  ASSERT_EQ(uom.suggest(result.key, kFakeNow + 1).candidate, "機油");
  ASSERT_TRUE(uom.refresh(result.key, "機油", kFakeNow + 100 * kDay));
  ASSERT_EQ(uom.suggest(walk, 1, kFakeNow + 200 * kDay).candidate, "機油");

  // The key is returned even when there is no suggestion.
  UserOverrideModel empty = MakeLongTermModel();
  auto none = empty.suggestWithKey(walk, 1, kFakeNow);
  ASSERT_TRUE(none.suggestion.empty());
  ASSERT_EQ(none.key, result.key);
}

TEST(UserOverrideModelTest, SerializeRoundTrip) {
  // Latest-pick-wins off so that keys hold several candidates.
  UserOverrideModel uom(3, kLongTermHalflife,
                        UserOverrideModel::Policy{/*latestPickWins=*/false,
                                                  /*observeRefresh=*/true});
  const double t1 = kFakeNow + 0.1;
  const double t2 = 1e9 / 3.0;
  const double t3 = 1657772432.123456789;
  uom.observe("(ㄐㄧ,機)-()-(ㄧㄡˊ,由)", "機油", t1, true);
  uom.observe("(ㄐㄧ,機)-()-(ㄧㄡˊ,由)", "油", t2);
  uom.observe("(ㄐㄧ,機)-()-(ㄧㄡˊ,由)", "油", t2 + 1);
  uom.observe("k2", "b", t3);
  uom.observe("k3", "c", t3 + 0.5);
  uom.observe("k2", "b2", t3 + 2.25);  // k2 becomes the most recent.

  std::string text = Serialize(uom);
  ASSERT_EQ(text.rfind("AFM-UOM 1\n", 0), 0u);

  UserOverrideModel copy(3, kLongTermHalflife,
                         UserOverrideModel::Policy{/*latestPickWins=*/false,
                                                   /*observeRefresh=*/true});
  copy.observe("stale", "s", kFakeNow);
  ASSERT_TRUE(Deserialize(&copy, text));
  ASSERT_EQ(copy.size(), 3u);
  ASSERT_TRUE(copy.suggest("stale", kFakeNow).empty());
  ASSERT_EQ(Serialize(copy), text);

  const std::vector<std::string> keys{"(ㄐㄧ,機)-()-(ㄧㄡˊ,由)", "k2", "k3"};
  for (double ts : {t3 + 3, t3 + 3 * kDay, t3 + 139 * kDay, t3 + 141 * kDay}) {
    for (const auto& key : keys) {
      auto a = uom.suggest(key, ts);
      auto b = copy.suggest(key, ts);
      ASSERT_EQ(a.candidate, b.candidate) << key << " " << ts;
      ASSERT_EQ(a.forceHighScoreOverride, b.forceHighScoreOverride);
    }
  }

  // Timestamps are bit-exact.
  {
    std::istringstream is(text);
    std::string line;
    std::getline(is, line);
    std::vector<double> parsed;
    while (std::getline(is, line)) {
      std::vector<std::string> fields;
      std::stringstream ls(line);
      std::string f;
      while (std::getline(ls, f, '\t')) {
        fields.push_back(f);
      }
      ASSERT_EQ(fields.size(), 5u);
      double d = 0;
      auto [ptr, ec] = std::from_chars(
          fields[3].data(), fields[3].data() + fields[3].size(), d);
      ASSERT_EQ(ec, std::errc());
      parsed.push_back(d);
    }
    // LRU order: k2 (b, b2), k3 (c), then the first key (機油, 油; candidates
    // in byte order).
    ASSERT_EQ(parsed.size(), 5u);
    ASSERT_EQ(std::bit_cast<uint64_t>(parsed[0]), std::bit_cast<uint64_t>(t3));
    ASSERT_EQ(std::bit_cast<uint64_t>(parsed[1]),
              std::bit_cast<uint64_t>(t3 + 2.25));
    ASSERT_EQ(std::bit_cast<uint64_t>(parsed[2]),
              std::bit_cast<uint64_t>(t3 + 0.5));
    ASSERT_EQ(std::bit_cast<uint64_t>(parsed[3]), std::bit_cast<uint64_t>(t1));
    ASSERT_EQ(std::bit_cast<uint64_t>(parsed[4]),
              std::bit_cast<uint64_t>(t2 + 1));
  }

  // Same LRU order: adding a new key evicts the same key in both.
  uom.observe("k4", "d", t3 + 10);
  copy.observe("k4", "d", t3 + 10);
  ASSERT_TRUE(uom.suggest("(ㄐㄧ,機)-()-(ㄧㄡˊ,由)", t3 + 11).empty());
  ASSERT_TRUE(copy.suggest("(ㄐㄧ,機)-()-(ㄧㄡˊ,由)", t3 + 11).empty());
  ASSERT_EQ(Serialize(uom), Serialize(copy));
}

TEST(UserOverrideModelTest, SerializeLongTermRoundTripKeepsLatestPick) {
  UserOverrideModel uom = MakeLongTermModel();
  uom.observe("ctx", "鍾", kFakeNow);
  uom.observe("ctx", "鍾", kFakeNow + 1);
  uom.observe("ctx", "中", kFakeNow + 2);
  UserOverrideModel copy = MakeLongTermModel();
  ASSERT_TRUE(Deserialize(&copy, Serialize(uom)));
  ASSERT_EQ(copy.suggest("ctx", kFakeNow + 3 * kDay).candidate, "中");
  ASSERT_EQ(Serialize(copy), Serialize(uom));
}

TEST(UserOverrideModelTest, SerializeEmptyModel) {
  UserOverrideModel uom = MakeLongTermModel();
  std::string text = Serialize(uom);
  ASSERT_EQ(text, "AFM-UOM 1\n");
  UserOverrideModel copy = MakeLongTermModel();
  copy.observe("k", "v", kFakeNow);
  ASSERT_TRUE(Deserialize(&copy, text));
  ASSERT_EQ(copy.size(), 0u);
  ASSERT_TRUE(copy.suggest("k", kFakeNow).empty());
}

TEST(UserOverrideModelTest, DeserializeMalformedLeavesEmptyModel) {
  const std::string good = "k1\tv1\t2\t1657772432.5\t0\n";
  const std::vector<std::string> malformed{
      // Wrong or missing header.
      "",
      "AFM-UOM 2\n" + good,
      "AFM-UOM 1 \n" + good,
      good,
      // Wrong field count.
      "AFM-UOM 1\nk1\tv1\t2\t1657772432.5\n",
      "AFM-UOM 1\nk1\tv1\t2\t1657772432.5\t0\textra\n",
      "AFM-UOM 1\nk1\n",
      // Non-numeric or invalid count.
      "AFM-UOM 1\nk1\tv1\tx\t1657772432.5\t0\n",
      "AFM-UOM 1\nk1\tv1\t\t1657772432.5\t0\n",
      "AFM-UOM 1\nk1\tv1\t-1\t1657772432.5\t0\n",
      "AFM-UOM 1\nk1\tv1\t0\t1657772432.5\t0\n",
      "AFM-UOM 1\nk1\tv1\t1.5\t1657772432.5\t0\n",
      // Non-numeric or invalid timestamp.
      "AFM-UOM 1\nk1\tv1\t2\tabc\t0\n",
      "AFM-UOM 1\nk1\tv1\t2\t\t0\n",
      "AFM-UOM 1\nk1\tv1\t2\tnan\t0\n",
      "AFM-UOM 1\nk1\tv1\t2\tinf\t0\n",
      // Invalid force flag.
      "AFM-UOM 1\nk1\tv1\t2\t1657772432.5\t2\n",
      "AFM-UOM 1\nk1\tv1\t2\t1657772432.5\ttrue\n",
      // Trailing garbage in a field or after the entries.
      "AFM-UOM 1\nk1\tv1\t2x\t1657772432.5\t0\n",
      "AFM-UOM 1\nk1\tv1\t2\t1657772432.5abc\t0\n",
      "AFM-UOM 1\nk1\tv1\t2\t1657772432.5 \t0\n",
      "AFM-UOM 1\n" + good + "garbage\n",
      "AFM-UOM 1\n" + good + "\n",
      "AFM-UOM 1\n" + good + "k2\tv2\t1\t1657772432.5\t0\ttrailing",
      // Empty key or candidate.
      "AFM-UOM 1\n\tv1\t2\t1657772432.5\t0\n",
      "AFM-UOM 1\nk1\t\t2\t1657772432.5\t0\n",
      // Duplicate candidate, and a key split into non-consecutive lines.
      "AFM-UOM 1\n" + good + good,
      "AFM-UOM 1\n" + good + "k2\tv2\t1\t1657772432.5\t0\n" +
          "k1\tv3\t1\t1657772432.5\t0\n",
  };

  // Sanity check: the good line alone is accepted.
  {
    UserOverrideModel uom = MakeLongTermModel();
    ASSERT_TRUE(Deserialize(&uom, "AFM-UOM 1\n" + good));
    ASSERT_EQ(uom.suggest("k1", 1657772432.5).candidate, "v1");
  }

  for (const auto& text : malformed) {
    UserOverrideModel uom = MakeLongTermModel();
    uom.observe("existing", "e", kFakeNow);
    ASSERT_FALSE(Deserialize(&uom, text)) << text;
    ASSERT_EQ(uom.size(), 0u) << text;
    ASSERT_TRUE(uom.suggest("existing", kFakeNow).empty());
    ASSERT_TRUE(uom.suggest("k1", 1657772432.5).empty()) << text;
    ASSERT_EQ(Serialize(uom), "AFM-UOM 1\n");
  }
}

TEST(UserOverrideModelTest, DeserializeEnforcesCapacity) {
  UserOverrideModel big = MakeLongTermModel(10);
  for (int i = 0; i < 5; ++i) {
    big.observe("k" + std::to_string(i), "v" + std::to_string(i),
                kFakeNow + i);
  }
  UserOverrideModel small = MakeLongTermModel(3);
  ASSERT_TRUE(Deserialize(&small, Serialize(big)));
  ASSERT_EQ(small.size(), 3u);
  ASSERT_TRUE(small.suggest("k0", kFakeNow + 10).empty());
  ASSERT_TRUE(small.suggest("k1", kFakeNow + 10).empty());
  ASSERT_EQ(small.suggest("k2", kFakeNow + 10).candidate, "v2");
  ASSERT_EQ(small.suggest("k3", kFakeNow + 10).candidate, "v3");
  ASSERT_EQ(small.suggest("k4", kFakeNow + 10).candidate, "v4");

  // The kept keys stay in LRU order: k2 is evicted next.
  small.observe("k5", "v5", kFakeNow + 11);
  ASSERT_TRUE(small.suggest("k2", kFakeNow + 12).empty());
  ASSERT_EQ(small.suggest("k3", kFakeNow + 12).candidate, "v3");
}

TEST(UserOverrideModelTest, SerializeSkipsEntriesWithSeparators) {
  UserOverrideModel uom(10, kLongTermHalflife,
                        UserOverrideModel::Policy{/*latestPickWins=*/false,
                                                  /*observeRefresh=*/true});
  uom.observe("good", "v", kFakeNow);
  uom.observe("bad\tkey", "v", kFakeNow);
  uom.observe("bad\nkey", "v", kFakeNow);
  uom.observe("good", "bad\tcandidate", kFakeNow);
  uom.observe("good", "bad\ncandidate", kFakeNow);
  uom.observe("other", "w", kFakeNow);

  std::string text = Serialize(uom);
  ASSERT_EQ(text.find("bad"), std::string::npos);

  UserOverrideModel copy(10, kLongTermHalflife,
                         UserOverrideModel::Policy{/*latestPickWins=*/false,
                                                   /*observeRefresh=*/true});
  ASSERT_TRUE(Deserialize(&copy, text));
  ASSERT_EQ(copy.size(), 2u);
  ASSERT_EQ(copy.suggest("other", kFakeNow).candidate, "w");
  ASSERT_EQ(copy.suggest("good", kFakeNow).candidate, "v");
}

}  // namespace McBopomofo
