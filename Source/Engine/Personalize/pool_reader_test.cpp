// Task 2.1: pool reader, eligibility, segmenting, dedup, split and late
// retractions (design D7).

#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "pool_reader.h"

namespace McBopomofo {
namespace Personalize {
namespace {

Vocab TestVocab() {
  Vocab v;
  for (const char* s : {"ㄘㄜˋ", "ㄕˋ", "ㄓㄨㄥ", "ㄨㄣˊ", "ㄕㄨ", "ㄖㄨˋ", "ㄏㄠˇ", "ㄉㄜ˙", "ㄋㄧˇ"})
    v.syllables.insert(s);
  for (const char* c : {"測", "試", "中", "文", "書", "入", "輸", "好", "的", "你"}) v.chars.insert(c);
  return v;
}

// One pool record line (no trailing newline).
std::string Rec(double t, bool manual, const std::vector<std::pair<std::string, std::string>>& nodes) {
  char tb[64];
  snprintf(tb, sizeof tb, "%.3f", t);
  std::string s = std::string("{\"v\":1,\"t\":") + tb + ",\"manual\":" + (manual ? "true" : "false") +
                  ",\"walk\":\"r\",\"app\":\"1.0\",\"model\":\"m\",\"nodes\":[";
  for (size_t i = 0; i < nodes.size(); ++i) {
    if (i) s += ",";
    s += "[\"" + nodes[i].first + "\",\"" + nodes[i].second + "\",\"w\"]";
  }
  return s + "]}";
}

std::string Retract(double t) {
  char tb[64];
  snprintf(tb, sizeof tb, "%.3f", t);
  return std::string("{\"v\":1,\"retract\":") + tb + "}";
}

PoolScan Scan(const std::string& bytes, const SegmentOptions& o = SegmentOptions()) {
  PoolScan s;
  ScanPoolBytes(bytes, TestVocab(), o, &s);
  return s;
}

std::string Join(const std::vector<std::string>& v) {
  std::string s;
  for (size_t i = 0; i < v.size(); ++i) s += (i ? " " : "") + v[i];
  return s;
}

TEST(PoolReaderTest, SyllableDetection) {
  EXPECT_TRUE(IsBopomofoSyllable("ㄓㄨㄥ"));
  EXPECT_TRUE(IsBopomofoSyllable("ㄉㄜ˙"));
  EXPECT_TRUE(IsBopomofoSyllable("ㄕˋ"));
  EXPECT_FALSE(IsBopomofoSyllable(""));
  EXPECT_FALSE(IsBopomofoSyllable("ˋ"));
  EXPECT_FALSE(IsBopomofoSyllable("_punctuation_,"));
  EXPECT_FALSE(IsBopomofoSyllable(" "));
  EXPECT_FALSE(IsBopomofoSyllable("ㄓㄨㄥa"));
}

TEST(PoolReaderTest, PartialLastLineIgnored) {
  std::string a = Rec(100.0, true, {{"ㄓㄨㄥ", "中"}, {"ㄨㄣˊ", "文"}}) + "\n";
  std::string b = Rec(101.0, true, {{"ㄕㄨ", "書"}, {"ㄖㄨˋ", "入"}});
  PoolScan s = Scan(a + b.substr(0, b.size() / 2));
  EXPECT_EQ(s.records, 1u);
  EXPECT_EQ(s.malformedLines, 0u);
  EXPECT_EQ(s.scanOffset, a.size());
  ASSERT_EQ(s.segments.size(), 1u);
  EXPECT_EQ(s.segments[0].target, "中文");
  // The full line once complete.
  s = Scan(a + b + "\n");
  EXPECT_EQ(s.segments.size(), 2u);
  EXPECT_EQ(s.scanOffset, a.size() + b.size() + 1);
}

TEST(PoolReaderTest, MultiSyllableNodeIsFlattened) {
  PoolScan s = Scan(Rec(5.0, false, {{"ㄘㄜˋ-ㄕˋ", "測試"}, {"ㄓㄨㄥ", "中"}}) + "\n");
  ASSERT_EQ(s.segments.size(), 1u);
  EXPECT_EQ(Join(s.segments[0].readings), "ㄘㄜˋ ㄕˋ ㄓㄨㄥ");
  EXPECT_EQ(s.segments[0].target, "測試中");
  EXPECT_EQ(s.segments[0].tMs, 5000);
}

TEST(PoolReaderTest, CharacterCountMismatchMakesRecordIneligible) {
  // Three syllables, two characters (e.g. a macro / replacement leftover).
  std::string bytes = Rec(1.0, true, {{"ㄘㄜˋ-ㄕˋ-ㄓㄨㄥ", "測試"}, {"ㄕㄨ", "書"}, {"ㄖㄨˋ", "入"}}) + "\n" +
                      Rec(2.0, true, {{"ㄓㄨㄥ", "中文"}, {"ㄕㄨ", "書"}}) + "\n" +
                      Rec(3.0, true, {{"ㄕㄨ", "書"}, {"ㄖㄨˋ", "入"}}) + "\n";
  PoolScan s = Scan(bytes);
  EXPECT_EQ(s.records, 3u);
  EXPECT_EQ(s.ineligibleRecords, 2u);
  ASSERT_EQ(s.segments.size(), 1u);  // the third record only
  EXPECT_EQ(s.segments[0].tMs, 3000);
}

TEST(PoolReaderTest, PunctuationSplitsSegments) {
  // ㄓㄨㄥ ㄨㄣˊ , ㄕㄨ ㄖㄨˋ -> two 2-syllable segments; the comma is not used.
  PoolScan s = Scan(Rec(7.0, true, {{"ㄓㄨㄥ", "中"}, {"ㄨㄣˊ", "文"}, {"_punctuation_,", "，"},
                                     {"ㄕㄨ", "書"}, {"ㄖㄨˋ", "入"}}) + "\n");
  ASSERT_EQ(s.segments.size(), 2u);
  EXPECT_EQ(s.segments[0].target, "中文");
  EXPECT_EQ(s.segments[1].target, "書入");
  EXPECT_EQ(Join(s.segments[1].readings), "ㄕㄨ ㄖㄨˋ");
  // A space node also separates; a single syllable left over is too short.
  s = Scan(Rec(8.0, true, {{"ㄓㄨㄥ", "中"}, {" ", " "}, {"ㄕㄨ", "書"}, {"ㄖㄨˋ", "入"}}) + "\n");
  ASSERT_EQ(s.segments.size(), 1u);
  EXPECT_EQ(s.segments[0].target, "書入");
  EXPECT_EQ(s.droppedSegments, 1u);
}

TEST(PoolReaderTest, UnknownSyllableOrCharacterDropsTheSegment) {
  std::string bytes =
      // unknown syllable ㄅㄚ (not in vocab) in the first segment only
      Rec(1.0, true, {{"ㄅㄚ", "中"}, {"ㄨㄣˊ", "文"}, {"_punctuation_,", "，"}, {"ㄕㄨ", "書"}, {"ㄖㄨˋ", "入"}}) +
      "\n" +
      // unknown character 忠
      Rec(2.0, true, {{"ㄓㄨㄥ", "忠"}, {"ㄨㄣˊ", "文"}}) + "\n";
  PoolScan s = Scan(bytes);
  ASSERT_EQ(s.segments.size(), 1u);
  EXPECT_EQ(s.segments[0].target, "書入");
  EXPECT_EQ(s.droppedSegments, 2u);
}

TEST(PoolReaderTest, LengthLimits) {
  SegmentOptions o;
  o.minSyllables = 2;
  o.maxSyllables = 3;
  PoolScan s = Scan(Rec(1.0, true, {{"ㄘㄜˋ-ㄕˋ", "測試"}, {"ㄓㄨㄥ", "中"}, {"ㄨㄣˊ", "文"}}) + "\n", o);
  EXPECT_TRUE(s.segments.empty());
  o.maxSyllables = 40;
  s = Scan(Rec(1.0, true, {{"ㄘㄜˋ-ㄕˋ", "測試"}, {"ㄓㄨㄥ", "中"}, {"ㄨㄣˊ", "文"}}) + "\n", o);
  EXPECT_EQ(s.segments.size(), 1u);
}

TEST(PoolReaderTest, DedupKeepsTheNewest) {
  std::string bytes = Rec(10.0, true, {{"ㄓㄨㄥ", "中"}, {"ㄨㄣˊ", "文"}}) + "\n" +
                      Rec(20.0, false, {{"ㄕㄨ", "書"}, {"ㄖㄨˋ", "入"}}) + "\n" +
                      Rec(30.0, false, {{"ㄓㄨㄥ", "中"}, {"ㄨㄣˊ", "文"}}) + "\n" +
                      // same readings, another target: not a duplicate
                      Rec(40.0, true, {{"ㄓㄨㄥ", "中"}, {"ㄨㄣˊ", "文"}}) + "\n";
  PoolScan s = Scan(bytes);
  ASSERT_EQ(s.segments.size(), 2u);
  EXPECT_EQ(s.duplicateSegments, 2u);
  EXPECT_EQ(s.segments[0].target, "書入");
  EXPECT_EQ(s.segments[0].tMs, 20000);
  EXPECT_EQ(s.segments[1].target, "中文");
  EXPECT_EQ(s.segments[1].tMs, 40000);
}

TEST(PoolReaderTest, MalformedLinesSkippedRestRead) {
  std::string bytes = "not json\n" + Rec(1.0, true, {{"ㄓㄨㄥ", "中"}, {"ㄨㄣˊ", "文"}}) + "\n" +
                      "{\"v\":2,\"t\":2.0,\"manual\":true,\"nodes\":[]}\n" +  // other version
                      "{\"v\":1,\"t\":\"x\",\"manual\":true,\"nodes\":[]}\n" +  // bad type
                      "{\"v\":1,\"t\":3.0,\"manual\":true,\"nodes\":[[1,2]]}\n" +  // bad node
                      "\n" + "[1,2,3]\n" + "{\"v\":1,\"t\":4.0,\"manual\":true,\"nodes\":[[\"ㄕㄨ\"" + "\n" +
                      Rec(5.0, false, {{"ㄕㄨ", "書"}, {"ㄖㄨˋ", "入"}}) + "\n";
  PoolScan s = Scan(bytes);
  EXPECT_EQ(s.malformedLines, 6u);
  EXPECT_EQ(s.records, 2u);
  ASSERT_EQ(s.segments.size(), 2u);
  EXPECT_EQ(s.segments[0].target, "中文");
  EXPECT_EQ(s.segments[1].target, "書入");
}

TEST(PoolReaderTest, RetractedRecordExcludedFromTrainAndTest) {
  std::string bytes;
  // 300 distinct-ish records: alternate 2 syllable patterns with distinct times.
  // Use t = i; segments differ by target ordering to avoid dedup: build from
  // a 10-char alphabet pair (i % 10, (i / 10) % 10) -> 100 distinct pairs.
  const char* syl[] = {"ㄘㄜˋ", "ㄕˋ", "ㄓㄨㄥ", "ㄨㄣˊ", "ㄕㄨ", "ㄖㄨˋ", "ㄏㄠˇ", "ㄉㄜ˙", "ㄋㄧˇ", "ㄕㄨ"};
  const char* chr[] = {"測", "試", "中", "文", "書", "入", "好", "的", "你", "輸"};
  for (int i = 0; i < 100; ++i) {
    int a = i % 10, b = i / 10;
    bytes += Rec(1000.0 + i, i % 2 == 0, {{syl[a], chr[a]}, {syl[b], chr[b]}}) + "\n";
  }
  bytes += Retract(1000.0 + 99) + "\n";  // newest record (would be in the test set)
  bytes += Retract(1000.0 + 3) + "\n";   // an old record (train)
  PoolScan s = Scan(bytes);
  EXPECT_EQ(s.retractedRecords, 2u);
  EXPECT_EQ(s.segments.size(), 98u);
  SplitOptions o;
  o.holdoutFraction = 0.1;
  o.holdoutMin = 20;
  Split sp = SplitSegments(s.segments, o);
  EXPECT_EQ(sp.test.size(), 20u);
  EXPECT_EQ(sp.train.size(), 78u);
  for (const auto* v : {&sp.train, &sp.test})
    for (const auto& seg : *v) {
      EXPECT_NE(seg.tMs, 1099000);
      EXPECT_NE(seg.tMs, 1003000);
    }
  EXPECT_FALSE(UsesRetracted(sp, s.retracted));
}

TEST(PoolReaderTest, LateRetractionFoundAfterScanOffset) {
  const std::string path = "/tmp/afm_pool_reader_test_" + std::to_string(getpid()) + ".jsonl";
  std::string bytes;
  for (int i = 0; i < 10; ++i)
    bytes += Rec(50.0 + i, true, {{"ㄓㄨㄥ", "中"}, {i % 2 ? "ㄨㄣˊ" : "ㄕㄨ", i % 2 ? "文" : "書"}}) + "\n";
  {
    std::ofstream f(path, std::ios::binary);
    f << bytes;
  }
  PoolScan s;
  std::string err;
  ASSERT_TRUE(ScanPool(path, TestVocab(), SegmentOptions(), &s, &err)) << err;
  EXPECT_EQ(s.scanOffset, bytes.size());
  ASSERT_EQ(s.segments.size(), 2u);  // dedup: two distinct segments (t 58, 59)
  SplitOptions o;
  o.holdoutMin = 1;
  Split sp = SplitSegments(s.segments, o);
  // Appended after the scan: a retraction of an unused record, then of a used one,
  // and a partial line.
  {
    std::ofstream f(path, std::ios::binary | std::ios::app);
    f << Retract(50.0) << "\n";
  }
  std::unordered_set<int64_t> late;
  ASSERT_TRUE(ScanRetractions(path, s.scanOffset, &late, &err)) << err;
  EXPECT_EQ(late.size(), 1u);
  EXPECT_FALSE(UsesRetracted(sp, late));  // record t=50 was deduplicated away
  {
    std::ofstream f(path, std::ios::binary | std::ios::app);
    f << Retract(59.0) << "\n" << "{\"v\":1,\"retract\":58.0";
  }
  late.clear();
  ASSERT_TRUE(ScanRetractions(path, s.scanOffset, &late, &err)) << err;
  EXPECT_EQ(late.size(), 2u);  // the partial line is not counted
  EXPECT_TRUE(UsesRetracted(sp, late));
  // A retraction before the offset is not "late".
  late.clear();
  ASSERT_TRUE(ScanRetractions(path, s.scanOffset + 1000000, &late, &err));
  EXPECT_TRUE(late.empty());
  std::remove(path.c_str());
}

TEST(PoolReaderTest, SplitSizes) {
  auto make = [](size_t n) {
    std::vector<Segment> v(n);
    for (size_t i = 0; i < n; ++i) v[i].tMs = static_cast<int64_t>(i);
    return v;
  };
  SplitOptions o;  // clamp(round(30%), 300, 400), train cap 2,000 (research 17)
  struct Case {
    size_t n, test, train;
  } cases[] = {{150, 150, 0}, {1000, 300, 700}, {2000, 400, 1600}, {3000, 400, 2000}};
  for (const auto& c : cases) {
    Split s = SplitSegments(make(c.n), o);
    EXPECT_EQ(s.test.size(), c.test) << c.n;
    EXPECT_EQ(s.train.size(), c.train) << c.n;
    if (!s.test.empty()) {
      EXPECT_EQ(s.test.back().tMs, static_cast<int64_t>(c.n - 1));  // newest in test
      EXPECT_EQ(s.test.front().tMs, static_cast<int64_t>(c.n - c.test));
    }
    if (!s.train.empty()) {
      EXPECT_EQ(s.train.back().tMs, static_cast<int64_t>(c.n - c.test - 1));
      // cap keeps the newest training segments
      EXPECT_EQ(s.train.front().tMs, static_cast<int64_t>(c.n - c.test - c.train));
    }
  }
  EXPECT_EQ(HoldoutSize(1200, o), 360u);
  // The spec's former max(10%, 200) rule is still expressible.
  SplitOptions old;
  old.holdoutFraction = 0.1;
  old.holdoutMin = 200;
  old.holdoutMax = 0;
  EXPECT_EQ(HoldoutSize(1000, old), 200u);
  EXPECT_EQ(HoldoutSize(2500, old), 250u);
  EXPECT_EQ(HoldoutSize(3000, old), 300u);
}

TEST(PoolReaderTest, LoadsTheRModelVocab) {
  Vocab v;
  std::string err;
  ASSERT_TRUE(LoadModelVocab(RWALK_MODEL_DIR, &v, &err)) << err;
  EXPECT_GT(v.syllables.size(), 1000u);
  EXPECT_GT(v.chars.size(), 5000u);
  EXPECT_TRUE(v.syllables.count("ㄓㄨㄥ"));
  EXPECT_TRUE(v.syllables.count("ㄉㄜ˙"));
  EXPECT_TRUE(v.chars.count("中"));
}

}  // namespace
}  // namespace Personalize
}  // namespace McBopomofo
