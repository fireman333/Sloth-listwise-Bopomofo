// Sentence-pool reader for personalized training (personal-finetune design D7):
// reads memory/sentences.jsonl (personal-memory D4 records plus retraction
// lines), applies eligibility, flattens multi-syllable nodes, cuts segments at
// non-syllable nodes, filters by the R vocabularies, deduplicates, and splits
// newest-first into the personal test set and the (capped) training set.
//
// Record: {"v":1,"t":<epoch s>,"manual":<bool>,...,"nodes":[[reading,raw,src],...]}
// Retraction: {"v":1,"retract":<t of the record>}
// Timestamps are compared in integer milliseconds (records are written with
// three decimals).

#ifndef SOURCE_ENGINE_PERSONALIZE_POOL_READER_H_
#define SOURCE_ENGINE_PERSONALIZE_POOL_READER_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace McBopomofo {
namespace Personalize {

struct Vocab {
  std::unordered_set<std::string> syllables;  // syl_vocab.json keys
  std::unordered_set<std::string> chars;      // tokenizer.json vocab + added tokens
};

// Loads syl_vocab.json and tokenizer/tokenizer.json of an R model directory.
bool LoadModelVocab(const std::string& modelDir, Vocab* out, std::string* error);

struct Segment {
  std::vector<std::string> readings;  // one syllable per character
  std::string target;                 // committed characters (UTF-8)
  int64_t tMs = 0;                    // commit time of the record
};

struct SegmentOptions {
  size_t minSyllables = 2;
  size_t maxSyllables = 40;
};

struct PoolScan {
  // Eligible, deduplicated segments, oldest first.
  std::vector<Segment> segments;
  // Byte offset just after the last complete line that was read.
  uint64_t scanOffset = 0;
  std::unordered_set<int64_t> retracted;  // record times (ms) with a retract line
  size_t lines = 0;
  size_t malformedLines = 0;
  size_t records = 0;
  size_t retractedRecords = 0;
  size_t ineligibleRecords = 0;    // node character count != syllable count
  size_t droppedSegments = 0;      // length or vocabulary filters
  size_t duplicateSegments = 0;
  size_t overlongLines = 0;        // longer than ReadOptions::maxLineBytes, skipped
  uint64_t windowStart = 0;        // first byte considered (> 0: pool above the cap)
};

// Bounds for reading the pool file (Codex review 2026-10-08, finding 3): the
// file is streamed line by line, never held whole in memory.
struct ReadOptions {
  // Only the newest maxPoolBytes before the end are read (the partial line at
  // the window start is skipped); 0 = no cap. 32 MiB is ~50k records, far more
  // than the newest 2,000 + 400 segments a run uses.
  uint64_t maxPoolBytes = 32ull << 20;
  // Lines longer than this are skipped (counted in overlongLines).
  size_t maxLineBytes = 64u << 10;
  // Stop at this byte offset instead of EOF (0 = EOF).
  uint64_t endOffset = 0;
};

// True when every code point of `s` is a Bopomofo letter or tone mark and it
// contains at least one letter.
bool IsBopomofoSyllable(std::string_view s);

// UTF-8 code points of s (invalid bytes become one-byte pieces).
std::vector<std::string> CodePoints(std::string_view s);

struct PoolNode {
  std::string reading;
  std::string raw;
};

// Cuts one record into candidate segments (before the length / vocabulary
// filters are applied, which happen in AppendSegments). Returns false when the
// record is ineligible (a syllable node whose character count differs from its
// syllable count).
bool RecordSegments(const std::vector<PoolNode>& nodes, std::vector<Segment>* out);

// Scans the complete lines of `bytes` (everything up to the last '\n').
void ScanPoolBytes(std::string_view bytes, const Vocab& vocab, const SegmentOptions& options,
                   PoolScan* out);
// Streams the complete lines of the file within the ReadOptions bounds.
bool ScanPool(const std::string& path, const Vocab& vocab, const SegmentOptions& options,
              PoolScan* out, std::string* error, const ReadOptions& read = ReadOptions());

// Defaults = research 17 (E7, task 1.7): hold-out clamp(round(0.30 N), 300, 400).
struct SplitOptions {
  double holdoutFraction = 0.30;
  size_t holdoutMin = 300;
  size_t holdoutMax = 400;  // 0 = no upper bound
  size_t trainCap = 2000;   // newest training segments kept; 0 = no cap
};

struct Split {
  std::vector<Segment> train;  // oldest first
  std::vector<Segment> test;   // oldest first
};

// Size of the personal test set for n segments:
// min(n, clamp(round(f * n), min, max)) (max ignored when 0).
size_t HoldoutSize(size_t n, const SplitOptions& options);

// `segments` must be oldest first (as returned by ScanPool).
Split SplitSegments(const std::vector<Segment>& segments, const SplitOptions& options);

// Record times with a retract line among the complete lines of
// [offset, EOF) of the pool file (late retractions, D7). Streamed; lines
// longer than maxLineBytes are skipped; maxPoolBytes / endOffset are ignored.
bool ScanRetractions(const std::string& path, uint64_t offset, std::unordered_set<int64_t>* out,
                     std::string* error, const ReadOptions& read = ReadOptions());
void ScanRetractionBytes(std::string_view bytes, std::unordered_set<int64_t>* out);

// True when any training or test segment comes from a retracted record.
bool UsesRetracted(const Split& split, const std::unordered_set<int64_t>& retracted);

}  // namespace Personalize
}  // namespace McBopomofo

#endif  // SOURCE_ENGINE_PERSONALIZE_POOL_READER_H_
