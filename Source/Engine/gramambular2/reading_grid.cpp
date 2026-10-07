// Copyright (c) 2022 and onwards Lukhnos Liu.
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

#include "reading_grid.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <memory>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace Formosa::Gramambular2 {

void ReadingGrid::clear() {
  cursor_ = 0;
  readings_.clear();
  spans_.clear();
}

void ReadingGrid::setCursor(size_t cursor) {
  assert(cursor <= readings_.size());
  cursor_ = cursor;
}

void ReadingGrid::setReadingSeparator(const std::string& separator) {
  separator_ = separator;
}

bool ReadingGrid::insertReading(const std::string& reading) {
  if (reading.empty() || reading == separator_) {
    return false;
  }

  std::string insertedReading = reading;
  auto unigrams = lm_.getUnigrams(insertedReading);
  if (unigrams.empty()) {
    return false;
  }

  readings_.insert(readings_.begin() + static_cast<ptrdiff_t>(cursor_),
                   std::move(insertedReading));
  expandGridAt(cursor_);
  insert(cursor_,
         std::make_shared<Node>(readings_[cursor_], 1, std::move(unigrams)));
  update(cursor_, EditType::kInsertion);

  // Cursor must only move after update().
  ++cursor_;
  return true;
}

bool ReadingGrid::deleteReadingBeforeCursor() {
  if (!cursor_) {
    return false;
  }

  readings_.erase(readings_.begin() + static_cast<ptrdiff_t>(cursor_ - 1),
                  readings_.begin() + static_cast<ptrdiff_t>(cursor_));
  // Cursor must decrement for grid-shrinking and update to work.
  --cursor_;
  shrinkGridAt(cursor_);
  update(cursor_, EditType::kDeletion);
  return true;
}

bool ReadingGrid::deleteReadingAfterCursor() {
  if (cursor_ == readings_.size()) {
    return false;
  }

  readings_.erase(readings_.begin() + static_cast<ptrdiff_t>(cursor_),
                  readings_.begin() + static_cast<ptrdiff_t>(cursor_ + 1));
  shrinkGridAt(cursor_);
  update(cursor_, EditType::kDeletion);
  return true;
}

std::optional<ReadingGrid::NodePtr> ReadingGrid::findInSpan(
    size_t cursor, const std::function<bool(const NodePtr&)>& predicate) const {
  assert(cursor <= readings_.size());
  std::vector<ReadingGrid::NodeInSpan> nodes =
      overlappingNodesAt(cursor == readings_.size() ? cursor - 1 : cursor);

  auto nodesIt = std::find_if(
      nodes.cbegin(), nodes.cend(),
      [&](const NodeInSpan& nodeInSpan) { return predicate(nodeInSpan.node); });

  return nodesIt == nodes.end()
             ? std::nullopt
             : std::optional<ReadingGrid::NodePtr>(nodesIt->node);
}

namespace {

int64_t GetEpochNowInMicroseconds() {
  auto now = std::chrono::system_clock::now();
  int64_t timestamp =
      std::chrono::time_point_cast<std::chrono::microseconds>(now)
          .time_since_epoch()
          .count();
  return timestamp;
}

}  // namespace

// Find the weightiest path in the grid graph. The path represents the most
// likely hidden chain of events from the observations.
// We use the Viterbi algorithm to compute such path.
// Instead of computing the path with the shortest distance, though, we compute
// the path with the longest distance (so the weightiest), since with log
// probability a larger value means a larger probability. The algorithm runs in
// O(|V| + |E|) time for G = (V, E) where G is a DAG. This means the walk is
// fairly economical even when the grid is large.
ReadingGrid::WalkResult ReadingGrid::walk() {
  WalkResult result;
  if (spans_.empty()) {
    return result;
  }
  int64_t start = GetEpochNowInMicroseconds();

  // Defines a state in the DP table. This structure tracks the maximum
  // accumulated score and the back-pointer required for path reconstruction in
  // the Viterbi algorithm.
  struct State {
    size_t fromIndex = 0;
    ReadingGrid::NodePtr fromNode = nullptr;
    double maxScore = -std::numeric_limits<double>::infinity();
  };

  const size_t readingLen = readings_.size();
  std::vector<State> viterbi(readingLen + 1);
  viterbi[0].maxScore = 0.0;

  // Iterate through the grid and compute the maximum accumulated score for each
  // reachable position. Since the grid is a lattice where edges only point
  // forward, processing nodes in index order is equivalent to processing them
  // in topological order.
  size_t reachableStates = 0;
  size_t evaluatedEdges = 0;
  for (size_t i = 0; i < readingLen; ++i) {
    ++reachableStates;

    const ReadingGrid::Span& span = spans_[i];
    const size_t maxSpanLen = span.maxLength();

    for (size_t spanLen = 1; spanLen <= maxSpanLen; ++spanLen) {
      const ReadingGrid::NodePtr& node = span.nodeOf(spanLen);
      if (node == nullptr) {
        continue;
      }
      ++evaluatedEdges;

      // Performs a relaxation on a transition. This updates the destination
      // state if the path through the current node yields a higher score than
      // the previously known best path. This is the core operation of the
      // Viterbi algorithm, adapted for finding the maximum likelihood path.
      double score = viterbi[i].maxScore + node->score();
      State& target = viterbi[i + spanLen];
      if (score > target.maxScore) {
        target.maxScore = score;
        target.fromNode = node;
        target.fromIndex = i;
      }
    }
  }
  // Vertices are the reachable states
  // Edges are the candidate word transitions
  result.vertices = reachableStates;
  result.edges = evaluatedEdges;

  // Reconstruct the most likely path by tracing back from the end of the grid
  // to the root using the back-pointers
  size_t totalReadingLen = 0;
  for (size_t curr = readingLen; curr > 0; curr = viterbi[curr].fromIndex) {
    assert(viterbi[curr].fromNode != nullptr);
    totalReadingLen += viterbi[curr].fromNode->spanningLength();
    result.nodes.emplace_back(std::move(viterbi[curr].fromNode));
  }
  std::reverse(result.nodes.begin(), result.nodes.end());
  assert(totalReadingLen == readingLen);
  result.totalReadings = totalReadingLen;

  result.elapsedMicroseconds = GetEpochNowInMicroseconds() - start;
  return result;
}

namespace {
void SplitCodePoints(const std::string& s, std::vector<std::string>* out);

// Bonus score of character ch at reading position pos. A user-phrase unigram
// gets the best score of the row (design D8: user phrases are never
// disadvantaged by the bonus); an empty row or a position past the end scores
// missingScore.
double PositionBonus(const ReadingGrid::PositionalCharScores& scores,
                     size_t pos, const std::string& ch, bool isUserPhrase) {
  if (pos >= scores.table.size()) {
    return scores.missingScore;
  }
  const auto& m = scores.table[pos];
  if (isUserPhrase) {
    if (m.empty()) {
      return scores.missingScore;
    }
    double best = -std::numeric_limits<double>::infinity();
    for (const auto& kv : m) {
      best = std::max(best, kv.second);
    }
    return best;
  }
  auto it = m.find(ch);
  return it == m.end() ? scores.missingScore : it->second;
}
}  // namespace

std::vector<ReadingGrid::WalkResult> ReadingGrid::walkNBest(size_t k) {
  return walkNBestImpl(k, 0.0, nullptr);
}

std::vector<ReadingGrid::WalkResult> ReadingGrid::walkNBestWithCharBonus(
    size_t k, double alpha, const PositionalCharScores& scores) {
  if (scores.table.size() != readings_.size()) {
    return walkNBest(k);
  }
  return walkNBestImpl(k, alpha, &scores);
}

std::vector<ReadingGrid::WalkResult> ReadingGrid::walkNBestImpl(
    size_t k, double alpha, const PositionalCharScores* scores) {
  std::vector<WalkResult> results;
  if (spans_.empty() || k == 0) {
    return results;
  }
  int64_t start = GetEpochNowInMicroseconds();

  // A partial path ending at some position, identified by its joined string.
  // Back-pointers (prevPos, prevEntry) let us rebuild the node sequence.
  // `text` points at the key of index[pos] (node-based map keys never move);
  // `node` points into spans_, which does not change during the walk.
  struct Entry {
    const std::string* text = nullptr;
    double score = 0;
    uint64_t seq = 0;  // Order in which the current score was achieved.
    size_t prevPos = 0;
    size_t prevEntry = 0;
    const NodePtr* node = nullptr;
    size_t unigramIndex = 0;
  };

  static const std::string kEmpty;
  const size_t readingLen = readings_.size();
  std::vector<std::vector<Entry>> table(readingLen + 1);
  // Per position: joined text -> index into table[pos] (dedup by text).
  std::vector<std::unordered_map<std::string, size_t>> index(readingLen + 1);
  table[0].push_back(Entry{&kEmpty});
  uint64_t seq = 0;
  size_t reachableStates = 0;
  size_t evaluatedEdges = 0;
  std::vector<double> optionAdds;
  std::vector<std::string> chars;
  std::vector<size_t> order;
  std::vector<size_t> byAdd;
  std::vector<size_t> kept;
  std::vector<size_t> keptForPrefix;
  std::vector<const std::string*> distinctValues;
  std::string text;

  // Total order on entries of one position: higher score first, then the
  // earlier-achieved one (seq is unique per position).
  auto better = [](const Entry& a, const Entry& b) {
    if (a.score != b.score) {
      return a.score > b.score;
    }
    return a.seq < b.seq;
  };

  for (size_t i = 0; i < readingLen; ++i) {
    if (table[i].empty()) {
      continue;
    }
    ++reachableStates;

    // Keep the top k distinct prefixes at this position. Keys are unique, so
    // a partial sort gives the same prefix as a full stable sort.
    const std::vector<Entry>& entries = table[i];
    order.resize(entries.size());
    for (size_t j = 0; j < order.size(); ++j) {
      order[j] = j;
    }
    auto byEntry = [&](size_t a, size_t b) {
      return better(entries[a], entries[b]);
    };
    if (order.size() > k) {
      std::partial_sort(order.begin(),
                        order.begin() + static_cast<ptrdiff_t>(k), order.end(),
                        byEntry);
      order.resize(k);
    } else {
      std::sort(order.begin(), order.end(), byEntry);
    }

    const Span& span = spans_[i];
    for (size_t spanLen = 1; spanLen <= span.maxLength(); ++spanLen) {
      const NodePtr& node = span.nodeOf(spanLen);
      if (node == nullptr) {
        continue;
      }
      ++evaluatedEdges;

      // The unigram indices this node offers. An overridden node offers only
      // its current unigram.
      const auto& unigrams = node->unigrams();
      size_t optionBegin = 0;
      size_t optionEnd = std::max<size_t>(unigrams.size(), 1);
      if (node->isOverridden() && !unigrams.empty()) {
        const std::string current = node->value();
        optionBegin = static_cast<size_t>(
            std::distance(unigrams.begin(),
                          std::find_if(unigrams.begin(), unigrams.end(),
                                       [&](const LanguageModel::Unigram& u) {
                                         return u.value() == current;
                                       })));
        optionEnd = optionBegin + 1;
      }

      // Per-option edge score, computed once per node: the unigram score,
      // plus alpha * sum of positional char scores when a bonus table is
      // given (same rule as walkWithCharBonus(); overridden/empty nodes keep
      // their stock score).
      optionAdds.clear();
      for (size_t u = optionBegin; u < optionEnd; ++u) {
        double add = (node->isOverridden() || unigrams.empty())
                         ? node->score()
                         : unigrams[u].score();
        if (scores != nullptr && alpha != 0 && !node->isOverridden() &&
            !unigrams.empty()) {
          double bonus = 0;
          SplitCodePoints(unigrams[u].value(), &chars);
          const bool userPhrase = unigrams[u].isUserPhrase();
          for (size_t j = 0; j < chars.size(); ++j) {
            const size_t pos = i + j;
            if (pos >= readingLen) {
              bonus += scores->missingScore;
              continue;
            }
            bonus += PositionBonus(*scores, pos, chars[j], userPhrase);
          }
          add += alpha * bonus;
        }
        optionAdds.push_back(add);
      }

      // Option pruning (exact). Sort options by (add desc, index asc) and
      // find the first m + 1 options that hold k distinct values; let
      // threshold = add of the (m+1)-th. For a prefix with score P, an option
      // u with P + add_u < P + threshold is dominated: either an option with
      // the same value scores strictly higher from the same prefix (so u
      // never sets its text's final entry), or k distinct texts from the same
      // prefix score strictly higher (so its text cannot be in the top k at
      // the target position). Skipping dominated options changes neither
      // the top-k entries nor their relative seq order. Floating-point
      // addition is monotone, so P + add is non-increasing along byAdd; ties
      // created by rounding at the threshold are kept per prefix below.
      const size_t n = optionEnd - optionBegin;
      bool prune = false;
      size_t cutoff = n;  // byAdd[0, cutoff) have add >= threshold
      double threshold = 0;
      kept.clear();
      if (n > k) {
        byAdd.resize(n);
        for (size_t j = 0; j < n; ++j) {
          byAdd[j] = j;
        }
        std::sort(byAdd.begin(), byAdd.end(), [&](size_t a, size_t b) {
          if (optionAdds[a] != optionAdds[b]) {
            return optionAdds[a] > optionAdds[b];
          }
          return a < b;
        });
        distinctValues.clear();
        size_t m = 0;
        for (; m < n; ++m) {
          const std::string& v = unigrams[optionBegin + byAdd[m]].value();
          if (std::none_of(distinctValues.begin(), distinctValues.end(),
                           [&](const std::string* s) { return *s == v; })) {
            distinctValues.push_back(&v);
            if (distinctValues.size() == k) {
              break;
            }
          }
        }
        if (m < n) {
          threshold = optionAdds[byAdd[m]];
          cutoff = m + 1;
          while (cutoff < n && optionAdds[byAdd[cutoff]] >= threshold) {
            ++cutoff;
          }
          prune = cutoff < n;
        }
      }
      if (prune) {
        kept.assign(byAdd.begin(),
                    byAdd.begin() + static_cast<ptrdiff_t>(cutoff));
        std::sort(kept.begin(), kept.end());
      } else {
        kept.resize(n);
        for (size_t j = 0; j < n; ++j) {
          kept[j] = j;
        }
      }

      std::vector<Entry>& target = table[i + spanLen];
      auto& targetIndex = index[i + spanLen];
      for (size_t idx : order) {
        const Entry& prefix = entries[idx];
        const std::vector<size_t>* options = &kept;
        if (prune) {
          const double t = prefix.score + threshold;
          if (prefix.score + optionAdds[byAdd[cutoff]] >= t) {
            keptForPrefix = kept;
            for (size_t j = cutoff;
                 j < n && prefix.score + optionAdds[byAdd[j]] >= t; ++j) {
              keptForPrefix.push_back(byAdd[j]);
            }
            std::sort(keptForPrefix.begin(), keptForPrefix.end());
            options = &keptForPrefix;
          }
        }
        for (size_t off : *options) {
          const size_t u = optionBegin + off;
          text.assign(*prefix.text);
          text.append(unigrams.empty() ? kEmpty : unigrams[u].value());
          const double score = prefix.score + optionAdds[off];
          auto [it, inserted] = targetIndex.try_emplace(text, target.size());
          if (inserted) {
            target.push_back(
                Entry{&it->first, score, seq++, i, idx, &node, u});
          } else if (score > target[it->second].score) {
            target[it->second] =
                Entry{&it->first, score, seq++, i, idx, &node, u};
          }
        }
      }
    }
  }

  const std::vector<Entry>& finals = table[readingLen];
  order.resize(finals.size());
  for (size_t j = 0; j < order.size(); ++j) {
    order[j] = j;
  }
  auto byFinal = [&](size_t a, size_t b) {
    return better(finals[a], finals[b]);
  };
  if (order.size() > k) {
    std::partial_sort(order.begin(),
                      order.begin() + static_cast<ptrdiff_t>(k), order.end(),
                      byFinal);
    order.resize(k);
  } else {
    std::sort(order.begin(), order.end(), byFinal);
  }

  for (size_t f : order) {
    const Entry& end = finals[f];
    WalkResult result;
    result.score = end.score;
    result.vertices = reachableStates;
    result.edges = evaluatedEdges;
    const Entry* e = &end;
    size_t pos = readingLen;
    while (pos > 0) {
      NodePtr node = *e->node;
      if (e->unigramIndex != 0 && !node->isOverridden()) {
        node = std::make_shared<Node>(*node);
        node->selectUnigramAt(e->unigramIndex);
      }
      result.totalReadings += node->spanningLength();
      result.nodes.push_back(std::move(node));
      const size_t prevPos = e->prevPos;
      e = &table[prevPos][e->prevEntry];
      pos = prevPos;
    }
    std::reverse(result.nodes.begin(), result.nodes.end());
    result.elapsedMicroseconds = GetEpochNowInMicroseconds() - start;
    results.push_back(std::move(result));
  }
  return results;
}

namespace {
// Splits UTF-8 into per-code-point strings (invalid lead bytes become
// one-byte pieces).
void SplitCodePoints(const std::string& s, std::vector<std::string>* out) {
  out->clear();
  size_t i = 0;
  while (i < s.size()) {
    unsigned char c = static_cast<unsigned char>(s[i]);
    size_t n = c < 0x80                ? 1
               : (c >> 5) == 0x6       ? 2
               : (c >> 4) == 0xE       ? 3
               : (c >> 3) == 0x1E      ? 4
                                       : 1;
    if (i + n > s.size()) {
      n = 1;
    }
    out->push_back(s.substr(i, n));
    i += n;
  }
}
}  // namespace

ReadingGrid::WalkResult ReadingGrid::walkWithCharBonus(
    double alpha, const PositionalCharScores& scores) {
  const size_t readingLen = readings_.size();
  if (spans_.empty() || scores.table.size() != readingLen) {
    return walk();
  }
  int64_t start = GetEpochNowInMicroseconds();

  struct State {
    size_t fromIndex = 0;
    NodePtr fromNode = nullptr;
    size_t unigramIndex = 0;
    bool useCurrent = true;  // keep the node's current unigram (no copy)
    double maxScore = -std::numeric_limits<double>::infinity();
  };
  std::vector<State> viterbi(readingLen + 1);
  viterbi[0].maxScore = 0.0;

  size_t reachableStates = 0;
  size_t evaluatedEdges = 0;
  std::vector<std::string> chars;
  for (size_t i = 0; i < readingLen; ++i) {
    ++reachableStates;
    const Span& span = spans_[i];
    for (size_t spanLen = 1; spanLen <= span.maxLength(); ++spanLen) {
      const NodePtr& node = span.nodeOf(spanLen);
      if (node == nullptr) {
        continue;
      }
      ++evaluatedEdges;

      // Best unigram of this node under the bonus. Overridden or empty nodes
      // keep stock semantics.
      double best = node->score();
      size_t bestIndex = 0;
      bool useCurrent = true;
      const auto& unigrams = node->unigrams();
      if (!node->isOverridden() && !unigrams.empty()) {
        best = -std::numeric_limits<double>::infinity();
        for (size_t u = 0; u < unigrams.size(); ++u) {
          double bonus = 0;
          if (alpha != 0) {
            SplitCodePoints(unigrams[u].value(), &chars);
            const bool userPhrase = unigrams[u].isUserPhrase();
            for (size_t j = 0; j < chars.size(); ++j) {
              const size_t pos = i + j;
              if (pos >= readingLen) {
                bonus += scores.missingScore;
                continue;
              }
              bonus += PositionBonus(scores, pos, chars[j], userPhrase);
            }
          }
          double x = unigrams[u].score() + alpha * bonus;
          if (x > best) {
            best = x;
            bestIndex = u;
          }
        }
        useCurrent = unigrams[bestIndex].value() == node->value() &&
                     unigrams[bestIndex].score() == node->score();
      }

      double score = viterbi[i].maxScore + best;
      State& target = viterbi[i + spanLen];
      if (score > target.maxScore) {
        target.maxScore = score;
        target.fromNode = node;
        target.fromIndex = i;
        target.unigramIndex = bestIndex;
        target.useCurrent = useCurrent;
      }
    }
  }

  WalkResult result;
  result.vertices = reachableStates;
  result.edges = evaluatedEdges;
  result.score = viterbi[readingLen].maxScore;
  size_t totalReadingLen = 0;
  for (size_t curr = readingLen; curr > 0; curr = viterbi[curr].fromIndex) {
    State& st = viterbi[curr];
    assert(st.fromNode != nullptr);
    NodePtr node = st.fromNode;
    if (!st.useCurrent) {
      node = std::make_shared<Node>(*node);
      node->selectUnigramAt(st.unigramIndex);
    }
    totalReadingLen += node->spanningLength();
    result.nodes.push_back(std::move(node));
  }
  std::reverse(result.nodes.begin(), result.nodes.end());
  assert(totalReadingLen == readingLen);
  result.totalReadings = totalReadingLen;
  result.elapsedMicroseconds = GetEpochNowInMicroseconds() - start;
  return result;
}

std::vector<std::vector<std::string>> ReadingGrid::charsAtPositions() const {
  const size_t n = readings_.size();
  std::vector<std::vector<std::string>> out(n);
  std::vector<std::unordered_set<std::string>> seen(n);
  std::vector<std::string> chars;
  for (size_t i = 0; i < spans_.size() && i < n; ++i) {
    const Span& span = spans_[i];
    for (size_t len = 1; len <= span.maxLength(); ++len) {
      const NodePtr& node = span.nodeOf(len);
      if (node == nullptr) {
        continue;
      }
      for (const auto& u : node->unigrams()) {
        SplitCodePoints(u.value(), &chars);
        for (size_t j = 0; j < chars.size() && i + j < n; ++j) {
          if (seen[i + j].insert(chars[j]).second) {
            out[i + j].push_back(chars[j]);
          }
        }
      }
    }
  }
  return out;
}

std::vector<ReadingGrid::Candidate> ReadingGrid::candidatesAt(size_t loc) {
  std::vector<ReadingGrid::Candidate> result;
  if (readings_.empty()) {
    return result;
  }

  if (loc > readings_.size()) {
    return result;
  }

  std::vector<NodeInSpan> nodes =
      overlappingNodesAt(loc == readings_.size() ? loc - 1 : loc);

  // Sort nodes by reading length.
  std::stable_sort(
      nodes.begin(), nodes.end(), [](const auto& n1, const auto& n2) {
        return n1.node->spanningLength() > n2.node->spanningLength();
      });

  for (const NodeInSpan& nodeInSpan : nodes) {
    for (const LanguageModel::Unigram& unigram : nodeInSpan.node->unigrams()) {
      result.emplace_back(nodeInSpan.node->reading(), unigram.value(),
                          unigram.rawValue());
    }
  }
  return result;
}

bool ReadingGrid::overrideCandidate(
    size_t loc, const ReadingGrid::Candidate& candidate,
    ReadingGrid::Node::OverrideType overrideType) {
  return overrideCandidate(loc, &candidate.reading, candidate.value,
                           overrideType);
}

bool ReadingGrid::overrideCandidate(
    size_t loc, const std::string& candidate,
    ReadingGrid::Node::OverrideType overrideType) {
  return overrideCandidate(loc, nullptr, candidate, overrideType);
}

void ReadingGrid::expandGridAt(size_t loc) {
  if (!loc || loc == spans_.size()) {
    spans_.insert(spans_.begin() + static_cast<ptrdiff_t>(loc), Span());
    return;
  }
  spans_.insert(spans_.begin() + static_cast<ptrdiff_t>(loc), Span());
  removeAffectedNodes(loc);
}

void ReadingGrid::shrinkGridAt(size_t loc) {
  if (loc == spans_.size()) {
    return;
  }
  spans_.erase(spans_.begin() + static_cast<ptrdiff_t>(loc));
  removeAffectedNodes(loc);
}

void ReadingGrid::removeAffectedNodes(size_t loc) {
  // Because of the expansion, certain spans now have "broken" nodes. We need
  // to remove those. For example, before:
  //
  // Span index 0   1   2   3
  //                (---)
  //                (-------)
  //            (-----------)
  //
  // After we've inserted a span at 2:
  //
  // Span index 0   1   2   3   4
  //                (---)
  //                (----   ----)
  //            (--------   ----)
  //
  // Similarly for shrinkage, before:
  //
  // Span index 0   1   2   3
  //                (---)
  //                (-------)
  //            (-----------)
  //
  // After we've deleted the span at 2:
  //
  // Span index 0   1   2   3   4
  //                (---)
  //                XXXXX
  //            XXXXXXXXX
  //
  if (spans_.empty() || loc == 0) {
    return;
  }
  size_t affectedLength = kMaximumSpanLength - 1;
  size_t begin = loc <= affectedLength ? 0 : loc - affectedLength;
  size_t end = loc >= 1 ? loc - 1 : 0;
  for (size_t i = begin; i <= end; ++i) {
    spans_[i].removeNodesOfOrLongerThan(loc - i + 1);
  }
}

void ReadingGrid::insert(size_t loc, const ReadingGrid::NodePtr& node) {
  assert(loc < spans_.size());
  spans_[loc].add(node);
}

std::string ReadingGrid::combineReading(
    std::vector<std::string>::const_iterator begin,
    std::vector<std::string>::const_iterator end) {
  std::string result;
  for (auto iter = begin; iter != end;) {
    result += *iter;
    ++iter;
    if (iter != end) {
      result += separator_;
    }
  }
  return result;
}

bool ReadingGrid::hasNodeAt(size_t loc, size_t readingLen,
                            const std::string& reading) {
  if (loc >= spans_.size()) {
    return false;
  }
  const NodePtr& n = spans_[loc].nodeOf(readingLen);
  if (n == nullptr) {
    return false;
  }
  return reading == n->reading();
}

void ReadingGrid::update(size_t loc, EditType editType) {
  // Spans that do not cross the edit retain their previous lookup result. A
  // node means that the lookup succeeded, while a null slot means that the
  // same reading was already looked up and did not exist. Only spans that
  // include an insertion or cross a deletion boundary need to be queried.
  size_t affectedLength = kMaximumSpanLength - 1;
  size_t begin = loc <= affectedLength ? 0 : loc - affectedLength;
  size_t end = editType == EditType::kInsertion ? loc + 1 : loc;
  end = std::min(end, readings_.size());

  for (size_t pos = begin; pos < end; pos++) {
    size_t minimumLength = loc - pos + 1;
    size_t maximumLength = std::min(kMaximumSpanLength, readings_.size() - pos);
    for (size_t len = minimumLength; len <= maximumLength; len++) {
      std::string combinedReading =
          combineReading(readings_.begin() + static_cast<ptrdiff_t>(pos),
                         readings_.begin() + static_cast<ptrdiff_t>(pos + len));

      if (!hasNodeAt(pos, len, combinedReading)) {
        auto unigrams = lm_.getUnigrams(combinedReading);
        if (unigrams.empty()) {
          continue;
        }

        insert(pos, std::make_shared<Node>(std::move(combinedReading), len,
                                           std::move(unigrams)));
      }
    }
  }
}

bool ReadingGrid::overrideCandidate(
    size_t loc, const std::string* reading, const std::string& value,
    ReadingGrid::Node::OverrideType overrideType) {
  std::optional<NodeInSpan> target = findOverrideTarget(loc, reading, value);
  if (!target.has_value()) {
    // Nothing gets overridden.
    return false;
  }

  NodeInSpan overridden = std::move(*target);
  bool selected = overridden.node->selectOverrideUnigram(value, overrideType);
  assert(selected);
  (void)selected;

  for (size_t i = overridden.spanIndex;
       i < overridden.spanIndex + overridden.node->spanningLength() &&
       i < spans_.size();
       ++i) {
    // We also need to reset *all* nodes that share the same location in the
    // span. For example, if previously the two walked nodes are "A BC" where
    // A and BC are two nodes with overrides. The user now chooses "DEF" which
    // is a node that shares the same span location with "A". The node with BC
    // will be reset as it's part of the overlapping node, but A is not.
    std::vector<NodeInSpan> nodes = overlappingNodesAt(i);
    for (NodeInSpan& nis : nodes) {
      if (nis.node != overridden.node) {
        nis.node->reset();
      }
    }
  }
  return true;
}

std::optional<ReadingGrid::NodeInSpan> ReadingGrid::findOverrideTarget(
    size_t loc, const std::string& value) const {
  return findOverrideTarget(loc, nullptr, value);
}

std::optional<ReadingGrid::NodeInSpan> ReadingGrid::findOverrideTarget(
    size_t loc, const std::string* reading, const std::string& value) const {
  if (loc > readings_.size()) {
    return std::nullopt;
  }

  std::vector<NodeInSpan> overlappingNodes =
      overlappingNodesAt(loc == readings_.size() ? loc - 1 : loc);
  for (NodeInSpan& nis : overlappingNodes) {
    if (reading != nullptr && nis.node->reading() != *reading) {
      continue;
    }

    const auto& unigrams = nis.node->unigrams();
    if (std::any_of(unigrams.cbegin(), unigrams.cend(),
                    [&value](const LanguageModel::Unigram& u) {
                      return u.value() == value;
                    })) {
      return nis;
    }
  }
  return std::nullopt;
}

std::vector<ReadingGrid::NodeInSpan> ReadingGrid::overlappingNodesAt(
    size_t loc) const {
  std::vector<ReadingGrid::NodeInSpan> results;

  if (spans_.empty() || loc >= spans_.size()) {
    return results;
  }

  // First, get all nodes from the span at location.
  for (size_t i = 1, len = spans_[loc].maxLength(); i <= len; ++i) {
    NodePtr ptr = spans_[loc].nodeOf(i);
    if (ptr != nullptr) {
      ReadingGrid::NodeInSpan element{std::move(ptr), loc};
      results.emplace_back(std::move(element));
    }
  }

  size_t begin = loc - std::min(loc, kMaximumSpanLength - 1);
  for (size_t i = begin; i < loc; ++i) {
    size_t beginLen = loc - i + 1;
    size_t endLen = spans_[i].maxLength();
    for (size_t j = beginLen; j <= endLen; ++j) {
      NodePtr ptr = spans_[i].nodeOf(j);
      if (ptr != nullptr) {
        ReadingGrid::NodeInSpan element{std::move(ptr), i};
        results.emplace_back(std::move(element));
      }
    }
  }

  return results;
}

LanguageModel::Unigram ReadingGrid::Node::currentUnigram() const {
  return unigrams_.empty() ? LanguageModel::Unigram{} : *unigramIter_;
}

std::string ReadingGrid::Node::value() const {
  return unigrams_.empty() ? "" : unigramIter_->value();
}

double ReadingGrid::Node::score() const {
  if (unigrams_.empty()) {
    return 0;
  }

  switch (overrideType_) {
    case OverrideType::kOverrideValueWithHighScore:
      return kOverridingScore;
    case OverrideType::kOverrideValueWithScoreFromTopUnigram:
      return unigrams_[0].score();
    case OverrideType::kNone:
    default:
      return unigramIter_->score();
  }
}

bool ReadingGrid::Node::isOverridden() const {
  return overrideType_ != OverrideType::kNone;
}

void ReadingGrid::Node::reset() {
  unigramIter_ = unigrams_.begin();
  overrideType_ = OverrideType::kNone;
}

bool ReadingGrid::Node::selectOverrideUnigram(
    const std::string& value, ReadingGrid::Node::OverrideType type) {
  assert(type != ReadingGrid::Node::OverrideType::kNone);
  for (auto it = unigrams_.begin(), end = unigrams_.end(); it != end; ++it) {
    if (value == it->value()) {
      unigramIter_ = it;
      overrideType_ = type;
      return true;
    }
  }
  return false;
}

bool ReadingGrid::Node::selectUnigramAt(size_t index) {
  if (index >= unigrams_.size()) {
    return false;
  }
  unigramIter_ = std::next(unigrams_.begin(), static_cast<ptrdiff_t>(index));
  return true;
}

std::vector<ReadingGrid::NodePtr>::const_iterator
ReadingGrid::WalkResult::findNodeAt(size_t cursor,
                                    size_t* outCursorPastNode) const {
  if (nodes.empty()) {
    return nodes.cend();
  }

  if (cursor > totalReadings) {
    return nodes.cend();
  }

  if (cursor == 0) {
    auto it = nodes.cbegin();
    if (outCursorPastNode != nullptr) {
      *outCursorPastNode = (*it)->spanningLength();
    }
    return it;
  }

  // Covers both the "cursor is right at end" and "cursor is one reading before
  // the end" cases.
  if (cursor >= totalReadings - 1) {
    if (outCursorPastNode != nullptr) {
      *outCursorPastNode = totalReadings;
    }
    return std::next(nodes.cbegin(), static_cast<ptrdiff_t>(nodes.size() - 1));
  }

  size_t accumulated = 0;
  for (auto i = nodes.cbegin(); i != nodes.cend(); ++i) {
    accumulated += (*i)->spanningLength();
    if (accumulated > cursor) {
      if (outCursorPastNode != nullptr) {
        *outCursorPastNode = accumulated;
      }
      return i;
    }
  }

  // Shouldn't happen.
  return nodes.cend();
}

std::vector<std::string> ReadingGrid::WalkResult::valuesAsStrings() const {
  std::vector<std::string> result;
  for (const NodePtr& node : nodes) {
    result.emplace_back(node->value());
  }
  return result;
}

std::vector<std::string> ReadingGrid::WalkResult::readingsAsStrings() const {
  std::vector<std::string> result;
  for (const NodePtr& node : nodes) {
    result.emplace_back(node->reading());
  }
  return result;
}

void ReadingGrid::Span::clear() {
  nodes_.fill(nullptr);
  maxLength_ = 0;
}

void ReadingGrid::Span::add(const ReadingGrid::NodePtr& node) {
  assert(node->spanningLength() > 0 &&
         node->spanningLength() <= kMaximumSpanLength);
  nodes_[node->spanningLength() - 1] = node;
  maxLength_ = std::max(maxLength_, node->spanningLength());
}

void ReadingGrid::Span::removeNodesOfOrLongerThan(size_t length) {
  assert(length > 0 && length <= kMaximumSpanLength);
  for (size_t i = length - 1; i < kMaximumSpanLength; ++i) {
    nodes_[i] = nullptr;
  }
  maxLength_ = 0;
  if (length == 1) {
    return;
  }

  size_t i = length - 2;
  while (true) {
    if (nodes_[i] != nullptr) {
      maxLength_ = i + 1;
      return;
    }

    if (i == 0) {
      return;
    }

    --i;
  }
}

const ReadingGrid::NodePtr& ReadingGrid::Span::nodeOf(size_t length) const {
  assert(length > 0 && length <= kMaximumSpanLength);
  return nodes_[length - 1];
}

std::vector<LanguageModel::Unigram>
ReadingGrid::ScoreRankedLanguageModel::getUnigrams(const std::string& reading) {
  auto unigrams = lm_->getUnigrams(reading);
  std::stable_sort(
      unigrams.begin(), unigrams.end(),
      [](const auto& u1, const auto& u2) { return u1.score() > u2.score(); });
  return unigrams;
}

bool ReadingGrid::ScoreRankedLanguageModel::hasUnigrams(
    const std::string& reading) {
  return lm_->hasUnigrams(reading);
}

}  // namespace Formosa::Gramambular2
