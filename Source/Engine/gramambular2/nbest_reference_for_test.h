// Test-only reference for ReadingGrid::walkNBestImpl(): the original
// (pre-optimization) N-best algorithm, kept verbatim except that it goes
// through ReadingGrid's public API. Tests compare the production
// walkNBest()/walkNBestWithCharBonus() against it for exact equality (texts,
// scores, order including seq tie-breaking, node/unigram selections).
// Do not include from production code.

#ifndef SRC_ENGINE_GRAMAMBULAR2_NBEST_REFERENCE_FOR_TEST_H_
#define SRC_ENGINE_GRAMAMBULAR2_NBEST_REFERENCE_FOR_TEST_H_

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "reading_grid.h"

namespace Formosa::Gramambular2::NBestReference {

inline void SplitCodePointsRef(const std::string& s,
                               std::vector<std::string>* out) {
  out->clear();
  size_t i = 0;
  while (i < s.size()) {
    unsigned char c = static_cast<unsigned char>(s[i]);
    size_t n = c < 0x80             ? 1
               : (c >> 5) == 0x6    ? 2
               : (c >> 4) == 0xE    ? 3
               : (c >> 3) == 0x1E   ? 4
                                    : 1;
    if (i + n > s.size()) {
      n = 1;
    }
    out->push_back(s.substr(i, n));
    i += n;
  }
}

// scores == nullptr: walkNBest(k). Otherwise walkNBestWithCharBonus(k, alpha,
// *scores) (the caller handles the table-size fallback, as the grid does).
inline std::vector<ReadingGrid::WalkResult> WalkNBestReference(
    const ReadingGrid& grid, size_t k, double alpha,
    const ReadingGrid::PositionalCharScores* scores) {
  using NodePtr = ReadingGrid::NodePtr;
  using Node = ReadingGrid::Node;
  using WalkResult = ReadingGrid::WalkResult;
  using Span = ReadingGrid::Span;
  if (scores != nullptr && scores->table.size() != grid.length()) {
    scores = nullptr;
    alpha = 0.0;
  }
  std::vector<WalkResult> results;
  const auto& spans = grid.spans();
  if (spans.empty() || k == 0) {
    return results;
  }

  struct Entry {
    std::string text;
    double score = 0;
    uint64_t seq = 0;
    size_t prevPos = 0;
    size_t prevEntry = 0;
    NodePtr node;
    size_t unigramIndex = 0;
  };

  const size_t readingLen = grid.readings().size();
  std::vector<std::vector<Entry>> table(readingLen + 1);
  table[0].push_back(Entry{});
  uint64_t seq = 0;
  size_t reachableStates = 0;
  size_t evaluatedEdges = 0;
  std::vector<double> optionAdds;
  std::vector<std::string> chars;

  for (size_t i = 0; i < readingLen; ++i) {
    if (table[i].empty()) {
      continue;
    }
    ++reachableStates;

    const std::vector<Entry>& entries = table[i];
    std::vector<size_t> order(entries.size());
    for (size_t j = 0; j < order.size(); ++j) {
      order[j] = j;
    }
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
      if (entries[a].score != entries[b].score) {
        return entries[a].score > entries[b].score;
      }
      return entries[a].seq < entries[b].seq;
    });
    if (order.size() > k) {
      order.resize(k);
    }

    const Span& span = spans[i];
    for (size_t spanLen = 1; spanLen <= span.maxLength(); ++spanLen) {
      const NodePtr& node = span.nodeOf(spanLen);
      if (node == nullptr) {
        continue;
      }
      ++evaluatedEdges;

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

      optionAdds.clear();
      for (size_t u = optionBegin; u < optionEnd; ++u) {
        double add = (node->isOverridden() || unigrams.empty())
                         ? node->score()
                         : unigrams[u].score();
        if (scores != nullptr && alpha != 0 && !node->isOverridden() &&
            !unigrams.empty()) {
          double bonus = 0;
          SplitCodePointsRef(unigrams[u].value(), &chars);
          for (size_t j = 0; j < chars.size(); ++j) {
            const size_t pos = i + j;
            if (pos >= readingLen) {
              bonus += scores->missingScore;
              continue;
            }
            const auto& m = scores->table[pos];
            if (unigrams[u].isUserPhrase()) {
              // User phrase: best score of the row (missingScore if empty).
              double best = scores->missingScore;
              bool any = false;
              for (const auto& kv : m) {
                if (!any || kv.second > best) {
                  best = kv.second;
                  any = true;
                }
              }
              bonus += best;
              continue;
            }
            auto it = m.find(chars[j]);
            bonus += it == m.end() ? scores->missingScore : it->second;
          }
          add += alpha * bonus;
        }
        optionAdds.push_back(add);
      }

      std::vector<Entry>& target = table[i + spanLen];
      for (size_t idx : order) {
        for (size_t u = optionBegin; u < optionEnd; ++u) {
          std::string value = unigrams.empty() ? "" : unigrams[u].value();
          double add = optionAdds[u - optionBegin];
          std::string text = entries[idx].text + value;
          double score = entries[idx].score + add;
          auto it =
              std::find_if(target.begin(), target.end(),
                           [&](const Entry& e) { return e.text == text; });
          if (it == target.end()) {
            target.push_back(
                Entry{std::move(text), score, seq++, i, idx, node, u});
          } else if (score > it->score) {
            *it = Entry{std::move(text), score, seq++, i, idx, node, u};
          }
        }
      }
    }
  }

  std::vector<Entry>& finals = table[readingLen];
  std::stable_sort(finals.begin(), finals.end(),
                   [](const Entry& a, const Entry& b) {
                     if (a.score != b.score) {
                       return a.score > b.score;
                     }
                     return a.seq < b.seq;
                   });
  if (finals.size() > k) {
    finals.resize(k);
  }

  for (const Entry& end : finals) {
    WalkResult result;
    result.score = end.score;
    result.vertices = reachableStates;
    result.edges = evaluatedEdges;
    const Entry* e = &end;
    size_t pos = readingLen;
    while (pos > 0) {
      NodePtr node = e->node;
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
    results.push_back(std::move(result));
  }
  return results;
}

// Exact comparison of two N-best lists. Returns "" when equal, otherwise a
// description of the first difference. Shared nodes must be the very same
// grid node (pointer equality); copied nodes must agree on reading, span,
// value, score, overridden flag and selected unigram.
inline bool IsGridNode(const ReadingGrid& grid,
                       const ReadingGrid::NodePtr& node) {
  for (const auto& span : grid.spans()) {
    for (size_t len = 1; len <= span.maxLength(); ++len) {
      if (span.nodeOf(len) == node) {
        return true;
      }
    }
  }
  return false;
}

inline std::string DiffNBest(const std::vector<ReadingGrid::WalkResult>& a,
                             const std::vector<ReadingGrid::WalkResult>& b,
                             const ReadingGrid* grid = nullptr) {
  if (a.size() != b.size()) {
    return "size " + std::to_string(a.size()) + " vs " +
           std::to_string(b.size());
  }
  for (size_t r = 0; r < a.size(); ++r) {
    const auto& x = a[r];
    const auto& y = b[r];
    const std::string at = "result " + std::to_string(r) + ": ";
    if (x.score != y.score) {
      return at + "score differs";
    }
    if (x.vertices != y.vertices || x.edges != y.edges ||
        x.totalReadings != y.totalReadings) {
      return at + "vertices/edges/totalReadings differ";
    }
    if (x.nodes.size() != y.nodes.size()) {
      return at + "node count differs";
    }
    for (size_t n = 0; n < x.nodes.size(); ++n) {
      const auto& p = x.nodes[n];
      const auto& q = y.nodes[n];
      if (p != q) {
        if (grid != nullptr &&
            IsGridNode(*grid, p) != IsGridNode(*grid, q)) {
          return at + "node " + std::to_string(n) + " shared vs copied";
        }
        if (p->reading() != q->reading() ||
            p->spanningLength() != q->spanningLength() ||
            p->value() != q->value() || p->score() != q->score() ||
            p->isOverridden() != q->isOverridden() ||
            p->unigrams().size() != q->unigrams().size()) {
          return at + "node " + std::to_string(n) + " differs (" +
                 p->value() + " vs " + q->value() + ")";
        }
      }
    }
  }
  return "";
}

}  // namespace Formosa::Gramambular2::NBestReference

#endif  // SRC_ENGINE_GRAMAMBULAR2_NBEST_REFERENCE_FOR_TEST_H_
