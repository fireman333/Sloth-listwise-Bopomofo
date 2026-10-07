// Copyright (c) 2017 ond onwards The McBopomofo Authors.
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
//

#include "UserOverrideModel.h"

#include <cassert>
#include <charconv>
#include <cmath>
#include <istream>
#include <list>
#include <ostream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "gramambular2/reading_grid.h"

namespace McBopomofo {

// Fully decay after about 20 generations.
static constexpr double kDecayThreshold = 1.0 / 1048576.0;

static constexpr char kEmptyNodeString[] = "()";

static constexpr char kSerializationHeader[] = "AFM-UOM 1";

// A scoring function that balances between "recent but infrequently observed"
// and "old but frequently observed".
static double Score(size_t eventCount, size_t totalCount, double eventTimestamp,
                    double timestamp, double lambda);

// Form the observation key from the nodes of a walk. This goes backward, but
// we are using a const_iterator, the "end" here should be a .cbegin() of a
// vector.
static std::string FormObservationKey(
    std::vector<Formosa::Gramambular2::ReadingGrid::NodePtr>::const_iterator
        head,
    std::vector<Formosa::Gramambular2::ReadingGrid::NodePtr>::const_iterator
        end);

UserOverrideModel::UserOverrideModel(size_t capacity, double decayConstant)
    : UserOverrideModel(capacity, decayConstant,
                        Policy{/*latestPickWins=*/false,
                               /*observeRefresh=*/false}) {}

UserOverrideModel::UserOverrideModel(size_t capacity, double decayConstant,
                                     Policy policy)
    : capacity_(capacity), policy_(policy) {
  assert(capacity_ > 0);
  // NOLINTNEXTLINE(readability-magic-numbers)
  decayExponent_ = log(0.5) / decayConstant;
}

void UserOverrideModel::observe(
    const Formosa::Gramambular2::ReadingGrid::WalkResult&
        walkBeforeUserOverride,
    const Formosa::Gramambular2::ReadingGrid::WalkResult& walkAfterUserOverride,
    size_t cursor, double timestamp) {
  // Sanity check.
  if (walkBeforeUserOverride.nodes.empty() ||
      walkAfterUserOverride.nodes.empty()) {
    return;
  }

  if (walkBeforeUserOverride.totalReadings !=
      walkAfterUserOverride.totalReadings) {
    return;
  }

  // We first infer what the user override is.
  size_t actualCursor = 0;
  auto currentNodeIt = walkAfterUserOverride.findNodeAt(cursor, &actualCursor);
  if (currentNodeIt == walkAfterUserOverride.nodes.cend()) {
    return;
  }

  // Based on previous analysis, we found it meaningless to handle phrases
  // over 3 characters.
  if ((*currentNodeIt)->spanningLength() > 3) {
    return;
  }

  // Now we need to find the head node in the previous walk (that is, before
  // the user override). Remember that actualCursor now is actually *past*
  // the current node, so we need to decrement by 1.
  if (actualCursor == 0) {
    // Shouldn't happen.
    return;
  }
  --actualCursor;
  auto prevHeadNodeIt = walkBeforeUserOverride.findNodeAt(actualCursor);
  if (prevHeadNodeIt == walkBeforeUserOverride.nodes.cend()) {
    return;
  }

  // Now we have everything. We want to handle the following cases:
  // (1) both prev and current head nodes represent an n-char phrase.
  // (2) current head node is a 2-/3-char phrase but prev head node and
  //     the nodes that lead to the prev head node are 1-char phrases.
  // (3) current head node is a 1-char phrase but the prev head node is
  //     a phrase of multi-char phrases.
  //
  // (1) is the simplest case. Our observation is based on the "walk before
  // user override", and we don't need to recommend force-high-score
  // overrides when we return such a suggestion. Example: overriding
  // "他姓[中]" with "他姓[鍾]".
  //
  // (2) is a case that UOM historically didn't handle properly. The
  // observation needs to be based on the walk before user override, but
  // we also need to recommend force-high-score override when we make the
  // suggestion, due to the fact (based on our data analysis) that many
  // n-char (esp. 2-char) phrases need to compete with individual chars
  // that together have higher score than the phrases themselves. Example:
  // overriding "增加[自][會]" with "增加[字彙]". Here [自][會] together have
  // higher score than the single unigram [字彙], and hence the boosting
  // here.
  //
  // (3) is a very special case where we need to base our observation on
  // the walk *after* user override. This is because when (3) happens, the
  // user intent is to break up a long phrase. We don't want to recommend
  // force-high-score overrides, which would cause the multi-char phrase
  // to lose over the user override all the time. For example (a somewhat
  // forced one): overriding "[三百元]" with "[參]百元".
  const auto& currentNode = *currentNodeIt;
  const auto& prevHeadNode = *prevHeadNodeIt;
  bool forceHighScoreOverride =
      currentNode->spanningLength() > prevHeadNode->spanningLength();
  bool breakingUp =
      currentNode->spanningLength() == 1 && prevHeadNode->spanningLength() > 1;

  auto nodeIter = breakingUp ? currentNodeIt : prevHeadNodeIt;
  auto endPoint = breakingUp ? walkAfterUserOverride.nodes.begin()
                             : walkBeforeUserOverride.nodes.begin();

  std::string key = FormObservationKey(nodeIter, endPoint);
  observe(key, currentNode->currentUnigram().value(), timestamp,
          forceHighScoreOverride);
}

UserOverrideModel::Suggestion UserOverrideModel::suggest(
    const Formosa::Gramambular2::ReadingGrid::WalkResult& currentWalk,
    size_t cursor, double timestamp) {
  return suggestWithKey(currentWalk, cursor, timestamp).suggestion;
}

UserOverrideModel::SuggestionWithKey UserOverrideModel::suggestWithKey(
    const Formosa::Gramambular2::ReadingGrid::WalkResult& currentWalk,
    size_t cursor, double timestamp) {
  auto nodeIter = currentWalk.findNodeAt(cursor);
  std::string key = FormObservationKey(nodeIter, currentWalk.nodes.begin());
  Suggestion suggestion = suggest(key, timestamp);
  return SuggestionWithKey{std::move(suggestion), std::move(key)};
}

void UserOverrideModel::observe(const std::string& key,
                                const std::string& candidate, double timestamp,
                                bool forceHighScoreOverride) {
  auto mapIter = lruMap_.find(key);
  if (mapIter == lruMap_.end()) {
    auto keyValuePair = KeyObservationPair(key, Observation());
    Observation& observation = keyValuePair.second;
    observation.update(candidate, timestamp, forceHighScoreOverride,
                       policy_.latestPickWins);

    lruList_.push_front(keyValuePair);
    auto listIter = lruList_.begin();
    auto lruKeyValue =
        std::pair<std::string, std::list<KeyObservationPair>::iterator>(
            key, listIter);
    lruMap_.insert(lruKeyValue);

    if (lruList_.size() > capacity_) {
      auto lastKeyValuePair = lruList_.end();
      --lastKeyValuePair;
      lruMap_.erase(lastKeyValuePair->first);
      lruList_.pop_back();
    }
  } else {
    auto listIter = mapIter->second;
    lruList_.splice(lruList_.begin(), lruList_, listIter);

    auto& keyValuePair = *listIter;
    Observation& observation = keyValuePair.second;
    observation.update(candidate, timestamp, forceHighScoreOverride,
                       policy_.latestPickWins);
  }
}

UserOverrideModel::Suggestion UserOverrideModel::suggest(const std::string& key,
                                                         double timestamp) {
  auto mapIter = lruMap_.find(key);
  if (mapIter == lruMap_.end()) {
    return UserOverrideModel::Suggestion{};
  }

  auto listIter = mapIter->second;
  auto& keyValuePair = *listIter;
  const Observation& observation = keyValuePair.second;

  auto best = observation.best(timestamp, decayExponent_);
  if (best == observation.overrides.cend()) {
    return UserOverrideModel::Suggestion{};
  }
  return UserOverrideModel::Suggestion{best->first,
                                       best->second.forceHighScoreOverride};
}

bool UserOverrideModel::refresh(const std::string& key,
                                const std::string& candidate,
                                double timestamp) {
  if (!policy_.observeRefresh) {
    return false;
  }

  auto mapIter = lruMap_.find(key);
  if (mapIter == lruMap_.end()) {
    return false;
  }

  auto listIter = mapIter->second;
  Observation& observation = listIter->second;
  auto best = observation.best(timestamp, decayExponent_);
  if (best == observation.overrides.cend() || best->first != candidate) {
    return false;
  }

  observation.overrides[candidate].timestamp = timestamp;
  lruList_.splice(lruList_.begin(), lruList_, listIter);
  return true;
}

namespace {

bool ContainsSeparator(const std::string& s) {
  return s.find_first_of("\t\n") != std::string::npos;
}

void SplitByTab(std::string_view line, std::vector<std::string_view>* out) {
  out->clear();
  size_t start = 0;
  while (true) {
    size_t pos = line.find('\t', start);
    if (pos == std::string_view::npos) {
      out->push_back(line.substr(start));
      return;
    }
    out->push_back(line.substr(start, pos - start));
    start = pos + 1;
  }
}

template <typename T>
bool ParseNumber(std::string_view field, T* out) {
  if (field.empty()) {
    return false;
  }
  const char* begin = field.data();
  const char* end = field.data() + field.size();
  auto [ptr, ec] = std::from_chars(begin, end, *out);
  return ec == std::errc() && ptr == end;
}

}  // namespace

void UserOverrideModel::serialize(std::ostream& os) const {
  os << kSerializationHeader << '\n';
  // Shortest round-trip representation, locale-independent.
  // NOLINTNEXTLINE(readability-magic-numbers)
  char buf[64];
  for (const auto& [key, observation] : lruList_) {
    if (ContainsSeparator(key)) {
      continue;
    }
    for (const auto& [candidate, o] : observation.overrides) {
      if (ContainsSeparator(candidate)) {
        continue;
      }
      auto [ptr, ec] = std::to_chars(buf, buf + sizeof(buf), o.timestamp);
      if (ec != std::errc()) {
        continue;
      }
      os << key << '\t' << candidate << '\t' << o.count << '\t'
         << std::string_view(buf, ptr - buf) << '\t'
         << (o.forceHighScoreOverride ? '1' : '0') << '\n';
    }
  }
}

bool UserOverrideModel::deserialize(std::istream& is) {
  lruList_.clear();
  lruMap_.clear();

  std::string line;
  if (!std::getline(is, line) || line != kSerializationHeader) {
    return false;
  }

  std::list<KeyObservationPair> list;
  std::map<std::string, std::list<KeyObservationPair>::iterator> map;
  std::vector<std::string_view> fields;
  // NOLINTNEXTLINE(readability-magic-numbers)
  constexpr size_t kFieldCount = 5;

  while (std::getline(is, line)) {
    SplitByTab(line, &fields);
    if (fields.size() != kFieldCount) {
      return false;
    }
    const std::string_view key = fields[0];
    const std::string_view candidate = fields[1];
    size_t count = 0;
    double timestamp = 0;
    if (key.empty() || candidate.empty() || !ParseNumber(fields[2], &count) ||
        count == 0 || !ParseNumber(fields[3], &timestamp) ||
        !std::isfinite(timestamp) || (fields[4] != "0" && fields[4] != "1")) {
      return false;
    }

    // serialize() writes the lines of a key consecutively.
    if (list.empty() || list.back().first != key) {
      std::string keyString(key);
      if (map.find(keyString) != map.end()) {
        return false;
      }
      list.emplace_back(keyString, Observation());
      map.emplace(std::move(keyString), std::prev(list.end()));
    }
    Observation& observation = list.back().second;
    auto [it, inserted] =
        observation.overrides.emplace(std::string(candidate), Override());
    if (!inserted) {
      return false;
    }
    it->second.count = count;
    it->second.timestamp = timestamp;
    it->second.forceHighScoreOverride = fields[4] == "1";
    observation.count += count;
  }

  if (is.bad()) {
    return false;
  }

  // Keep the most recently used keys.
  while (list.size() > capacity_) {
    map.erase(list.back().first);
    list.pop_back();
  }

  lruList_ = std::move(list);
  lruMap_ = std::move(map);
  return true;
}

void UserOverrideModel::Observation::update(const std::string& candidate,
                                            double timestamp,
                                            bool forceHighScoreOverride,
                                            bool latestPickWins) {
  if (latestPickWins) {
    for (auto it = overrides.begin(); it != overrides.end();) {
      if (it->first == candidate) {
        ++it;
      } else {
        it = overrides.erase(it);
      }
    }
  }

  count++;
  auto& o = overrides[candidate];
  o.timestamp = timestamp;
  o.count++;
  o.forceHighScoreOverride = forceHighScoreOverride;

  if (latestPickWins) {
    count = o.count;
  }
}

std::map<std::string, UserOverrideModel::Override>::const_iterator
UserOverrideModel::Observation::best(double timestamp,
                                     double decayExponent) const {
  auto result = overrides.cend();
  double score = 0;
  for (auto i = overrides.cbegin(); i != overrides.cend(); ++i) {
    const Override& o = i->second;
    double overrideScore =
        Score(o.count, count, o.timestamp, timestamp, decayExponent);
    if (overrideScore == 0.0) {
      continue;
    }

    if (overrideScore > score) {
      result = i;
      score = overrideScore;
    }
  }
  return result;
}

static double Score(size_t eventCount, size_t totalCount, double eventTimestamp,
                    double timestamp, double lambda) {
  double decay = exp((timestamp - eventTimestamp) * lambda);
  if (decay < kDecayThreshold) {
    return 0.0;
  }

  double prob =
      static_cast<double>(eventCount) / static_cast<double>(totalCount);
  return prob * decay;
}

static std::string CombineReadingValue(const std::string& reading,
                                       const std::string& value) {
  return std::string("(") + reading + "," + value + ")";
}

static bool IsPunctuation(
    const Formosa::Gramambular2::ReadingGrid::NodePtr& node) {
  const std::string& reading = node->reading();
  return !reading.empty() && reading[0] == '_';
}

static std::string FormObservationKey(
    std::vector<Formosa::Gramambular2::ReadingGrid::NodePtr>::const_iterator
        head,
    std::vector<Formosa::Gramambular2::ReadingGrid::NodePtr>::const_iterator
        end) {
  // Using the top unigram from the head node. Recall that this is an
  // observation for *before* the user override, and when we provide
  // a suggestion, this head node is never overridden yet.
  std::string headStr =
      CombineReadingValue((*head)->reading(), (*head)->unigrams()[0].value());

  // For the next two nodes, use their current unigram values. If it's a
  // punctuation, we ignore the reading and the value altogether and treat
  // it as if it's like the beginning of the sentence.
  std::string prevStr;
  bool prevIsPunctuation = false;
  if (head != end) {
    --head;
    prevIsPunctuation = IsPunctuation(*head);
    if (prevIsPunctuation) {
      prevStr = kEmptyNodeString;
    } else {
      prevStr = CombineReadingValue((*head)->reading(),
                                    (*head)->currentUnigram().value());
    }
  } else {
    prevStr = kEmptyNodeString;
  }

  std::string anteriorStr;
  if (head != end && !prevIsPunctuation) {
    --head;
    if (IsPunctuation((*head))) {
      anteriorStr = kEmptyNodeString;
    } else {
      anteriorStr = CombineReadingValue((*head)->reading(),
                                        (*head)->currentUnigram().value());
    }
  } else {
    anteriorStr = kEmptyNodeString;
  }

  return anteriorStr + "-" + prevStr + "-" + headStr;
}

}  // namespace McBopomofo
