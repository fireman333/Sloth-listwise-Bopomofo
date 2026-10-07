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

#ifndef SRC_ENGINE_USEROVERRIDEMODEL_H_
#define SRC_ENGINE_USEROVERRIDEMODEL_H_

#include <iosfwd>
#include <list>
#include <map>
#include <string>
#include <utility>

#include "gramambular2/reading_grid.h"

namespace McBopomofo {

class UserOverrideModel {
 public:
  // Behavior switches on top of the stock model. The stock constructor keeps
  // both off.
  struct Policy {
    // When a candidate is observed for a key, drop all other candidates of
    // that key, so the latest manual pick is always the one suggested.
    bool latestPickWins;
    // Enables refresh(); with this off, refresh() is a no-op.
    bool observeRefresh;
  };

  UserOverrideModel(size_t capacity, double decayConstant);
  UserOverrideModel(size_t capacity, double decayConstant, Policy policy);

  struct Suggestion {
    Suggestion() = default;
    Suggestion(std::string c, bool f)
        : candidate(std::move(c)), forceHighScoreOverride(f) {}
    std::string candidate;
    bool forceHighScoreOverride = false;

    [[nodiscard]] bool empty() const { return candidate.empty(); }
  };

  struct SuggestionWithKey {
    Suggestion suggestion;
    // The observation key used for the lookup; set even if the suggestion is
    // empty.
    std::string key;
  };

  void observe(const Formosa::Gramambular2::ReadingGrid::WalkResult&
                   walkBeforeUserOverride,
               const Formosa::Gramambular2::ReadingGrid::WalkResult&
                   walkAfterUserOverride,
               size_t cursor, double timestamp);

  Suggestion suggest(
      const Formosa::Gramambular2::ReadingGrid::WalkResult& currentWalk,
      size_t cursor, double timestamp);

  void observe(const std::string& key, const std::string& candidate,
               double timestamp, bool forceHighScoreOverride = false);

  Suggestion suggest(const std::string& key, double timestamp);

  // Same as suggest(currentWalk, cursor, timestamp), but also returns the key
  // so that the caller can later refresh() it.
  SuggestionWithKey suggestWithKey(
      const Formosa::Gramambular2::ReadingGrid::WalkResult& currentWalk,
      size_t cursor, double timestamp);

  // If the policy enables refresh, the key exists, and its best candidate at
  // the timestamp is the given candidate, sets that candidate's timestamp to
  // the given one and moves the key to the front of the LRU list. Counts are
  // not changed. Returns whether anything changed.
  bool refresh(const std::string& key, const std::string& candidate,
               double timestamp);

  // Writes the model as text: a header line "AFM-UOM 1", then one line per
  // (key, candidate) in LRU order (most recently used key first):
  // key \t candidate \t count \t timestamp \t force(0|1). Entries whose key
  // or candidate contain a tab or a newline are skipped.
  void serialize(std::ostream& os) const;

  // Replaces the content of the model with what serialize() wrote. All or
  // nothing: on any malformed input, the model is left empty and false is
  // returned. If there are more keys than the capacity, the most recently used
  // ones are kept. Capacity, half-life and policy are not changed.
  bool deserialize(std::istream& is);

  [[nodiscard]] size_t capacity() const { return capacity_; }
  [[nodiscard]] Policy policy() const { return policy_; }
  // Number of keys (contexts) in the model.
  [[nodiscard]] size_t size() const { return lruList_.size(); }
  // Removes all keys; capacity, half-life and policy are kept.
  void clear() {
    lruList_.clear();
    lruMap_.clear();
  }

 private:
  struct Override {
    size_t count = 0;
    double timestamp = 0;
    bool forceHighScoreOverride = false;
  };

  struct Observation {
    size_t count;
    std::map<std::string, Override> overrides;

    Observation() : count(0) {}
    void update(const std::string& candidate, double timestamp,
                bool forceHighScoreOverride, bool latestPickWins);
    // The candidate with the highest non-zero score at the timestamp, or
    // overrides.end().
    [[nodiscard]] std::map<std::string, Override>::const_iterator best(
        double timestamp, double decayExponent) const;
  };

  typedef std::pair<std::string, Observation> KeyObservationPair;

  size_t capacity_;
  double decayExponent_;
  Policy policy_;
  std::list<KeyObservationPair> lruList_;
  std::map<std::string, std::list<KeyObservationPair>::iterator> lruMap_;
};

}  // namespace McBopomofo

#endif  // SRC_ENGINE_USEROVERRIDEMODEL_H_
