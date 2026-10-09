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

#include "McBopomofoLM.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "UTF8Helper.h"
#include "gramambular2/reading_grid.h"

namespace McBopomofo {

static constexpr std::string_view kMacroPrefix = "MACRO@";
static constexpr double kMacroScore = -8.0;

void McBopomofoLM::loadLanguageModel(const char* languageModelDataPath) {
  if (languageModelDataPath) {
    languageModel_.close();
    languageModel_.open(languageModelDataPath);
  }
}

bool McBopomofoLM::isDataModelLoaded() const {
  return languageModel_.isLoaded();
}

void McBopomofoLM::loadAssociatedPhrasesV2(const char* associatedPhrasesPath) {
  if (associatedPhrasesPath) {
    associatedPhrasesV2_.close();
    associatedPhrasesV2_.open(associatedPhrasesPath);
  }
}

void McBopomofoLM::loadUserPhrases(const char* userPhrasesDataPath,
                                   const char* excludedPhrasesDataPath) {
  userPhrases_.close();
  excludedPhrases_.close();

  if (userPhrasesDataPath) {
    userPhrasesDataPath_ = userPhrasesDataPath;
    userPhrases_.open(userPhrasesDataPath);
  } else {
    userPhrasesDataPath_.reset();
  }

  if (excludedPhrasesDataPath) {
    excludedPhrasesDataPath_ = excludedPhrasesDataPath;
    excludedPhrases_.open(excludedPhrasesDataPath);
  } else {
    excludedPhrasesDataPath_.reset();
  }
}

bool McBopomofoLM::isAssociatedPhrasesV2Loaded() const {
  return associatedPhrasesV2_.isLoaded();
}

void McBopomofoLM::loadPhraseReplacementMap(const char* phraseReplacementPath) {
  phraseReplacement_.close();

  if (phraseReplacementPath) {
    phraseReplacementPath_ = phraseReplacementPath;
    phraseReplacement_.open(phraseReplacementPath);
  } else {
    phraseReplacementPath_.reset();
  }
}

static McBopomofoLM::IssueType TranslateIssue(
    ByteBlockBackedDictionary::Issue::Type t) {
  switch (t) {
    case ByteBlockBackedDictionary::Issue::Type::MISSING_SECOND_COLUMN:
      return McBopomofoLM::IssueType::MISSING_SECOND_COLUMN;
    case ByteBlockBackedDictionary::Issue::Type::NULL_CHARACTER_IN_TEXT:
      return McBopomofoLM::IssueType::NULL_CHARACTER_IN_TEXT;
    default:
      // should not happen
      return McBopomofoLM::IssueType::NO_ISSUE;
  }
}

std::vector<McBopomofoLM::UserFileIssue> McBopomofoLM::getUserFileIssues()
    const {
  std::vector<McBopomofoLM::UserFileIssue> issues;

  if (userPhrasesDataPath_.has_value()) {
    for (const auto& issue : userPhrases_.getParsingIssues()) {
      issues.emplace_back(McBopomofoLM::UserFileType::USER_PHRASES,
                          userPhrasesDataPath_.value(),
                          TranslateIssue(issue.type), issue.lineNumber);
    }
  }

  if (excludedPhrasesDataPath_.has_value()) {
    for (const auto& issue : excludedPhrases_.getParsingIssues()) {
      issues.emplace_back(McBopomofoLM::UserFileType::EXCLUDED_PHRASES,
                          excludedPhrasesDataPath_.value(),
                          TranslateIssue(issue.type), issue.lineNumber);
    }
  }

  if (phraseReplacementPath_.has_value()) {
    for (const auto& issue : phraseReplacement_.getParsingIssues()) {
      issues.emplace_back(McBopomofoLM::UserFileType::PHRASE_REPLACEMENT_MAP,
                          phraseReplacementPath_.value(),
                          TranslateIssue(issue.type), issue.lineNumber);
    }
  }

  return issues;
}

std::vector<Formosa::Gramambular2::LanguageModel::Unigram>
McBopomofoLM::getUnigrams(const std::string& key) {
  if (key == " ") {
    std::vector<Formosa::Gramambular2::LanguageModel::Unigram> spaceUnigrams;
    spaceUnigrams.emplace_back(" ", 0);
    return spaceUnigrams;
  }

  std::vector<Formosa::Gramambular2::LanguageModel::Unigram> allUnigrams;
  std::vector<Formosa::Gramambular2::LanguageModel::Unigram> userUnigrams;

  std::unordered_set<std::string> excludedValues;
  std::unordered_set<std::string> insertedValues;

  if (excludedPhrases_.hasUnigrams(key)) {
    std::vector<Formosa::Gramambular2::LanguageModel::Unigram>
        excludedUnigrams = excludedPhrases_.getUnigrams(key);
    std::transform(excludedUnigrams.begin(), excludedUnigrams.end(),
                   std::inserter(excludedValues, excludedValues.end()),
                   [](const Formosa::Gramambular2::LanguageModel::Unigram& u) {
                     return u.value();
                   });
  }

  if (userPhrases_.hasUnigrams(key)) {
    std::vector<Formosa::Gramambular2::LanguageModel::Unigram> rawUserUnigrams =
        userPhrases_.getUnigrams(key);
    // Flag user phrases so R-walk does not disadvantage them (design D8).
    userUnigrams = filterAndTransformUnigrams(rawUserUnigrams, excludedValues,
                                              insertedValues,
                                              /*isUserPhrase=*/true);
  }

  std::vector<Formosa::Gramambular2::LanguageModel::Unigram> rawGlobalUnigrams;
  if (languageModel_.hasUnigrams(key)) {
    rawGlobalUnigrams = languageModel_.getUnigrams(key);
    if (!readingPolicyExclusions_.empty()) {
      // Reading policy applies to built-in entries only, on raw values
      // (before phrase replacement, macros and the external converter).
      std::vector<std::string> syllables = AssociatedPhrasesV2::SplitReadings(key);
      bool keyHasPolicySyllable = false;
      for (const auto& syllable : syllables) {
        for (const auto& ex : readingPolicyExclusions_) {
          keyHasPolicySyllable |= syllable == ex.syllable;
        }
      }
      if (keyHasPolicySyllable) {
        rawGlobalUnigrams.erase(
            std::remove_if(
                rawGlobalUnigrams.begin(), rawGlobalUnigrams.end(),
                [&](const Formosa::Gramambular2::LanguageModel::Unigram& u) {
                  return violatesReadingPolicy(syllables, u.value());
                }),
            rawGlobalUnigrams.end());
      }
    }
  }
  appendToneVariantUnigrams(key, rawGlobalUnigrams);
  if (!rawGlobalUnigrams.empty()) {
    allUnigrams = filterAndTransformUnigrams(rawGlobalUnigrams, excludedValues,
                                             insertedValues);
  }

  // This relies on the fact that we always use the default separator.
  bool isKeyMultiSyllable =
      key.find(Formosa::Gramambular2::ReadingGrid::kDefaultSeparator) !=
      std::string::npos;

  // If key is multi-syllabic (for example, ㄉㄨㄥˋ-ㄈㄢˋ), we just
  // insert all collected userUnigrams on top of the unigrams fetched from
  // the database. If key is mono-syllabic (for example, ㄉㄨㄥˋ), then
  // we'll have to rewrite the collected userUnigrams.
  //
  // This is because, by default, user unigrams have a score of 0, which
  // guarantees that grid walks will choose them. This is problematic,
  // however, when a single-syllabic user phrase is competing with other
  // multisyllabic phrases that start with the same syllable. For example,
  // if a user has 丼 for ㄉㄨㄥˋ, and because that unigram has a score
  // of 0, no other phrases in the database that start with ㄉㄨㄥˋ would
  // be able to compete with it. Without the rewrite, ㄉㄨㄥˋ-ㄗㄨㄛˋ
  // would always result in "丼" + "作" instead of "動作" because the
  // node for "丼" would dominate the walk.
  if (isKeyMultiSyllable || allUnigrams.empty()) {
    allUnigrams.insert(allUnigrams.begin(), userUnigrams.begin(),
                       userUnigrams.end());
  } else if (!userUnigrams.empty()) {
    // Find the highest score from the existing allUnigrams.
    double topScore = std::numeric_limits<double>::lowest();
    for (const auto& unigram : allUnigrams) {
      topScore = std::max(topScore, unigram.score());
    }

    // Boost by a very small number. This is the score for user phrases.
    constexpr double epsilon = 0.000000001;
    double boostedScore = topScore + epsilon;

    std::vector<Formosa::Gramambular2::LanguageModel::Unigram>
        rewrittenUserUnigrams;
    for (const auto& unigram : userUnigrams) {
      // Keep the user-phrase flag through the score rewrite (rawValue stays
      // empty, as upstream).
      rewrittenUserUnigrams.emplace_back(unigram.value(), boostedScore,
                                         std::string(),
                                         unigram.isUserPhrase());
    }
    allUnigrams.insert(allUnigrams.begin(), rewrittenUserUnigrams.begin(),
                       rewrittenUserUnigrams.end());
  }

  return allUnigrams;
}

bool McBopomofoLM::hasUnigrams(const std::string& key) {
  if (key == " ") {
    return true;
  }

  if (!excludedPhrases_.hasUnigrams(key)) {
    if (userPhrases_.hasUnigrams(key) || languageModel_.hasUnigrams(key)) {
      return true;
    }
    bool found = false;
    forEachToneVariantUnigram(key, [&](const auto&) {
      found = true;
      return false;
    });
    return found;
  }

  return !getUnigrams(key).empty();
}

namespace {
struct ToneForms {
  const char* character;
  std::vector<std::string> syllables;
};

const std::vector<ToneForms>& YiBuToneForms() {
  static const std::vector<ToneForms> forms{{"一", {"ㄧ", "ㄧˊ", "ㄧˋ"}},
                                            {"不", {"ㄅㄨˋ", "ㄅㄨˊ"}}};
  return forms;
}

// At most this many 一 / 不 positions of a key are expanded (design D2).
constexpr size_t kMaxToneVariantPositions = 3;
}  // namespace

void McBopomofoLM::forEachToneVariantUnigram(
    const std::string& key,
    const std::function<bool(
        const Formosa::Gramambular2::LanguageModel::Unigram&)>& visit) {
  if (key.find(Formosa::Gramambular2::ReadingGrid::kDefaultSeparator) ==
      std::string::npos) {
    return;
  }
  std::vector<std::string> syllables = AssociatedPhrasesV2::SplitReadings(key);
  // (position, tone forms) for each 一 / 不 syllable in the key.
  std::vector<std::pair<size_t, const ToneForms*>> positions;
  for (size_t i = 0; i < syllables.size() &&
                     positions.size() < kMaxToneVariantPositions;
       ++i) {
    for (const auto& forms : YiBuToneForms()) {
      if (std::find(forms.syllables.begin(), forms.syllables.end(),
                    syllables[i]) != forms.syllables.end()) {
        positions.emplace_back(i, &forms);
      }
    }
  }
  if (positions.empty()) {
    return;
  }

  std::vector<std::string> alt = syllables;
  std::vector<size_t> choice(positions.size(), 0);
  while (true) {
    for (size_t p = 0; p < positions.size(); ++p) {
      alt[positions[p].first] = positions[p].second->syllables[choice[p]];
    }
    std::string altKey = AssociatedPhrasesV2::CombineReadings(alt);
    if (alt != syllables && languageModel_.hasUnigrams(altKey)) {
      for (const auto& u : languageModel_.getUnigrams(altKey)) {
        std::vector<std::string> chars = Split(u.value());
        bool keep =
            chars.size() == alt.size() &&
            std::all_of(positions.begin(), positions.end(), [&](const auto& p) {
              return alt[p.first] == syllables[p.first] ||
                     chars[p.first] == p.second->character;
            });
        if (keep && !violatesReadingPolicy(alt, u.value()) && !visit(u)) {
          return;
        }
      }
    }
    // Next combination (odometer over the tone forms of each position).
    size_t p = 0;
    while (p < positions.size() &&
           ++choice[p] == positions[p].second->syllables.size()) {
      choice[p] = 0;
      ++p;
    }
    if (p == positions.size()) {
      return;
    }
  }
}

void McBopomofoLM::appendToneVariantUnigrams(
    const std::string& key,
    std::vector<Formosa::Gramambular2::LanguageModel::Unigram>& rawUnigrams) {
  bool changed = false;
  forEachToneVariantUnigram(key, [&](const auto& u) {
    auto it = std::find_if(rawUnigrams.begin(), rawUnigrams.end(),
                           [&](const auto& x) { return x.value() == u.value(); });
    if (it == rawUnigrams.end()) {
      rawUnigrams.push_back(u);
      changed = true;
    } else if (u.score() > it->score()) {
      *it = u;
      changed = true;
    }
    return true;
  });
  if (changed) {
    std::stable_sort(rawUnigrams.begin(), rawUnigrams.end(),
                     [](const auto& a, const auto& b) {
                       return a.score() > b.score();
                     });
  }
}

std::string McBopomofoLM::getReading(const std::string& value) const {
  std::vector<ParselessLM::FoundReading> foundReadings =
      languageModel_.getReadings(value);
  double topScore = std::numeric_limits<double>::lowest();
  std::string topValue;
  for (const auto& foundReading : foundReadings) {
    if (foundReading.score > topScore) {
      topValue = foundReading.reading;
      topScore = foundReading.score;
    }
  }
  return topValue;
}

std::vector<AssociatedPhrasesV2::Phrase> McBopomofoLM::findAssociatedPhrasesV2(
    const std::string& prefixValue,
    const std::vector<std::string>& prefixReadings) const {
  std::vector<AssociatedPhrasesV2::Phrase> phrases =
      associatedPhrasesV2_.findPhrases(prefixValue, prefixReadings);
  if (readingPolicyExclusions_.empty()) {
    return phrases;
  }
  std::vector<AssociatedPhrasesV2::Phrase> kept;
  kept.reserve(phrases.size());
  for (const auto& phrase : phrases) {
    if (!violatesReadingPolicy(phrase.readings, phrase.value)) {
      kept.push_back(phrase);
    }
  }
  return kept;
}

void McBopomofoLM::setReadingPolicyExclusions(
    std::vector<ReadingPolicyExclusion> exclusions) {
  readingPolicyExclusions_ = std::move(exclusions);
}

const std::vector<McBopomofoLM::ReadingPolicyExclusion>&
McBopomofoLM::readingPolicyExclusions() const {
  return readingPolicyExclusions_;
}

bool McBopomofoLM::violatesReadingPolicy(
    const std::vector<std::string>& syllables, const std::string& value) const {
  if (readingPolicyExclusions_.empty()) {
    return false;
  }
  std::vector<std::string> chars = Split(value);
  if (chars.size() != syllables.size()) {
    return false;
  }
  for (size_t i = 0; i < chars.size(); ++i) {
    for (const auto& ex : readingPolicyExclusions_) {
      if (syllables[i] == ex.syllable && chars[i] == ex.character) {
        return true;
      }
    }
  }
  return false;
}

void McBopomofoLM::setPhraseReplacementEnabled(bool enabled) {
  phraseReplacementEnabled_ = enabled;
}

bool McBopomofoLM::phraseReplacementEnabled() const {
  return phraseReplacementEnabled_;
}

void McBopomofoLM::setExternalConverterEnabled(bool enabled) {
  externalConverterEnabled_ = enabled;
}

bool McBopomofoLM::externalConverterEnabled() const {
  return externalConverterEnabled_;
}

void McBopomofoLM::setExternalConverter(
    std::function<std::string(const std::string&)> externalConverter) {
  externalConverter_ = std::move(externalConverter);
}

void McBopomofoLM::setMacroConverter(
    std::function<std::string(const std::string&)> macroConverter) {
  macroConverter_ = std::move(macroConverter);
}

std::string McBopomofoLM::convertMacro(const std::string& input) const {
  if (macroConverter_ != nullptr) {
    return macroConverter_(input);
  }
  return input;
}

std::vector<Formosa::Gramambular2::LanguageModel::Unigram>
McBopomofoLM::filterAndTransformUnigrams(
    const std::vector<Formosa::Gramambular2::LanguageModel::Unigram>& unigrams,
    const std::unordered_set<std::string>& excludedValues,
    std::unordered_set<std::string>& insertedValues,
    bool isUserPhrase) const {
  std::vector<Formosa::Gramambular2::LanguageModel::Unigram> results;

  for (auto&& unigram : unigrams) {
    // excludedValues filters out the unigrams with the original value.
    // insertedValues filters out the ones with the converted value
    const std::string& rawValue = unigram.value();
    if (excludedValues.find(rawValue) != excludedValues.end()) {
      continue;
    }

    std::string value = rawValue;
    if (phraseReplacementEnabled_) {
      std::string replacement = phraseReplacement_.valueForKey(value);
      if (!replacement.empty()) {
        if (value != replacement) {
          value = replacement;
        }
      }
    }
    if (macroConverter_ != nullptr) {
      std::string replacement = macroConverter_(value);
      if (value != replacement) {
        value = replacement;
      }
    }

    // Check if the string is an unsupported macro
    if (unigram.score() == kMacroScore && value.size() > kMacroPrefix.size() &&
        value.compare(0, kMacroPrefix.size(), kMacroPrefix) == 0) {
      continue;
    }

    if (externalConverterEnabled_ && externalConverter_ != nullptr) {
      std::string replacement = externalConverter_(value);
      if (value != replacement) {
        value = replacement;
      }
    }
    if (insertedValues.find(value) == insertedValues.end()) {
      results.emplace_back(value, unigram.score(), rawValue,
                           isUserPhrase || unigram.isUserPhrase());
      insertedValues.insert(value);
    }
  }
  return results;
}

void McBopomofoLM::loadLanguageModel(std::unique_ptr<ParselessPhraseDB> db) {
  languageModel_.close();
  languageModel_.open(std::move(db));
}

void McBopomofoLM::loadAssociatedPhrasesV2(
    std::unique_ptr<ParselessPhraseDB> db) {
  associatedPhrasesV2_.close();
  associatedPhrasesV2_.open(std::move(db));
}

void McBopomofoLM::loadUserPhrases(const char* data, size_t length) {
  userPhrases_.close();
  userPhrases_.load(data, length);
}

void McBopomofoLM::loadExcludedPhrases(const char* data, size_t length) {
  excludedPhrases_.close();
  excludedPhrases_.load(data, length);
}

void McBopomofoLM::loadPhraseReplacementMap(const char* data, size_t length) {
  phraseReplacement_.close();
  phraseReplacement_.load(data, length);
}

}  // namespace McBopomofo
