// N-best candidates and n-gram path scores for the training loss
// (personal-finetune design D2): segment readings -> ReadingGrid on the
// shipped McBopomofo LM (no user phrases, no selection memory) ->
// walkNBest(k) texts and WalkResult.score, plus the gold index. Matches
// train/personalize/prep_user.py (bench/src/decoder.py decode(k=50)), E5.

#ifndef SOURCE_ENGINE_PERSONALIZE_NBEST_FEATURES_H_
#define SOURCE_ENGINE_PERSONALIZE_NBEST_FEATURES_H_

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "McBopomofoLM.h"

namespace McBopomofo {
namespace Personalize {

struct NBestFeatures {
  std::vector<std::string> texts;  // best first
  std::vector<double> scores;      // n-gram path scores (WalkResult.score)
  int gold = -1;                   // index of the target in texts, -1 if absent
};

// Applies the IME's reading policy (LanguageModelManager
// applyReadingPolicyAcceptingYanReadingForAi:): unless accepted, ㄧㄢˊ is not
// a reading of 癌.
void ApplyReadingPolicy(McBopomofoLM* lm, bool acceptYanReadingForAi);

// False when a reading cannot be inserted into the grid (out is cleared).
bool BuildNBest(const std::shared_ptr<McBopomofoLM>& lm, const std::vector<std::string>& readings,
                const std::string& target, size_t k, NBestFeatures* out);

}  // namespace Personalize
}  // namespace McBopomofo

#endif  // SOURCE_ENGINE_PERSONALIZE_NBEST_FEATURES_H_
