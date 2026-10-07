#include "nbest_features.h"

#include "gramambular2/reading_grid.h"

namespace McBopomofo {
namespace Personalize {

using Formosa::Gramambular2::ReadingGrid;

void ApplyReadingPolicy(McBopomofoLM* lm, bool acceptYanReadingForAi) {
  std::vector<McBopomofoLM::ReadingPolicyExclusion> ex;
  if (!acceptYanReadingForAi) ex.push_back({"ㄧㄢˊ", "癌"});
  lm->setReadingPolicyExclusions(std::move(ex));
}

bool BuildNBest(const std::shared_ptr<McBopomofoLM>& lm, const std::vector<std::string>& readings,
                const std::string& target, size_t k, NBestFeatures* out) {
  *out = NBestFeatures();
  if (!lm || readings.empty()) return false;
  ReadingGrid grid(lm);
  for (const auto& r : readings) {
    if (!grid.insertReading(r)) return false;
  }
  for (auto& w : grid.walkNBest(k)) {
    std::string text;
    for (const auto& n : w.nodes) text += n->value();
    if (out->gold < 0 && text == target) out->gold = static_cast<int>(out->texts.size());
    out->texts.push_back(std::move(text));
    out->scores.push_back(w.score);
  }
  return !out->texts.empty();
}

}  // namespace Personalize
}  // namespace McBopomofo
