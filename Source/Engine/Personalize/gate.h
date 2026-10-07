// Acceptance gate statistics (personal-finetune design D10): exact McNemar
// test on paired per-sentence correctness and the adoption rules, with
// thresholds from the personalization pack's pack.json.

#ifndef SOURCE_ENGINE_PERSONALIZE_GATE_H_
#define SOURCE_ENGINE_PERSONALIZE_GATE_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace McBopomofo {
namespace Personalize {

// Two-sided exact McNemar p-value: binomial(n = b + c, 0.5) on the discordant
// pairs, min(1, 2 * P(X <= min(b, c))); 1 when b + c == 0. Same definition as
// train/ranker/ranker_lib.mcnemar_exact.
double McNemarExactTwoSided(uint64_t b, uint64_t c);
// One-sided: P(X >= wins) for X ~ binomial(wins + losses, 0.5) (H1: the
// candidate is better); 1 when wins + losses == 0.
double McNemarExactOneSided(uint64_t wins, uint64_t losses);

// Defaults = research 17 (E7, task 1.7): +2 pt floor and exact one-sided
// McNemar p < 0.05, general >= -1.0 pt on CV-1,005, not worse than active.
struct GateThresholds {
  double minPersonalGainPt = 2.0;   // rule 1: candidate - factory on personal test
  double pMax = 0.05;               // rule 1: p < pMax
  bool oneSided = true;             // McNemar sidedness
  double maxGeneralDropPt = 1.0;    // rule 2: candidate - factory >= -maxGeneralDropPt
  bool notWorseThanActive = true;   // rule 3
};

// Data-selection values that the pack may also override (spec "initial values").
struct DataThresholds {
  size_t minEligible = 1000;
  double holdoutFraction = 0.30;
  size_t holdoutMin = 300;
  size_t holdoutMax = 400;
  size_t trainCap = 2000;
  size_t minSyllables = 2;
  size_t maxSyllables = 40;
};

// Reads the "gate" and "data" objects of a pack.json text; absent keys keep
// their defaults. Keys (all optional):
//   gate: min_personal_gain_pt, p_max, mcnemar ("two-sided" | "one-sided"),
//         max_general_drop_pt, not_worse_than_active
//   data: min_eligible, holdout_fraction, holdout_min, holdout_max, train_cap,
//         min_syllables, max_syllables
// False (with *error) on malformed JSON, a wrong type or an out-of-range value.
bool ParsePackThresholds(const std::string& packJson, GateThresholds* gate, DataThresholds* data,
                         std::string* error);

struct GateInput {
  // Per-sentence correctness, same order and length within a set.
  std::vector<bool> factoryPersonal;
  std::vector<bool> candidatePersonal;
  std::vector<bool> activePersonal;  // empty = no personalized version in use
  std::vector<bool> factoryGeneral;
  std::vector<bool> candidateGeneral;
};

struct GateResult {
  bool valid = false;  // false: sizes mismatch or empty sets (reason "failed:gate")
  size_t nPersonal = 0, nGeneral = 0;
  double factoryPersonalPct = 0, candidatePersonalPct = 0, activePersonalPct = -1;
  double factoryGeneralPct = 0, candidateGeneralPct = 0;
  double deltaPersonalPt = 0, deltaGeneralPt = 0;
  uint64_t wins = 0, losses = 0;  // personal discordant pairs: candidate only / factory only
  double p = 1;
  bool adopted = false;
  // Primary reason: adopted, general_regression, too_small_gain, not_significant,
  // worse_than_active, failed:gate.
  std::string reason;
  std::vector<std::string> failedRules;  // every failing rule, in the order above
};

// Rule order for the primary reason: general_regression (safety first),
// too_small_gain, not_significant, worse_than_active.
GateResult EvaluateGate(const GateInput& input, const GateThresholds& thresholds);

}  // namespace Personalize
}  // namespace McBopomofo

#endif  // SOURCE_ENGINE_PERSONALIZE_GATE_H_
