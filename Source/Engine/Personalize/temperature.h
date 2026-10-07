// Precise-mode temperature refit for a personalized R (personal-finetune
// design D12): C++ port of step (1) of bench/src/precise_calib.py.
//   p_top1(T) = 1 / sum_{j<K} exp((r_j - r_0) / T)
//   T = argmin over the log-T grid (1201 points, log 0.02 .. log 50) of the
//       binary log-loss of p_top1 vs [R-walk pick correct]
// globally or per syllable-count bucket (mode taken from the factory config).

#ifndef SOURCE_ENGINE_PERSONALIZE_TEMPERATURE_H_
#define SOURCE_ENGINE_PERSONALIZE_TEMPERATURE_H_

#include <cstddef>
#include <vector>

namespace McBopomofo {
namespace Personalize {

struct CalibRow {
  size_t nSyl = 0;
  std::vector<double> rScores;  // R-walk N-best path scores, best first
  bool top1Correct = false;     // rScores[0]'s candidate == gold
};

// np.linspace(log(0.02), log(50.0), 1201).
const std::vector<double>& LogTGrid();

double PTop1(const std::vector<double>& r, double T, size_t K);
// Mean binary log-loss with p clipped to [1e-12, 1 - 1e-12].
double LogLoss(const std::vector<double>& ps, const std::vector<bool>& ys);
// Expected calibration error, `bins` equal-width bins (last bin closed).
double Ece(const std::vector<double>& ps, const std::vector<bool>& ys, int bins = 10);

// Global temperature (1.0 for no rows).
double FitTemperature(const std::vector<CalibRow>& rows, size_t K);

size_t BucketOf(size_t nSyl, const std::vector<size_t>& bucketUpper);

// bucketUpper.size() + 1 temperatures; all equal when !perBucket.
std::vector<double> FitTemperatures(const std::vector<CalibRow>& rows, size_t K,
                                    const std::vector<size_t>& bucketUpper, bool perBucket);

struct CalibQuality {
  double eceBefore = 0, eceAfter = 0;          // T = 1 vs fitted temperatures
  double logLossBefore = 0, logLossAfter = 0;
};
CalibQuality EvaluateCalibration(const std::vector<CalibRow>& rows, size_t K,
                                 const std::vector<size_t>& bucketUpper,
                                 const std::vector<double>& temperatures);

}  // namespace Personalize
}  // namespace McBopomofo

#endif  // SOURCE_ENGINE_PERSONALIZE_TEMPERATURE_H_
