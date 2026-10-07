// AFMPersonalTrainer pipeline (personal-finetune design D1, D6, D7, D10,
// D12, D13): scan -> nbest -> train -> merge -> late-retraction check ->
// gate (re-reading the staged weights.bin) -> calib -> staging files.
//
// Files (all under the memory dir, dirs 0700, files 0600):
//   <memory>/sentences.jsonl                      read only (pool)
//   <memory>/r-personal/status.json               progress, atomic
//   <memory>/r-personal/staging/<run>/weights.bin  candidate weights (D9)
//   <memory>/r-personal/staging/<run>/precise-r.conf  (adopted runs only, D12)
//   <memory>/r-personal/staging/<run>/report.json  outcome (no text)
// The helper never writes manifest.json and never deletes versions (D1).

#ifndef SOURCE_PERSONALTRAINER_PIPELINE_H_
#define SOURCE_PERSONALTRAINER_PIPELINE_H_

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>

#include "system_probes.h"

namespace McBopomofo {
namespace Trainer {

// Process exit codes (report.json "result" in brackets).
enum ExitCode : int {
  kExitFinished = 0,      // gate evaluated: adopted | not_significant | too_small_gain |
                          // general_regression | worse_than_active
  kExitUsage = 2,         // bad arguments (no report)
  kExitNotEnoughData = 3, // not_enough_data
  kExitDeferred = 4,      // deferred (automatic run: battery / Low Power Mode / thermal)
  kExitCancelled = 5,     // cancelled (SIGTERM)
  kExitRetracted = 6,     // discarded:retracted
  kExitFailed = 7,        // failed:<phase>
  kExitMemory = 8,        // failed:memory (resident memory above the limit)
  kExitPack = 9,          // failed:pack (pack missing / invalid / other factory model)
};

struct PipelineArgs {
  std::string memoryDir;     // <root>/memory
  std::string modelDir;      // <root>/models/r-walk
  std::string packDir;       // default <modelDir>/personalize
  std::string dataTxt;       // the IME bundle's data.txt
  std::string runId;
  std::string stagingDir;    // default <memory>/r-personal/staging/<run>
  std::string activeWeights; // weights.bin of the version in use (gate rule 3), optional
  uint64_t generation = 0;
  bool automatic = false;
  bool acceptYanForAi = false;
  double alpha = -1;         // < 0: pack recipe alpha (0.5)
  int threads = 4;           // onnxruntime intra-op threads of the gate
  // Test / experiment overrides (0 = pack value).
  double lr = 0;
  int maxSteps = 0;
  size_t replayLimit = 0;
};

struct PipelineHooks {
  SystemProbes probes = SystemProbes::Real();
  const std::atomic<bool>* cancel = nullptr;  // set by SIGTERM
  double pollSeconds = 10;                    // power / thermal poll interval (automatic runs)
  uint64_t rssLimitBytes = 2ull << 30;
  // Called after each training step (tests: inject a retraction, a signal...).
  std::function<void(int step)> onTrainStep;
  bool verbose = false;  // numbers-only progress lines on stderr
};

int RunPipeline(const PipelineArgs& args, const PipelineHooks& hooks);

// --check-pack: pack.json, file hashes, factory SHA, train-base.bin.
// Returns 0 (OK) or kExitPack; *summary receives a one-line JSON.
int CheckPack(const std::string& packDir, const std::string& modelDir, std::string* summary);

// 64-bit FNV-1a of the run id (training seed).
uint64_t SeedForRun(const std::string& runId);

}  // namespace Trainer
}  // namespace McBopomofo

#endif  // SOURCE_PERSONALTRAINER_PIPELINE_H_
