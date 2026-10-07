// Self-limits of the trainer helper (personal-finetune design D6, task 3.4):
// power source, Low Power Mode, thermal state (polled in automatic runs) and
// the process's resident memory. Injectable for tests.

#ifndef SOURCE_PERSONALTRAINER_SYSTEM_PROBES_H_
#define SOURCE_PERSONALTRAINER_SYSTEM_PROBES_H_

#include <cstdint>
#include <functional>
#include <string>

namespace McBopomofo {
namespace Trainer {

struct SystemProbes {
  std::function<bool()> onAC;           // true when on AC power
  std::function<bool()> lowPowerMode;   // true when Low Power Mode is on
  std::function<int()> thermalState;    // 0 nominal, 1 fair, 2 serious, 3 critical
  std::function<uint64_t()> residentBytes;

  static SystemProbes Real();
};

// Empty string when an automatic run may continue, else the failing condition
// ("battery", "low_power_mode", "thermal").
std::string DeferReason(const SystemProbes& probes);

}  // namespace Trainer
}  // namespace McBopomofo

#endif  // SOURCE_PERSONALTRAINER_SYSTEM_PROBES_H_
