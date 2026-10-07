#include "system_probes.h"

#import <Foundation/Foundation.h>
#include <IOKit/ps/IOPSKeys.h>
#include <IOKit/ps/IOPowerSources.h>
#include <mach/mach.h>

namespace McBopomofo {
namespace Trainer {

SystemProbes SystemProbes::Real() {
  SystemProbes p;
  p.onAC = [] {
    CFTypeRef info = IOPSCopyPowerSourcesInfo();
    if (info == nullptr) return true;  // desktop Mac without a power-source API
    CFStringRef type = IOPSGetProvidingPowerSourceType(info);
    bool ac = type == nullptr || CFStringCompare(type, CFSTR(kIOPMACPowerKey), 0) == kCFCompareEqualTo;
    CFRelease(info);
    return ac;
  };
  p.lowPowerMode = [] {
    @autoreleasepool {
      return static_cast<bool>([[NSProcessInfo processInfo] isLowPowerModeEnabled]);
    }
  };
  p.thermalState = [] {
    @autoreleasepool {
      return static_cast<int>([[NSProcessInfo processInfo] thermalState]);
    }
  };
  p.residentBytes = [] {
    mach_task_basic_info_data_t info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&info),
                  &count) != KERN_SUCCESS)
      return static_cast<uint64_t>(0);
    return static_cast<uint64_t>(info.resident_size);
  };
  return p;
}

std::string DeferReason(const SystemProbes& probes) {
  if (probes.onAC && !probes.onAC()) return "battery";
  if (probes.lowPowerMode && probes.lowPowerMode()) return "low_power_mode";
  // D6: thermal state must stay <= fair (1).
  if (probes.thermalState && probes.thermalState() > 1) return "thermal";
  return "";
}

}  // namespace Trainer
}  // namespace McBopomofo
