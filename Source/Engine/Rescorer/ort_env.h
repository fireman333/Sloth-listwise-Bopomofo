#ifndef SOURCE_ENGINE_RESCORER_ORT_ENV_H_
#define SOURCE_ENGINE_RESCORER_ORT_ENV_H_

#include <onnxruntime_cxx_api.h>

#include <cstdlib>

namespace McBopomofo {

// The prebuilt onnxruntime bundles Microsoft 1DS telemetry that posts usage
// events over HTTPS. Opt out before the first Env exists (the env var is read
// when telemetry starts) and again on the Env, so nothing leaves the device.
// The app sets the env var at launch; this fallback covers CLI tools/tests.
inline Ort::Env MakeOfflineOrtEnv(const char* logId) {
  if (getenv("ORT_DISABLE_TELEMETRY") == nullptr) {
    setenv("ORT_DISABLE_TELEMETRY", "1", 1);
  }
  Ort::Env env{ORT_LOGGING_LEVEL_ERROR, logId};
  env.DisableTelemetryEvents();
  return env;
}

}  // namespace McBopomofo

#endif  // SOURCE_ENGINE_RESCORER_ORT_ENV_H_
