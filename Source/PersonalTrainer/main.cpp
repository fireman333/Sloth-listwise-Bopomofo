// AFMPersonalTrainer — personalized R fine-tune helper (personal-finetune).
//
// Usage:
//   AFMPersonalTrainer --memory-dir <root>/memory --model-dir <root>/models/r-walk
//       --run-id <id> [--generation N] [--mode auto|manual] [--accept-yan-for-ai 0|1]
//       [--alpha a] [--threads n] [--pack-dir d] [--data-txt f] [--staging-dir d]
//       [--active-weights v<N>/weights.bin] [--deadline-seconds s] [--parent-pid p]
//       [--verbose]
//       [--lr x] [--max-steps n] [--replay-limit n]        (experiments / tests)
//   AFMPersonalTrainer --check-pack --model-dir <dir> [--pack-dir d]
//
// Exit codes: see pipeline.h (0 finished, 2 usage, 3 not_enough_data,
// 4 deferred, 5 cancelled, 6 discarded:retracted, 7 failed:<phase>,
// 8 failed:memory, 9 failed:pack). SIGTERM -> exit 5 within 2 s.
// Own watchdog (Codex review 2026-10-08, finding 3): after --deadline-seconds
// (default 600) -> exit 7 "failed:timeout" with no candidate files left; when
// the parent IME (--parent-pid, default the parent at start) exits -> the
// staging dir is removed and the helper exits 5, so an IME crash never leaves a
// training helper behind.

#include <mach-o/dyld.h>
#include <signal.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cctype>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>

#include "pipeline.h"

using namespace McBopomofo::Trainer;

namespace {

std::atomic<bool> gCancel{false};
std::atomic<bool> gTimedOut{false};
std::atomic<bool> gOrphaned{false};

void OnTerm(int) { gCancel.store(true); }

std::string ExecutableDir() {
  char buf[PATH_MAX];
  uint32_t size = sizeof buf;
  if (_NSGetExecutablePath(buf, &size) != 0) return "";
  char real[PATH_MAX];
  if (realpath(buf, real) == nullptr) return "";
  std::string p = real;
  return p.substr(0, p.rfind('/'));
}

int Usage(const char* msg) {
  fprintf(stderr, "AFMPersonalTrainer: %s\n", msg);
  return kExitUsage;
}

}  // namespace

int main(int argc, char** argv) {
  PipelineArgs a;
  bool checkPack = false, verbose = false;
  double deadlineSeconds = 600;
  pid_t parentPid = getppid();
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto val = [&]() -> const char* { return i + 1 < argc ? argv[++i] : nullptr; };
    if (k == "--check-pack") {
      checkPack = true;
      continue;
    }
    if (k == "--verbose") {
      verbose = true;
      continue;
    }
    const char* v = val();
    if (v == nullptr) return Usage(("missing value for " + k).c_str());
    if (k == "--memory-dir") a.memoryDir = v;
    else if (k == "--model-dir") a.modelDir = v;
    else if (k == "--pack-dir") a.packDir = v;
    else if (k == "--data-txt") a.dataTxt = v;
    else if (k == "--run-id") a.runId = v;
    else if (k == "--staging-dir") a.stagingDir = v;
    else if (k == "--active-weights") a.activeWeights = v;
    else if (k == "--generation") a.generation = strtoull(v, nullptr, 10);
    else if (k == "--mode") {
      if (strcmp(v, "auto") != 0 && strcmp(v, "manual") != 0) return Usage("--mode auto|manual");
      a.automatic = strcmp(v, "auto") == 0;
    } else if (k == "--accept-yan-for-ai") a.acceptYanForAi = atoi(v) != 0;
    else if (k == "--alpha") a.alpha = atof(v);
    else if (k == "--threads") a.threads = atoi(v);
    else if (k == "--lr") a.lr = atof(v);
    else if (k == "--max-steps") a.maxSteps = atoi(v);
    else if (k == "--replay-limit") a.replayLimit = strtoull(v, nullptr, 10);
    else if (k == "--deadline-seconds") deadlineSeconds = atof(v);
    else if (k == "--parent-pid") parentPid = static_cast<pid_t>(atoi(v));
    else return Usage(("unknown argument " + k).c_str());
  }
  if (a.modelDir.empty()) return Usage("--model-dir is required");
  if (checkPack) {
    std::string summary;
    int rc = CheckPack(a.packDir.empty() ? a.modelDir + "/personalize" : a.packDir, a.modelDir,
                       &summary);
    printf("%s\n", summary.c_str());
    return rc;
  }
  if (a.memoryDir.empty() || a.runId.empty()) return Usage("--memory-dir and --run-id are required");
  for (char c : a.runId)
    if (!(isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_' || c == '.'))
      return Usage("--run-id: [A-Za-z0-9._-] only");
  if (a.threads < 1 || a.threads > 16) return Usage("--threads 1..16");
  if (!(deadlineSeconds > 0) || deadlineSeconds > 24 * 3600) return Usage("--deadline-seconds 0..86400");
  if (a.dataTxt.empty()) a.dataTxt = ExecutableDir() + "/../Resources/data.txt";

  struct sigaction sa;
  memset(&sa, 0, sizeof sa);
  sa.sa_handler = OnTerm;
  sigaction(SIGTERM, &sa, nullptr);
  sigaction(SIGINT, &sa, nullptr);
  // Watchdog: SIGTERM, the wall-clock deadline or the parent's exit set the
  // flags the pipeline checks between steps / sentences (< 0.5 s apart); if
  // the process is inside a long call (e.g. a session load) leave anyway after
  // 1.8 s (deadline / orphaned: remove the candidate files or staging first).
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(static_cast<int64_t>(deadlineSeconds * 1000));
  const bool watchParent = parentPid > 1;
  // Our real parent: reparenting (getppid changes) is immediate; any other pid
  // (tests, wrappers): polled for existence.
  const bool isParent = parentPid == getppid();
  const std::string staging = a.stagingDir.empty() ? a.memoryDir + "/r-personal/staging/" + a.runId : a.stagingDir;
  std::thread([deadline, watchParent, isParent, parentPid, staging] {
    while (!gCancel.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      if (std::chrono::steady_clock::now() >= deadline) {
        gTimedOut.store(true);
        gCancel.store(true);
      } else if (watchParent && (isParent ? getppid() != parentPid
                                          : (kill(parentPid, 0) != 0 && errno == ESRCH))) {
        gOrphaned.store(true);
        gCancel.store(true);
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1800));
    if (gOrphaned.load()) {
      std::error_code ec;
      std::filesystem::remove_all(staging, ec);
    } else if (gTimedOut.load()) {
      for (const char* f : {"weights.bin", "weights.bin.tmp", "precise-r.conf", "precise-r.conf.tmp"})
        unlink((staging + "/" + f).c_str());
    }
    _exit(gTimedOut.load() && !gOrphaned.load() ? kExitFailed : kExitCancelled);
  }).detach();

  PipelineHooks h;
  h.cancel = &gCancel;
  h.timedOut = &gTimedOut;
  h.orphaned = &gOrphaned;
  h.verbose = verbose;
  return RunPipeline(a, h);
}
