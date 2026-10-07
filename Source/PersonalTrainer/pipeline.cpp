#include "pipeline.h"

#include <fcntl.h>
#include <os/log.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <memory>
#include <sstream>
#include <vector>

#include "McBopomofoLM.h"
#include "Personalize/gate.h"
#include "Personalize/nbest_features.h"
#include "Personalize/personal_weights.h"
#include "Personalize/pool_reader.h"
#include "Personalize/temperature.h"
#include "gramambular2/reading_grid.h"
#include "mlx_trainer.h"
#include "neural_walk.h"
#include "nlohmann/json.hpp"
#include "precise_walk.h"
#include "sloth_rescorer.h"
#include "train_base.h"
#include "train_data.h"

namespace McBopomofo {
namespace Trainer {

using nlohmann::json;
using Formosa::Gramambular2::ReadingGrid;
namespace pz = McBopomofo::Personalize;

uint64_t SeedForRun(const std::string& runId) {
  uint64_t h = 1469598103934665603ULL;
  for (unsigned char c : runId) {
    h ^= c;
    h *= 1099511628211ULL;
  }
  return h;
}

namespace {

constexpr const char* kSubsystem = "io.github.fireman333.inputmethod.Sloth-listwise-Bopomofo.trainer";

os_log_t Log() {
  static os_log_t log = os_log_create(kSubsystem, "run");
  return log;
}

using Clock = std::chrono::steady_clock;
double SecSince(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

std::string IsoNow() {
  char buf[32];
  time_t t = time(nullptr);
  struct tm tm;
  gmtime_r(&t, &tm);
  strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", &tm);
  return buf;
}

double R3(double x) { return std::round(x * 1000.0) / 1000.0; }

bool MakeDir700(const std::string& path) {
  if (mkdir(path.c_str(), 0700) == 0) return true;
  struct stat st;
  if (errno == EEXIST && stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
    chmod(path.c_str(), 0700);
    return true;
  }
  return false;
}

bool MakeDirs700(const std::string& path) {
  std::string cur;
  size_t i = 0;
  while (i <= path.size()) {
    size_t j = path.find('/', i);
    if (j == std::string::npos) j = path.size();
    cur = path.substr(0, j);
    if (!cur.empty()) {
      struct stat st;
      if (stat(cur.c_str(), &st) != 0 && !MakeDir700(cur)) return false;
    }
    i = j + 1;
  }
  return true;
}

// Atomic text write (<path>.tmp + rename), mode 0600.
bool WriteFile600(const std::string& path, const std::string& text) {
  std::string tmp = path + ".tmp";
  int fd = open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
  if (fd < 0) return false;
  fchmod(fd, 0600);
  size_t off = 0;
  while (off < text.size()) {
    ssize_t n = write(fd, text.data() + off, text.size() - off);
    if (n <= 0) {
      close(fd);
      unlink(tmp.c_str());
      return false;
    }
    off += static_cast<size_t>(n);
  }
  fsync(fd);
  close(fd);
  return rename(tmp.c_str(), path.c_str()) == 0;
}

std::string RealDir(const std::string& p) {
  char buf[PATH_MAX];
  if (realpath(p.c_str(), buf) == nullptr) return "";
  return buf;
}

bool Inside(const std::string& child, const std::string& parent) {
  return !parent.empty() && child.size() > parent.size() &&
         child.compare(0, parent.size(), parent) == 0 && child[parent.size()] == '/';
}

struct Sentence {
  std::vector<std::string> readings;
  std::string text;
};

bool ReadSentences(const std::string& path, std::vector<Sentence>* out, std::string* error) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    *error = "cannot open " + path;
    return false;
  }
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    json j = json::parse(line, nullptr, false);
    if (j.is_discarded() || !j.contains("text") || !j.contains("reading") ||
        !j["text"].is_string() || !j["reading"].is_array()) {
      *error = "malformed line in " + path;
      return false;
    }
    Sentence s;
    s.text = j["text"].get<std::string>();
    for (const auto& r : j["reading"]) {
      if (!r.is_string()) {
        *error = "malformed reading in " + path;
        return false;
      }
      s.readings.push_back(r.get<std::string>());
    }
    out->push_back(std::move(s));
  }
  return true;
}

struct Pack {
  json meta;
  pz::GateThresholds gate;
  pz::DataThresholds data;
  Recipe recipe;
  double alpha = 0.5;
  size_t calibK = 10;
  std::vector<size_t> bucketUpper;
  bool perBucket = false;
  std::string version;
  std::string factoryShaHex;
  std::shared_ptr<const TrainBase> base;
};

int LoadPack(const std::string& packDir, const std::string& modelDir, Pack* pack, std::string* error,
             bool loadBase) {
  std::string text;
  if (!pz::ReadFileBytes(packDir + "/pack.json", &text, error)) return kExitPack;
  pack->meta = json::parse(text, nullptr, false);
  const json& m = pack->meta;
  if (m.is_discarded() || !m.is_object() || !m.contains("factory_sha256") ||
      !m["factory_sha256"].is_string() || !m.contains("files") || !m["files"].is_object()) {
    *error = "pack.json malformed";
    return kExitPack;
  }
  if (!pz::ParsePackThresholds(text, &pack->gate, &pack->data, error)) return kExitPack;
  pack->version = m.value("pack_version", "");
  pack->factoryShaHex = m["factory_sha256"].get<std::string>();
  pz::Sha256Digest modelSha;
  if (!pz::Sha256File(modelDir + "/onnx/model_quantized.onnx", &modelSha, error)) return kExitPack;
  if (pz::ToHex(modelSha) != pack->factoryShaHex) {
    *error = "pack is for another factory model";
    return kExitPack;
  }
  for (const char* f : {"train-base.bin", "replay.jsonl", "gate-general.jsonl", "calib.jsonl"}) {
    if (!m["files"].contains(f) || !m["files"][f].contains("sha256")) {
      *error = std::string("pack.json lists no ") + f;
      return kExitPack;
    }
    pz::Sha256Digest d;
    if (!pz::Sha256File(packDir + "/" + f, &d, error)) return kExitPack;
    if (pz::ToHex(d) != m["files"][f]["sha256"].get<std::string>()) {
      *error = std::string("hash mismatch: ") + f;
      return kExitPack;
    }
  }
  if (m.contains("recipe") && m["recipe"].is_object()) {
    const json& r = m["recipe"];
    Recipe& R = pack->recipe;
    R.rank = r.value("rank", R.rank);
    R.loraScale = r.value("lora_scale", R.loraScale);
    R.lr = r.value("lr", R.lr);
    R.beta1 = r.value("beta1", R.beta1);
    R.beta2 = r.value("beta2", R.beta2);
    R.eps = r.value("eps", R.eps);
    R.warmupFrac = r.value("warmup_frac", R.warmupFrac);
    R.lrFloor = r.value("lr_floor", R.lrFloor);
    R.epochs = r.value("epochs", R.epochs);
    R.minSteps = r.value("min_steps", R.minSteps);
    R.batchUser = r.value("batch_user", R.batchUser);
    R.batchReplay = r.value("batch_replay", R.batchReplay);
    R.clip = r.value("clip", R.clip);
    R.mleWeight = r.value("mle_weight", R.mleWeight);
    R.ngramWeight = r.value("ngram_weight", R.ngramWeight);
    R.missingLogP = r.value("missing_logp", R.missingLogP);
    R.cacheMB = r.value("cache_mb", R.cacheMB);
    pack->alpha = r.value("alpha", pack->alpha);
    if (R.rank <= 0 || R.rank > 64 || R.batchUser <= 0 || R.batchReplay < 0 || !(R.lr > 0)) {
      *error = "pack recipe out of range";
      return kExitPack;
    }
  }
  if (m.contains("calib") && m["calib"].is_object()) {
    const json& c = m["calib"];
    pack->calibK = c.value("k", pack->calibK);
    pack->perBucket = c.value("per_bucket", pack->perBucket);
    if (c.contains("bucket_upper") && c["bucket_upper"].is_array())
      for (const auto& b : c["bucket_upper"]) pack->bucketUpper.push_back(b.get<size_t>());
  }
  if (loadBase) {
    pack->base = ReadTrainBase(packDir + "/train-base.bin", error);
    if (!pack->base) return kExitPack;
    if (pz::ToHex(pack->base->factorySha) != pack->factoryShaHex) {
      *error = "train-base.bin is for another factory model";
      return kExitPack;
    }
  }
  return 0;
}

class Runner {
 public:
  Runner(const PipelineArgs& a, const PipelineHooks& h) : args_(a), hooks_(h) {}

  int Run() {
    t0_ = Clock::now();
    report_ = {{"format", 1},          {"run", args_.runId},
               {"generation", args_.generation}, {"mode", args_.automatic ? "auto" : "manual"},
               {"started", IsoNow()},  {"pid", getpid()}};
    int rc;
    try {
      rc = Pipeline();
    } catch (const std::exception& e) {
      os_log_error(Log(), "exception in %{public}s: %{public}s", phase_.c_str(), e.what());
      report_["error"] = "exception";
      rc = End(kExitFailed, "failed:" + (phase_.empty() ? std::string("setup") : phase_));
    }
    report_["finished"] = IsoNow();
    timing_["total_s"] = R3(SecSince(t0_));
    report_["timing"] = timing_;
    report_["peak_rss_mb"] = R3(peakRss_ / 1048576.0);
    report_["exit_code"] = rc;
    if (!stagingDir_.empty()) WriteFile600(stagingDir_ + "/report.json", report_.dump(1) + "\n");
    Status(rc == kExitFinished ? "done" : "ended", 0, 0);
    os_log(Log(), "run end: result %{public}s exit %d total %.1f s rss %.0f MB",
           report_.value("result", "").c_str(), rc, SecSince(t0_), peakRss_ / 1048576.0);
    if (hooks_.verbose)
      fprintf(stderr, "result %s exit %d total %.1f s peak rss %.0f MB\n",
              report_.value("result", "").c_str(), rc, SecSince(t0_), peakRss_ / 1048576.0);
    return rc;
  }

 private:
  int End(int rc, const std::string& result) {
    report_["result"] = result;
    return rc;
  }
  int Fail(const std::string& phase, const std::string& why) {
    report_["error"] = why;  // file / rule names only, never text
    return End(kExitFailed, "failed:" + phase);
  }

  void Status(const std::string& phase, int step, int steps) {
    if (statusPath_.empty()) return;
    json s = {{"run", args_.runId}, {"phase", phase},    {"step", step},
              {"steps", steps},     {"eligible", eligible_}, {"started", report_["started"]},
              {"pid", getpid()},    {"generation", args_.generation}};
    WriteFile600(statusPath_, s.dump() + "\n");
  }

  void Phase(const std::string& phase) {
    phase_ = phase;
    Status(phase, 0, 0);
    os_log(Log(), "phase %{public}s", phase.c_str());
    if (hooks_.verbose) fprintf(stderr, "[%6.1f s] phase %s\n", SecSince(t0_), phase.c_str());
  }

  // 0 = continue; else the exit code to stop with (result set).
  int Limits() {
    uint64_t rss = hooks_.probes.residentBytes ? hooks_.probes.residentBytes() : 0;
    if (rss > peakRss_) peakRss_ = rss;
    if (hooks_.cancel && hooks_.cancel->load()) return End(kExitCancelled, "cancelled");
    if (rss > hooks_.rssLimitBytes) return End(kExitMemory, "failed:memory");
    if (args_.automatic) {
      double now = SecSince(t0_);
      if (now - lastPoll_ >= hooks_.pollSeconds || lastPoll_ < 0) {
        lastPoll_ = now;
        std::string why = DeferReason(hooks_.probes);
        if (!why.empty()) {
          report_["deferred_because"] = why;
          return End(kExitDeferred, "deferred");
        }
      }
    }
    return 0;
  }

  int Pipeline() {
    // ---- arguments and files -------------------------------------------------
    std::string mem = RealDir(args_.memoryDir);
    if (mem.empty()) return Fail("setup", "memory dir missing");
    std::string rp = mem + "/r-personal";
    if (!MakeDir700(rp)) return Fail("setup", "cannot create r-personal");
    statusPath_ = rp + "/status.json";
    std::string staging = args_.stagingDir.empty() ? rp + "/staging/" + args_.runId : args_.stagingDir;
    if (!MakeDirs700(staging)) return Fail("setup", "cannot create staging");
    stagingDir_ = RealDir(staging);
    if (!Inside(stagingDir_, mem)) {
      stagingDir_.clear();
      return Fail("setup", "staging dir outside the memory dir");
    }
    report_["eval_config"] = json::object();
    Phase("scan");
    if (int rc = Limits()) return rc;
    auto ts = Clock::now();
    std::string packDir = args_.packDir.empty() ? args_.modelDir + "/personalize" : args_.packDir;
    Pack pack;
    std::string err;
    if (int rc = LoadPack(packDir, args_.modelDir, &pack, &err, true)) {
      report_["error"] = err;
      return End(rc, "failed:pack");
    }
    Recipe recipe = pack.recipe;
    if (args_.lr > 0) recipe.lr = args_.lr;
    const double alpha = args_.alpha >= 0 ? args_.alpha : pack.alpha;
    report_["eval_config"] = {{"factory_sha256", pack.factoryShaHex},
                              {"alpha", alpha},
                              {"accept_yan_for_ai", args_.acceptYanForAi},
                              {"pack_version", pack.version},
                              {"calibration_version", pack.version}};
    pz::Vocab poolVocab;
    if (!pz::LoadModelVocab(args_.modelDir, &poolVocab, &err)) return Fail("scan", err);
    pz::SegmentOptions so;
    so.minSyllables = pack.data.minSyllables;
    so.maxSyllables = pack.data.maxSyllables;
    pz::PoolScan scan;
    const std::string poolPath = mem + "/sentences.jsonl";
    if (!pz::ScanPool(poolPath, poolVocab, so, &scan, &err)) {
      report_["eligible"] = 0;
      report_["error"] = "pool unreadable";
      return End(kExitNotEnoughData, "not_enough_data");
    }
    eligible_ = scan.segments.size();
    report_["scan_offset"] = scan.scanOffset;
    report_["eligible"] = eligible_;
    report_["pool"] = {{"lines", scan.lines},
                       {"malformed_lines", scan.malformedLines},
                       {"records", scan.records},
                       {"retracted_records", scan.retractedRecords},
                       {"ineligible_records", scan.ineligibleRecords},
                       {"dropped_segments", scan.droppedSegments},
                       {"duplicate_segments", scan.duplicateSegments}};
    if (eligible_ < pack.data.minEligible) {
      report_["min_eligible"] = pack.data.minEligible;
      return End(kExitNotEnoughData, "not_enough_data");
    }
    pz::SplitOptions sp;
    sp.holdoutFraction = pack.data.holdoutFraction;
    sp.holdoutMin = pack.data.holdoutMin;
    sp.holdoutMax = pack.data.holdoutMax;
    sp.trainCap = pack.data.trainCap;
    pz::Split split = pz::SplitSegments(scan.segments, sp);
    report_["n_train"] = split.train.size();
    report_["n_test"] = split.test.size();
    timing_["scan_s"] = R3(SecSince(ts));

    // ---- N-best features ----------------------------------------------------
    Phase("nbest");
    ts = Clock::now();
    auto lm = std::make_shared<McBopomofoLM>();
    lm->loadLanguageModel(args_.dataTxt.c_str());
    if (!lm->isDataModelLoaded()) return Fail("nbest", "data.txt not loaded");
    pz::ApplyReadingPolicy(lm.get(), args_.acceptYanForAi);
    ModelVocab vocab;
    if (!LoadTrainerVocab(args_.modelDir, pack.base->config.nSyl, &vocab, &err))
      return Fail("nbest", err);
    std::vector<EncodedRow> userRows, replayRows;
    size_t nbFail = 0;
    pz::NBestFeatures nb;
    for (size_t i = 0; i < split.train.size(); ++i) {
      const auto& s = split.train[i];
      if (i % 100 == 0)
        if (int rc = Limits()) return rc;
      if (!pz::BuildNBest(lm, s.readings, s.target, 50, &nb)) {
        ++nbFail;
        continue;
      }
      userRows.push_back(EncodeRow(vocab, s.readings, s.target, nb.texts, nb.scores));
    }
    std::vector<Sentence> replay;
    if (!ReadSentences(packDir + "/replay.jsonl", &replay, &err)) return Fail("nbest", err);
    if (args_.replayLimit > 0 && replay.size() > args_.replayLimit) replay.resize(args_.replayLimit);
    for (size_t i = 0; i < replay.size(); ++i) {
      if (i % 100 == 0)
        if (int rc = Limits()) return rc;
      if (!pz::BuildNBest(lm, replay[i].readings, replay[i].text, 50, &nb)) {
        ++nbFail;
        continue;
      }
      replayRows.push_back(EncodeRow(vocab, replay[i].readings, replay[i].text, nb.texts, nb.scores));
    }
    size_t goldIn = 0;
    for (const auto& r : userRows) goldIn += r.target >= 0;
    report_["nbest"] = {{"user_rows", userRows.size()},
                        {"replay_rows", replayRows.size()},
                        {"failed", nbFail},
                        {"user_gold_in_top50", goldIn}};
    timing_["nbest_s"] = R3(SecSince(ts));

    // ---- training ------------------------------------------------------------
    Phase("train");
    ts = Clock::now();
    int steps = StepsFor(recipe, userRows.size());
    if (args_.maxSteps > 0) steps = std::min(steps, args_.maxSteps);
    const uint64_t seed = SeedForRun(args_.runId);
    LoraTrainer trainer(pack.base, vocab, recipe);
    if (!trainer.ok()) return Fail("train", trainer.error());
    trainer.InitLora(seed);
    int stopRc = 0;
    double firstLoss = NAN, lastLoss = NAN;
    bool trained = trainer.Train(
        userRows, replayRows, steps, seed,
        [&](const TrainProgress& p) {
          lastLoss = p.loss;
          if (p.step % 10 == 0 || p.step == p.steps) Status("train", p.step, p.steps);
          if (hooks_.onTrainStep) hooks_.onTrainStep(p.step);
          if (hooks_.verbose && (p.step % 25 == 0 || p.step == p.steps))
            fprintf(stderr, "[%6.1f s] step %d/%d loss %.4f\n", SecSince(t0_), p.step, p.steps, p.loss);
          stopRc = Limits();
          return stopRc == 0;
        },
        &firstLoss);
    report_["train"] = {{"steps", steps},
                        {"seed", seed},
                        {"lr", recipe.lr},
                        {"trainable", trainer.trainableParameters()},
                        {"first_loss", std::isfinite(firstLoss) ? json(R3(firstLoss)) : json(nullptr)},
                        {"last_loss", std::isfinite(lastLoss) ? json(R3(lastLoss)) : json(nullptr)},
                        {"ms_per_step", R3(1000.0 * SecSince(ts) / std::max(1, steps))}};
    timing_["train_s"] = R3(SecSince(ts));
    if (!trained) {
      if (stopRc) return stopRc;
      return Fail("train", trainer.error());
    }
    os_log(Log(), "trained %d steps, first loss %.4f", steps, firstLoss);

    // ---- merge + quantize + staged weights ---------------------------------------
    Phase("merge");
    ts = Clock::now();
    pz::PersonalWeights pw;
    if (!trainer.Merge(&pw, &err)) return Fail("merge", err);
    const std::string weightsPath = stagingDir_ + "/weights.bin";
    if (!pz::WriteWeightsFile(weightsPath, pw, &err)) return Fail("merge", err);
    pz::Sha256Digest wsha;
    if (pz::Sha256File(weightsPath, &wsha, &err)) report_["weights_sha256"] = pz::ToHex(wsha);
    timing_["merge_s"] = R3(SecSince(ts));
    if (int rc = Limits()) return rc;

    // ---- late retractions (D7) -------------------------------------------------
    std::unordered_set<int64_t> late;
    if (!pz::ScanRetractions(poolPath, scan.scanOffset, &late, &err)) return Fail("retract", err);
    report_["late_retractions"] = late.size();
    if (pz::UsesRetracted(split, late)) return End(kExitRetracted, "discarded:retracted");

    // ---- gate (D10): IME's ORT + R-walk on the staged weights.bin bytes ----------
    Phase("gate");
    ts = Clock::now();
    std::vector<Sentence> general;
    if (!ReadSentences(packDir + "/gate-general.jsonl", &general, &err)) return Fail("gate", err);
    SlothRescorer::Options fo;
    fo.intraOpThreads = args_.threads;
    SlothRescorer factory(args_.modelDir, fo);
    SlothRescorer::Options co = fo;
    co.overridesPath = weightsPath;
    SlothRescorer candidate(args_.modelDir, co);
    if (!factory.ready() || !candidate.ready()) return Fail("gate", "R model not ready");
    if (!candidate.personalized()) return Fail("gate", "weights not applied: " + candidate.personalizationError());
    std::unique_ptr<SlothRescorer> active;
    if (!args_.activeWeights.empty()) {
      SlothRescorer::Options ao = fo;
      ao.overridesPath = args_.activeWeights;
      active = std::make_unique<SlothRescorer>(args_.modelDir, ao);
      if (!active->personalized()) {
        report_["active_error"] = active->personalizationError();
        active.reset();  // rule 3 skipped when the active version cannot be loaded
      }
    }
    NeuralWalkOptions wo;
    wo.alpha = alpha;
    size_t skipped = 0;
    auto walkCorrect = [&](SlothRescorer* model, const std::vector<std::string>& readings,
                           const std::string& target) {
      ReadingGrid grid(lm);
      for (const auto& r : readings)
        if (!grid.insertReading(r)) {
          ++skipped;
          return false;
        }
      auto w = NeuralWalk(grid, model, wo);
      std::string text;
      for (const auto& n : w.nodes) text += n->value();
      return text == target;
    };
    pz::GateInput gi;
    for (size_t i = 0; i < split.test.size(); ++i) {
      if (i % 50 == 0)
        if (int rc = Limits()) return rc;
      const auto& s = split.test[i];
      gi.factoryPersonal.push_back(walkCorrect(&factory, s.readings, s.target));
      gi.candidatePersonal.push_back(walkCorrect(&candidate, s.readings, s.target));
      if (active) gi.activePersonal.push_back(walkCorrect(active.get(), s.readings, s.target));
    }
    for (size_t i = 0; i < general.size(); ++i) {
      if (i % 50 == 0)
        if (int rc = Limits()) return rc;
      gi.factoryGeneral.push_back(walkCorrect(&factory, general[i].readings, general[i].text));
      gi.candidateGeneral.push_back(walkCorrect(&candidate, general[i].readings, general[i].text));
    }
    pz::GateResult g = pz::EvaluateGate(gi, pack.gate);
    report_["gate"] = {{"n_personal", g.nPersonal},
                       {"n_general", g.nGeneral},
                       {"factory_personal_pct", R3(g.factoryPersonalPct)},
                       {"candidate_personal_pct", R3(g.candidatePersonalPct)},
                       {"active_personal_pct", active ? json(R3(g.activePersonalPct)) : json(nullptr)},
                       {"factory_general_pct", R3(g.factoryGeneralPct)},
                       {"candidate_general_pct", R3(g.candidateGeneralPct)},
                       {"delta_personal_pt", R3(g.deltaPersonalPt)},
                       {"delta_general_pt", R3(g.deltaGeneralPt)},
                       {"wins", g.wins},
                       {"losses", g.losses},
                       {"p", g.p},
                       {"mcnemar", pack.gate.oneSided ? "one-sided" : "two-sided"},
                       {"failed_rules", g.failedRules},
                       {"unwalkable", skipped}};
    timing_["gate_s"] = R3(SecSince(ts));
    if (!g.valid) return End(kExitFailed, "failed:gate");
    os_log(Log(), "gate: personal %.2f -> %.2f, general %.2f -> %.2f, p %.4g, %{public}s",
           g.factoryPersonalPct, g.candidatePersonalPct, g.factoryGeneralPct,
           g.candidateGeneralPct, g.p, g.reason.c_str());
    if (!g.adopted) return End(kExitFinished, g.reason);

    // ---- precise-mode temperatures (D12), adopted candidates only -----------------
    Phase("calib");
    ts = Clock::now();
    std::vector<Sentence> calib;
    if (!ReadSentences(packDir + "/calib.jsonl", &calib, &err)) return Fail("calib", err);
    PreciseConfig pc;
    pc.alpha = alpha;
    pc.k = pack.calibK;
    std::vector<pz::CalibRow> rows;
    for (size_t i = 0; i < calib.size(); ++i) {
      if (i % 50 == 0)
        if (int rc = Limits()) return rc;
      ReadingGrid grid(lm);
      bool ok = true;
      for (const auto& r : calib[i].readings) ok = grid.insertReading(r) && ok;
      if (!ok) continue;
      auto scores = ComputeCharScores(grid, MakeNeuralScorer(&candidate), wo);
      auto set = PreciseCandidates(grid, scores, pc);
      if (set.empty()) continue;
      pz::CalibRow row;
      row.nSyl = calib[i].readings.size();
      row.rScores = set.rScores;
      row.top1Correct = set.texts[0] == calib[i].text;
      rows.push_back(std::move(row));
    }
    auto temps = pz::FitTemperatures(rows, pack.calibK, pack.bucketUpper, pack.perBucket);
    auto q = pz::EvaluateCalibration(rows, pack.calibK, pack.bucketUpper, temps);
    std::ostringstream conf;
    conf.precision(17);
    conf << "# personalized R temperatures (personal-finetune D12); overlay on precise.conf\n"
         << "run = " << args_.runId << "\nbucket_upper =";
    for (size_t b : pack.bucketUpper) conf << " " << b;
    conf << "\ntemperatures =";
    for (double t : temps) conf << " " << t;
    conf << "\n";
    if (!WriteFile600(stagingDir_ + "/precise-r.conf", conf.str()))
      return Fail("calib", "cannot write precise-r.conf");
    report_["calib"] = {{"n", rows.size()},
                        {"k", pack.calibK},
                        {"per_bucket", pack.perBucket},
                        {"temperatures", temps},
                        {"ece_before", R3(q.eceBefore)},
                        {"ece_after", R3(q.eceAfter)},
                        {"logloss_before", R3(q.logLossBefore)},
                        {"logloss_after", R3(q.logLossAfter)}};
    timing_["calib_s"] = R3(SecSince(ts));
    if (int rc = Limits()) return rc;
    return End(kExitFinished, "adopted");
  }

  const PipelineArgs& args_;
  const PipelineHooks& hooks_;
  json report_;
  json timing_ = json::object();
  std::string statusPath_, stagingDir_, phase_;
  size_t eligible_ = 0;
  uint64_t peakRss_ = 0;
  double lastPoll_ = -1;
  Clock::time_point t0_;
};

}  // namespace

int RunPipeline(const PipelineArgs& args, const PipelineHooks& hooks) {
  umask(077);
  Runner r(args, hooks);
  try {
    return r.Run();
  } catch (const std::exception& e) {
    os_log_error(Log(), "exception: %{public}s", e.what());
    return kExitFailed;
  }
}

int CheckPack(const std::string& packDir, const std::string& modelDir, std::string* summary) {
  Pack pack;
  std::string err;
  int rc = LoadPack(packDir, modelDir, &pack, &err, true);
  json j;
  if (rc == 0) {
    j = {{"ok", true},
         {"pack_version", pack.version},
         {"factory_sha256", pack.factoryShaHex},
         {"base_tensors", pack.base->tensors.size()}};
  } else {
    j = {{"ok", false}, {"error", err}};
  }
  if (summary) *summary = j.dump();
  return rc;
}

}  // namespace Trainer
}  // namespace McBopomofo
