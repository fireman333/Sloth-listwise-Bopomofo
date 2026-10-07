// Tests of the trainer helper (personal-finetune tasks 3.2 - 3.5).

#include <fcntl.h>
#include <gtest/gtest.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>

#include "Personalize/personal_weights.h"
#include "Personalize/pool_reader.h"
#include "mlx_trainer.h"
#include "nlohmann/json.hpp"
#include "pipeline.h"
#include "system_probes.h"
#include "train_base.h"
#include "train_data.h"

namespace fs = std::filesystem;
using nlohmann::json;
using namespace McBopomofo::Trainer;

namespace {

std::string ReadText(const std::string& p) {
  std::ifstream in(p, std::ios::binary);
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

const ModelVocab& Vocab() {
  static ModelVocab v = [] {
    ModelVocab m;
    std::string err;
    EXPECT_TRUE(LoadTrainerVocab(RWALK_MODEL_DIR, 1539, &m, &err)) << err;
    return m;
  }();
  return v;
}

std::vector<EncodedRow> FixedRows(json* meta) {
  json j = json::parse(ReadText(std::string(TRAINER_TESTDATA) + "/fixed_batch.json"));
  std::vector<EncodedRow> rows;
  for (const auto& r : j["rows"]) {
    EncodedRow e;
    e.syl = r["syl"].get<std::vector<int32_t>>();
    e.gold = r["gold"].get<std::vector<int32_t>>();
    e.cands = r["cands"].get<std::vector<std::vector<int32_t>>>();
    e.target = r["target"].get<int32_t>();
    e.ngram = r["ngram"].get<std::vector<float>>();
    rows.push_back(std::move(e));
  }
  j.erase("rows");
  *meta = j;
  return rows;
}

// Fixture base (fp32 clean_v1) with the torch LoRA A matrices stored as lora_A.<i>.
std::shared_ptr<const TrainBase> FixtureBase(std::vector<std::vector<float>>* loraA) {
  std::string err;
  auto base = ReadTrainBase(FIXTURE_BASE_FP32, &err);
  EXPECT_TRUE(base) << err << " (run Source/PersonalTrainer/tools/make_trainer_fixture.py)";
  if (!base) return nullptr;
  loraA->clear();
  for (int i = 0; i < 84; ++i) {
    const BaseTensor* t = base->find("lora_A." + std::to_string(i));
    EXPECT_NE(t, nullptr);
    if (t) loraA->push_back(t->data);
  }
  return base;
}

std::string TempDir(const char* tag) {
  std::string d = (fs::temp_directory_path() / ("afm-trainer-" + std::string(tag) + "-" +
                                                std::to_string(getpid())))
                      .string();
  fs::remove_all(d);
  fs::create_directories(d);
  return d;
}

}  // namespace

// --- 3.2 numerics ---------------------------------------------------------------

TEST(TrainBaseTest, RoundTripAndCorruption) {
  TrainBase b;
  b.config = {10, 20, 8, 1, 2, 1, 16};
  BaseTensor f;
  f.name = "norm.w";
  f.dims = {8};
  f.data.assign(8, 1.5f);
  BaseTensor l;
  l.name = "blocks.0.attn.q.weight";
  l.kind = TensorKind::kLinearInt8;
  l.dims = {3, 2};  // [out, in]
  l.linear.name = "val_1";
  l.linear.in = 2;
  l.linear.out = 3;
  l.linear.scales = {0.5f, 1.0f, 2.0f};
  l.linear.values = {1, -2, 3, 4, -5, 6};  // [in, out]
  BaseTensor u;
  u.name = "embed.weight";
  u.kind = TensorKind::kUint8Zp;
  u.dims = {2, 2};
  u.scale = 0.25f;
  u.zeroPoint = 128;
  u.u8 = {128, 132, 124, 255};
  b.tensors = {f, l, u};
  std::string s = SerializeTrainBase(b);
  std::string err;
  auto r = ParseTrainBase(s, &err);
  ASSERT_TRUE(r) << err;
  EXPECT_EQ(SerializeTrainBase(*r), s);
  // torch layout of the linear: W[j][i] = q[i][j] * s[j]
  auto w = DequantizedTorchLayout(*r->find("blocks.0.attn.q.weight"));
  EXPECT_FLOAT_EQ(w[0 * 2 + 1], 4 * 0.5f);   // j=0, i=1
  EXPECT_FLOAT_EQ(w[2 * 2 + 0], 3 * 2.0f);   // j=2, i=0
  auto e = DequantizedTorchLayout(*r->find("embed.weight"));
  EXPECT_FLOAT_EQ(e[1], 1.0f);
  EXPECT_FLOAT_EQ(e[2], -1.0f);
  for (size_t cut : {size_t(10), s.size() / 2, s.size() - 1}) {
    EXPECT_FALSE(ParseTrainBase(s.substr(0, cut), &err));
  }
  std::string bad = s;
  bad[20] ^= 1;
  EXPECT_FALSE(ParseTrainBase(bad, &err));
}

TEST(TrainDataTest, EncodeMatchesFixtureRow) {
  // pool row 0 of the fixed batch re-encoded from text with the C++ vocab is
  // checked indirectly by the loss test; here: unknown syllable -> 1, OOV char -> -1 / -100.
  auto r = EncodeRow(Vocab(), {"ㄋㄧˇ", "zz"}, "你\uE000", {"你\uE000", "妳\uE000"}, {-1.0, -2.0});
  EXPECT_EQ(r.syl[1], 1);
  EXPECT_EQ(r.target, 0);
  EXPECT_EQ(r.gold[1], -100);
  EXPECT_EQ(r.cands[0][1], -1);
  EXPECT_GE(r.cands[1][0], 0);
  EXPECT_FLOAT_EQ(r.ngram[1], -2.0f);
}

TEST(MlxTrainerTest, FixedBatchMatchesPyTorch) {
  json meta;
  auto rows = FixedRows(&meta);
  ASSERT_EQ(rows.size(), 32u);
  std::vector<std::vector<float>> A;
  auto base = FixtureBase(&A);
  ASSERT_TRUE(base);
  LoraTrainer t(base, Vocab(), Recipe());
  ASSERT_TRUE(t.ok()) << t.error();
  ASSERT_TRUE(t.SetLoraA(A));
  EXPECT_EQ(t.trainableParameters(), 454656u);
  std::vector<const EncodedRow*> batch;
  for (auto& r : rows) batch.push_back(&r);
  auto v = t.Evaluate(batch);
  // 16 §3 / research 16: loss 0.6777, LoRA-B grad norm 2.5893 (4 decimals).
  EXPECT_NEAR(v.loss, meta["loss"].get<double>(), 5e-5) << v.loss;
  EXPECT_NEAR(v.gradNormB, meta["gradnorm_B"].get<double>(), 5e-5) << v.gradNormB;
  EXPECT_NEAR(v.listwise, meta["l_list"].get<double>(), 1e-4);
  EXPECT_NEAR(v.mle, meta["l_mle"].get<double>(), 1e-4);
  printf("fixed batch: loss %.5f (torch %.5f) gradnorm_B %.5f (torch %.5f)\n", v.loss,
         meta["loss"].get<double>(), v.gradNormB, meta["gradnorm_B"].get<double>());
}

TEST(MlxTrainerTest, ScheduleMatchesPython) {
  Recipe r;
  const int steps = 188;  // warm = 9
  EXPECT_NEAR(ScheduledLr(r, 0, steps), 5e-4 / 9, 1e-12);
  EXPECT_NEAR(ScheduledLr(r, 8, steps), 5e-4, 1e-12);
  EXPECT_NEAR(ScheduledLr(r, 9, steps), 5e-4, 1e-12);
  EXPECT_NEAR(ScheduledLr(r, 187, steps),
              5e-4 * (0.1 + 0.9 * 0.5 * (1 + std::cos(M_PI * 178.0 / 179.0))), 1e-12);
  EXPECT_EQ(StepsFor(r, 1000), 188);
  EXPECT_EQ(StepsFor(r, 100), 60);
}

TEST(MlxTrainerTest, NonFiniteLossAborts) {
  json meta;
  auto rows = FixedRows(&meta);
  std::vector<std::vector<float>> A;
  auto base = FixtureBase(&A);
  ASSERT_TRUE(base);
  Recipe r;
  r.lr = 1e30;  // diverges within a few steps
  r.batchReplay = 0;
  r.warmupFrac = 0;
  LoraTrainer t(base, Vocab(), r);
  ASSERT_TRUE(t.ok());
  int steps = 0;
  bool ok = t.Train(rows, {}, 20, 1, [&](const TrainProgress& p) {
    steps = p.step;
    return true;
  });
  EXPECT_FALSE(ok);
  EXPECT_EQ(t.error(), "non-finite loss");
  EXPECT_LT(steps, 20);
}

// --- 3.3 / 3.4 pipeline ----------------------------------------------------------

namespace {

struct SimRow {
  std::vector<std::string> reading;
  std::string text;
};

std::vector<SimRow> LoadSim(const std::string& file, size_t n) {
  std::vector<SimRow> out;
  std::ifstream in(std::string(SIM_DATA_DIR) + "/" + file);
  std::string line;
  while (out.size() < n && std::getline(in, line)) {
    json j = json::parse(line);
    out.push_back({j["reading"].get<std::vector<std::string>>(), j["text"].get<std::string>()});
  }
  return out;
}

constexpr double kT0 = 1790000000.0;

// Simulated pool in the IME's record format: pool rows (oldest) then test rows.
std::string WriteSimPool(const char* tag, size_t nPool, size_t nTest) {
  std::string dir = TempDir(tag);
  std::string mem = dir + "/memory";
  fs::create_directories(mem);
  chmod(mem.c_str(), 0700);
  auto rows = LoadSim("user_pool.jsonl", nPool);
  auto test = LoadSim("user_test.jsonl", nTest);
  rows.insert(rows.end(), test.begin(), test.end());
  std::ofstream f(mem + "/sentences.jsonl", std::ios::binary);
  for (size_t i = 0; i < rows.size(); ++i) {
    json nodes = json::array();
    auto cps = McBopomofo::Personalize::CodePoints(rows[i].text);
    for (size_t k = 0; k < rows[i].reading.size(); ++k)
      nodes.push_back({rows[i].reading[k], cps[k], "w"});
    char t[32];
    snprintf(t, sizeof t, "%.3f", kT0 + 60.0 * i);
    f << "{\"v\":1,\"t\":" << t << ",\"manual\":false,\"walk\":\"r\",\"app\":\"sim\","
      << "\"model\":\"m\",\"nodes\":" << nodes.dump() << "}\n";
  }
  return mem;
}

PipelineArgs BaseArgs(const std::string& mem, const char* run) {
  PipelineArgs a;
  a.memoryDir = mem;
  a.modelDir = RWALK_MODEL_DIR;
  a.packDir = PACK_DIR;
  a.dataTxt = DATA_TXT;
  a.runId = run;
  a.maxSteps = 4;
  a.replayLimit = 100;
  return a;
}

PipelineHooks QuietHooks() {
  PipelineHooks h;
  h.probes.onAC = [] { return true; };
  h.probes.lowPowerMode = [] { return false; };
  h.probes.thermalState = [] { return 0; };
  h.probes.residentBytes = SystemProbes::Real().residentBytes;
  return h;
}

json Report(const std::string& mem, const char* run) {
  return json::parse(ReadText(mem + "/r-personal/staging/" + run + "/report.json"));
}

}  // namespace

TEST(PipelineTest, CheckPack) {
  std::string s;
  EXPECT_EQ(CheckPack(PACK_DIR, RWALK_MODEL_DIR, &s), 0) << s;
  EXPECT_EQ(CheckPack(std::string(PACK_DIR) + "/missing", RWALK_MODEL_DIR, &s), kExitPack);
  // A pack for another factory model is refused.
  std::string other = TempDir("othermodel");
  fs::create_directories(other + "/onnx");
  std::ofstream(other + "/onnx/model_quantized.onnx") << "not the factory model";
  EXPECT_EQ(CheckPack(PACK_DIR, other, &s), kExitPack);
  EXPECT_NE(s.find("another factory model"), std::string::npos) << s;
  fs::remove_all(fs::path(other).parent_path() / fs::path(other).filename());
}

TEST(PipelineTest, NotEnoughData) {
  std::string mem = WriteSimPool("small", 500, 0);
  EXPECT_EQ(RunPipeline(BaseArgs(mem, "small"), QuietHooks()), kExitNotEnoughData);
  json r = Report(mem, "small");
  EXPECT_EQ(r["result"], "not_enough_data");
  EXPECT_LT(r["eligible"].get<int>(), 1000);
}

TEST(PipelineTest, DeferredInAutomaticMode) {
  std::string mem = WriteSimPool("defer", 1000, 400);
  struct Case {
    const char* run;
    bool ac, lpm;
    int thermal;
    const char* why;
  } cases[] = {{"d-bat", false, false, 0, "battery"},
               {"d-lpm", true, true, 0, "low_power_mode"},
               {"d-hot", true, false, 2, "thermal"}};
  for (const auto& c : cases) {
    PipelineArgs a = BaseArgs(mem, c.run);
    a.automatic = true;
    PipelineHooks h = QuietHooks();
    h.probes.onAC = [&] { return c.ac; };
    h.probes.lowPowerMode = [&] { return c.lpm; };
    h.probes.thermalState = [&] { return c.thermal; };
    EXPECT_EQ(RunPipeline(a, h), kExitDeferred) << c.run;
    json r = Report(mem, c.run);
    EXPECT_EQ(r["result"], "deferred");
    EXPECT_EQ(r["deferred_because"], c.why);
  }
  // Manual runs ignore power conditions (spec "Training triggers"); a mid-run
  // power loss in an automatic run defers at the next poll.
  PipelineArgs a = BaseArgs(mem, "d-mid");
  a.automatic = true;
  PipelineHooks h = QuietHooks();
  std::atomic<bool> unplugged{false};
  h.pollSeconds = 0;
  h.probes.onAC = [&] { return !unplugged.load(); };
  h.onTrainStep = [&](int step) {
    if (step == 2) unplugged = true;
  };
  EXPECT_EQ(RunPipeline(a, h), kExitDeferred);
  json r = Report(mem, "d-mid");
  EXPECT_EQ(r["train"]["steps"], 4);
  EXPECT_FALSE(fs::exists(mem + "/r-personal/staging/d-mid/weights.bin"));
}

TEST(PipelineTest, MemoryLimitAborts) {
  std::string mem = WriteSimPool("rss", 1000, 400);
  PipelineHooks h = QuietHooks();
  h.probes.residentBytes = [] { return uint64_t(3) << 30; };  // 3 GB > 2 GB
  EXPECT_EQ(RunPipeline(BaseArgs(mem, "rss"), h), kExitMemory);
  EXPECT_EQ(Report(mem, "rss")["result"], "failed:memory");
}

TEST(PipelineTest, CancelDuringTrain) {
  std::string mem = WriteSimPool("cancel", 1000, 400);
  std::atomic<bool> cancel{false};
  PipelineHooks h = QuietHooks();
  h.cancel = &cancel;
  h.onTrainStep = [&](int step) {
    if (step == 2) cancel = true;
  };
  EXPECT_EQ(RunPipeline(BaseArgs(mem, "cancel"), h), kExitCancelled);
  EXPECT_EQ(Report(mem, "cancel")["result"], "cancelled");
}

TEST(PipelineTest, RetractionDuringTrainDiscardsRun) {
  std::string mem = WriteSimPool("retract", 1000, 400);
  // The newest record belongs to the personal test set.
  char t[32];
  snprintf(t, sizeof t, "%.3f", kT0 + 60.0 * 1399);
  PipelineHooks h = QuietHooks();
  h.onTrainStep = [&](int step) {
    if (step == 1) {
      std::ofstream f(mem + "/sentences.jsonl", std::ios::app);
      f << "{\"v\":1,\"retract\":" << t << "}\n";
    }
  };
  EXPECT_EQ(RunPipeline(BaseArgs(mem, "retract"), h), kExitRetracted);
  json r = Report(mem, "retract");
  EXPECT_EQ(r["result"], "discarded:retracted");
  EXPECT_EQ(r["late_retractions"], 1);
  // The scan offset is the pool size before the retraction was appended.
  EXPECT_LT(r["scan_offset"].get<uint64_t>(), fs::file_size(mem + "/sentences.jsonl"));
}

TEST(PipelineTest, StagingOutsideMemoryRefused) {
  std::string mem = WriteSimPool("outside", 10, 0);
  PipelineArgs a = BaseArgs(mem, "outside");
  a.stagingDir = TempDir("elsewhere");
  EXPECT_EQ(RunPipeline(a, QuietHooks()), kExitFailed);
}

// SIGTERM to the real helper during `train` -> exit 5 within 2 s.
TEST(HelperProcessTest, SigtermDuringTrainExitsWithin2s) {
  std::string mem = WriteSimPool("sigterm", 1000, 400);
  pid_t pid = fork();
  ASSERT_GE(pid, 0);
  if (pid == 0) {
    int devnull = open("/dev/null", O_WRONLY);
    dup2(devnull, 1);
    dup2(devnull, 2);
    execl(TRAINER_EXE, TRAINER_EXE, "--memory-dir", mem.c_str(), "--model-dir", RWALK_MODEL_DIR,
          "--pack-dir", PACK_DIR, "--data-txt", DATA_TXT, "--run-id", "sigterm", "--replay-limit",
          "200", (char*)nullptr);
    _exit(127);
  }
  const std::string status = mem + "/r-personal/status.json";
  bool inTrain = false;
  for (int i = 0; i < 1200 && !inTrain; ++i) {  // <= 120 s
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    json s = json::parse(ReadText(status), nullptr, false);
    inTrain = !s.is_discarded() && s.value("phase", "") == "train" && s.value("step", 0) >= 10;
    int st;
    if (waitpid(pid, &st, WNOHANG) == pid) FAIL() << "helper exited before train";
  }
  ASSERT_TRUE(inTrain);
  auto t0 = std::chrono::steady_clock::now();
  kill(pid, SIGTERM);
  int st = 0;
  waitpid(pid, &st, 0);
  double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  printf("SIGTERM -> exit in %.3f s\n", sec);
  EXPECT_LE(sec, 2.0);
  ASSERT_TRUE(WIFEXITED(st));
  EXPECT_EQ(WEXITSTATUS(st), kExitCancelled);
  EXPECT_EQ(Report(mem, "sigterm")["result"], "cancelled");
}
