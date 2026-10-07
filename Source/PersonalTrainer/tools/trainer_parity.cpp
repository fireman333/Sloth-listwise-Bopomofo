// Task 3.2 parity run: the C++ MLX trainer on the spike's pool-1,000 rows
// (fp32 clean_v1 base, 3,000 replay rows, 188 steps), merged + per-channel
// int8 quantized like the MLX Python run of research 16 §3, written as a
// weights.bin for `rescorer_bench --walk-mode --override-weights` (ORT eval).
//
//   trainer_parity <fixture_base_fp32.bin> <pack train-base.bin> <parity_rows.json>
//                  <model dir> <out weights.bin> [steps=188] [seed=0]

#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>

#include "Personalize/quantize.h"
#include "mlx_trainer.h"
#include "nlohmann/json.hpp"

using namespace McBopomofo::Trainer;
namespace pz = McBopomofo::Personalize;

int main(int argc, char** argv) {
  if (argc < 6) {
    fprintf(stderr, "usage: %s fp32_base pack_base rows.json model_dir out.bin [steps] [seed]\n", argv[0]);
    return 2;
  }
  std::string err;
  auto fp32 = ReadTrainBase(argv[1], &err);
  auto pack = ReadTrainBase(argv[2], &err);
  if (!fp32 || !pack) {
    fprintf(stderr, "%s\n", err.c_str());
    return 1;
  }
  const int steps = argc > 6 ? atoi(argv[6]) : 188;
  const uint64_t seed = argc > 7 ? strtoull(argv[7], nullptr, 10) : 0;
  std::ifstream in(argv[3]);
  std::stringstream ss;
  ss << in.rdbuf();
  auto j = nlohmann::json::parse(ss.str());
  auto rows = [](const nlohmann::json& a) {
    std::vector<EncodedRow> out;
    for (const auto& r : a) {
      EncodedRow e;
      e.syl = r["syl"].get<std::vector<int32_t>>();
      e.gold = r["gold"].get<std::vector<int32_t>>();
      e.cands = r["cands"].get<std::vector<std::vector<int32_t>>>();
      e.target = r["target"].get<int32_t>();
      e.ngram = r["ngram"].get<std::vector<float>>();
      out.push_back(std::move(e));
    }
    return out;
  };
  auto pool = rows(j["pool"]), replay = rows(j["replay"]);
  ModelVocab vocab;
  if (!LoadTrainerVocab(argv[4], fp32->config.nSyl, &vocab, &err)) {
    fprintf(stderr, "%s\n", err.c_str());
    return 1;
  }
  LoraTrainer t(fp32, vocab, Recipe());
  if (!t.ok()) {
    fprintf(stderr, "%s\n", t.error().c_str());
    return 1;
  }
  t.InitLora(seed);
  auto t0 = std::chrono::steady_clock::now();
  double first = 0;
  bool ok = t.Train(pool, replay, steps, seed, [](const TrainProgress& p) {
    if (p.step % 25 == 0 || p.step == p.steps) fprintf(stderr, "step %d loss %.4f\n", p.step, p.loss);
    return true;
  }, &first);
  double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  if (!ok) {
    fprintf(stderr, "train failed: %s\n", t.error().c_str());
    return 1;
  }
  // Merge on the fp32 base (as mlx_probe.py --save), quantize, name by the pack's ONNX names.
  pz::PersonalWeights pw;
  pw.factorySha = pack->factorySha;
  const char* attn[4] = {"q", "k", "v", "o"};
  const char* ffn[3] = {"w1", "w3", "w2"};
  size_t li = 0;
  for (int b = 0; b < fp32->config.depth; ++b)
    for (int k = 0; k < 7; ++k, ++li) {
      std::string name = "blocks." + std::to_string(b) + (k < 4 ? std::string(".attn.") + attn[k]
                                                                  : std::string(".ffn.") + ffn[k - 4]) +
                         ".weight";
      const BaseTensor* w = fp32->find(name);
      const BaseTensor* q = pack->find(name);
      if (!w || !q || q->kind != TensorKind::kLinearInt8) {
        fprintf(stderr, "missing %s\n", name.c_str());
        return 1;
      }
      const uint32_t out = w->dims[0], inn = w->dims[1];
      std::vector<float> wt(size_t(inn) * out);  // [in, out]
      for (uint32_t o = 0; o < out; ++o)
        for (uint32_t i = 0; i < inn; ++i) wt[size_t(i) * out + o] = w->data[size_t(o) * inn + i];
      auto A = t.loraA(li), B = t.loraB(li);
      pz::MergeLoRA(&wt, inn, out, A.data(), B.data(), 8, 2.0f);
      pz::QuantizedLinear ql;
      ql.name = q->linear.name;
      pz::QuantizePerChannel(wt.data(), inn, out, &ql);
      pw.tensors.push_back(std::move(ql));
    }
  if (!pz::WriteWeightsFile(argv[5], pw, &err)) {
    fprintf(stderr, "%s\n", err.c_str());
    return 1;
  }
  printf("{\"steps\":%d,\"first_loss\":%.4f,\"train_s\":%.1f,\"ms_per_step\":%.1f}\n", steps, first, sec,
         1000 * sec / steps);
  return 0;
}
