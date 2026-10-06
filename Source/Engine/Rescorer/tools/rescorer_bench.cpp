// rescorer_bench: end-to-end C++ parity benchmark for the rescoring pipeline.
//
// For every sentence of the test set it builds a real ReadingGrid on top of
// McBopomofoLM (data.txt), runs walk() / walkNBest(k), then scores the N-best
// with the Sloth (D) and Qwen (B) rescorers and the D->B cascade (DB).
//
// Usage: rescorer_bench [--testset f] [--data data.txt] [--out-dir dir]
//                       [--sloth-dir dir] [--qwen gguf] [--k 10] [--limit n]
//                       [--skip-sloth] [--skip-qwen]
// Outputs (in out-dir): result_cpp_{A,D,B,DB}.jsonl, cpp_nbest.jsonl

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "McBopomofoLM.h"
#include "gramambular2/reading_grid.h"
#include "nlohmann/json.hpp"
#include "rescorer.h"
#ifdef RESCORER_BENCH_WITH_SLOTH
#include "sloth_rescorer.h"
#endif
#ifdef RESCORER_BENCH_WITH_QWEN
#include "qwen_rescorer.h"
#endif

using json = nlohmann::json;
using Clock = std::chrono::steady_clock;
using Formosa::Gramambular2::ReadingGrid;

static double MsSince(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

static double Round3(double v) { return std::round(v * 1000.0) / 1000.0; }

static size_t ArgMax(const std::vector<double>& v) {
  return static_cast<size_t>(std::max_element(v.begin(), v.end()) - v.begin());
}

struct Item {
  std::string id, category, text;
  std::vector<std::string> reading;
};

int main(int argc, char** argv) {
  std::string root = BENCH_ROOT;  // "<repo>/" (AFM Input directory)
  std::string testset = root + "bench/data/testset.jsonl";
  std::string dataPath = root + "AFMBopomofo/Source/Data/data.txt";
  std::string outDir = root + "bench/out";
  std::string slothDir = root + "bench/out/_models/sloth";
  std::string qwenPath = root + "models/Qwen3-0.6B-Base-Q8_0.gguf";
  size_t k = 10;
  size_t limit = 0;
  bool skipSloth = false, skipQwen = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        fprintf(stderr, "missing value for %s\n", a.c_str());
        exit(2);
      }
      return argv[++i];
    };
    if (a == "--testset") testset = next();
    else if (a == "--data") dataPath = next();
    else if (a == "--out-dir") outDir = next();
    else if (a == "--sloth-dir") slothDir = next();
    else if (a == "--qwen") qwenPath = next();
    else if (a == "--k") k = std::stoul(next());
    else if (a == "--limit") limit = std::stoul(next());
    else if (a == "--skip-sloth") skipSloth = true;
    else if (a == "--skip-qwen") skipQwen = true;
    else {
      fprintf(stderr, "unknown arg %s\n", a.c_str());
      return 2;
    }
  }

  // ---- load test set
  std::vector<Item> items;
  {
    std::ifstream f(testset);
    if (!f) { fprintf(stderr, "cannot open %s\n", testset.c_str()); return 1; }
    std::string line;
    while (std::getline(f, line)) {
      if (line.empty()) continue;
      json j = json::parse(line);
      Item it;
      it.id = j["id"];
      it.category = j.value("category", "");
      it.text = j["text"];
      it.reading = j["reading"].get<std::vector<std::string>>();
      items.push_back(std::move(it));
      if (limit && items.size() >= limit) break;
    }
  }

  // ---- language model
  auto t0 = Clock::now();
  auto lm = std::make_shared<McBopomofo::McBopomofoLM>();
  lm->loadLanguageModel(dataPath.c_str());
  if (!lm->isDataModelLoaded()) {
    fprintf(stderr, "cannot load LM %s\n", dataPath.c_str());
    return 1;
  }
  fprintf(stderr, "LM loaded in %.0f ms\n", MsSince(t0));

  // ---- rescorers
  std::unique_ptr<McBopomofo::Rescorer> sloth, qwen;
#ifdef RESCORER_BENCH_WITH_SLOTH
  if (!skipSloth) {
    t0 = Clock::now();
    sloth = std::make_unique<McBopomofo::SlothRescorer>(slothDir);
    fprintf(stderr, "Sloth %s (%.0f ms)\n", sloth->ready() ? "ready" : "FAILED", MsSince(t0));
    if (!sloth->ready()) sloth.reset();
  }
#endif
#ifdef RESCORER_BENCH_WITH_QWEN
  if (!skipQwen) {
    t0 = Clock::now();
    qwen = std::make_unique<McBopomofo::QwenRescorer>(qwenPath);
    fprintf(stderr, "Qwen %s (%.0f ms)\n", qwen->ready() ? "ready" : "FAILED", MsSince(t0));
    if (!qwen->ready()) qwen.reset();
  }
#endif

  auto openOut = [&](const std::string& name) {
    return std::ofstream(outDir + "/" + name, std::ios::binary);
  };
  std::ofstream fNb = openOut("cpp_nbest.jsonl");
  std::ofstream fA = openOut("result_cpp_A.jsonl");
  std::ofstream fD, fB, fDB;
  if (sloth) fD = openOut("result_cpp_D.jsonl");
  if (qwen) fB = openOut("result_cpp_B.jsonl");
  if (sloth && qwen) fDB = openOut("result_cpp_DB.jsonl");

  // Warm-up (not recorded): one real-sized request per rescorer.
  {
    McBopomofo::RescoreRequest w;
    w.readings = items[0].reading;
    w.candidates = {items[0].text, items[0].text};
    if (sloth) sloth->score(w);
    if (qwen) { qwen->score(w); qwen->score(w); }
  }

  size_t skipped = 0;
  for (const Item& it : items) {
    // ---- grid + walk (A) and N-best
    auto tg = Clock::now();
    ReadingGrid grid(lm);
    bool ok = true;
    for (const std::string& r : it.reading) ok = grid.insertReading(r) && ok;
    if (!ok) {
      fprintf(stderr, "WARN %s: unknown reading, skipped\n", it.id.c_str());
      ++skipped;
      continue;
    }
    auto walk = grid.walk();
    double tWalk = MsSince(tg);
    std::string a;
    for (auto& n : walk.nodes) a += n->value();

    auto tn = Clock::now();
    auto nbest = grid.walkNBest(k);
    double tNbest = MsSince(tn);
    std::vector<std::string> cands;
    json nbJson = json::array();
    for (auto& r : nbest) {
      std::string text;
      for (auto& n : r.nodes) text += n->value();
      cands.push_back(text);
      nbJson.push_back({{"text", text},
                        {"segments", r.valuesAsStrings()},
                        {"ngram_score", std::round(r.score * 1e6) / 1e6}});
    }
    fNb << json{{"id", it.id}, {"gold", it.text}, {"category", it.category},
                {"reading", it.reading}, {"decode_ms", Round3(tNbest)},
                {"nbest", nbJson}}.dump() << "\n";
    fA << json{{"id", it.id}, {"choice", a}, {"latency_ms", Round3(tWalk)}}.dump() << "\n";

    McBopomofo::RescoreRequest req;
    req.readings = it.reading;
    req.candidates = cands;

    // ---- D
    std::vector<double> dScores;
    double tD = 0;
    size_t dBest = 0;
    if (sloth) {
      auto t = Clock::now();
      dScores = sloth->score(req);
      tD = MsSince(t);
      if (dScores.size() != cands.size()) { dScores.assign(cands.size(), 0.0); }
      dBest = ArgMax(dScores);
      fD << json{{"id", it.id}, {"choice", cands[dBest]}, {"latency_ms", Round3(tD)},
                 {"scores", dScores}}.dump() << "\n";
    }

    // ---- B
    std::vector<double> bScores;
    double tB = 0;
    if (qwen) {
      auto t = Clock::now();
      bScores = qwen->score(req);
      tB = MsSince(t);
      if (bScores.size() != cands.size()) { bScores.assign(cands.size(), 0.0); }
      fB << json{{"id", it.id}, {"choice", cands[ArgMax(bScores)]},
                 {"latency_ms", Round3(tB)}, {"scores", bScores}}.dump() << "\n";
    }

    // ---- DB cascade: D orders the candidates, B re-ranks them; final = B.
    if (sloth && qwen) {
      std::vector<size_t> order(cands.size());
      std::iota(order.begin(), order.end(), 0);
      std::stable_sort(order.begin(), order.end(),
                       [&](size_t x, size_t y) { return dScores[x] > dScores[y]; });
      McBopomofo::RescoreRequest req2;
      req2.readings = it.reading;
      for (size_t o : order) req2.candidates.push_back(cands[o]);
      auto t = Clock::now();
      std::vector<double> s2 = qwen->score(req2);
      double tB2 = MsSince(t);
      std::string choice = cands[dBest];
      if (s2.size() == req2.candidates.size()) choice = req2.candidates[ArgMax(s2)];
      fDB << json{{"id", it.id}, {"choice", choice}, {"latency_ms", Round3(tD + tB2)},
                  {"latency_d_ms", Round3(tD)}, {"latency_b_ms", Round3(tB2)},
                  {"d_choice", cands[dBest]}}.dump() << "\n";
    }
  }
  fprintf(stderr, "done: %zu items, %zu skipped\n", items.size() - skipped, skipped);
  return 0;
}
