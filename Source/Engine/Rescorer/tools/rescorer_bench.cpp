// rescorer_bench: end-to-end C++ parity benchmark for the rescoring pipeline.
//
// For every sentence of the test set it builds a real ReadingGrid on top of
// McBopomofoLM (data.txt), runs walk() / walkNBest(k), then scores the N-best
// with the Sloth (D) and Qwen (B) rescorers and the D->B cascade (DB).
//
// Usage: rescorer_bench [--testset f] [--data data.txt] [--out-dir dir]
//                       [--sloth-dir dir] [--qwen gguf] [--electra-dir dir]
//                       [--k 10] [--limit n] [--skip-sloth] [--skip-qwen]
//                       [--nbest file.jsonl] [--tag _lex]
//                       [--w 0.5] [--sloth-threads 1] [--electra-threads 1]
//                       [--electra-onnx model_int8.onnx]
//                       [--parallel] [--latency 1,2,4]
// Outputs (in out-dir): result_cpp_{A,D,B,DB,E,DE}<tag>.jsonl, cpp_nbest.jsonl
//   --nbest   take candidates from an existing nbest jsonl (e.g. out/nbest_lex.jsonl)
//             instead of walkNBest (no grid; A = nbest[0]); use with --tag.
//   --electra-dir enables E and the D+E ensemble (z-score, weight w on D, 1-w on E).
//   --latency "t1,t2,.." instead of the normal run: time D / E / D+E (sequential and
//             parallel) on the 10-candidate sentences for each thread count; prints a
//             JSON summary with the machine load average.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <map>
#include <memory>
#include <numeric>
#include <thread>
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
#ifdef RESCORER_BENCH_WITH_ELECTRA
#include "electra_rescorer.h"
#include "ensemble_rescorer.h"
#endif
#include <cmath>
#include <cstdlib>
#include <ctime>

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

static double Pct(std::vector<double> v, double q) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  size_t i = static_cast<size_t>(std::llround(q * (v.size() - 1)));
  return v[std::min(i, v.size() - 1)];
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
  bool skipSloth = false, skipQwen = false, parallel = false;
  std::string electraOnnx = "model_int8.onnx";
  std::string electraDir, nbestFile, tag, latencyList;
  double wD = 0.5;
  int slothThreads = 1, electraThreads = 1;
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
    else if (a == "--electra-dir") electraDir = next();
    else if (a == "--electra-onnx") electraOnnx = next();
    else if (a == "--nbest") nbestFile = next();
    else if (a == "--tag") tag = next();
    else if (a == "--w") wD = std::stod(next());
    else if (a == "--sloth-threads") slothThreads = std::stoi(next());
    else if (a == "--electra-threads") electraThreads = std::stoi(next());
    else if (a == "--parallel") parallel = true;
    else if (a == "--latency") latencyList = next();
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
    McBopomofo::SlothRescorer::Options so;
    so.intraOpThreads = slothThreads;
    sloth = std::make_unique<McBopomofo::SlothRescorer>(slothDir, so);
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

  std::shared_ptr<McBopomofo::Rescorer> slothS(std::move(sloth));
  sloth.reset();
  std::shared_ptr<McBopomofo::Rescorer> electra;
  std::unique_ptr<McBopomofo::EnsembleRescorer> de;
#ifdef RESCORER_BENCH_WITH_ELECTRA
  if (!electraDir.empty()) {
    t0 = Clock::now();
    McBopomofo::ElectraRescorer::Options eo;
    eo.intraOpThreads = electraThreads;
    eo.onnxFile = electraOnnx;
    electra = std::make_shared<McBopomofo::ElectraRescorer>(electraDir, eo);
    fprintf(stderr, "Electra %s (%.0f ms)\n", electra->ready() ? "ready" : "FAILED", MsSince(t0));
    if (!electra->ready()) electra.reset();
  }
  if (electra && slothS) {
    McBopomofo::EnsembleRescorer::Options eo;
    eo.parallel = parallel;
    de = std::make_unique<McBopomofo::EnsembleRescorer>(
        std::vector<McBopomofo::EnsembleRescorer::Member>{{slothS, wD, "D"}, {electra, 1.0 - wD, "E"}}, eo);
  }
#endif

  // ---- nbest override (candidates from file)
  std::map<std::string, std::vector<std::string>> nbFromFile;
  if (!nbestFile.empty()) {
    std::ifstream f(nbestFile);
    if (!f) { fprintf(stderr, "cannot open %s\n", nbestFile.c_str()); return 1; }
    std::string line;
    while (std::getline(f, line)) {
      if (line.empty()) continue;
      json j = json::parse(line);
      auto& v = nbFromFile[j["id"].get<std::string>()];
      for (auto& c : j["nbest"]) {
        if (v.size() < k) v.push_back(c["text"].get<std::string>());
      }
    }
  }

  // ---- latency mode
  if (!latencyList.empty()) {
#if defined(RESCORER_BENCH_WITH_ELECTRA) && defined(RESCORER_BENCH_WITH_SLOTH)
    std::vector<int> threads;
    {
      size_t p = 0;
      while (p < latencyList.size()) {
        size_t q = latencyList.find(',', p);
        if (q == std::string::npos) q = latencyList.size();
        threads.push_back(std::stoi(latencyList.substr(p, q - p)));
        p = q + 1;
      }
    }
    // 10-candidate requests: take real candidates from walkNBest.
    std::vector<McBopomofo::RescoreRequest> reqs;
    for (const Item& it : items) {
      std::vector<std::string> cands;
      if (!nbFromFile.empty()) {
        cands = nbFromFile[it.id];
      } else {
        ReadingGrid grid(lm);
        bool ok = true;
        for (const std::string& r : it.reading) ok = grid.insertReading(r) && ok;
        if (!ok) continue;
        for (auto& r : grid.walkNBest(10)) {
          std::string text;
          for (auto& n : r.nodes) text += n->value();
          cands.push_back(text);
        }
      }
      if (cands.size() != 10) continue;
      McBopomofo::RescoreRequest rq;
      rq.readings = it.reading;
      rq.candidates = cands;
      reqs.push_back(std::move(rq));
    }
    double la[3];
    getloadavg(la, 3);
    json out = json::object();
    out["n_requests"] = reqs.size();
    out["loadavg_start"] = {la[0], la[1], la[2]};
    out["hw_threads"] = std::thread::hardware_concurrency();
    json rows = json::array();
    for (int th : threads) {
      McBopomofo::SlothRescorer::Options so2;
      so2.intraOpThreads = th;
      McBopomofo::ElectraRescorer::Options eo2;
      eo2.intraOpThreads = th;
      eo2.onnxFile = electraOnnx;
      auto d2 = std::make_shared<McBopomofo::SlothRescorer>(slothDir, so2);
      auto e2 = std::make_shared<McBopomofo::ElectraRescorer>(electraDir, eo2);
      if (!d2->ready() || !e2->ready()) { fprintf(stderr, "latency: model load failed\n"); return 1; }
      McBopomofo::EnsembleRescorer seq({{d2, wD, "D"}, {e2, 1.0 - wD, "E"}});
      McBopomofo::EnsembleRescorer::Options po;
      po.parallel = true;
      McBopomofo::EnsembleRescorer par({{d2, wD, "D"}, {e2, 1.0 - wD, "E"}}, po);
      for (int i = 0; i < 10; ++i) { d2->score(reqs[0]); e2->score(reqs[0]); seq.score(reqs[0]); par.score(reqs[0]); }
      std::vector<double> ld, le, ls, lp, cd, ce, cs, cp;  // wall ms, process-CPU ms
      auto cpuNow = [] { return 1000.0 * static_cast<double>(std::clock()) / CLOCKS_PER_SEC; };
      auto timeIt = [&](auto&& fn, std::vector<double>& wall, std::vector<double>& cpu) {
        double c0 = cpuNow();
        auto t = Clock::now();
        fn();
        wall.push_back(MsSince(t));
        cpu.push_back(cpuNow() - c0);
      };
      for (int rep = 0; rep < 3; ++rep) {
        for (auto& rq : reqs) {
          timeIt([&] { d2->score(rq); }, ld, cd);
          timeIt([&] { e2->score(rq); }, le, ce);
          timeIt([&] { seq.score(rq); }, ls, cs);
          timeIt([&] { par.score(rq); }, lp, cp);
        }
      }
      auto row = [&](const char* name, std::vector<double>& v, std::vector<double>& c) {
        return json{{"model", name}, {"threads", th}, {"p50", Round3(Pct(v, .5))}, {"p95", Round3(Pct(v, .95))},
                    {"max", Round3(Pct(v, 1.0))}, {"cpu_p50", Round3(Pct(c, .5))}, {"cpu_p95", Round3(Pct(c, .95))},
                    {"n", v.size()}};
      };
      rows.push_back(row("D", ld, cd));
      rows.push_back(row("E", le, ce));
      rows.push_back(row("D+E_sequential", ls, cs));
      rows.push_back(row("D+E_parallel", lp, cp));
    }
    getloadavg(la, 3);
    out["loadavg_end"] = {la[0], la[1], la[2]};
    out["rows"] = rows;
    printf("%s\n", out.dump(1).c_str());
    return 0;
#else
    fprintf(stderr, "latency mode needs Sloth + Electra backends\n");
    return 2;
#endif
  }

  auto openOut = [&](const std::string& name) {
    return std::ofstream(outDir + "/" + name, std::ios::binary);
  };
  std::ofstream fNb, fA, fD, fB, fDB, fE, fDE;
  if (nbFromFile.empty()) { fNb = openOut("cpp_nbest.jsonl"); fA = openOut("result_cpp_A" + tag + ".jsonl"); }
  McBopomofo::Rescorer* sloth_ = slothS.get();
  if (sloth_) fD = openOut("result_cpp_D" + tag + ".jsonl");
  if (qwen) fB = openOut("result_cpp_B" + tag + ".jsonl");
  if (sloth_ && qwen) fDB = openOut("result_cpp_DB" + tag + ".jsonl");
  if (electra) fE = openOut("result_cpp_E" + tag + ".jsonl");
  if (de) fDE = openOut("result_cpp_DE" + tag + ".jsonl");

  // Warm-up (not recorded): one real-sized request per rescorer.
  {
    McBopomofo::RescoreRequest w;
    w.readings = items[0].reading;
    w.candidates = {items[0].text, items[0].text};
    if (sloth_) sloth_->score(w);
    if (electra) { electra->score(w); }
    if (de) { de->score(w); }
    if (qwen) { qwen->score(w); qwen->score(w); }
  }

  size_t skipped = 0;
  for (const Item& it : items) {
    std::vector<std::string> cands;
    if (!nbFromFile.empty()) {
      auto fit = nbFromFile.find(it.id);
      if (fit == nbFromFile.end() || fit->second.empty()) {
        fprintf(stderr, "WARN %s: not in nbest file, skipped\n", it.id.c_str());
        ++skipped;
        continue;
      }
      cands = fit->second;
    } else {
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
    }

    McBopomofo::RescoreRequest req;
    req.readings = it.reading;
    req.candidates = cands;

    // ---- D
    std::vector<double> dScores;
    double tD = 0;
    size_t dBest = 0;
    if (sloth_) {
      auto t = Clock::now();
      dScores = sloth_->score(req);
      tD = MsSince(t);
      if (dScores.size() != cands.size()) { dScores.assign(cands.size(), 0.0); }
      dBest = ArgMax(dScores);
      fD << json{{"id", it.id}, {"choice", cands[dBest]}, {"latency_ms", Round3(tD)},
                 {"scores", dScores}}.dump() << "\n";
    }

    // ---- E and D+E ensemble
    if (electra) {
      auto t = Clock::now();
      std::vector<double> eScores = electra->score(req);
      double tE = MsSince(t);
      if (eScores.size() != cands.size()) { eScores.assign(cands.size(), 0.0); }
      fE << json{{"id", it.id}, {"choice", cands[ArgMax(eScores)]}, {"latency_ms", Round3(tE)},
                 {"scores", eScores}}.dump() << "\n";
    }
    if (de) {
      auto t = Clock::now();
      std::vector<double> s = de->score(req);
      double tDE = MsSince(t);
      std::string choice = cands[0];
      if (s.size() == cands.size()) choice = cands[ArgMax(s)];
      fDE << json{{"id", it.id}, {"choice", choice}, {"latency_ms", Round3(tDE)},
                  {"active", de->lastActiveMembers()}, {"scores", s}}.dump() << "\n";
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
    if (sloth_ && qwen) {
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
