// Standalone test for QwenRescorer (no gtest).
// Usage: qwen_rescorer_test <model.gguf> <nbest.jsonl> <python_result.jsonl>
// Exit code 0 = all checks passed. Prints latency / load time / RSS.
#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "qwen_rescorer.h"

using McBopomofo::QwenRescorer;
using McBopomofo::RescoreRequest;
using nlohmann::json;

static std::vector<json> LoadJsonl(const std::string& p) {
  std::vector<json> v;
  std::ifstream f(p);
  std::string line;
  while (std::getline(f, line))
    if (!line.empty()) v.push_back(json::parse(line));
  return v;
}

static double Pct(std::vector<double> x, double q) {
  std::sort(x.begin(), x.end());
  return x[std::min(x.size() - 1, (size_t)std::llround(q * (x.size() - 1)))];
}

int main(int argc, char** argv) {
  if (argc < 4) { fprintf(stderr, "usage: %s model nbest py_result\n", argv[0]); return 2; }
  int fails = 0;
  auto check = [&](bool ok, const char* name) {
    printf("[%s] %s\n", ok ? "PASS" : "FAIL", name);
    if (!ok) ++fails;
  };

  // (2) error path
  {
    QwenRescorer bad("/nonexistent/model.gguf");
    RescoreRequest r; r.candidates = {"你好"};
    check(!bad.ready() && bad.score(r).empty(), "bad path -> ready()==false, score() empty");
  }

  auto t0 = std::chrono::steady_clock::now();
  QwenRescorer rs(argv[1]);
  double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  check(rs.ready(), "model loads");
  if (!rs.ready()) return 1;

  auto nb = LoadJsonl(argv[2]);
  auto py = LoadJsonl(argv[3]);

  // (1) parity on first 5
  int agree = 0;
  for (int i = 0; i < 5; ++i) {
    RescoreRequest req;
    for (auto& c : nb[i]["nbest"]) {
      if (req.candidates.size() == 10) break;
      req.candidates.push_back(c["text"].get<std::string>());
    }
    auto sc = rs.score(req);
    bool sized = sc.size() == req.candidates.size();
    size_t best = sized ? std::max_element(sc.begin(), sc.end()) - sc.begin() : 0;
    std::string choice = req.candidates[best];
    bool same = choice == py[i]["choice"].get<std::string>();
    agree += same;
    double maxd = 0;
    auto& ps = py[i]["scores"];
    printf("id=%s top1=%s py=%s %s\n  cpp:", nb[i]["id"].get<std::string>().c_str(), choice.c_str(),
           py[i]["choice"].get<std::string>().c_str(), same ? "MATCH" : "DIFF");
    for (size_t k = 0; k < sc.size(); ++k) printf(" %.3f", sc[k]);
    printf("\n  py: ");
    for (size_t k = 0; k < sc.size(); ++k) {
      printf(" %.3f", ps[k].get<double>());
      maxd = std::max(maxd, std::abs(sc[k] - ps[k].get<double>()));
    }
    printf("\n  max|diff|=%.3f\n", maxd);
  }
  check(agree == 5, "top-1 matches Python on first 5");

  // (3) latency over all sentences, 5 warm-up excluded
  std::vector<double> lat;
  for (size_t i = 0; i < nb.size(); ++i) {
    RescoreRequest req;
    for (auto& c : nb[i]["nbest"]) {
      if (req.candidates.size() == 10) break;
      req.candidates.push_back(c["text"].get<std::string>());
    }
    auto a = std::chrono::steady_clock::now();
    auto sc = rs.score(req);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
    if (sc.empty()) { check(false, "score() non-empty during latency run"); break; }
    if (i >= 5) lat.push_back(ms);
  }
  struct rusage ru; getrusage(RUSAGE_SELF, &ru);
  printf("load=%.2fs  n=%zu  p50=%.1fms p95=%.1fms max=%.1fms  RSS peak=%.0fMB\n", load_s, lat.size(),
         Pct(lat, .5), Pct(lat, .95), *std::max_element(lat.begin(), lat.end()), ru.ru_maxrss / 1e6);
  check(Pct(lat, .95) < 80.0, "p95 < 80ms");
  printf(fails ? "FAILED\n" : "ALL PASSED\n");
  return fails ? 1 : 0;
}
