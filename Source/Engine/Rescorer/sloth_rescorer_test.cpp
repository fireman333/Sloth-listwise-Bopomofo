#include "sloth_rescorer.h"

#include "Personalize/onnx_initializers.h"
#include "Personalize/personal_weights.h"
#include "Personalize/quantize.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <string>
#include <vector>

namespace McBopomofo {
namespace {

std::string ModelDir() {
  const char* e = std::getenv("SLOTH_MODEL_DIR");
  return e ? e : SLOTH_MODEL_DIR;
}

std::vector<std::string> Split(const std::string& s, char d) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, d)) out.push_back(tok);
  return out;
}

struct Case {
  std::string id;
  RescoreRequest req;
  std::vector<double> expected;
};

// Fixture lines (tab separated): id, readings (space separated), then
// repeated (candidate, python score) pairs. Produced by bench/src/rescore_sloth.py.
std::vector<Case> LoadFixture() {
  std::vector<Case> cases;
  std::ifstream in(SLOTH_FIXTURE);
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    auto f = Split(line, '\t');
    Case c;
    c.id = f[0];
    c.req.readings = Split(f[1], ' ');
    for (size_t i = 2; i + 1 < f.size(); i += 2) {
      c.req.candidates.push_back(f[i]);
      c.expected.push_back(std::stod(f[i + 1]));
    }
    cases.push_back(std::move(c));
  }
  return cases;
}

TEST(SlothRescorerTest, MatchesPythonScores) {
  SlothRescorer r(ModelDir());
  ASSERT_TRUE(r.ready());
  auto cases = LoadFixture();
  ASSERT_EQ(cases.size(), 5u);
  double maxDiff = 0;
  for (const auto& c : cases) {
    auto s = r.score(c.req);
    ASSERT_EQ(s.size(), c.expected.size()) << c.id;
    for (size_t i = 0; i < s.size(); ++i) {
      double d = std::fabs(s[i] - c.expected[i]);
      maxDiff = std::max(maxDiff, d);
      EXPECT_LT(d, 1e-3) << c.id << " cand " << i << " cpp=" << s[i] << " py=" << c.expected[i];
    }
    // Same argmax as Python.
    auto am = [](const std::vector<double>& v) {
      return std::max_element(v.begin(), v.end()) - v.begin();
    };
    EXPECT_EQ(am(s), am(c.expected)) << c.id;
  }
  std::printf("[ INFO ] max |cpp - python| over fixture = %.3g\n", maxDiff);
}

TEST(SlothRescorerTest, BadModelPathNotReady) {
  SlothRescorer r("/nonexistent/path/to/sloth");
  EXPECT_FALSE(r.ready());
  RescoreRequest req;
  req.readings = {"ㄧ", "ㄕㄥ"};
  req.candidates = {"一生", "醫生"};
  EXPECT_TRUE(r.score(req).empty());
}

TEST(SlothRescorerTest, EdgeCases) {
  SlothRescorer r(ModelDir());
  ASSERT_TRUE(r.ready());
  RescoreRequest req;
  EXPECT_TRUE(r.score(req).empty());
  req.readings = {"ㄧ", "ㄕㄥ"};
  EXPECT_TRUE(r.score(req).empty());  // no candidates
  req.candidates = {"醫生", "一聲", "ab"};  // third has non-vocab chars
  auto s = r.score(req);
  ASSERT_EQ(s.size(), 3u);
  EXPECT_LT(s[2], s[0] - 20);  // out-of-vocab chars get -30 each
}

TEST(SlothRescorerTest, LatencyTop10) {
  SlothRescorer r(ModelDir());
  ASSERT_TRUE(r.ready());
  auto cases = LoadFixture();
  ASSERT_FALSE(cases.empty());
  for (int i = 0; i < 20; ++i) r.score(cases[0].req);  // warm up
  std::vector<double> ms;
  for (int rep = 0; rep < 20; ++rep) {
    for (const auto& c : cases) {
      auto t0 = std::chrono::steady_clock::now();
      auto s = r.score(c.req);
      auto t1 = std::chrono::steady_clock::now();
      ASSERT_EQ(s.size(), c.req.candidates.size());
      ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
  }
  std::sort(ms.begin(), ms.end());
  std::printf("[ INFO ] 10-candidate score(): n=%zu p50=%.3f ms p95=%.3f ms max=%.3f ms\n",
              ms.size(), ms[ms.size() / 2], ms[static_cast<size_t>(ms.size() * 0.95)], ms.back());
  EXPECT_LT(ms[ms.size() / 2], 10.0);
}


// ---------------------------------------------------------------------------
// Personalized weights (personal-finetune task 2.4, design D11), on the R
// export (train/ranker/export/clean_v1).

namespace P = Personalize;

const std::string kROnnx = std::string(R_MODEL_DIR) + "/onnx/model_quantized.onnx";

struct Factory {
  std::string bytes;
  P::Sha256Digest sha{};
  std::vector<P::QuantizedLinear> linears;
};

const Factory& LoadFactory() {
  static Factory* f = [] {
    auto* x = new Factory();
    std::string err;
    P::OnnxModelInfo info;
    if (!P::ReadFileBytes(kROnnx, &x->bytes, &err) || !P::ScanOnnxModel(x->bytes, &info, &err) ||
        !P::FactoryLinears(info, &x->linears, &err))
      std::fprintf(stderr, "factory load failed: %s\n", err.c_str());
    x->sha = P::Sha256(x->bytes.data(), x->bytes.size());
    return x;
  }();
  return *f;
}

std::shared_ptr<P::PersonalWeights> FactoryWeights() {
  auto w = std::make_shared<P::PersonalWeights>();
  w->factorySha = LoadFactory().sha;
  w->tensors = LoadFactory().linears;
  return w;
}

// The deterministic personalized weights of tools/make_fixtures.py
// (perturbed_linears): t-th MatMulInteger weight, rank 8, scale 2,
// A[k,i] = ((i*31 + k*17 + t*7) % 13 - 6) / 64, B[j,k] = ((j*7 + k*5 + t*3) % 11 - 5) / 1024.
std::shared_ptr<P::PersonalWeights> PerturbedWeights() {
  auto w = std::make_shared<P::PersonalWeights>();
  w->factorySha = LoadFactory().sha;
  const uint32_t r = 8;
  uint32_t t = 0;
  for (const auto& lin : LoadFactory().linears) {
    std::vector<float> x = P::Dequantize(lin);
    std::vector<float> A(static_cast<size_t>(r) * lin.in), B(static_cast<size_t>(lin.out) * r);
    for (uint32_t k = 0; k < r; ++k)
      for (uint32_t i = 0; i < lin.in; ++i)
        A[k * lin.in + i] = static_cast<float>(static_cast<int>((i * 31 + k * 17 + t * 7) % 13) - 6) / 64.0f;
    for (uint32_t j = 0; j < lin.out; ++j)
      for (uint32_t k = 0; k < r; ++k)
        B[j * r + k] = static_cast<float>(static_cast<int>((j * 7 + k * 5 + t * 3) % 11) - 5) / 1024.0f;
    P::MergeLoRA(&x, lin.in, lin.out, A.data(), B.data(), r, 2.0f);
    P::QuantizedLinear q;
    q.name = lin.name;
    P::QuantizePerChannel(x.data(), lin.in, lin.out, &q);
    w->tensors.push_back(std::move(q));
    ++t;
  }
  return w;
}

struct OverrideFixture {
  std::string fileSha;
  std::vector<Case> cases;
};

OverrideFixture LoadOverrideFixture() {
  OverrideFixture f;
  std::ifstream in(PERSONAL_OVERRIDE_FIXTURE);
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    auto p = Split(line, '\t');
    if (p[0] == "#") {
      f.fileSha = p[1];
      continue;
    }
    Case c;
    c.id = p[0];
    c.req.readings = Split(p[1], ' ');
    for (size_t i = 2; i + 1 < p.size(); i += 2) {
      c.req.candidates.push_back(p[i]);
      c.expected.push_back(std::stod(p[i + 1]));
    }
    f.cases.push_back(std::move(c));
  }
  return f;
}

SlothRescorer::Options WithWeights(std::shared_ptr<const P::PersonalWeights> w) {
  SlothRescorer::Options o;
  o.overrides = std::move(w);
  return o;
}

// Max |a - b| of score() over the fixture cases.
double MaxScoreDiff(SlothRescorer& a, SlothRescorer& b, const std::vector<Case>& cases) {
  double m = 0;
  for (const auto& c : cases) {
    auto x = a.score(c.req), y = b.score(c.req);
    EXPECT_EQ(x.size(), y.size());
    for (size_t i = 0; i < std::min(x.size(), y.size()); ++i) m = std::max(m, std::fabs(x[i] - y[i]));
  }
  return m;
}

TEST(SlothPersonalTest, NoOverridesIsFactory) {
  SlothRescorer r(R_MODEL_DIR);
  ASSERT_TRUE(r.ready());
  EXPECT_FALSE(r.personalized());
  EXPECT_EQ(r.personalizationError(), "");
}

TEST(SlothPersonalTest, IdenticalTensorsGiveIdenticalLogits) {
  ASSERT_EQ(LoadFactory().linears.size(), 84u);
  SlothRescorer plain(R_MODEL_DIR);
  SlothRescorer same(R_MODEL_DIR, WithWeights(FactoryWeights()));
  ASSERT_TRUE(same.ready());
  ASSERT_TRUE(same.personalized()) << same.personalizationError();
  auto fx = LoadOverrideFixture();
  ASSERT_FALSE(fx.cases.empty());
  EXPECT_EQ(MaxScoreDiff(plain, same, fx.cases), 0.0);
  for (const auto& c : fx.cases) {
    auto a = plain.charLogProbs(c.req.readings), b = same.charLogProbs(c.req.readings);
    ASSERT_EQ(a.size(), b.size());
    for (size_t t = 0; t < a.size(); ++t) EXPECT_EQ(a[t], b[t]) << c.id << " " << t;
  }
}

TEST(SlothPersonalTest, PersonalizedOverrideMatchesPythonOrt) {
  auto w = PerturbedWeights();
  auto fx = LoadOverrideFixture();
  ASSERT_EQ(fx.cases.size(), 5u);
  // The C++ writer produces the same bytes as the independent Python writer.
  std::string bytes = P::SerializeWeights(*w);
  EXPECT_EQ(P::ToHex(P::Sha256(bytes.data(), bytes.size())), fx.fileSha);

  SlothRescorer plain(R_MODEL_DIR);
  SlothRescorer pers(R_MODEL_DIR, WithWeights(w));
  ASSERT_TRUE(pers.personalized()) << pers.personalizationError();
  double maxDiff = 0;
  for (const auto& c : fx.cases) {
    auto s = pers.score(c.req);
    ASSERT_EQ(s.size(), c.expected.size());
    for (size_t i = 0; i < s.size(); ++i) {
      maxDiff = std::max(maxDiff, std::fabs(s[i] - c.expected[i]));
      EXPECT_NEAR(s[i], c.expected[i], 1e-4) << c.id << " cand " << i;
    }
  }
  const double vsFactory = MaxScoreDiff(plain, pers, fx.cases);
  std::printf("[ INFO ] personalized: max |cpp - python-ort| = %.3g, max |personal - factory| = %.3g\n",
              maxDiff, vsFactory);
  EXPECT_GT(vsFactory, 0.05);  // the overrides reach the MatMulInteger weights

  // Same result through a weights.bin path.
  const std::string path = "/tmp/afm_sloth_personal_" + std::to_string(getpid()) + ".bin";
  std::string err;
  ASSERT_TRUE(P::WriteWeightsFile(path, *w, &err)) << err;
  SlothRescorer::Options o;
  o.overridesPath = path;
  SlothRescorer viaFile(R_MODEL_DIR, o);
  ASSERT_TRUE(viaFile.personalized()) << viaFile.personalizationError();
  EXPECT_EQ(MaxScoreDiff(pers, viaFile, fx.cases), 0.0);
  std::remove(path.c_str());
}

// Every failed check: ready, factory outputs, not personalized, an error.
void ExpectFactoryWithError(SlothRescorer& r, const char* what) {
  SCOPED_TRACE(what);
  ASSERT_TRUE(r.ready());
  EXPECT_FALSE(r.personalized());
  EXPECT_NE(r.personalizationError(), "");
  SlothRescorer plain(R_MODEL_DIR);
  EXPECT_EQ(MaxScoreDiff(plain, r, LoadOverrideFixture().cases), 0.0);
}

TEST(SlothPersonalTest, WrongShaFallsBackToFactory) {
  auto w = PerturbedWeights();
  w->factorySha[0] ^= 1;
  SlothRescorer r(R_MODEL_DIR, WithWeights(w));
  ExpectFactoryWithError(r, "wrong sha");
  EXPECT_NE(r.personalizationError().find("factory"), std::string::npos);
}

TEST(SlothPersonalTest, UnknownNameFallsBackToFactory) {
  // ORT would silently ignore this name (E1); the rescorer must not.
  auto w = PerturbedWeights();
  w->tensors[5].name = "val_999999";
  SlothRescorer r(R_MODEL_DIR, WithWeights(w));
  ExpectFactoryWithError(r, "unknown name");
  EXPECT_EQ(r.personalizationError(), "unknown initializer val_999999");
}

TEST(SlothPersonalTest, MissingTensorOrWrongShapeFallsBackToFactory) {
  auto w = PerturbedWeights();
  w->tensors.pop_back();
  SlothRescorer missing(R_MODEL_DIR, WithWeights(w));
  ExpectFactoryWithError(missing, "83 tensors");

  w = PerturbedWeights();
  std::swap(w->tensors[0].name, w->tensors[1].name);  // [256,256] vs [256,64]
  SlothRescorer shape(R_MODEL_DIR, WithWeights(w));
  ExpectFactoryWithError(shape, "shape");
  EXPECT_NE(shape.personalizationError().find("shape mismatch"), std::string::npos);
}

TEST(SlothPersonalTest, TruncatedFileFallsBackToFactory) {
  const std::string path = "/tmp/afm_sloth_truncated_" + std::to_string(getpid()) + ".bin";
  std::string bytes = P::SerializeWeights(*PerturbedWeights());
  {
    std::ofstream f(path, std::ios::binary);
    f.write(bytes.data(), static_cast<std::streamsize>(bytes.size() / 2));
  }
  SlothRescorer::Options o;
  o.overridesPath = path;
  SlothRescorer r(R_MODEL_DIR, o);
  ExpectFactoryWithError(r, "truncated");
  SlothRescorer::Options none;
  none.overridesPath = path + ".missing";
  SlothRescorer m(R_MODEL_DIR, none);
  ExpectFactoryWithError(m, "missing file");
  std::remove(path.c_str());
}

}  // namespace
}  // namespace McBopomofo
