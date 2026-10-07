// Tests for the personalized weights file (task 2.2), the ONNX initializer
// scan and the merge / per-channel int8 quantizer (task 2.3).

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <unistd.h>
#include <sys/stat.h>
#include <vector>

#include "nlohmann/json.hpp"
#include "onnx_initializers.h"
#include "personal_weights.h"
#include "quantize.h"

namespace McBopomofo {
namespace Personalize {
namespace {

const std::string kOnnx = std::string(RWALK_MODEL_DIR) + "/onnx/model_quantized.onnx";

std::string TempPath(const std::string& name) {
  return "/tmp/afm_personalize_test_" + std::to_string(getpid()) + "_" + name;
}

QuantizedLinear MakeLinear(const std::string& name, uint32_t in, uint32_t out, uint32_t seed) {
  QuantizedLinear t;
  t.name = name;
  t.in = in;
  t.out = out;
  std::mt19937 rng(seed);
  for (uint32_t j = 0; j < out; ++j) t.scales.push_back(std::uniform_real_distribution<float>(0, 0.01f)(rng));
  for (size_t k = 0; k < static_cast<size_t>(in) * out; ++k)
    t.values.push_back(static_cast<int8_t>(std::uniform_int_distribution<int>(-127, 127)(rng)));
  return t;
}

PersonalWeights SmallWeights() {
  PersonalWeights w;
  for (size_t i = 0; i < w.factorySha.size(); ++i) w.factorySha[i] = static_cast<uint8_t>(i * 7);
  w.tensors.push_back(MakeLinear("val_1", 4, 3, 1));
  w.tensors.push_back(MakeLinear("val_22", 3, 5, 2));
  w.tensors.push_back(MakeLinear("mul_7", 2, 2, 3));
  return w;
}

WeightsReadOptions AnyCount() {
  WeightsReadOptions o;
  o.expectedTensors = 0;
  return o;
}

void ExpectSame(const PersonalWeights& a, const PersonalWeights& b) {
  EXPECT_EQ(a.factorySha, b.factorySha);
  ASSERT_EQ(a.tensors.size(), b.tensors.size());
  for (size_t i = 0; i < a.tensors.size(); ++i) {
    EXPECT_EQ(a.tensors[i].name, b.tensors[i].name);
    EXPECT_EQ(a.tensors[i].in, b.tensors[i].in);
    EXPECT_EQ(a.tensors[i].out, b.tensors[i].out);
    EXPECT_EQ(a.tensors[i].values, b.tensors[i].values);
    ASSERT_EQ(a.tensors[i].scales.size(), b.tensors[i].scales.size());
    EXPECT_EQ(0, std::memcmp(a.tensors[i].scales.data(), b.tensors[i].scales.data(),
                             4 * a.tensors[i].scales.size()));
  }
}

// Re-seals a modified body with a valid trailer hash.
std::string Reseal(std::string bytes) {
  bytes.resize(bytes.size() - 32);
  Sha256Digest h = Sha256(bytes.data(), bytes.size());
  bytes.append(reinterpret_cast<const char*>(h.data()), 32);
  return bytes;
}

// ---------------------------------------------------------------------------
// 2.2 weights file

TEST(PersonalWeightsTest, Sha256KnownVectors) {
  EXPECT_EQ(ToHex(Sha256("", 0)), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  EXPECT_EQ(ToHex(Sha256("abc", 3)), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

TEST(PersonalWeightsTest, RoundTripBitExact) {
  PersonalWeights w = SmallWeights();
  const std::string path = TempPath("rt.bin");
  std::string err;
  ASSERT_TRUE(WriteWeightsFile(path, w, &err)) << err;
  struct stat st {};
  ASSERT_EQ(0, stat(path.c_str(), &st));
  EXPECT_EQ(st.st_mode & 0777, 0600u);
  auto r = ReadWeightsFile(path, AnyCount(), &err);
  ASSERT_TRUE(r) << err;
  ExpectSame(w, *r);
  // Re-serializing the parsed weights reproduces the file bytes.
  std::string bytes;
  ASSERT_TRUE(ReadFileBytes(path, &bytes, &err));
  EXPECT_EQ(SerializeWeights(*r), bytes);
  std::remove(path.c_str());
}

TEST(PersonalWeightsTest, RejectsTruncated) {
  std::string bytes = SerializeWeights(SmallWeights());
  std::string err;
  for (size_t n : {size_t{0}, size_t{7}, size_t{40}, bytes.size() / 2, bytes.size() - 1}) {
    EXPECT_FALSE(ParseWeights(bytes.data(), n, AnyCount(), &err)) << n;
  }
  // Truncated but re-sealed (valid trailer): the tensor data runs out.
  std::string cut = bytes.substr(0, bytes.size() - 32 - 5);
  cut.append(32, '\0');
  EXPECT_FALSE(ParseWeights(cut.data(), cut.size(), AnyCount(), &err));
  cut = Reseal(cut);
  EXPECT_FALSE(ParseWeights(cut.data(), cut.size(), AnyCount(), &err));
  EXPECT_EQ(err, "truncated");
}

TEST(PersonalWeightsTest, RejectsWrongMagic) {
  std::string bytes = SerializeWeights(SmallWeights());
  bytes[7] = '2';
  bytes = Reseal(bytes);
  std::string err;
  EXPECT_FALSE(ParseWeights(bytes.data(), bytes.size(), AnyCount(), &err));
  EXPECT_EQ(err, "bad magic");
}

TEST(PersonalWeightsTest, RejectsWrongTrailer) {
  std::string bytes = SerializeWeights(SmallWeights());
  std::string err;
  std::string a = bytes;
  a.back() ^= 1;  // trailer itself
  EXPECT_FALSE(ParseWeights(a.data(), a.size(), AnyCount(), &err));
  EXPECT_EQ(err, "trailer hash mismatch");
  std::string b = bytes;
  b[60] ^= 0x10;  // a body byte (tensor data) without resealing
  EXPECT_FALSE(ParseWeights(b.data(), b.size(), AnyCount(), &err));
  EXPECT_EQ(err, "trailer hash mismatch");
}

TEST(PersonalWeightsTest, RejectsWrongTensorCount) {
  std::string bytes = SerializeWeights(SmallWeights());
  std::string err;
  WeightsReadOptions o;  // expects 84
  EXPECT_FALSE(ParseWeights(bytes.data(), bytes.size(), o, &err));
  o.expectedTensors = 3;
  EXPECT_TRUE(ParseWeights(bytes.data(), bytes.size(), o, &err)) << err;
  // Header count 4 with 3 tensors present (re-sealed).
  std::string c = bytes;
  c[40] = 4;
  c = Reseal(c);
  EXPECT_FALSE(ParseWeights(c.data(), c.size(), AnyCount(), &err));
  // Header count 2 with 3 present -> trailing bytes.
  c = bytes;
  c[40] = 2;
  c = Reseal(c);
  EXPECT_FALSE(ParseWeights(c.data(), c.size(), AnyCount(), &err));
}

TEST(PersonalWeightsTest, RejectsWrongShape) {
  PersonalWeights w = SmallWeights();
  std::string bytes = SerializeWeights(w);
  std::string err;
  std::unordered_map<std::string, std::pair<uint32_t, uint32_t>> shapes = {
      {"val_1", {4, 3}}, {"val_22", {3, 5}}, {"mul_7", {2, 2}}};
  WeightsReadOptions o = AnyCount();
  o.expectedShapes = &shapes;
  EXPECT_TRUE(ParseWeights(bytes.data(), bytes.size(), o, &err)) << err;
  shapes["val_22"] = {5, 3};  // transposed
  EXPECT_FALSE(ParseWeights(bytes.data(), bytes.size(), o, &err));
  EXPECT_EQ(err, "shape mismatch for val_22");
  shapes.erase("val_22");
  EXPECT_FALSE(ParseWeights(bytes.data(), bytes.size(), o, &err));
  // Zero dimension and a dimension above maxDim.
  PersonalWeights z = w;
  z.tensors[0].in = 0;
  z.tensors[0].values.clear();
  std::string zb = SerializeWeights(z);
  ASSERT_FALSE(zb.empty());
  EXPECT_FALSE(ParseWeights(zb.data(), zb.size(), AnyCount(), &err));
  WeightsReadOptions small = AnyCount();
  small.maxDim = 4;
  EXPECT_FALSE(ParseWeights(bytes.data(), bytes.size(), small, &err));  // val_22 out = 5
  // Inconsistent buffers are not serialized.
  PersonalWeights bad = w;
  bad.tensors[1].scales.pop_back();
  EXPECT_TRUE(SerializeWeights(bad).empty());
}

TEST(PersonalWeightsTest, RejectsBadScalesAndNames) {
  std::string err;
  PersonalWeights w = SmallWeights();
  w.tensors[1].scales[2] = NAN;
  std::string b = SerializeWeights(w);
  EXPECT_FALSE(ParseWeights(b.data(), b.size(), AnyCount(), &err));
  w = SmallWeights();
  w.tensors[2].name = "val_1";  // duplicate
  b = SerializeWeights(w);
  EXPECT_FALSE(ParseWeights(b.data(), b.size(), AnyCount(), &err));
  w = SmallWeights();
  w.tensors[2].name = "val 1";  // space
  b = SerializeWeights(w);
  EXPECT_FALSE(ParseWeights(b.data(), b.size(), AnyCount(), &err));
}

TEST(PersonalWeightsTest, FuzzNeverThrows) {
  std::mt19937 rng(42);
  const std::string valid = SerializeWeights(SmallWeights());
  std::string err;
  size_t accepted = 0;
  for (int i = 0; i < 1000; ++i) {
    std::string s;
    if (i % 2 == 0) {
      // Random bytes, sometimes with the right magic, up to 4 KB.
      size_t n = std::uniform_int_distribution<size_t>(0, 4096)(rng);
      s.resize(n);
      for (auto& c : s) c = static_cast<char>(rng());
      if (i % 4 == 0 && n >= 8) std::memcpy(s.data(), kWeightsMagic, 8);
      if (i % 8 == 0) s = Reseal(s.size() >= 32 ? s : s + std::string(32, '\0'));
    } else {
      // Mutated valid file, re-sealed so that the structure checks run.
      s = valid;
      int flips = std::uniform_int_distribution<int>(1, 8)(rng);
      for (int f = 0; f < flips; ++f) {
        size_t pos = std::uniform_int_distribution<size_t>(0, s.size() - 33)(rng);
        s[pos] = static_cast<char>(rng());
      }
      if (i % 3 == 0) s.resize(std::uniform_int_distribution<size_t>(40, s.size())(rng));
      s = Reseal(s);
    }
    std::shared_ptr<const PersonalWeights> r;
    EXPECT_NO_THROW(r = ParseWeights(s.data(), s.size(), AnyCount(), &err));
    accepted += r != nullptr;
  }
  // Mutations of scale / value bytes alone are still structurally valid.
  EXPECT_LT(accepted, 1000u);
}

// ---------------------------------------------------------------------------
// ONNX scan + 2.3 quantizer

class FactoryModelTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    std::string err;
    bytes_ = new std::string();
    ASSERT_TRUE(ReadFileBytes(kOnnx, bytes_, &err)) << err;
    info_ = new OnnxModelInfo();
    ASSERT_TRUE(ScanOnnxModel(*bytes_, info_, &err)) << err;
    linears_ = new std::vector<QuantizedLinear>();
    ASSERT_TRUE(FactoryLinears(*info_, linears_, &err)) << err;
  }
  static void TearDownTestSuite() {
    delete linears_;
    delete info_;
    delete bytes_;
  }
  static std::string* bytes_;
  static OnnxModelInfo* info_;
  static std::vector<QuantizedLinear>* linears_;
};
std::string* FactoryModelTest::bytes_ = nullptr;
OnnxModelInfo* FactoryModelTest::info_ = nullptr;
std::vector<QuantizedLinear>* FactoryModelTest::linears_ = nullptr;

TEST_F(FactoryModelTest, ScanFindsTheR84Linears) {
  EXPECT_EQ(info_->matMulIntegerWeights.size(), kRLinearCount);
  ASSERT_EQ(linears_->size(), kRLinearCount);
  EXPECT_EQ(info_->initializers.size(), 332u);
  size_t total = 0, scales = 0;
  std::map<std::pair<uint32_t, uint32_t>, int> shapes;
  for (const auto& t : *linears_) {
    total += t.values.size();
    scales += t.scales.size();
    ++shapes[{t.in, t.out}];
  }
  EXPECT_EQ(total, 9043968u);  // research 16 section 0
  EXPECT_EQ(scales, 29184u);
  EXPECT_EQ((shapes[{256, 256}]), 24);
  EXPECT_EQ((shapes[{256, 64}]), 24);
  EXPECT_EQ((shapes[{256, 768}]), 24);
  EXPECT_EQ((shapes[{768, 256}]), 12);
  EXPECT_EQ(linears_->front().name, "val_22");
  // The file has 86 2-D 8-bit tensors (research 17, E2): the 84 LoRA linears
  // (int8, per-channel, zero point 0) plus embed.weight / head.weight, which
  // are per-tensor uint8 with a non-zero zero point, are not LoRA targets and
  // are never part of weights.bin (they keep the factory bytes).
  size_t eightBit = 0;
  for (const auto& t : info_->initializers)
    eightBit += (t.dataType == kOnnxInt8 || t.dataType == kOnnxUint8) && t.dims.size() == 2;
  EXPECT_EQ(eightBit, 86u);
  for (const char* n : {"embed.weight_quantized", "head.weight_quantized"}) {
    const OnnxInitializer* e = info_->find(n);
    ASSERT_NE(e, nullptr) << n;
    EXPECT_EQ(e->dataType, kOnnxUint8) << n;
    for (const auto& l : *linears_) EXPECT_NE(l.name + "_quantized", n);
  }
}

TEST_F(FactoryModelTest, ScanRejectsGarbage) {
  OnnxModelInfo m;
  std::string err;
  EXPECT_FALSE(ScanOnnxModel(bytes_->substr(0, bytes_->size() / 2), &m, &err));
  EXPECT_FALSE(ScanOnnxModel("", &m, &err));
  std::mt19937 rng(7);
  for (int i = 0; i < 200; ++i) {
    std::string s(std::uniform_int_distribution<size_t>(0, 2048)(rng), '\0');
    for (auto& c : s) c = static_cast<char>(rng());
    EXPECT_NO_THROW(ScanOnnxModel(s, &m, &err));
  }
}

TEST_F(FactoryModelTest, DequantizeQuantizeReproducesShippedTensors) {
  // Every linear (the task asks for >= 2 fixture tensors; the shipped model
  // file itself is the fixture), bit-exact values and scales.
  size_t mismatches = 0;
  for (const auto& t : *linears_) {
    std::vector<float> w = Dequantize(t);
    QuantizedLinear q;
    QuantizePerChannel(w.data(), t.in, t.out, &q);
    mismatches += q.values != t.values;
    mismatches += std::memcmp(q.scales.data(), t.scales.data(), 4 * t.out) != 0;
  }
  EXPECT_EQ(mismatches, 0u);
}

TEST_F(FactoryModelTest, ZeroLoRADeltaGivesIdenticalBytes) {
  const uint32_t r = 8;
  PersonalWeights a, b;
  Sha256Digest sha = Sha256(bytes_->data(), bytes_->size());
  a.factorySha = b.factorySha = sha;
  for (const auto& t : *linears_) {
    std::vector<float> w = Dequantize(t);
    std::vector<float> A(static_cast<size_t>(r) * t.in), B(static_cast<size_t>(t.out) * r, 0.0f);
    for (size_t i = 0; i < A.size(); ++i) A[i] = std::sin(0.37f * static_cast<float>(i));
    MergeLoRA(&w, t.in, t.out, A.data(), B.data(), r, 2.0f);
    QuantizedLinear q;
    q.name = t.name;
    QuantizePerChannel(w.data(), t.in, t.out, &q);
    a.tensors.push_back(q);
    b.tensors.push_back(t);
  }
  EXPECT_EQ(SerializeWeights(a), SerializeWeights(b));
  // And the factory linears re-read from a weights file parse with the R shapes.
  std::string bytes = SerializeWeights(b);
  auto shapes = LinearShapes(*linears_);
  WeightsReadOptions o;
  o.expectedShapes = &shapes;
  std::string err;
  EXPECT_TRUE(ParseWeights(bytes.data(), bytes.size(), o, &err)) << err;
  EXPECT_NEAR(static_cast<double>(bytes.size()) / 1e6, 9.2, 0.1);
}

TEST(QuantizeTest, AllZeroChannelHasNoNaN) {
  // 3 x 2, column 1 all zero.
  std::vector<float> w = {0.25f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f};
  QuantizedLinear q;
  QuantizePerChannel(w.data(), 3, 2, &q);
  EXPECT_FLOAT_EQ(q.scales[0], 1.0f / 127.0f);
  EXPECT_EQ(q.scales[1], 0.0f);
  for (float s : q.scales) EXPECT_TRUE(std::isfinite(s));
  EXPECT_EQ(q.values, (std::vector<int8_t>{32, 0, -127, 0, 127, 0}));
  for (float x : Dequantize(q)) EXPECT_TRUE(std::isfinite(x));
}

TEST(QuantizeTest, MergeMatchesNumPyFixture) {
  std::ifstream in(std::string(PERSONALIZE_TESTDATA) + "/merge_fixture.json");
  ASSERT_TRUE(in);
  std::stringstream ss;
  ss << in.rdbuf();
  auto j = nlohmann::json::parse(ss.str());
  const uint32_t nIn = j["in"], nOut = j["out"], r = j["r"];
  std::vector<float> w = j["W"].get<std::vector<float>>();
  std::vector<float> A = j["A"].get<std::vector<float>>();
  std::vector<float> B = j["B"].get<std::vector<float>>();
  std::vector<double> expected = j["merged"].get<std::vector<double>>();
  MergeLoRA(&w, nIn, nOut, A.data(), B.data(), r, j["scale"].get<float>());
  ASSERT_EQ(w.size(), expected.size());
  double maxDiff = 0;
  for (size_t i = 0; i < w.size(); ++i) maxDiff = std::max(maxDiff, std::fabs(w[i] - expected[i]));
  EXPECT_LT(maxDiff, 1e-6);
  QuantizedLinear q;
  QuantizePerChannel(w.data(), nIn, nOut, &q);
  std::vector<int> qs(q.values.begin(), q.values.end());
  EXPECT_EQ(qs, j["q"].get<std::vector<int>>());
  std::vector<double> sc = j["scales"].get<std::vector<double>>();
  for (uint32_t c = 0; c < nOut; ++c) EXPECT_EQ(static_cast<double>(q.scales[c]), sc[c]);
}

}  // namespace
}  // namespace Personalize
}  // namespace McBopomofo
