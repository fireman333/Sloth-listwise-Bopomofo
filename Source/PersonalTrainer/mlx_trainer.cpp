#include "mlx_trainer.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

#include "Personalize/quantize.h"
#include "mlx/mlx.h"

namespace mx = mlx::core;

namespace McBopomofo {
namespace Trainer {

double ScheduledLr(const Recipe& r, int s, int steps) {
  const int warm = std::max(1, static_cast<int>(r.warmupFrac * steps));
  if (s < warm) return r.lr * (s + 1) / warm;
  const double prog = static_cast<double>(s - warm) / std::max(1, steps - warm);
  return r.lr * (r.lrFloor + (1.0 - r.lrFloor) * 0.5 * (1.0 + std::cos(M_PI * prog)));
}

int StepsFor(const Recipe& r, size_t nTrain) {
  const int s = static_cast<int>(std::ceil(nTrain * r.epochs / r.batchUser));
  return std::max(r.minSteps, s);
}

namespace {

struct Linear {
  std::string torchName;
  std::string onnxName;  // empty for fp32 fixtures
  int in = 0, out = 0;
  mx::array wt = mx::array(0.0f);  // [in, out] (= W^T)
};

struct BlockW {
  mx::array n1 = mx::array(0.0f), n2 = mx::array(0.0f), qn = mx::array(0.0f),
            kn = mx::array(0.0f);
  int lin[7];  // indices into linears: q, k, v, o, w1, w3, w2
};

mx::array HostArray(const std::vector<float>& v, mx::Shape shape) {
  return mx::array(v.data(), std::move(shape), mx::float32);
}

}  // namespace

struct LoraTrainer::Impl {
  std::shared_ptr<const TrainBase> base;
  const ModelVocab* vocab = nullptr;
  Recipe recipe;
  ModelConfig cfg;
  std::string error;
  bool ok = false;

  mx::array embed = mx::array(0.0f), hintNone = mx::array(0.0f), embedNorm = mx::array(0.0f),
            norm = mx::array(0.0f), headT = mx::array(0.0f);
  std::vector<BlockW> blocks;
  std::vector<Linear> linears;
  std::vector<mx::array> params;  // A0, B0, A1, B1, ... (2 per linear)
  std::vector<mx::array> m, v;    // AdamW state

  bool Load() {
    cfg = base->config;
    auto tensor = [&](const std::string& name, std::vector<uint32_t> dims, mx::array* out) {
      const BaseTensor* t = base->find(name);
      if (t == nullptr) {
        error = "missing tensor " + name;
        return false;
      }
      if (t->dims != dims) {
        error = "bad shape for " + name;
        return false;
      }
      mx::Shape s(dims.begin(), dims.end());
      *out = HostArray(DequantizedTorchLayout(*t), s);
      return true;
    };
    const uint32_t d = cfg.dim, dh = cfg.dim / cfg.heads;
    if (!tensor("embed.weight", {uint32_t(cfg.nSyl), d}, &embed) ||
        !tensor("hint_none", {d}, &hintNone) || !tensor("embed_norm.w", {d}, &embedNorm) ||
        !tensor("norm.w", {d}, &norm))
      return false;
    mx::array head(0.0f);
    if (!tensor("head.weight", {uint32_t(cfg.nChar), d}, &head)) return false;
    headT = mx::copy(mx::transpose(head));
    const char* attnNames[4] = {"q", "k", "v", "o"};
    const char* ffnNames[3] = {"w1", "w3", "w2"};
    for (int i = 0; i < cfg.depth; ++i) {
      BlockW b;
      std::string p = "blocks." + std::to_string(i) + ".";
      if (!tensor(p + "n1.w", {d}, &b.n1) || !tensor(p + "n2.w", {d}, &b.n2) ||
          !tensor(p + "attn.qn.w", {dh}, &b.qn) || !tensor(p + "attn.kn.w", {dh}, &b.kn))
        return false;
      for (int j = 0; j < 7; ++j) {
        std::string name = j < 4 ? p + "attn." + attnNames[j] + ".weight"
                                 : p + "ffn." + ffnNames[j - 4] + ".weight";
        const BaseTensor* t = base->find(name);
        if (t == nullptr || t->dims.size() != 2) {
          error = "missing linear " + name;
          return false;
        }
        Linear l;
        l.torchName = name;
        l.out = static_cast<int>(t->dims[0]);
        l.in = static_cast<int>(t->dims[1]);
        if (t->kind == TensorKind::kLinearInt8) {
          l.onnxName = t->linear.name;
          // ONNX layout [in, out] = W^T directly.
          l.wt = HostArray(Personalize::Dequantize(t->linear), {l.in, l.out});
        } else {
          mx::array w = HostArray(DequantizedTorchLayout(*t), {l.out, l.in});
          l.wt = mx::copy(mx::transpose(w));
        }
        b.lin[j] = static_cast<int>(linears.size());
        linears.push_back(std::move(l));
      }
      blocks.push_back(std::move(b));
    }
    mx::eval(embed, hintNone, embedNorm, norm, headT);
    for (auto& b : blocks) mx::eval(b.n1, b.n2, b.qn, b.kn);
    for (auto& l : linears) mx::eval(l.wt);
    return true;
  }

  static mx::array RmsNorm(const mx::array& x, const mx::array& w) {
    return x * mx::rsqrt(mx::mean(x * x, -1, true) + mx::array(1e-6f)) * w;
  }

  mx::array Lin(const mx::array& x, int li, const std::vector<mx::array>& p) const {
    const mx::array& A = p[2 * li];
    const mx::array& B = p[2 * li + 1];
    mx::array y = mx::matmul(x, linears[li].wt);
    mx::array lo = mx::matmul(mx::matmul(x, mx::transpose(A)), mx::transpose(B));
    return y + lo * mx::array(recipe.loraScale);
  }

  static mx::array Rope(const mx::array& x, int T, int dh) {
    const int half = dh / 2;
    mx::array freq = mx::array(1.0f) /
                     mx::power(mx::array(10000.0f),
                               mx::astype(mx::arange(0, half), mx::float32) /
                                   mx::array(static_cast<float>(half)));
    mx::array pos = mx::astype(mx::arange(0, T), mx::float32);
    mx::array ang = mx::expand_dims(pos, 1) * mx::expand_dims(freq, 0);  // [T, half]
    mx::array cs = mx::concatenate({mx::cos(ang), mx::cos(ang)}, -1);
    mx::array sn = mx::concatenate({mx::sin(ang), mx::sin(ang)}, -1);
    cs = mx::reshape(cs, {1, 1, T, dh});
    sn = mx::reshape(sn, {1, 1, T, dh});
    mx::array x1 = mx::slice(x, {0, 0, 0, 0}, {x.shape(0), x.shape(1), T, half});
    mx::array x2 = mx::slice(x, {0, 0, 0, half}, {x.shape(0), x.shape(1), T, dh});
    return x * cs + mx::concatenate({mx::negative(x2), x1}, -1) * sn;
  }

  mx::array Forward(const mx::array& syl, const mx::array& amask,
                    const std::vector<mx::array>& p) const {
    const int B = syl.shape(0), T = syl.shape(1);
    const int h = cfg.heads, kv = cfg.kv, dh = cfg.dim / cfg.heads;
    mx::array x = mx::take(embed, syl, 0) + hintNone;
    x = RmsNorm(x, embedNorm);
    mx::array kmask = mx::where(amask, mx::array(0.0f), mx::array(-1e9f));
    kmask = mx::reshape(kmask, {B, 1, 1, T});
    const float attScale = std::pow(static_cast<float>(dh), -0.5f);
    for (const auto& b : blocks) {
      mx::array y = RmsNorm(x, b.n1);
      mx::array q = mx::transpose(mx::reshape(Lin(y, b.lin[0], p), {B, T, h, dh}), {0, 2, 1, 3});
      mx::array k = mx::transpose(mx::reshape(Lin(y, b.lin[1], p), {B, T, kv, dh}), {0, 2, 1, 3});
      mx::array v = mx::transpose(mx::reshape(Lin(y, b.lin[2], p), {B, T, kv, dh}), {0, 2, 1, 3});
      q = Rope(RmsNorm(q, b.qn), T, dh);
      k = Rope(RmsNorm(k, b.kn), T, dh);
      const int rep = h / kv;
      k = mx::repeat(k, rep, 1);
      v = mx::repeat(v, rep, 1);
      mx::array att = mx::matmul(q, mx::transpose(k, {0, 1, 3, 2})) * mx::array(attScale) + kmask;
      mx::array o = mx::reshape(mx::transpose(mx::matmul(mx::softmax(att, -1), v), {0, 2, 1, 3}),
                                {B, T, h * dh});
      x = x + Lin(o, b.lin[3], p);
      y = RmsNorm(x, b.n2);
      mx::array g1 = Lin(y, b.lin[4], p);
      mx::array silu = g1 * mx::sigmoid(g1);
      x = x + Lin(silu * Lin(y, b.lin[5], p), b.lin[6], p);
    }
    return mx::matmul(RmsNorm(x, norm), headT);  // [B, T, V]
  }

  struct BatchArrays {
    mx::array syl = mx::array(0), mask = mx::array(0), maskf = mx::array(0.0f),
              gold = mx::array(0), cand = mx::array(0), cv = mx::array(0), tgt = mx::array(0),
              ng = mx::array(0.0f), allowed = mx::array(0), sel = mx::array(0),
              candOk = mx::array(0), valid = mx::array(0.0f);
    bool anySel = false;
  };

  BatchArrays Collate(const std::vector<const EncodedRow*>& items) const {
    const int B = static_cast<int>(items.size());
    int T = 1, K = 1;
    for (const auto* x : items) {
      T = std::max(T, static_cast<int>(x->syl.size()));
      K = std::max(K, static_cast<int>(x->cands.size()));
    }
    const int V = cfg.nChar;
    std::vector<int32_t> syl(B * T, 0), gold(B * T, -100), cand(size_t(B) * K * T, -1),
        tgt(B, -1);
    std::vector<uint8_t> mask(B * T, 0), cv(B * K, 0);
    std::vector<float> ng(B * K, 0.0f);
    for (int i = 0; i < B; ++i) {
      const EncodedRow& x = *items[i];
      const int n = static_cast<int>(x.syl.size());
      for (int t = 0; t < n; ++t) {
        syl[i * T + t] = x.syl[t];
        mask[i * T + t] = 1;
        gold[i * T + t] = t < static_cast<int>(x.gold.size()) ? x.gold[t] : -100;
      }
      for (size_t k = 0; k < x.cands.size(); ++k) {
        ng[i * K + k] = static_cast<float>(recipe.ngramWeight) * x.ngram[k];
        cv[i * K + k] = 1;
        for (int t = 0; t < n; ++t) cand[(size_t(i) * K + k) * T + t] = x.cands[k][t];
      }
      tgt[i] = x.target;
    }
    // allowed [B, T, V] = legal(syl) U candidate characters (non-differentiable).
    std::vector<uint8_t> allowed(size_t(B) * T * V, 0);
    for (int i = 0; i < B; ++i)
      for (int t = 0; t < T; ++t) {
        uint8_t* row = allowed.data() + (size_t(i) * T + t) * V;
        int32_t s = syl[i * T + t];
        const std::vector<int32_t>* lg =
            (s >= 0 && s < static_cast<int32_t>(vocab->legal.size())) ? &vocab->legal[s] : nullptr;
        if (lg == nullptr || lg->empty()) {
          std::fill(row, row + V, 1);
        } else {
          for (int32_t c : *lg)
            if (c >= 0 && c < V) row[c] = 1;
        }
        for (int k = 0; k < K; ++k) {
          int32_t c = cand[(size_t(i) * K + k) * T + t];
          if (cv[i * K + k] && c >= 0 && c < V) row[c] = 1;
        }
      }
    // cand_ok [B, T, K]
    std::vector<uint8_t> ok(size_t(B) * T * K, 0);
    for (int i = 0; i < B; ++i)
      for (int k = 0; k < K; ++k)
        for (int t = 0; t < T; ++t)
          ok[(size_t(i) * T + t) * K + k] = cv[i * K + k] && cand[(size_t(i) * K + k) * T + t] >= 0;
    std::vector<int32_t> sel;
    for (int i = 0; i < B; ++i)
      if (tgt[i] >= 0) sel.push_back(i);
    std::vector<float> validf(B * T), maskf(B * T);
    for (int i = 0; i < B * T; ++i) {
      validf[i] = gold[i] >= 0 ? 1.0f : 0.0f;
      maskf[i] = mask[i] ? 1.0f : 0.0f;
    }
    BatchArrays a;
    auto boolArr = [](const std::vector<uint8_t>& v, mx::Shape s) {
      return mx::astype(mx::array(v.data(), std::move(s), mx::uint8), mx::bool_);
    };
    a.syl = mx::array(syl.data(), {B, T}, mx::int32);
    a.mask = boolArr(mask, {B, T});
    a.maskf = HostArray(maskf, {B, T});
    a.gold = mx::array(gold.data(), {B * T}, mx::int32);
    a.valid = HostArray(validf, {B * T});
    a.cand = mx::array(cand.data(), {B, K, T}, mx::int32);
    a.cv = boolArr(cv, {B, K});
    a.tgt = mx::array(tgt.data(), {B}, mx::int32);
    a.ng = HostArray(ng, {B, K});
    a.allowed = boolArr(allowed, {B, T, V});
    a.candOk = boolArr(ok, {B, T, K});
    a.anySel = !sel.empty();
    if (a.anySel) a.sel = mx::array(sel.data(), {static_cast<int>(sel.size())}, mx::int32);
    return a;
  }

  // Returns {loss, listwise, mle}.
  std::vector<mx::array> Loss(const BatchArrays& a, const std::vector<mx::array>& p) const {
    mx::array lg = Forward(a.syl, a.mask, p);  // [B, T, V]
    const mx::array ninf(-std::numeric_limits<float>::infinity());
    mx::array lse = mx::logsumexp(mx::where(a.allowed, lg, ninf), -1, false);  // [B, T]
    mx::array cc = mx::transpose(mx::maximum(a.cand, mx::array(0)), {0, 2, 1});  // [B, T, K]
    mx::array g = mx::take_along_axis(lg, cc, 2) - mx::expand_dims(lse, -1);
    g = mx::where(a.candOk, g, mx::array(recipe.missingLogP)) * mx::expand_dims(a.maskf, -1);
    mx::array s = mx::where(a.cv, mx::sum(g, 1, false), ninf) + a.ng;  // [B, K]
    mx::array lList = mx::array(0.0f);
    if (a.anySel) {
      mx::array ss = mx::take(s, a.sel, 0);
      mx::array ts = mx::take(a.tgt, a.sel, 0);
      mx::array picked = mx::squeeze(mx::take_along_axis(ss, mx::expand_dims(ts, -1), 1), -1);
      lList = mx::mean(mx::logsumexp(ss, -1, false) - picked);
    }
    const int V = cfg.nChar;
    mx::array lg2 = mx::reshape(lg, {-1, V});
    mx::array gi = mx::maximum(a.gold, mx::array(0));
    mx::array ce = mx::logsumexp(lg2, -1, false) -
                   mx::squeeze(mx::take_along_axis(lg2, mx::expand_dims(gi, -1), 1), -1);
    mx::array lMle = mx::sum(ce * a.valid) / mx::sum(a.valid);
    mx::array loss = lList + mx::array(static_cast<float>(recipe.mleWeight)) * lMle;
    return {loss, lList, lMle};
  }

  std::pair<std::vector<mx::array>, std::vector<mx::array>> ValueAndGrad(const BatchArrays& a) {
    std::vector<int> argnums(params.size());
    std::iota(argnums.begin(), argnums.end(), 0);
    auto fn = [this, &a](const std::vector<mx::array>& p) { return Loss(a, p); };
    return mx::value_and_grad(fn, argnums)(params);
  }
};

LoraTrainer::LoraTrainer(std::shared_ptr<const TrainBase> base, const ModelVocab& vocab,
                         Recipe recipe)
    : impl_(new Impl) {
  mx::set_default_device(mx::Device::cpu);
  mx::set_cache_limit(recipe.cacheMB << 20);
  impl_->base = std::move(base);
  impl_->vocab = &vocab;
  impl_->recipe = recipe;
  if (!impl_->base) {
    impl_->error = "no training base";
    return;
  }
  try {
    impl_->ok = impl_->Load();
  } catch (const std::exception& e) {
    impl_->error = std::string("load: ") + e.what();
    impl_->ok = false;
  }
  if (impl_->ok && static_cast<int>(vocab.legal.size()) != impl_->cfg.nSyl) {
    impl_->error = "vocabulary / model size mismatch";
    impl_->ok = false;
  }
  if (impl_->ok) InitLora(0);
}

LoraTrainer::~LoraTrainer() = default;
bool LoraTrainer::ok() const { return impl_->ok; }
const std::string& LoraTrainer::error() const { return impl_->error; }

void LoraTrainer::InitLora(uint64_t seed) {
  auto& I = *impl_;
  I.params.clear();
  mx::array key = mx::random::key(seed);
  for (const auto& l : I.linears) {
    auto ks = mx::random::split(key);
    key = ks.first;
    mx::array A = mx::random::normal({I.recipe.rank, l.in}, mx::float32, 0.0f,
                                     1.0f / std::sqrt(static_cast<float>(l.in)), ks.second);
    I.params.push_back(A);
    I.params.push_back(mx::zeros({l.out, I.recipe.rank}, mx::float32));
  }
  mx::eval(I.params);
  I.m.clear();
  I.v.clear();
}

bool LoraTrainer::SetLoraA(const std::vector<std::vector<float>>& a) {
  auto& I = *impl_;
  if (a.size() != I.linears.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    const auto& l = I.linears[i];
    if (a[i].size() != static_cast<size_t>(I.recipe.rank) * l.in) return false;
    I.params[2 * i] = HostArray(a[i], {I.recipe.rank, l.in});
    I.params[2 * i + 1] = mx::zeros({l.out, I.recipe.rank}, mx::float32);
  }
  mx::eval(I.params);
  return true;
}

LossValues LoraTrainer::Evaluate(const std::vector<const EncodedRow*>& batch) {
  auto& I = *impl_;
  auto a = I.Collate(batch);
  auto [vals, grads] = I.ValueAndGrad(a);
  mx::array nb(0.0f), na(0.0f);
  for (size_t i = 0; i < grads.size(); ++i) {
    mx::array sq = mx::sum(grads[i] * grads[i]);
    na = na + sq;
    if (i % 2 == 1) nb = nb + sq;
  }
  mx::eval(vals[0], vals[1], vals[2], nb, na);
  LossValues r;
  r.loss = vals[0].item<float>();
  r.listwise = vals[1].item<float>();
  r.mle = vals[2].item<float>();
  r.gradNormB = std::sqrt(nb.item<float>());
  r.gradNorm = std::sqrt(na.item<float>());
  return r;
}

bool LoraTrainer::Train(const std::vector<EncodedRow>& user, const std::vector<EncodedRow>& replay,
                        int steps, uint64_t seed,
                        const std::function<bool(const TrainProgress&)>& cont, double* firstLoss) {
  auto& I = *impl_;
  const Recipe& R = I.recipe;
  if (user.empty() || steps <= 0) {
    I.error = "no training data";
    return false;
  }
  if (R.batchReplay > 0 && replay.size() < static_cast<size_t>(R.batchReplay)) {
    I.error = "replay set too small";
    return false;
  }
  std::mt19937_64 rng(seed ^ 0x9e3779b97f4a7c15ULL);
  I.m.clear();
  I.v.clear();
  for (const auto& p : I.params) {
    I.m.push_back(mx::zeros_like(p));
    I.v.push_back(mx::zeros_like(p));
  }
  std::vector<size_t> order;
  std::vector<size_t> repIdx(replay.size());
  std::iota(repIdx.begin(), repIdx.end(), 0);
  const float b1 = static_cast<float>(R.beta1), b2 = static_cast<float>(R.beta2);
  const float eps = static_cast<float>(R.eps);
  for (int step = 0; step < steps; ++step) {
    if (order.size() < static_cast<size_t>(R.batchUser)) {
      std::vector<size_t> idx(user.size());
      std::iota(idx.begin(), idx.end(), 0);
      std::shuffle(idx.begin(), idx.end(), rng);
      order.insert(order.end(), idx.begin(), idx.end());
    }
    std::vector<const EncodedRow*> batch;
    const size_t nu = std::min(order.size(), static_cast<size_t>(R.batchUser));
    for (size_t i = 0; i < nu; ++i) batch.push_back(&user[order[i]]);
    order.erase(order.begin(), order.begin() + nu);
    // random.sample: batchReplay distinct rows (partial Fisher-Yates).
    for (int i = 0; i < R.batchReplay; ++i) {
      std::uniform_int_distribution<size_t> d(i, repIdx.size() - 1);
      std::swap(repIdx[i], repIdx[d(rng)]);
      batch.push_back(&replay[repIdx[i]]);
    }
    auto a = I.Collate(batch);
    auto [vals, grads] = I.ValueAndGrad(a);
    mx::array nsq(0.0f);
    for (const auto& g : grads) nsq = nsq + mx::sum(g * g);
    mx::array total = mx::sqrt(nsq);
    mx::array normalizer =
        mx::minimum(mx::array(static_cast<float>(R.clip)) / (total + mx::array(1e-6f)),
                    mx::array(1.0f));
    const float lr = static_cast<float>(ScheduledLr(R, step, steps));
    for (size_t i = 0; i < I.params.size(); ++i) {
      mx::array g = grads[i] * normalizer;
      I.m[i] = mx::array(b1) * I.m[i] + mx::array(1.0f - b1) * g;
      I.v[i] = mx::array(b2) * I.v[i] + mx::array(1.0f - b2) * mx::square(g);
      I.params[i] = I.params[i] - mx::array(lr) * I.m[i] / (mx::sqrt(I.v[i]) + mx::array(eps));
    }
    std::vector<mx::array> toEval = I.params;
    toEval.insert(toEval.end(), I.m.begin(), I.m.end());
    toEval.insert(toEval.end(), I.v.begin(), I.v.end());
    toEval.push_back(vals[0]);
    mx::eval(toEval);
    const double loss = vals[0].item<float>();
    if (!std::isfinite(loss)) {
      I.error = "non-finite loss";
      return false;
    }
    if (step == 0 && firstLoss) *firstLoss = loss;
    if (cont && !cont({step + 1, steps, loss})) {
      I.error = "cancelled";
      return false;
    }
  }
  return true;
}

bool LoraTrainer::Merge(Personalize::PersonalWeights* out, std::string* error) const {
  const auto& I = *impl_;
  out->factorySha = I.base->factorySha;
  out->tensors.clear();
  for (size_t li = 0; li < I.linears.size(); ++li) {
    const Linear& l = I.linears[li];
    const BaseTensor* t = I.base->find(l.torchName);
    if (t == nullptr || t->kind != TensorKind::kLinearInt8) {
      if (error) *error = "linear " + l.torchName + " is not an int8 factory linear";
      return false;
    }
    std::vector<float> w = Personalize::Dequantize(t->linear);
    std::vector<float> A = loraA(li), B = loraB(li);
    Personalize::MergeLoRA(&w, t->linear.in, t->linear.out, A.data(), B.data(),
                           static_cast<uint32_t>(I.recipe.rank), I.recipe.loraScale);
    Personalize::QuantizedLinear q;
    q.name = t->linear.name;
    Personalize::QuantizePerChannel(w.data(), t->linear.in, t->linear.out, &q);
    out->tensors.push_back(std::move(q));
  }
  return true;
}

size_t LoraTrainer::trainableParameters() const {
  size_t n = 0;
  for (const auto& p : impl_->params) n += p.size();
  return n;
}

std::vector<float> LoraTrainer::loraA(size_t i) const {
  mx::array a = impl_->params[2 * i];
  mx::eval(a);
  return std::vector<float>(a.data<float>(), a.data<float>() + a.size());
}

std::vector<float> LoraTrainer::loraB(size_t i) const {
  mx::array b = impl_->params[2 * i + 1];
  mx::eval(b);
  return std::vector<float>(b.data<float>(), b.data<float>() + b.size());
}

}  // namespace Trainer
}  // namespace McBopomofo
