#include "train_data.h"

#include <algorithm>
#include <fstream>
#include <set>
#include <sstream>

#include "Personalize/pool_reader.h"
#include "nlohmann/json.hpp"

namespace McBopomofo {
namespace Trainer {

int32_t ModelVocab::sylId(const std::string& s) const {
  auto it = syl.find(s);
  return it == syl.end() ? 1 : it->second;
}

int32_t ModelVocab::charId(const std::string& c) const {
  auto it = chr.find(c);
  return it == chr.end() ? -1 : it->second;
}

bool LoadTrainerVocab(const std::string& dir, int32_t nSyl, ModelVocab* out, std::string* error) {
  using nlohmann::json;
  auto load = [&](const std::string& p, json* j) {
    std::ifstream in(p, std::ios::binary);
    if (!in) {
      if (error) *error = "cannot open " + p;
      return false;
    }
    std::stringstream ss;
    ss << in.rdbuf();
    *j = json::parse(ss.str(), nullptr, false);
    if (j->is_discarded()) {
      if (error) *error = "bad json " + p;
      return false;
    }
    return true;
  };
  *out = ModelVocab();
  out->nSyl = nSyl;
  json syl, tok;
  if (!load(dir + "/syl_vocab.json", &syl) || !load(dir + "/tokenizer/tokenizer.json", &tok))
    return false;
  if (!syl.is_object()) {
    if (error) *error = "bad syl_vocab.json";
    return false;
  }
  for (auto it = syl.begin(); it != syl.end(); ++it) {
    if (!it.value().is_number_integer()) continue;
    out->syl[it.key()] = it.value().get<int32_t>();
  }
  if (tok.contains("model") && tok["model"].contains("vocab") && tok["model"]["vocab"].is_object()) {
    const auto& v = tok["model"]["vocab"];
    for (auto it = v.begin(); it != v.end(); ++it)
      if (it.value().is_number_integer()) out->chr[it.key()] = it.value().get<int32_t>();
  }
  // tokenizers' token_to_id: added tokens take precedence over the model vocab.
  if (tok.contains("added_tokens") && tok["added_tokens"].is_array()) {
    for (const auto& t : tok["added_tokens"])
      if (t.contains("content") && t["content"].is_string() && t.contains("id") &&
          t["id"].is_number_integer())
        out->chr[t["content"].get<std::string>()] = t["id"].get<int32_t>();
  }
  if (out->syl.empty() || out->chr.empty()) {
    if (error) *error = "empty vocabulary";
    return false;
  }
  out->legal.assign(nSyl, {});
  std::ifstream in(dir + "/phonetic_table.tsv");
  if (!in) {
    if (error) *error = "cannot open phonetic_table.tsv";
    return false;
  }
  std::string line;
  while (std::getline(in, line)) {
    auto tab = line.find('\t');
    if (tab == std::string::npos) continue;
    std::string s = line.substr(0, tab);
    std::string chars = line.substr(tab + 1);
    auto t2 = chars.find('\t');  // only the second column holds characters
    if (t2 != std::string::npos) chars = chars.substr(0, t2);
    auto it = out->syl.find(s);
    if (it == out->syl.end() || it->second < 0 || it->second >= nSyl) continue;
    std::set<int32_t> ids;
    for (const auto& c : Personalize::CodePoints(chars)) {
      int32_t id = out->charId(c);
      if (id >= 0) ids.insert(id);
    }
    // Python keeps the last line of a duplicated syllable.
    out->legal[it->second].assign(ids.begin(), ids.end());
  }
  return true;
}

EncodedRow EncodeRow(const ModelVocab& vocab, const std::vector<std::string>& readings,
                     const std::string& target, const std::vector<std::string>& nbest,
                     const std::vector<double>& ngram, size_t maxK) {
  EncodedRow r;
  const size_t T = readings.size();
  for (const auto& s : readings) r.syl.push_back(vocab.sylId(s));
  auto ids = [&](const std::string& text) {
    auto cps = Personalize::CodePoints(text);
    std::vector<int32_t> v(T, -1);
    for (size_t t = 0; t < T && t < cps.size(); ++t) v[t] = vocab.charId(cps[t]);
    return v;
  };
  const size_t K = std::min(maxK, nbest.size());
  for (size_t k = 0; k < K; ++k) {
    r.cands.push_back(ids(nbest[k]));
    r.ngram.push_back(k < ngram.size() ? static_cast<float>(ngram[k]) : 0.0f);
    if (r.target < 0 && nbest[k] == target) r.target = static_cast<int32_t>(k);
  }
  for (const auto& c : Personalize::CodePoints(target)) {
    int32_t id = vocab.charId(c);
    r.gold.push_back(id < 0 ? -100 : id);
  }
  r.gold.resize(T, -100);
  return r;
}

}  // namespace Trainer
}  // namespace McBopomofo
