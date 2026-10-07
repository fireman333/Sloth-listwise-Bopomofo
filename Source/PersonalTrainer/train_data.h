// Encoded training rows for the R fine-tune (personal-finetune design D8):
// port of train/ranker/ranker_lib.py Vocab / legal_matrix / encode
// (inject_gold = False, max_k = 50), using the files of an R model directory
// (syl_vocab.json, tokenizer/tokenizer.json, phonetic_table.tsv).

#ifndef SOURCE_PERSONALTRAINER_TRAIN_DATA_H_
#define SOURCE_PERSONALTRAINER_TRAIN_DATA_H_

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace McBopomofo {
namespace Trainer {

struct ModelVocab {
  std::unordered_map<std::string, int32_t> syl;   // syl_vocab.json
  std::unordered_map<std::string, int32_t> chr;   // tokenizer vocab + added tokens
  // Legal character ids per syllable id (phonetic_table.tsv); empty = every
  // character is legal (pad / unk / syllables without a legal set).
  std::vector<std::vector<int32_t>> legal;
  int32_t nSyl = 0;

  int32_t sylId(const std::string& s) const;   // 1 (= <unk>) when unknown
  int32_t charId(const std::string& c) const;  // -1 when unknown
};

// nSyl = the model's syllable vocabulary size (legal has nSyl entries).
bool LoadTrainerVocab(const std::string& modelDir, int32_t nSyl, ModelVocab* out,
                      std::string* error);

struct EncodedRow {
  std::vector<int32_t> syl;                 // [T]
  std::vector<int32_t> gold;                // [T], -100 = out of vocabulary
  std::vector<std::vector<int32_t>> cands;  // [K][T], -1 = OOV / padding
  std::vector<float> ngram;                 // [K]
  int32_t target = -1;                      // index of the gold text in cands
};

// readings: one syllable per character of `target`; nbest texts and their
// n-gram path scores (best first).
EncodedRow EncodeRow(const ModelVocab& vocab, const std::vector<std::string>& readings,
                     const std::string& target, const std::vector<std::string>& nbest,
                     const std::vector<double>& ngram, size_t maxK = 50);

}  // namespace Trainer
}  // namespace McBopomofo

#endif  // SOURCE_PERSONALTRAINER_TRAIN_DATA_H_
