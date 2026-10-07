#include "personal_weights.h"

#include <CommonCrypto/CommonDigest.h>
#include <sys/stat.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <unordered_set>

namespace McBopomofo {
namespace Personalize {

Sha256Digest Sha256(const void* data, size_t size) {
  Sha256Digest d{};
  CC_SHA256_CTX ctx;
  CC_SHA256_Init(&ctx);
  const auto* p = static_cast<const uint8_t*>(data);
  // CC_LONG is 32-bit: feed in chunks.
  while (size > 0) {
    const size_t n = size > (1u << 30) ? (1u << 30) : size;
    CC_SHA256_Update(&ctx, p, static_cast<CC_LONG>(n));
    p += n;
    size -= n;
  }
  CC_SHA256_Final(d.data(), &ctx);
  return d;
}

bool ReadFileBytes(const std::string& path, std::string* out, std::string* error) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    if (error) *error = "cannot open " + path;
    return false;
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  if (in.bad()) {
    if (error) *error = "read error " + path;
    return false;
  }
  *out = ss.str();
  return true;
}

bool Sha256File(const std::string& path, Sha256Digest* out, std::string* error) {
  std::string bytes;
  if (!ReadFileBytes(path, &bytes, error)) return false;
  *out = Sha256(bytes.data(), bytes.size());
  return true;
}

std::string ToHex(const Sha256Digest& digest) {
  static const char* kHex = "0123456789abcdef";
  std::string s;
  s.reserve(64);
  for (uint8_t b : digest) {
    s += kHex[b >> 4];
    s += kHex[b & 15];
  }
  return s;
}

namespace {

void PutU32(std::string& s, uint32_t v) {
  for (int i = 0; i < 4; ++i) s += static_cast<char>((v >> (8 * i)) & 0xFF);
}

class Cursor {
 public:
  Cursor(const uint8_t* p, size_t n) : p_(p), n_(n) {}
  bool U32(uint32_t* v) {
    if (n_ - pos_ < 4) return false;
    *v = static_cast<uint32_t>(p_[pos_]) | (static_cast<uint32_t>(p_[pos_ + 1]) << 8) |
         (static_cast<uint32_t>(p_[pos_ + 2]) << 16) | (static_cast<uint32_t>(p_[pos_ + 3]) << 24);
    pos_ += 4;
    return true;
  }
  bool Bytes(void* dst, size_t n) {
    if (n_ - pos_ < n) return false;
    if (n) std::memcpy(dst, p_ + pos_, n);
    pos_ += n;
    return true;
  }
  size_t remaining() const { return n_ - pos_; }

 private:
  const uint8_t* p_;
  size_t n_;
  size_t pos_ = 0;
};

}  // namespace

std::string SerializeWeights(const PersonalWeights& w) {
  std::string s;
  size_t total = 8 + 32 + 4 + 32;
  for (const auto& t : w.tensors) {
    if (t.scales.size() != t.out || t.values.size() != static_cast<size_t>(t.in) * t.out) return {};
    total += 4 + t.name.size() + 8 + 4 * t.scales.size() + t.values.size();
  }
  s.reserve(total);
  s.append(kWeightsMagic, 8);
  s.append(reinterpret_cast<const char*>(w.factorySha.data()), 32);
  PutU32(s, static_cast<uint32_t>(w.tensors.size()));
  for (const auto& t : w.tensors) {
    PutU32(s, static_cast<uint32_t>(t.name.size()));
    s += t.name;
    PutU32(s, t.in);
    PutU32(s, t.out);
    // float32 little-endian == host layout on arm64 / x86_64.
    s.append(reinterpret_cast<const char*>(t.scales.data()), 4 * t.scales.size());
    s.append(reinterpret_cast<const char*>(t.values.data()), t.values.size());
  }
  Sha256Digest h = Sha256(s.data(), s.size());
  s.append(reinterpret_cast<const char*>(h.data()), 32);
  return s;
}

bool WriteWeightsFile(const std::string& path, const PersonalWeights& w, std::string* error) {
  std::string bytes = SerializeWeights(w);
  if (bytes.empty()) {
    if (error) *error = "inconsistent tensor buffers";
    return false;
  }
  const std::string tmp = path + ".tmp";
  FILE* f = std::fopen(tmp.c_str(), "wb");
  if (!f) {
    if (error) *error = "cannot create " + tmp;
    return false;
  }
  chmod(tmp.c_str(), 0600);
  bool ok = std::fwrite(bytes.data(), 1, bytes.size(), f) == bytes.size();
  ok = (std::fflush(f) == 0) && ok;
  ok = (std::fclose(f) == 0) && ok;
  if (!ok || std::rename(tmp.c_str(), path.c_str()) != 0) {
    std::remove(tmp.c_str());
    if (error) *error = "cannot write " + path;
    return false;
  }
  return true;
}

std::shared_ptr<const PersonalWeights> ParseWeights(const void* data, size_t size,
                                                     const WeightsReadOptions& opt,
                                                     std::string* error) {
  auto fail = [&](const std::string& m) -> std::shared_ptr<const PersonalWeights> {
    if (error) *error = m;
    return nullptr;
  };
  try {
    const auto* p = static_cast<const uint8_t*>(data);
    if (p == nullptr || size < 8 + 32 + 4 + 32) return fail("truncated");
    if (std::memcmp(p, kWeightsMagic, 8) != 0) return fail("bad magic");
    const size_t body = size - 32;
    Sha256Digest h = Sha256(p, body);
    if (std::memcmp(h.data(), p + body, 32) != 0) return fail("trailer hash mismatch");

    Cursor c(p + 8, body - 8);
    auto w = std::make_shared<PersonalWeights>();
    c.Bytes(w->factorySha.data(), 32);
    uint32_t count = 0;
    c.U32(&count);
    if (count == 0) return fail("no tensors");
    if (opt.expectedTensors != 0 && count != opt.expectedTensors)
      return fail("tensor count " + std::to_string(count) + " != " +
                  std::to_string(opt.expectedTensors));
    // Each tensor needs at least 4 + 1 + 8 + 4 + 1 bytes.
    if (count > c.remaining() / 18) return fail("tensor count exceeds file size");
    std::unordered_set<std::string> names;
    w->tensors.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
      QuantizedLinear t;
      uint32_t nameLen = 0;
      if (!c.U32(&nameLen)) return fail("truncated");
      if (nameLen == 0 || nameLen > 256) return fail("bad name length");
      t.name.resize(nameLen);
      if (!c.Bytes(t.name.data(), nameLen)) return fail("truncated");
      for (char ch : t.name) {
        if (static_cast<unsigned char>(ch) < 0x21 || static_cast<unsigned char>(ch) > 0x7E)
          return fail("bad name");
      }
      if (!names.insert(t.name).second) return fail("duplicate tensor " + t.name);
      if (!c.U32(&t.in) || !c.U32(&t.out)) return fail("truncated");
      if (t.in == 0 || t.out == 0 || t.in > opt.maxDim || t.out > opt.maxDim)
        return fail("bad shape for " + t.name);
      if (opt.expectedShapes) {
        auto it = opt.expectedShapes->find(t.name);
        if (it == opt.expectedShapes->end()) return fail("unexpected tensor " + t.name);
        if (it->second.first != t.in || it->second.second != t.out)
          return fail("shape mismatch for " + t.name);
      }
      const size_t nv = static_cast<size_t>(t.in) * t.out;
      if (c.remaining() < 4 * static_cast<size_t>(t.out) + nv) return fail("truncated");
      t.scales.resize(t.out);
      c.Bytes(t.scales.data(), 4 * static_cast<size_t>(t.out));
      for (float s : t.scales) {
        if (!std::isfinite(s) || s < 0) return fail("bad scale in " + t.name);
      }
      t.values.resize(nv);
      c.Bytes(t.values.data(), nv);
      w->tensors.push_back(std::move(t));
    }
    if (c.remaining() != 0) return fail("trailing bytes");
    return w;
  } catch (...) {
    return fail("exception");
  }
}

std::shared_ptr<const PersonalWeights> ReadWeightsFile(const std::string& path,
                                                        const WeightsReadOptions& options,
                                                        std::string* error) {
  std::string bytes;
  if (!ReadFileBytes(path, &bytes, error)) return nullptr;
  return ParseWeights(bytes.data(), bytes.size(), options, error);
}

}  // namespace Personalize
}  // namespace McBopomofo
