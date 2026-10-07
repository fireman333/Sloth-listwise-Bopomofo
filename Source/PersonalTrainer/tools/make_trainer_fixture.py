#!/usr/bin/env python3
"""Fixtures for the AFMPersonalTrainer gtests (personal-finetune task 3.2).

  ../bench/.venv/bin/python Source/PersonalTrainer/tools/make_trainer_fixture.py   (from AFMBopomofo/)

Writes
  Source/PersonalTrainer/testdata/fixed_batch.json  the 16 §3 fixed batch (pool[:16] + replay[:16]
      of research/spike_runtime/data/rows.pkl, encoded rows only, no text) + the PyTorch reference
      loss / LoRA-B grad norm
  research/spike_runtime/data/fixture_base_fp32.bin  train-base.bin format (AFMRTB01, every tensor fp32 =
      the clean_v1 fp32 checkpoint of the spike, factory SHA zero) plus the 84 LoRA A initial matrices
      as extra fp32 tensors "lora_A.<i>" (torch add_lora order). ~47 MB, kept outside the repo.
  research/spike_runtime/data/parity_rows.json  encoded pool-1,000 + 3,000 replay rows (no text) for
      tools/trainer_parity (188-step run vs the MLX Python run)
"""
import hashlib, json, os, pickle, struct
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
ROOT = os.path.dirname(REPO)
D = os.path.join(ROOT, "research", "spike_runtime", "data")

rows = pickle.load(open(os.path.join(D, "rows.pkl"), "rb"))
batch = rows["pool"][:16] + rows["replay"][:16]
ref = json.load(open(os.path.join(D, "torch_ref.json")))
out = {"loss": ref["loss"], "l_list": ref["l_list"], "l_mle": ref["l_mle"], "gradnorm_B": ref["gradnorm_B"],
       "rows": [{"syl": r["syl"], "gold": r["gold"], "cands": r["cands"], "target": r["target"],
                 "ngram": [float(x) for x in r["ngram"]]} for r in batch]}
with open(os.path.join(HERE, "..", "testdata", "fixed_batch.json"), "w") as f:
    json.dump(out, f, separators=(",", ":"))

cfg = json.load(open(os.path.join(D, "config.json")))
w = np.load(os.path.join(D, "weights.npz"))
A = np.load(os.path.join(D, "lora_A.npz"))
buf = bytearray(b"AFMRTB01") + bytes(32)
c = json.dumps({k: cfg[k] for k in ("n_syl", "n_char", "dim", "depth", "heads", "kv", "ffn")}).encode()
buf += struct.pack("<I", len(c)) + c
tensors = [(k, w[k].astype(np.float32)) for k in w.files] + [(f"lora_A.{k}", A[k].astype(np.float32)) for k in A.files]
buf += struct.pack("<I", len(tensors))
for name, arr in tensors:
    nb = name.encode()
    buf += struct.pack("<I", len(nb)) + nb + bytes([0]) + struct.pack("<I", arr.ndim)
    buf += b"".join(struct.pack("<I", d) for d in arr.shape) + np.ascontiguousarray(arr).astype("<f4").tobytes()
buf += hashlib.sha256(buf).digest()
with open(os.path.join(D, "fixture_base_fp32.bin"), "wb") as f:
    f.write(buf)
print("fixed batch rows", len(batch), "base tensors", len(tensors), "bytes", len(buf))

# Parity run of task 3.2 (tools/trainer_parity.cpp): the spike's encoded pool-1,000 + 3,000 replay rows.
enc = lambda r: {"syl": r["syl"], "gold": r["gold"], "cands": r["cands"], "target": r["target"],
                 "ngram": [float(x) for x in r["ngram"]]}
with open(os.path.join(D, "parity_rows.json"), "w") as f:
    json.dump({"pool": [enc(r) for r in rows["pool"]], "replay": [enc(r) for r in rows["replay"]]}, f,
              separators=(",", ":"))
print("parity rows", len(rows["pool"]), len(rows["replay"]))
