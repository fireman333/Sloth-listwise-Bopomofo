#!/usr/bin/env python3
"""Generates the gtest fixtures of Source/Engine/Personalize (personal-finetune group 2) and the
Rescorer personalized-override fixture. Run from `AFM Input/`:

  bench/.venv/bin/python AFMBopomofo/Source/Engine/Personalize/tools/make_fixtures.py

Outputs (no typed text from real pools; sentences come from bench sets and the simulated pool):
  Personalize/testdata/merge_fixture.json         numpy LoRA merge W + 2 * (B A)^T and its per-channel int8
  Personalize/testdata/mcnemar_fixture.tsv        b, c, ranker_lib.mcnemar_exact, scipy one-sided p
  Personalize/testdata/calib_fit_fixture.jsonl    precise-mode fit set (general dev + calib): n_syl, R top-10
                                                  path scores, top-1 correct; header = precise_calib.py values
  Personalize/testdata/nbest_fixture.jsonl        20 E5 rows (10 simulated pool, Qwen3-4B-generated + 10 Common Voice
                                                  replay; see testdata/SOURCES.md): readings, target,
                                                  walkNBest(50) texts / scores (== bench/src/decoder.py, E5)
  Rescorer/testdata/personal_override_fixture.tsv Python-ORT 1.24 candidate scores (sloth_fixture rows) with a
                                                  deterministic personalized override (see PerturbedLinears in
                                                  sloth_rescorer_test.cpp); header line = SHA-256 of the
                                                  AFMRPW01 file this script writes for the same tensors
"""
import hashlib, json, math, os, struct, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PZ = os.path.dirname(HERE)
ENGINE = os.path.dirname(PZ)
ROOT = os.path.normpath(os.path.join(ENGINE, "..", "..", ".."))  # AFM Input/
TD = os.path.join(PZ, "testdata")
RTD = os.path.join(ENGINE, "Rescorer", "testdata")
RMODEL = os.path.join(ROOT, "train/ranker/export/clean_v1")
ONNX = os.path.join(RMODEL, "onnx/model_quantized.onnx")
sys.path.insert(0, os.path.join(ROOT, "train/ranker"))
sys.path.insert(0, os.path.join(ROOT, "bench/src"))


def quant_pc(w):
    amax = np.abs(w).max(axis=0)
    scale = (amax / np.float32(127.0)).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(scale > 0, np.rint(w / scale), 0)
    return np.clip(q, -127, 127).astype(np.int8), scale


def merge_fixture():
    rng = np.random.default_rng(20261008)
    n_in, n_out, r = 12, 10, 8
    W = rng.normal(0, 0.05, (n_in, n_out)).astype(np.float32)
    A = rng.normal(0, 1 / math.sqrt(n_in), (r, n_in)).astype(np.float32)
    B = rng.normal(0, 0.02, (n_out, r)).astype(np.float32)
    delta = (B.astype(np.float64) @ A.astype(np.float64)).T  # [in, out]
    merged = W + (2.0 * delta).astype(np.float32)
    q, s = quant_pc(merged)
    json.dump({"in": n_in, "out": n_out, "r": r, "scale": 2.0, "W": W.ravel().tolist(), "A": A.ravel().tolist(),
               "B": B.ravel().tolist(), "merged": merged.astype(np.float64).ravel().tolist(),
               "q": q.astype(int).ravel().tolist(), "scales": s.astype(np.float64).ravel().tolist()},
              open(os.path.join(TD, "merge_fixture.json"), "w"))


def mcnemar_fixture():
    from ranker_lib import mcnemar_exact
    from scipy.stats import binomtest
    tables = [(0, 0), (6, 0), (0, 5), (1, 1), (10, 3), (3, 9), (50, 40), (120, 80), (200, 170), (500, 430),
              (27, 12), (900, 860)]
    with open(os.path.join(TD, "mcnemar_fixture.tsv"), "w") as f:
        f.write("# b\tc\ttwo_sided(ranker_lib.mcnemar_exact)\tone_sided P(X>=b), X~Bin(b+c,0.5)\n")
        for b, c in tables:
            one = binomtest(b, b + c, 0.5, alternative="greater").pvalue if b + c else 1.0
            f.write(f"{b}\t{c}\t{mcnemar_exact(b, c)!r}\t{float(one)!r}\n")


def calib_fixture():
    import precise_calib as pc
    rows = pc.load_set("dev", os.path.join(pc.GEN, "dev.jsonl")) + \
        pc.load_set("calib", os.path.join(pc.P, "calib.jsonl"))
    K = 10
    tg = pc.fit_T(rows, K)
    tb = pc.fit_temps(rows, K, True)
    ys = [r["cands"][0] == r["gold"] for r in rows]
    p1 = [pc.p_top1(r["r"], 1.0, K) for r in rows]
    pg = [pc.p_top1(r["r"], tg, K) for r in rows]
    pb = [pc.p_top1(r["r"], pc.temp_for(tb, r["n"]), K) for r in rows]
    conf = open(os.path.join(pc.P, "precise.conf")).read()
    factory_T = float([l for l in conf.splitlines() if l.startswith("temperatures")][0].split("=")[1].split()[0])
    head = {"k": K, "bucket_upper": pc.BUCKET_UPPER, "factory_T": factory_T, "py_T_global": tg, "py_T_bucket": tb,
            "ece_T1": pc.ece(p1, ys)[0], "ece_global": pc.ece(pg, ys)[0], "ece_bucket": pc.ece(pb, ys)[0],
            "logloss_T1": pc.logloss(p1, ys), "logloss_global": pc.logloss(pg, ys)}
    with open(os.path.join(TD, "calib_fit_fixture.jsonl"), "w") as f:
        f.write(json.dumps(head) + "\n")
        for r, y in zip(rows, ys):
            f.write(json.dumps({"n": int(r["n"]), "r": [float(x) for x in r["r"][:K]], "y": bool(y)}) + "\n")
    print("calib: factory T", factory_T, "py T", tg, "bucket", tb)


def nbest_fixture():
    """Licence-clean rows only (publish audit 2026-10-08): pool rows whose simulated sentence was generated by local
    Qwen3-4B (engine mlx:*, Apache-2.0; AFM-generated rows excluded), replay rows from Common Voice zh-TW (CC0) via
    e5_compare.py prep-cv (the original E5 replay sample is all Tatoeba)."""
    e5 = os.path.join(ROOT, "train/personalize/e5")
    engine = {}
    for l in open(os.path.join(ROOT, "train/sloth/corpus/work/med_clean.jsonl"), encoding="utf-8"):
        j = json.loads(l)
        engine.setdefault(j["text"], j.get("engine", ""))
    rows = [json.loads(l) for l in open(os.path.join(e5, "rows.jsonl"), encoding="utf-8")]
    cv = [json.loads(l) for l in open(os.path.join(e5, "rows_cv.jsonl"), encoding="utf-8")]
    cpp = {}
    for fn in ("cpp_nbest_full_e5.jsonl", "cpp_nbest_full_e5cv.jsonl"):
        cpp.update({j["id"]: j for j in (json.loads(l) for l in open(os.path.join(e5, fn), encoding="utf-8"))})
    from decoder import Decoder
    dec = Decoder()
    pool = [r for r in rows if r["id"].startswith("pool") and engine.get(r["text"], "").startswith("mlx:")][:10]
    rep = cv[:10]
    assert len(pool) == 10 and len(rep) == 10
    with open(os.path.join(TD, "nbest_fixture.jsonl"), "w", encoding="utf-8") as f:
        for r in pool + rep:
            c = cpp[r["id"]]
            py = dec.decode(r["reading"], k=50)
            texts = [x["text"] for x in py]
            assert texts == c["nbest"], r["id"]  # E5: identical (C++ == Python)
            gold = texts.index(r["text"]) if r["text"] in texts else -1
            f.write(json.dumps({"id": r["id"], "reading": r["reading"], "text": r["text"], "nbest": c["nbest"],
                                "ngram": [float(s) for s in c["ngram"]], "gold": gold}, ensure_ascii=False) + "\n")


def perturbed_linears():
    """Deterministic personalized weights shared with sloth_rescorer_test.cpp: for the t-th MatMulInteger weight
    (graph order), W' = dequant(W) + 2 * (B A)^T with A[k,i] = ((i*31 + k*17 + t*7) % 13 - 6) / 64,
    B[j,k] = ((j*7 + k*5 + t*3) % 11 - 5) / 1024, rank 8 (dyadic values: the merge is exact in fp32/fp64)."""
    import onnx
    from onnx import numpy_helper as nh
    g = onnx.load(ONNX).graph
    init = {i.name: i for i in g.initializer}
    out = []
    for t, n in enumerate([n for n in g.node if n.op_type == "MatMulInteger"]):
        base = n.input[1][: -len("_quantized")]
        q = nh.to_array(init[base + "_quantized"]).astype(np.float32)
        s = nh.to_array(init[base + "_scale"]).astype(np.float32)
        w = q * s
        n_in, n_out = w.shape
        i, k = np.meshgrid(np.arange(n_in), np.arange(8), indexing="xy")  # [8, in]
        A = (((i * 31 + k * 17 + t * 7) % 13) - 6) / 64.0
        j, k2 = np.meshgrid(np.arange(8), np.arange(n_out), indexing="xy")  # [out, 8] (j=rank idx, k2=out idx)
        B = (((k2 * 7 + j * 5 + t * 3) % 11) - 5) / 1024.0
        delta = (B @ A).T  # [in, out] float64, exact
        w2 = w + (2.0 * delta).astype(np.float32)
        qq, ss = quant_pc(w2)
        out.append((base, qq, ss))
    return out


def override_fixture():
    import onnxruntime as ort
    import rescore_sloth as rs
    lin = perturbed_linears()
    # AFMRPW01 bytes (independent writer) -> SHA-256 of the whole file
    fsha = hashlib.sha256(open(ONNX, "rb").read()).digest()
    body = b"AFMRPW01" + fsha + struct.pack("<I", len(lin))
    for base, q, s in lin:
        nb = base.encode()
        body += struct.pack("<I", len(nb)) + nb + struct.pack("<II", *q.shape) + s.astype("<f4").tobytes() + q.tobytes()
    body += hashlib.sha256(body).digest()
    file_sha = hashlib.sha256(body).hexdigest()

    rs.MD = RMODEL
    plain = rs.Sloth("int8", threads=1)
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    keep = []
    for base, q, s in lin:
        for nm, arr in ((base + "_quantized", q), (base + "_scale", s)):
            v = ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(arr)); keep.append(v); so.add_initializer(nm, v)
    pers = rs.Sloth("int8", threads=1)
    pers.sess = ort.InferenceSession(ONNX, so, providers=["CPUExecutionProvider"])
    lines = [l.rstrip("\n").split("\t") for l in open(os.path.join(RTD, "sloth_fixture.tsv"), encoding="utf-8") if l.strip()]
    maxdiff = 0.0
    with open(os.path.join(RTD, "personal_override_fixture.tsv"), "w", encoding="utf-8") as f:
        f.write(f"#\t{file_sha}\n")
        for p in lines[:12]:
            rid, reading, cands = p[0], p[1].split(" "), p[2::2]
            ps = pers.score_cands(reading, cands)[0]
            fs = plain.score_cands(reading, cands)[0]
            maxdiff = max(maxdiff, max(abs(a - b) for a, b in zip(ps, fs)))
            f.write("\t".join([rid, p[1]] + [x for c, s in zip(cands, ps) for x in (c, f"{s:.9f}")]) + "\n")
    print("override fixture: file sha", file_sha, "max |personal - factory| score diff", round(maxdiff, 3))


if __name__ == "__main__":
    os.makedirs(TD, exist_ok=True)
    merge_fixture(); mcnemar_fixture(); calib_fixture(); nbest_fixture(); override_fixture()
    print("ok")
