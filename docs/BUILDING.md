# Building Sloth-listwise Bopomofo (this fork)

The upstream McBopomofo build (open `McBopomofo.xcodeproj`, build the `McBopomofo` scheme) still applies, but this
fork links native libraries and reads model files that are **not stored in this repository**. They live in sibling
folders of the repository, inside one workspace folder (called `AFM Input/` below; any name works).

## Workspace layout

```
AFM Input/
├── AFMBopomofo/                  this repository
├── vendor/
│   ├── onnxruntime/onnxruntime-osx-arm64-1.30.0/   ONNX Runtime prebuilt (include/, lib/)
│   ├── llama.cpp/                                  llama.cpp source, commit 5e03bdd
│   └── mlx-0.32.3-cpu-static/                      MLX 0.32.3 static CPU-only install (include/, lib/libmlx.a)
├── models/                       Qwen3-0.6B-Base-Q8_0.gguf (precise mode, optional)
├── train/ranker/export/clean_v1/ R ranker export (onnx/model_quantized.onnx, syl_vocab.json,
│                                 phonetic_table.tsv, tokenizer/tokenizer.json)
├── train/personalize/pack/out/   personalization pack (optional)
└── bench/                        Python venv (cmake, ninja), benchmark data, test-only models
```

The Xcode project and the CMake files refer to these folders with relative paths
(`$(SRCROOT)/../vendor/...`, `${CMAKE_CURRENT_LIST_DIR}/../../../../vendor/...`), so the layout must match.

Requirements: Apple Silicon, macOS 26 or later, Xcode 27 (the project file is Xcode 27 format), CMake ≥ 3.16 and
Ninja (any install; the scripts fall back to `bench/.venv`), GoogleTest for the C++ tests.

## Third-party libraries

| Library | Version | Where | How to get it |
|---|---|---|---|
| ONNX Runtime | 1.30.0 osx-arm64 | `vendor/onnxruntime/onnxruntime-osx-arm64-1.30.0/` | Download `onnxruntime-osx-arm64-1.30.0.tgz` from the ONNX Runtime GitHub releases (https://github.com/microsoft/onnxruntime/releases/tag/v1.30.0) and unpack it into `vendor/onnxruntime/`. The app links and embeds `lib/libonnxruntime.1.30.0.dylib`. In code, create the environment with `MakeOfflineOrtEnv` (`Source/Engine/Rescorer/ort_env.h`) so the prebuilt package's telemetry stays off. |
| llama.cpp / ggml | commit `5e03bdd` | `vendor/llama.cpp/` | `git clone https://github.com/ggml-org/llama.cpp vendor/llama.cpp && git -C vendor/llama.cpp checkout 5e03bdd`, then from the repository run `tools/build_llama.sh`. It builds static `libllama.a` / `libggml*.a` (Metal on, curl / OpenSSL off) into `AFMBopomofo/build-llama/`, which the Xcode project links, and checks that no network symbols are linked. nlohmann/json is taken from `vendor/llama.cpp/vendor/`. |
| MLX | 0.32.3 (tag `v0.32.3`) | `vendor/mlx-0.32.3-cpu-static/` | Only needed for the `AFMPersonalTrainer` helper. Clone https://github.com/ml-explore/mlx at `v0.32.3` and configure with `-DMLX_BUILD_METAL=OFF -DMLX_BUILD_CPU=ON -DBUILD_SHARED_LIBS=OFF -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_EXAMPLES=OFF -DMLX_BUILD_BENCHMARKS=OFF -DMLX_BUILD_PYTHON_BINDINGS=OFF -DMLX_BUILD_SAFETENSORS=OFF -DMLX_BUILD_GGUF=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_DEPLOYMENT_TARGET=26.0 -DCMAKE_INSTALL_PREFIX=<AFM Input>/vendor/mlx-0.32.3-cpu-static`, then `cmake --build . && cmake --install .`. |

Licences for all of these are listed in `ACKNOWLEDGEMENTS.md`.

## Building the app

1. Set up `vendor/` as above and run `tools/build_llama.sh`.
2. Open `McBopomofo.xcodeproj`, build the `McBopomofo` scheme (it also builds the `AFMPersonalTrainer` helper via
   CMake), or run `xcodebuild -project McBopomofo.xcodeproj -scheme McBopomofo -configuration Debug build`.
3. Install the app into `~/Library/Input Methods` as described in the upstream README.

## Model files and the personalization pack (not in this repository)

Model weights and the personalization pack are not committed. The app works without them: when no model is
installed it falls back to the stock McBopomofo walk.

- **R ranker** (default mode, ~14 MB int8 ONNX): a SlothLM-E 12M model
  (https://huggingface.co/Luigi/slothlm-e-12m-zhuyin, Apache-2.0) fine-tuned by this project. The training
  pipeline is not part of this repository and the fine-tuned export is not distributed yet. Given an export directory
  (`onnx/model_quantized.onnx`, `syl_vocab.json`, `phonetic_table.tsv`, `tokenizer/tokenizer.json`), install it with
  `tools/install_models.sh [target-dir] [source-dir]`; the default target is
  `~/Library/Application Support/Sloth-listwise-Bopomofo/models/r-walk`.
- **Precise mode** (optional, ~600 MB): Qwen3-0.6B-Base in GGUF Q8_0 (https://huggingface.co/Qwen/Qwen3-0.6B-Base,
  Apache-2.0; convert with llama.cpp's `convert_hf_to_gguf.py` and `llama-quantize ... Q8_0`, or use a published
  Q8_0 GGUF) plus a calibration file `precise.conf`. Install both with
  `tools/install_models.sh --with-qwen --qwen-gguf <model.gguf> --precise-conf <precise.conf>`.
- **Personalization pack** (optional, needed for on-device fine-tuning): `train-base.bin`, `replay.jsonl`,
  `gate-general.jsonl`, `calib.jsonl`, `pack.json`, `NOTICE`, built by the project's pack builder from licence-clean
  text only (Common Voice, Taiwan government open data, locally generated text) and installed next to the R model.
  It is bound to one R model by SHA-256 (`pack.json` `factory_sha256`). It is not distributed yet.

## Tests

| Suite | Configure / run |
|---|---|
| Engine (incl. `Personalize`) | `cmake -S Source -B build-engine -G Ninja -DCMAKE_CXX_COMPILER=/usr/bin/clang++ [-DGTest_DIR=...]`, `cmake --build build-engine`, `ctest --test-dir build-engine` |
| Rescorer | same with `-S Source/Engine/Rescorer -B build-rescorer` |
| Trainer | same with `-S Source/PersonalTrainer -B build-trainer` (needs MLX) |
| App | `xcodebuild test -project McBopomofo.xcodeproj -scheme McBopomofo -configuration Debug -destination 'platform=macOS'` |

Some Rescorer and app tests need model files from the workspace: `bench/out/_models/sloth` (original SlothLM-E ONNX),
`train/ranker/export/{clean_v1,v2_listmle05_goldmle}`, `models/` (Qwen GGUF), `models/electra`. The paths are CMake
cache variables (`SLOTH_MODEL_DIR`, `RWALK_MODEL_DIR`, `QWEN_MODEL_DIR`, `ELECTRA_MODEL_DIR`) and can be overridden.
Test fixtures and their sentence sources are documented in each `testdata/SOURCES.md`.
