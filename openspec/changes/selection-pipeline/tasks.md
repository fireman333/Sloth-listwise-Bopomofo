# Tasks

> 前置：`app-identity` change 完成後才做第 6 組（安裝實測）。第 1 組不碰 Xcode project，可與 `app-identity` 平行；第 2–5 組會改 `project.pbxproj`，排在 `app-identity` 的 pbxproj 改動之後。

## 1. Engine: score cache and split walk (C++, gtest)

- [x] 1.1 In `Source/Engine/Rescorer/neural_walk.h`, add `ComputeCharScores(grid, model/scorer, options)` and `WalkWithScores(grid, scores, alpha)`; reimplement `NeuralWalk` on top of them with identical behavior. Verify: existing `neural_walk_test` passes unchanged with the default `RWALK_MODEL_DIR` (the model its `rwalk_fixture.tsv` was generated from; clean_v1 parity is covered by 5.4), and the new model-free tests pass when built with `-DRWALK_MODEL_DIR=<AFM Input>/train/ranker/export/clean_v1`.
- [x] 1.2 Add header-only `NeuralScoreCache` (key = readings + `charsAtPositions()`, one entry, empty tables never stored) and gtests: identical key → hit; same length but different readings → miss; changed per-position chars → miss; failed (empty) forward not cached. Verify: `ctest` in the Rescorer build passes.
- [x] 1.3 Add pure `NeuralCircuitBreaker` (3 consecutive > 10 ms → tripped; record only for keystrokes with inference) with gtests for: 3 slow → tripped; slow, fast, slow, slow → not tripped; tripped is sticky. Verify: `ctest` passes.

## 2. Xcode build integration

- [x] 2.1 Add `sloth_rescorer.{h,cpp}`, `rescorer.h`, `neural_walk.h` to the `McBopomofo` target; set `HEADER_SEARCH_PATHS` to `"$(SRCROOT)/Source/Engine"` and `"$(SRCROOT)/../vendor/onnxruntime/onnxruntime-osx-arm64-1.30.0/include"` (quoted); define `ORT_API_MANUAL_INIT`; raise `MACOSX_DEPLOYMENT_TARGET` to ≥ 14 unless app-identity already set 26.0. Verify: `xcodebuild -scheme McBopomofo -configuration Debug build` succeeds.
- [x] 2.2 Embed the dylib as `Contents/Frameworks/libonnxruntime.1.dylib` (Copy Files, Code Sign On Copy), weak-link it (`-weak_library`), add `@executable_path/../Frameworks` to `LD_RUNPATH_SEARCH_PATHS`. Verify: `otool -L` shows `@rpath/libonnxruntime.1.dylib` marked weak; `codesign --verify --deep` passes; a copy of the built app with the dylib deleted still launches (log shows selector state `failed`).

## 3. Model holder and loading (model-loading spec)

- [x] 3.1 Use app-identity's `UserPhraseLocationHelper.rWalkModelDirectory` (`<app-support root>/models/r-walk/`) as the only model-dir source, and add `tools/install_models.sh` copying only `onnx/model_quantized.onnx`, `syl_vocab.json`, `phonetic_table.tsv`, `tokenizer/tokenizer.json` from clean_v1. Verify: script lists the 4 files (≈ 14 MB total) and `xcodebuild build` still succeeds.
- [x] 3.2 Implement `AFMNeuralSelector` per design D2 (injectable dir + scorer, `+resetForTesting`, heap-allocated never-freed, `OrtGetApiBase` presence check + `Ort::InitApi()`, background load publishing only while loading, `os_log` once for missing dir / load failure). Verify XCTests: nonexistent dir → `failed`, no crash; truncated copy of the onnx file → `failed`; real dir → `ready` within a timeout (`XCTSkip` if the dir is absent).
- [x] 3.3 Trigger loading at startup / activation without blocking, and on activation when the switch turns on after starting off. Verify: XCTest shows activation returns before a slow fake load completes; record measured real load time in `../PLAN.md`.

## 4. Preferences (hidden defaults)

- [x] 4.1 Add `AFMNeuralWalkEnabled` (Bool, default YES) and `AFMNeuralWalkAlpha` (Double, default 0.5) in `Source/Preferences.swift` with the `@UserDefault` pattern; read them on activation / `syncWithPreferences`. Verify: `PreferencesTests` cover defaults and overrides; `xcodebuild test` passes.

## 5. KeyHandler integration (selection-pipeline spec)

- [x] 5.1 Route `- (void)_walk` through `AFMNeuralSelector` in Bopomofo mode; stock `_grid->walk()` for Plain Bopomofo, disabled, alpha 0, not ready, failed, tripped, and > 40 syllables. Verify: all existing `KeyHandlerBopomofoTests` / `KeyHandlerPlainBopomofoTests` pass with the selector reset to no model (stock behavior unchanged).
- [x] 5.2 Wire `NeuralScoreCache` and `NeuralCircuitBreaker` into the selector. Verify XCTests with a counting fake scorer: one forward per new reading, zero for cursor moves; fake slow scorer ×3 → stock walk afterwards; fake scorer returning empty → stock walk, next keystroke retries.
- [x] 5.3 Behavior XCTests with a fake scorer: manual candidate pick survives subsequent typing; UOM suggestion applied with the stock override type; commit then continue typing leaves committed text unchanged; Plain Bopomofo with the model loaded equals stock; disabled and alpha 0 with the model loaded equal stock; > 40 syllables uses stock. Verify: `xcodebuild test` passes.
- [x] 5.4 Parity with the real model: generate a golden fixture once with `rescorer_bench --walk-mode --alpha 0.5` on a few `../bench/data/general/dev` sentences where R-walk differs from stock, check it in under `Source/` test resources; XCTest types the readings (empty user phrases, cleared UOM, associated phrases off) and asserts the composing buffer equals the fixture. `XCTSkip` if the model dir is absent. Verify: test passes on the dev machine.

## 6. End-to-end check (after app-identity)

- [ ] 6.1 Build Release, install side by side with stock McBopomofo, real typing smoke test in TextEdit: R-walk active (log); `defaults write … AFMNeuralWalkEnabled -bool NO` + reactivation restores stock behavior; deleting the model dir + `pkill -x Sloth-listwise-Bopomofo` (or bundle-id terminate) keeps typing working. Record results in `../PLAN.md`.
  - Real-typing smoke test 2026-10-07 (installed Release b5824b5, TextEdit): R-walk on → 「儘管我台語說得不好」 (fixture gd0002 R-walk result); `defaults write AFMNeuralWalkEnabled NO` + Ctrl-Space away/back → 「…說的…」 (stock); YES + Ctrl-Space → 「…說得…」 again. Note: programmatic `TISSelectInputSource` switching does not call activateServer, so it can't be used to test this.
- [ ] 6.2 Measure in-process per-keystroke latency (inference + walk) up to 40 syllables and IME RSS; confirm p95 < 10 ms. Record in `../PLAN.md` / `../bench/UNIFIED.md`.
  - Test-host numbers recorded 2026-10-07 (Release, in-process XCTest: after /simplify: p95 3.86 ms, max 5.41 ms, phys_footprint ≈ 132 MB after test (+77–98 MB from model load), breaker not tripped; see `../PLAN.md`, `../bench/UNIFIED.md`). Installed-IME measurement still pending, so 6.2 stays unticked.

## 7. Docs

- [x] 7.1 Update `openspec/project.md` and `openspec/config.yaml` `context` fallback chain to "precise mode → R-walk → stock walk" (D removed). Verify: `openspec validate --all` passes.
