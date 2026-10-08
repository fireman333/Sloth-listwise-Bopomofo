# Proposal: personal-finetune

## Why

The fast loop (change `personal-memory`) remembers exact picks in their exact context, but R itself never learns the user's vocabulary, so a medical term picked once is still mis-ranked in a new sentence. The feasibility study (`../research/14_personalization_finetune_feasibility.md`) shows that a small LoRA fine-tune of R on the user's own sentence pool adds +6.8 pt on held-out user text on top of a fast-loop approximation at pool 1,000 (p<0.001), with general accuracy within −1.0 pt, and survives int8 re-quantization (§7); the runtime study (`../research/16_on_device_training_runtime.md`) shows it can run in ~30 s on the CPU with MLX and that, in a Python onnxruntime spike, merged weights can replace the shipped weights at session creation without editing the model file. The C++ onnxruntime path and the accuracy of fine-tuned int8 weights in it are **not yet confirmed** (research 16 §5 risk 4); a C++ go/no-go experiment is the first task of this change. This change is the "slow loop" of the on-device personalization plan (grill 2026-10-07, `~/.claude/scratch/grilled-AFM-本機微調R-2026-10-07.md`; `../PLAN.md` "On-device personalization").

## What Changes

- New trainer helper: a separate executable bundled in the app that fine-tunes R on this Mac from the sentence pool collected by `personal-memory` (LoRA r=8 on the 84 linear layers + 1:1 general replay, CPU-only MLX, hand-written AdamW / clipping / cosine schedule, capped buffer cache). It runs at background priority, never inside the input-method process, never blocks typing, and has no network access.
- Training data: sentences committed with a manual correction, and sentences whose default pick was kept and not deleted right after commit. New: a commit that is followed immediately by Backspace/Delete passed through to the application is marked in the pool as retracted and excluded (the pool did not record this before).
- Triggers: automatic when on AC power and idle (at most once a day, only when enough new sentences arrived), plus a manual 「立即學習」 button in the 「選字模型」 tab with progress and cancel. Training starts only when at least 1,000 eligible sentences exist (initial value).
- Acceptance gate (initial values, to be calibrated): the newest 30% of eligible segments, clamped to 300–400, are held out (set by the power simulation, task 1.7 / research 17); a new version is adopted only if personal held-out accuracy improves by ≥ +2 pt with p < 0.05 (paired exact one-sided McNemar) over the factory R and general accuracy (shipped general test set) drops by ≤ 1.0 pt; otherwise the current model stays.
- Weight round trip: LoRA is merged into the 84 linear layers and re-quantized per output channel with the shipped rule (this rule reproduces the shipped int8 file exactly only for a zero LoRA delta; every candidate is validated through the real C++ onnxruntime gate, never assumed equivalent); the input method loads the shipped `model_quantized.onnx` unchanged and replaces those weights at session creation. Personalized weights are stored as separate files under `memory/r-personal/`.
- Versions and rollback: the active and the previous personalized version are kept; one click returns to the factory model or to the previous version. Personalized weights are bound to the factory R, the R-walk weight (alpha), the reading-policy setting and the pack version; on any mismatch they are not loaded (factory R is used) and retraining is allowed at once. Personalized weights built for another factory R are deleted; after a factory R update the pool is retrained on the new R. Clearing learning data also deletes personalized weights and returns to the factory model immediately.
- Precise mode: Qwen is not personalized. When R runs with personalized weights, precise mode uses a gate calibration (temperatures) refitted by the helper for that version; without a matching calibration precise mode is unavailable and R-walk continues.
- The input method ships a small personalization pack next to the R model (training base weights, general replay sentences, general gate and calibration sets) built only from redistributable sources (no c4, Wikipedia, CC BY-ND, NC or Apple Foundation Models output; no Tatoeba, see design D4).
- Priority is unchanged: user phrase > fast-loop memory > (personalized) R.

## Capabilities

### New Capabilities
- `personal-finetune`: on-device fine-tuning of R from the sentence pool — eligibility and the retraction marker, triggers, process isolation and resource limits, acceptance gate, versions / rollback / factory update / clearing, status reporting, on-device guarantees.

### Modified Capabilities
- `model-loading`: "Model location" — R may run with adopted personalized weights from `memory/r-personal/` built for the installed factory model; "Load failure handling" — invalid or incompatible personalized weights fall back to the factory R, not to the stock walk.
- `precise-mode`: "Heavy model and calibration files" — with personalized R, the gate uses the calibration refitted for that version; missing or mismatched → precise mode unavailable.
- `preferences`: "Selection-model preferences tab" gains the personalization section (status, last result, 「立即學習」 / cancel, 「改回出廠模型」, 「改用上一版」). This delta restates the requirement as modified by `personal-memory`, so `personal-memory` must be archived first (design, Migration Plan).
- `personal-memory` (not yet in main specs; **archiving it first is task 0.1, a hard prerequisite**): its "Clearing learning data" requirement must also cover personalized weights. Recorded in design D13 and task 9.3; the delta is written after `personal-memory` is archived.

## Impact

- New C++ code (gtest-covered) under `Source/Engine/Personalize/`: pool reader / eligibility / split, personalized-weights file format, merge + int8 quantizer, McNemar gate, temperature refit; `SlothRescorer::Options` gains weight overrides via `OrtApi::AddInitializer` (no effect when absent).
- New helper target `AFMPersonalTrainer` (C++, links McBopomofoLMLib + sloth_rescorer + neural walk + statically linked CPU-only MLX 0.32.x, MIT), embedded in `Contents/MacOS/`, ad-hoc signed with the app; uses the app's embedded `libonnxruntime.1.dylib`.
- App: training controller (scheduler, power / idle checks, spawn, status, cancel, adoption), `AFMNeuralSelector` reload with personalized weights and model identifier, `AFMPreciseController` calibration overlay, KeyHandler retraction marker, `LanguageModelManager.clearLearningData()` extension, Preferences UI and strings.
- Tooling: `../train/personalize/pack/` scripts to build the personalization pack with licence and leak checks.
- Data: `~/Library/Application Support/Sloth-listwise-Bopomofo/memory/r-personal/` (~9.2 MB per version, two versions kept, mode 0600); pack in `models/r-walk/personalize/` (~15–20 MB).
- Size: helper + static MLX CPU ≈ 20 MB added to the app.
- Latency: nothing added to the keystroke path except the O(1) retraction check; training CPU load must keep keystroke p95 < 10 ms (measured, task 7.1).
</content>
</invoke>
