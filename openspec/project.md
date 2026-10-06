# AFM Input (AFMBopomofo) — Project Context

> 由 `/spec init` 產生（2026-10-07）。手動維護。研究、實驗與數據在 `../PLAN.md`、`../bench/UNIFIED.md`、`../bench/FINDINGS.md`，不寫進 spec。

## Purpose

macOS 注音輸入法（fork 小麥注音 McBopomofo），每鍵用本機小模型 R（SlothLM-E 微調，嵌進選字 walk，約 1.4 ms）選字；選配「精準模式」在背景以校準閘門呼叫 Qwen3-0.6B 合併打分（一般句 82.5%）；記住使用者手動選字；目標公開發佈，所有資料與模型授權須可再散佈。

## Target Users

- 第一階段：開發者本人（台灣醫學系畢業，日常打繁中、醫學文字）
- 公開發佈後：使用標準注音鍵盤的台灣繁中使用者；之後以領域包（照 macOS 注音「特殊字彙」14 類）擴充專業使用者

## Stack & Constraints

- App：Swift + Objective-C++（InputMethodKit），fork 自 openvanilla/McBopomofo（MIT），分支 `afm-rescoring`
- 選字引擎：C++（`Source/Engine/gramambular2`、`Source/Engine/Rescorer`）
- 推論：onnxruntime 1.30（R / D / ELECTRA int8，CPU）；llama.cpp + Metal（Qwen3-0.6B-Base q8_0）
- 平台：Apple Silicon、macOS 26+；開發用 Xcode 27
- 延遲：每鍵 p95 < 10 ms（R-walk 實測約 1.3/2.1 ms）；精準模式在背景
- 只改組字區，絕不改已送出文字
- 訓練資料禁止：c4、維基百科（CC BY-SA）、CC BY-ND、NC、Apple Foundation Models 生成文字
- 不 sandbox（沿用上游）

## Non-Functional Requirements

- 打字流暢優先：任何模型問題都不得阻塞按鍵處理
- 記憶體：預設模式只載 R（int8 約 13 MB）；精準模式另需約 1 GB（Qwen）
- 讀音政策：「癌」預設 ㄞˊ，ㄧㄢˊ 僅使用者開啟時接受；「一」「不」本調與變調皆接受

## Failure Modes & Constraints

- 模型載入失敗 / 推論失敗 / 逾時 → 依序退回：精準模式 → R-walk → D → 原版 walk；不得讓輸入法無法打字
- 背景重排結果過期（使用者已繼續輸入）→ 丟棄，不套用
- 使用者手動選字（UserOverrideModel）永遠優先於模型分數

## Out of Scope

- 雲端推論（Jev 等）：打字內容不得離開本機
- 修改已送出到應用程式的文字
- 讀取其他 App 的前文當上下文（隱私考量；未來若做需另開 change）
- 非 Apple Silicon、macOS 26 以下

## Roadmap

- Phase 2（本階段）：選字管線接進 App（R-walk 每鍵 + 精準模式背景）、改名與 bundle id、模型檔存放與載入、偏好設定、安裝與實際打字測試
- 之後：蒸餾 Qwen 進 R、注音組合式輸入（含簡拼）、14 個領域包（西醫詞庫已有雛形）、簽章與公證、公開發佈

## Deploy & Distribution

- 開發期：本機 build，安裝到 `~/Library/Input Methods`
- 模型檔：`~/Library/Application Support/AFMInput/`
- 發佈：簽章 + notarization（之後階段）

## Key People & Sources

- Owner：使用者本人
- 上游：openvanilla/McBopomofo（MIT）；SlothLM-E（Luigi/slothlm-e-12m-zhuyin，Apache-2.0；訓練程式無 LICENSE，發佈前需聯絡作者）
- 授權紀錄：`../train/ranker/DATA_LICENSES.md`、`../lexicon/LICENSES.md`、`../lexicon/MODEL_LICENSES.md`

## Conventions

- Capability slug：kebab-case（例：selection-pipeline、composing-buffer、user-override、reading-policy、preferences、model-loading）
- Commit message 英文；與使用者互動繁體中文
- 引擎改動需有 gtest；benchmark 用 `rescorer_bench`，主要指標為 `../bench/data/general/test.jsonl`

## Related Projects / References

- `../PLAN.md`（決策與進度）、`../bench/UNIFIED.md`（方案比較總表）
- 競品：workfunction/McBopomofoLM（`../research/08_mcbopomofolm_teardown.md`）
