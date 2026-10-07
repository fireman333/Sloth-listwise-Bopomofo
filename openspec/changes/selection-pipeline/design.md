# Design: selection-pipeline

## Context

動機見 proposal.md - Why。現況（2026-10-07 調查 + 審查 agent 複核）：

- 引擎已具備：`ReadingGrid::walkWithCharBonus(alpha, PositionalCharScores)`（`Source/Engine/gramambular2/reading_grid.h:250`）；`McBopomofo::NeuralWalk(grid, SlothRescorer*, options)`（`Source/Engine/Rescorer/neural_walk.h`，header-only），在模型為 null / 未 ready / alpha 0 / 空 grid / 表長不符時一律退回 `grid.walk()`（`:53`、`:70`）；user-overridden node 不加分（`reading_grid.cpp:308-311`）。`SlothRescorer` 建構子吞掉例外（`sloth_rescorer.cpp:421-448`），失敗時 `ready()==false`，`charLogProbs()` 失敗回空表；它沒有額外的本地依賴（JSON 讀取為 inline），`Ort::Env` 是每個實例自己的成員，不是全域。
- App 端：唯一的 walk 呼叫點是 `KeyHandler.mm:2536-2539` 的 `- (void)_walk`，有 16 處呼叫；主要輸入路徑是 `insertReading` → `_walk` → UOM `suggest` → 可能 `overrideCandidate`（`kOverrideValueWithScoreFromTopUnigram`，`:546`）+ 再 `_walk`。整條流程在 main thread 同步執行。
- `reading_grid.cpp` 已由 Xcode 直接編譯；App target 沒有 `HEADER_SEARCH_PATHS`，靠 headermap 平面 include（`KeyHandler.mm:31`）。Rescorer / onnxruntime 尚未進 Xcode project。
- onnxruntime 預編譯版 `../vendor/onnxruntime/onnxruntime-osx-arm64-1.30.0/`；dylib 的 install name 是 `@rpath/libonnxruntime.1.dylib`，minos 14.0（App 目前 deployment target 13.0）。
- `McBopomofoTests` 是 hosted test（`TEST_HOST=McBopomofo.app`），不需另加原始碼或 dylib；無 entitlements、ad-hoc 簽章、未開 hardened runtime。
- 模型：`../train/ranker/export/clean_v1/`。`SlothRescorer` 實際只讀 `onnx/model_quantized.onnx`（13 MB）、`syl_vocab.json`、`phonetic_table.tsv`、`tokenizer/tokenizer.json`。調校值 alpha = 0.5。
- 實測（C++，1 thread，安靜機器）：R-walk 每鍵 p50 1.3 ms / p95 2.1 ms。模型初始化成本未量。

## Goals / Non-Goals

**Goals:**
- Bopomofo 模式每鍵改用 R-walk，main thread 同步執行。
- 模型在背景 queue 載入；任何失敗（含 dylib 不存在）都退回原版 walk，不崩潰。
- 模型不存在 / 關閉 / alpha 0 時，行為與上游完全一致。

**Non-Goals:**
- 候選字窗（`candidatesAt`）排序維持上游順序，不用 R 重排。
- 精準模式 / Qwen / llama.cpp / Metal：不進本 change 的 build。
- 模型下載、打包、簽章、公證。

## Decisions

> D1、D3、D4 的門檻與做法已於 2026-10-07 以決策清單呈現給使用者，使用者指示審查無誤後直接 apply。

### D1. 推論放在 main thread 同步執行
p95 2.1 ms 遠低於 10 ms 預算；同步可直接沿用 `_walk` → UOM → `buildInputtingState` 流程，不必處理結果過期的競態。
- 替代方案：背景 queue + 回推狀態——留給精準模式（30–50 ms）。

### D2. 新增 Objective-C++ 類別 `AFMNeuralSelector`（暫名），可注入

> 2026-10-07 /simplify 後實作：`sharedSelector` 為 class property（static strong，可直接指定），只有一個 designated init（接受 loader block；nil 目錄 → loader 失敗 → `failed`）；測試以 ObjC block 注入 scorer（`initWithScoreBlockForTesting:loadDelay:`）再包成 `NeuralScorer`；Bopomofo 模式判斷移到 `KeyHandler._walk`，selector 介面為 `walkGrid:`。斷路器對推論失敗（空表）的那次也計時。
- 狀態：loading / ready / failed / disabled / tripped。
- 建構時可注入：模型目錄、以及一個 scorer 函式（`std::function<CharLogProbTable(readings, extraChars)>`）；正式環境用 `SlothRescorer::charLogProbs` 包裝，測試用假 scorer。提供 `+resetForTesting`。
- 背景載入：serial queue 上建 `SlothRescorer`，block 持有強參照；完成後 `dispatch_async(main)` 發佈，且只在狀態仍為 loading 時發佈。推論永遠只在 main thread，不需要鎖。
- selector 與 `SlothRescorer` 以 heap 配置、刻意永不釋放（避免 IME 結束時 C++ 靜態解構順序問題）。
- `walkGrid:inputMode:` 給 `KeyHandler._walk` 呼叫，回傳 `WalkResult`。
- 替代方案：塞進 `LanguageModelManager`——會讓之後精準模式的生命週期更亂，獨立類別較好拆。

### D3. 分數表快取 `NeuralScoreCache`（header-only，放在 `neural_walk.h`）
- key = `grid.readings()` + `grid.charsAtPositions()`；只存 1 筆；**空表（推論失敗）不存**，下一鍵會重試。
- 正確性依據：分數表是 (readings, 各位置候選字) 的純函式；override / UOM / boost 只改 node 分數，不改表；排除詞會改 `charsAtPositions`，因此會使快取失效。
- 同時在 `neural_walk.h` 拆出 `ComputeCharScores` 與 `WalkWithScores` 兩步，原 `NeuralWalk` 改用它們重寫但行為不變。
- 放在 C++ header 的理由：可用 gtest 直接測。

### D4. 慢推論斷路器 `NeuralCircuitBreaker`（純類別）
- 每個「有做推論」的按鍵量一次「推論 + 加分 walk」總時間，呼叫 `recordKeystrokeMs:`；連續 3 次 > 10 ms → tripped，本 process 之後全用原版 walk，`os_log` 記一筆。
- 沒做推論的按鍵（快取命中、游標移動、超過 40 音節）不呼叫 `record`，既不延長也不重置連續計數。
- 純類別、時間由外部餵入 → 測試不需要假時鐘。
- 為什麼不在單鍵內中途放棄：onnxruntime 同步 `Run` 無法廉價中斷；連續超時代表機器狀態異常，停用比每鍵賭運氣穩定。

### D5. 退回順序 R-walk → 原版 walk（不保留 D 層）
使用者 2026-10-07 決定。project.md 退回順序同步更新。

### D6. 建置：Rescorer 原始碼直接加進 McBopomofo target；onnxruntime 弱連結
- 加入 target：`sloth_rescorer.{h,cpp}`、`rescorer.h`、`neural_walk.h`。
- `HEADER_SEARCH_PATHS` 加（加引號，路徑含空白）：`"$(SRCROOT)/Source/Engine"`、`"$(SRCROOT)/../vendor/onnxruntime/onnxruntime-osx-arm64-1.30.0/include"`。
- dylib 以 **`libonnxruntime.1.dylib`** 這個檔名放進 `Contents/Frameworks/`（與 install name 一致），Code Sign On Copy；`LD_RUNPATH_SEARCH_PATHS` 加 `@executable_path/../Frameworks`。
- **弱連結**（`-weak_library`）：dylib 不見時 dyld 不會讓 IME 啟動失敗。
- 定義 `ORT_API_MANUAL_INIT`：onnxruntime C++ header 預設會在靜態初始化時呼叫 `OrtGetApiBase()`，弱連結下 dylib 缺席會在啟動時崩潰；改為 selector 先檢查 `&OrtGetApiBase != nullptr`，再呼叫 `Ort::InitApi()`，否則狀態設為 failed。
- `MACOSX_DEPLOYMENT_TARGET` 提高到 ≥ 14（若 `app-identity` 已提高到 26.0 則沿用）。
- 替代方案：CMake 靜態庫 + Xcode 連結——使用者 2026-10-07 選擇直接加原始碼。

### D7. 開關：隱藏 UserDefaults key
`AFMNeuralWalkEnabled`（Bool，預設 YES）、`AFMNeuralWalkAlpha`（Double，預設 0.5），照 `Preferences.swift` 的 `@UserDefault` 模式加 `@objc static var`，不加 UI。每次 activate / `syncWithPreferences` 讀取。開關一開始是關的也要能在之後打開：switch 為 NO 時不載模型；在 activate 時若發現 switch 變成 YES 且模型尚未載入，就觸發 `startLoadingIfNeeded`。

### D8. Plain Bopomofo 模式不套用
`_inputMode == InputModePlainBopomofo` 時直接 `_grid->walk()`，與上游 UOM 跳過的判斷一致。

### D9. 模型目錄
`<app-support root>/models/r-walk/`，app-support 根目錄由 `app-identity` change 定義（`~/Library/Application Support/Sloth-listwise-Bopomofo/`）；路徑常數由 app-identity 的 `UserPhraseLocationHelper.rWalkModelDirectory` 提供，本 change 不另定義。開發期由 `tools/install_models.sh` 從 `../train/ranker/export/clean_v1/` 複製 4 個必要檔案。

### D10. 神經長度上限 40 音節
組字區超過 40 音節時該鍵直接用原版 walk（`SlothRescorer` 對長度沒有上限，長句 forward 成本隨長度上升，若計入斷路器容易誤停用）。

## Risks / Trade-offs

- [IME process 內 onnxruntime 首次載入可能較慢（數百 ms）] → 背景載入，期間用原版 walk；task 中實測並記錄。
- [dylib 缺失或簽章問題] → 弱連結 + `ORT_API_MANUAL_INIT` + 執行期檢查；task 2.2 實測刪掉 dylib 後 App 仍能啟動。
- [main thread 偶發慢 forward] → 40 音節上限（D10）+ 斷路器（D4）；task 6.2 量 40 音節的延遲。
- [快取用到舊分數] → key 含完整 readings 與各位置候選字（D3），同長度不同讀音也會失效；gtest 覆蓋。
- [記憶體：onnxruntime + 13 MB 模型常駐] → 可接受；task 6.2 量 RSS。
- [與已安裝小麥注音衝突] → 依賴 `app-identity`；在那之前只做 build 與 XCTest，不安裝。
- [XCTest 的 host 是真的 App，會載入真實模型目錄] → selector 可注入目錄與 scorer，測試用假 scorer；需要真模型的測試在目錄不存在時 `XCTSkip`。

## Migration Plan

- 無資料遷移。模型目錄不存在 = 原版行為。
- 回退：`defaults write <bundle id> AFMNeuralWalkEnabled -bool NO` 後重新切換輸入法；或刪除模型目錄後以 `pkill -x Sloth-listwise-Bopomofo`（名稱超過 `killall` 的 16 字元比對上限）讓 IME 重啟。

## Open Questions

- 初始化時間與 RSS 實測數字（task 中量測後記錄）。
