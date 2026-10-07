# Proposal: selection-pipeline

## Why

Phase 1 已在 C++ 引擎完成 R-walk（SlothLM-E 微調模型 clean_v1 的逐字分數嵌進 McBopomofo walk；一般句測試集 76.4% vs 原版 62.2%，每鍵 p95 約 2 ms），但 App 仍呼叫原版 `walk()`，使用者打字時完全用不到。Phase 2 的第一步是把每鍵的 R-walk 接進輸入法本體，並確保任何模型問題都不會讓輸入法無法打字。

## What Changes

- 組字（Bopomofo 模式）每次重新 walk 時改用 R-walk：同一組讀音只跑一次模型 forward，結果供同一按鍵內多次 walk 重用。
- 模型在背景載入；載入完成前、載入失敗、模型被關閉、或推論連續過慢時，一律退回原版 walk。退回順序定為 **R-walk → 原版 walk**（不保留 D 層；project.md 的退回順序一併更新為「精準模式 → R-walk → 原版 walk」）。
- 使用者手動選字（含 UserOverrideModel 的建議）永遠優先於模型分數。
- 純注音模式（Plain Bopomofo）維持原版行為，不套用 R。
- 開關與 alpha 以隱藏的 UserDefaults key 控制（`defaults write`），預設開啟；不加偏好設定 UI。
- 推論程式（Rescorer 原始碼）直接加進 Xcode App target，`libonnxruntime.dylib` 內嵌於 App bundle。

**不在本 change**：精準模式（背景 Qwen，另開 change）、改名與 bundle id（另開 `app-identity` change，**先做**，本 change 的安裝實測依賴它）、偏好設定 UI、候選字窗排序、蒸餾與新模型。

## Capabilities

### New Capabilities
- `selection-pipeline`：每鍵選字的決策順序——R-walk、退回原版、使用者覆寫優先、純注音模式、每鍵延遲預算與過慢時的處理。
- `model-loading`：R 模型的存放位置、背景載入、載入失敗或檔案缺失時的行為、開關。

### Modified Capabilities
（無——目前尚無 main spec）

## Impact

- 程式：`Source/KeyHandler.mm`（`_walk`）、新增一個 Objective-C++ 的模型持有者 / 載入器、`Source/Preferences.swift`（隱藏 key）、`Source/Engine/Rescorer/`（`sloth_rescorer`、`neural_walk.h` 進 App target）。
- 建置：`McBopomofo.xcodeproj` 加入 Rescorer 原始碼、onnxruntime header 搜尋路徑、`libonnxruntime.dylib` 內嵌 + rpath + codesign。
- 執行期：IME process 實測多約 77–98 MB（phys_footprint，測完約 132 MB；原估 15–60 MB 偏低，onnxruntime 記憶體設定調整無明顯效果，待之後另查）。
- 模型檔：`~/Library/Application Support/Sloth-listwise-Bopomofo/models/r-walk/`（根目錄由 app-identity 定義；開發期以 script 從 `../train/ranker/export/clean_v1/` 複製）；不打包進 App。
- 依賴：`app-identity` change 完成後才能和已安裝的小麥注音並存安裝實測。
