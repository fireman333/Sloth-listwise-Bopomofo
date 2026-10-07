# Proposal: precise-mode-and-preferences

## Why

精準模式（R-walk 的 N-best 經校準閘門，約四成句子在背景呼叫 Qwen3-0.6B 合併打分）在 C++ 已完成，一般句測試集 82.5%（R-walk 76.5%、Qwen 單獨 77.1%），但輸入法本體還用不到；而且它需要一個使用者看得到的開關。同時，「癌」唸 ㄧㄢˊ 的讀音政策（預設只收 ㄞˊ）早已決定，卻還沒在輸入法裡生效——上游詞庫的 `ㄧㄢˊ 癌` 分數甚至高於 `ㄞˊ 癌`。R-walk 的開關目前也只能用 `defaults write` 改。這三件事都需要偏好設定介面，所以合成一個 change。

## What Changes

- **精準模式接進輸入法**：Bopomofo 模式每鍵仍先同步跑 R-walk（組字區立刻顯示 R 的結果）；若精準模式開啟、Qwen 已載入、且閘門判定 R 不夠有把握，就在背景 queue 呼叫 Qwen。結果回來時，若組字區的讀音與候選字都沒變、使用者也沒在選字，就把組字區換成合併打分的結果；否則丟棄。使用者已手動選的字永遠保留。
- **Qwen 模型**：不打包進 App；由 `tools/install_models.sh` 放進 `<app-support>/models/qwen/`（GGUF q8_0 約 640 MB + 校準參數 `precise.conf`）。精準模式預設關閉；打開才在背景載入（約 1 GB 記憶體），關閉即釋放。檔案缺失、載入失敗、校準檔錯誤 → 精準模式不啟用，退回 R-walk。退回順序定為 **精準模式 → R-walk → 原版 walk**。
- **讀音政策**：新增執行期的讀音過濾層。「癌」對應 ㄧㄢˊ 的條目（單字與詞，例：癌、抗癌、致癌、血癌）預設不出現在組字與候選字中，Bopomofo 與純注音模式都套用，聯想詞也一樣；使用者在偏好設定打開後才接受。不修改 `data.txt`。
- **偏好設定新分頁「選字模型」**：R 神經選字開關、精準模式開關與模型狀態（關閉 / 載入中 / 可用 / 找不到模型 / 載入失敗 / 太慢已停用）、「癌」接受 ㄧㄢˊ 開關、領域包區塊（14 類，灰色佔位，標示尚未提供）。R 的權重 alpha 仍只用隱藏 defaults key。
- **使用者詞不被 R 蓋掉**：使用者自己加的詞在 R 的模型加分上取各位置最高分，原版會選它時 R-walk 也選它（實機測試發現罕見字使用者詞被 R 蓋掉後加入）。
- **建置**：llama.cpp 以 CMake 預先編成靜態庫（Metal library 內嵌），Xcode App target 連結；`precise_walk`、`qwen_rescorer` 原始碼進 App target。

**不在本 change**：蒸餾、新模型、領域包內容與載入、候選字窗以模型重排、App 內下載模型、簽章公證。

## Capabilities

### New Capabilities
- `precise-mode`：背景 Qwen 重排——觸發條件（閘門）、非同步執行與過期結果的丟棄、合併打分結果套用到組字區的條件、與使用者選字的優先序、Qwen 模型與校準檔的位置、載入 / 釋放 / 失敗行為、不連網、記憶體。
- `reading-policy`：讀音政策——「癌」預設只接受 ㄞˊ、ㄧㄢˊ 由使用者開啟；過濾範圍（單字與詞、兩種輸入模式）；不影響已送出文字。
- `preferences`：「選字模型」分頁的項目、預設值、模型狀態顯示、設定生效時機、領域包佔位。

### Modified Capabilities
- `selection-pipeline`：使用者詞不吃模型扣分（實機測試後加入，見 design D8）；退回順序改為精準模式 → R-walk → 原版 walk；組字區除了按鍵當下，也可在背景結果回來時更新（仍只限組字區）；斷路器計時納入 N-best 與閘門，跳脫時精準模式一併停用；純注音模式與「與上游相同」的條文加上讀音政策例外。
- `model-loading`：R-walk 開關改為也可由偏好設定介面切換，並明確切換的生效時機；「與上游相同」加上讀音政策例外。

## Impact

- 程式：`Source/AFMNeuralSelector.{h,mm}`（提供 N-best / 閘門所需的 R 分數）、新增背景精準模式控制器（Obj-C++）、`Source/KeyHandler.mm`（`_walk` 後觸發背景請求、套用結果的回呼）、`Source/InputMethodController.swift`（背景結果更新 marked text）、`Source/Engine/McBopomofoLM.{h,cpp}`（讀音過濾層）、`Source/Preferences.swift`、`Source/PreferencesUI/*`、`Localizable.strings`（en / zh-Hant）。
- 引擎：`Source/Engine/Rescorer/precise_walk.{h,cpp}` 可能需改為接受外部提供的 R 分數表（重用 `NeuralScoreCache`，避免同一讀音跑兩次 forward）。
- 建置：新增 `tools/build_llama.sh`；`project.pbxproj` 加入 llama/ggml 靜態庫、Metal / Accelerate framework、header 路徑。App 體積增加（llama.cpp 靜態庫約數 MB，模型不進 bundle）。
- 工具：`tools/install_models.sh` 加入 Qwen GGUF 與 `precise.conf`。
- 執行期：精準模式開啟時常駐記憶體約 +1 GB；背景使用 GPU（Metal）。
