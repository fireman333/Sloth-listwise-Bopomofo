# Design: precise-mode-and-preferences

## Context

動機見 proposal.md - Why。現況（2026-10-07 調查）：

- **精準模式引擎已完成，但只在 CMake**：`Source/Engine/Rescorer/precise_walk.{h,cpp}`。`PreciseSelect(readings, candidates, rScores, heavy, cfg)` 是「給定 R 的 N-best 分數 → 校準、閘門、合併」的純函式（heavy 呼叫除外），`PreciseWalk` 則自己跑 forward + `walkNBestWithCharBonus`。`PreciseConfig::LoadFile` 讀 `key = value` 校準檔，遇到未知 key 就失敗。校準值在 `../bench/out/general/precise/precise.conf`：tau 0.8、beta 0.7、溫度 1.0814、k 10、timeout 1000 ms。`QwenRescorer(gguf, n_gpu_layers=99)`（`qwen_rescorer.h`）走 Metal，**不是 thread-safe**，呼叫端要自己序列化；載入失敗時 `ready()==false`。llama.cpp 只在 `Rescorer/CMakeLists.txt`（`RESCORER_QWEN`、`GGML_METAL_EMBED_LIBRARY`）；`project.pbxproj` 完全沒有 llama / ggml。
- **實測延遲**（C++，1 thread，1,495 句）：Qwen 單次 p50 31.4 ms、p95 52.3 ms、max 67.8 ms；閘門開啟比例 40.1%。閘門關閉時，含 N-best 的每鍵 p95 為 6.7 ms（句級量測）；不含 heavy 的每鍵 p95 為 5.8 ms。
- **R-walk 在 App 中**：`AFMNeuralSelector`（`Source/AFMNeuralSelector.mm`）。推論與狀態都在 main thread，只有載入在 serial queue；`NeuralScoreCache` 以 readings + `charsAtPositions` 為 key；斷路器 `NeuralCircuitBreaker` 也已實作。呼叫點只有一個：`KeyHandler.mm:2541` 的 `_walk`。
- **App 沒有任何非同步更新組字區的路徑**，也沒有 generation counter。最接近的寫法是 boost / exclude 選單的回呼（`KeyHandler.mm:1564-1590`）：以 weak self 呼叫 `_walk` → `buildInputtingState` → `stateCallback`。IMC 端只有 `handle(state:client:)` 一個入口（`InputMethodController.swift:404`）。
- **詞庫**：`data.txt:163377 ㄧㄢˊ 癌 -4.349`（高於 `:152386 ㄞˊ 癌 -5.042`）。含「癌」的 93 行中有 45 行唸 ㄧㄢˊ（抗癌、致癌、血癌、罹癌…）。`data-plain-bpmf.txt` 兩種讀音都有。`McBopomofoLM::getUnigrams(key)`（`McBopomofoLM.cpp:139`）會依 key 合併使用者詞、排除詞與主詞庫；key 用 `-` 分隔音節。
- **偏好設定**是 SwiftUI：`Source/PreferencesUI/PreferencesView.swift` 有三個分頁（basic、userPhrases、advanced）；`PreferencesModel.swift` 是 view model；`Preferences.swift` 用 `@UserDefault` 宣告 key。R-walk 的 key（`AFMNeuralWalkEnabled`、`AFMNeuralWalkAlpha`）目前沒有 UI。
- **模型**：`~/Library/Application Support/Sloth-listwise-Bopomofo/models/r-walk/` 已安裝。Qwen 檔案 `../models/Qwen3-0.6B-Base-Q8_0.gguf`（639 MB）尚未安裝。

## Goals / Non-Goals

**Goals:**
- 同一組讀音在 App 裡只跑一次 R forward：N-best、閘門與每鍵 R-walk 共用同一張分數表。
- 背景 heavy 推論嚴格序列化，最多一個執行中、一個待處理（新的覆蓋舊的）。
- 讀音過濾只在主詞庫與聯想詞那一層生效，精準模式與 R-walk 不需要知道它存在。
- 精準模式關閉或尚未 ready 時，按鍵路徑與現行版本相同（N-best 不計算）。

**Non-Goals:**
- 中途中斷正在跑的 Qwen 推論（llama.cpp decode 無法廉價中斷，只丟棄結果）。
- 候選字窗以 Qwen 重排。
- 跨重新 walk 保留精準模式的選擇（每次重新 walk 都從 R-walk 開始，見 Risks）。
- 「轉注音」服務的讀音政策（見 D4 非目標）。
- 領域包的資料格式與載入。

## Decisions

> 2026-10-07 依 fresh-context 審查（34 項，P1 ×5）修訂。審查原文摘要見本節末「審查處理」。

### D0. 讀音政策是 config 規則「與上游一致」的明文例外
`config.yaml` design 規則要求「alpha=0 / 無模型時行為與上游 McBopomofo 相同」。讀音政策（使用者決定）預設移除 ㄧㄢˊ 的「癌」，這不受模型影響。因此相關 requirement 一律寫成「與上游相同，但 reading-policy 移除的條目除外」，同時把 config 規則改成同樣措辭。

### D1. 前景：先照常 walk，按鍵事件結束時才算 N-best 和閘門
- `_walk` 不變，仍用 `walkWithCharBonus` 決定顯示結果，精準模式不在 `_walk` 裡做任何事。原因是 `_walk` 每個按鍵最多被呼叫兩次（`KeyHandler.mm:545` 和 UOM 建議後的 `:553`），在裡面送請求會導致重複送出，甚至形成循環。
- **送請求的時機**：每個按鍵事件結束後呼叫一次 `_maybeRequestPrecise`。`handleInput` 約有 46 個 return，候選字選取等路徑也不經過它，所以 hook 定為兩處：
  - (a) IMC 在 `keyHandler.handle(...)` 回傳之後（`InputMethodController.swift:278` 附近）呼叫。
  - (b) IMC 在候選字選取後的 `fixNode…` 之後（`InputMethodController+CandidateControllerDelegate.swift:58`、`:108`）呼叫。
  - boost／exclude 回呼與 `handleForceCommit` 不送請求，因為它們會遞增 generation，或者已經送出。
  - `_maybeRequestPrecise` 只在目前 `state is Inputting` 時執行。只有當精準模式 controller 是 ready、R 也是 ready、本次 walk 確實用了 R-walk、而且是 Bopomofo 模式時才會執行：
  1. 從 `NeuralScoreCache` 取出同一張分數表（不重跑 forward）。
  2. 跑 `walkNBestWithCharBonus(k, alpha, table)`。
  3. 過濾掉任何改動「固定位置」的候選（見 D3）。
  4. 確保目前顯示的結果排第 0 名。若它不在 N-best 裡就不送請求；若同分順序不同，就把它移到第 0 名。
  5. 用 `PreciseSelect(..., heavy=nullptr)` 算 pTop1 和閘門。
  6. 閘門開啟、候選至少 2 個、且請求 key 不等於上一次送出或套用的 key，才送出請求。
- `precise_walk` 新增 `PreciseCandidates(grid, table, cfg)` 入口：輸入外部分數表，輸出 N-best 文字、R 分數和路徑。`PreciseWalk` 改用這個入口實作，行為不變。
- **時間計入斷路器**：`walkGrid` 遇到快取 miss 時，只先把推論時間暫存為 `pendingBreakerMs`，不立即記錄。等到事件結束的 hook，加上 N-best 與閘門的時間後一起記錄一次。若 hook 沒有執行（精準模式未 ready 或非 Inputting 狀態），就在下一次 `walkGrid` 開頭或 hook 中先把暫存值補記。這樣每一筆快取 miss 都恰好記錄一次，快取命中的按鍵仍不記錄。
- 預算：閘門關閉時每鍵 p95 6.7 ms。若實測超過，改成只在快取 miss 的按鍵才算 N-best（task 5.5）。
- 替代方案：在 `_walk` 裡算。不採用，原因見上（重複請求、套用後再觸發）。

### D2. 背景：新增 Obj-C++ 類別 `AFMPreciseController`（全域單例）
- 擁有 `QwenRescorer`、`PreciseConfig`、serial queue `<bundleid>.Precise`，以及狀態（off / loading / ready / notFound / failed / tripped）。狀態只在 main thread 改變，改變時發出通知。
- **epoch**：每次 `setEnabled:` 改變值時遞增 `sessionEpoch`。載入完成、推論結果回到 main 時，epoch 不符的一律丟棄。這可以處理「開→關→開」時舊 scorer 已被釋放的競態。
- **載入順序**：先在 queue 上讀 `precise.conf`（`LoadFile`），失敗時，檔案不存在記為 notFound，格式錯記為 failed；conf 裡的 alpha 和 defaults 的 alpha 不同也記為 failed。conf 讀成功後才載入 GGUF，失敗記為 failed。成功後把 `PreciseConfig` 以值拷貝送到 main，再發布 ready。前景閘門只讀 main 上的這份拷貝。
- **請求合併**：main thread 上有 `pending` 與 `running` 兩個欄位，新的請求直接覆蓋 `pending`，所有 client 共用這一份（全域最新優先）。
  - 請求內容只有字串與數值：readings、N-best 文字、R 分數、請求 key、epoch、發出者 KeyHandler 的 weak reference。grid node 不跨執行緒，N-best 的 `WalkResult` 留在發出者身上，依 key 對應（審查第 17 項）。
- **完成時**：queue 上執行 `PreciseSelect(..., heavy=qwen)` 並自行計時。結果分三種：
  - 逾時：連續第 3 次時狀態改為 tripped，之後切換開關也不清除，直到 process 重啟。
  - 無法打分（空結果，例如超過 512 token）：視為失敗，保留 R 的結果，不計入逾時次數。
  - 成功：回到 main，交給發出者的 `applyPreciseResult:`。
- **關閉精準模式**：epoch 加一、清掉 `pending`、狀態改為 off，並在 queue 上釋放 `QwenRescorer`。釋放排在執行中的推論之後，不需要鎖。
- **啟動時已是開啟**：`KeyHandler syncWithPreferences`（activation 時）呼叫 `[AFMPreciseController.shared setEnabled:Preferences.preciseModeEnabled]`，值相同時重複呼叫不做任何事。`AppDelegate` 啟動時也呼叫一次。
- **log**：只記錄狀態變化與耗時，不寫入 readings 或候選字文字。
- 替代方案：把精準模式併進 `AFMNeuralSelector`。不採用，因為兩者生命週期不同（R 永不釋放，Qwen 可開可關）。

- **D2 補充（2026-10-07，真 Qwen 延遲量測後；只調整背景執行時機，不改 spec）**：
  - Precise queue 用 `QOS_CLASS_UTILITY`。每次推論的 block 以 `DISPATCH_BLOCK_ENFORCE_QOS_CLASS` 執行，不繼承 main thread 的 user-interactive。ggml 的 worker thread 由這條 queue 的 thread 建立，會繼承 QoS（`ps -M` 實測忙碌 thread 的優先度為 20），所以不需修改 vendor。
  - 啟動延遲 `startDelay`（預設 40 ms）：每個按鍵事件呼叫 `noteKeyEvent`（KeyHandler 的 end-of-event hook）。距離上一個按鍵事件未滿 40 ms 時不開始新的推論，請求先排隊，仍是最新者優先；後續按鍵事件會把開始時間往後推。已開始的推論不會中斷。
  - 理由：真 Qwen 與 R 同時跑時，R 的斷路器在每一輪都會跳脫。只降 QoS（utility 或 background、4 或 2 threads）仍會跳脫；加上啟動延遲後，在連續打字的 harness 中 Qwen 只在停頓時執行，0 次跳脫，p95 約 4.5 ms。
  - 也考慮過：Qwen 改 2 threads，實測比 4 threads 慢，不採用。
  - 未解（需另做決定）：按鍵之間只要 CPU 閒置 20–150 ms，下一次 R 推論就慢 3–5 倍（精準模式關閉時也一樣），斷路器在真實打字節奏下可能跳脫。詳見 ../PLAN.md。

### D3. 套用結果：請求 key 加上 IMC 端的狀態檢查
- **固定位置**：手動選字（`kOverrideValueWithHighScore`）與 UOM 建議在 grid 上無法區分（`KeyHandler.mm:551`），所以 spec 把兩者都視為「固定位置」。指紋是各位置 override 的類型加上值。
- **兩種 key（再審 N2）**：
  - **請求 key**（用於 `preciseOverride` 比對與去重）：readings、`charsAtPositions`、固定位置指紋、KeyHandler 的 `compositionGeneration`、`LanguageModelManager` 的全域 `reloadGeneration`，不含顯示文字。
  - **顯示文字**：只放在請求裡，供 IMC 端檢查。
  - `compositionGeneration` 遞增的時機：commit、clear、切換輸入模式、開啟候選字窗、boost／exclude 選單。
  - `reloadGeneration` 是全域計數，每次語言模型重新載入時遞增，例如使用者詞檔變動（再審 N3）。
  - 目前顯示結果若來自 `preciseOverride`，就不送請求。
- **套用流程**：
  1. 發出者在 main 上確認 key 仍相符（重算目前 grid 的 key）、epoch 相符、固定位置沒被改動、合併結果和顯示的不同。
  2. 通過後，透過 delegate 交給 IMC，由 IMC 確認目前 `state is InputState.Inputting` 且 `composingBuffer` 等於請求時的文字，同時這個 controller 仍是 active client（`currentClient` 非 nil）。
  3. 都成立才設定 `preciseOverride = (key, WalkResult)`，跑 `_walk`。`_walk` 只在 key 相符時直接回傳這個 `WalkResult`，不做推論。接著 `buildInputtingState`，再呼叫 `handle(state:)`。
  - Marking、AssociatedPhrases、CustomMenu、ChoosingCandidate、Big5、Number 等狀態一律丟棄（審查第 3 項）。
- **不寫入 grid override、不呼叫 UOM observe**，所以合併結果不會被學成使用者的選擇。
- **之後任何重新 walk**（新的音、刪除、選字、UOM 建議）只要讓 key 不再相符，`_walk` 就清掉 `preciseOverride`，整句回到 R-walk。這是 spec 明定的行為（審查第 15 項），相關風險見下方 Risks。
- 套用的成本：一次 `_walk`（直接回傳）加 `buildInputtingState` 加 `setMarkedText`，p95 < 10 ms，task 5.3 實測。
- **不會重複請求**：顯示結果來自 `preciseOverride` 時不送；此外 KeyHandler 只記住「最近一次」送出或套用的請求 key，相同就不送。這與 spec 的 "most recent" 一致（再審 N5）。附帶效果：如果同一個組字區的請求被全域最新優先機制丟掉（例如另一個 App 送了更新的請求），組字區沒變時不會重送，要等下一次 key 改變才會再送。

### D4. 讀音政策：在 `McBopomofoLM` 加一層「讀音與字的配對過濾」，聯想詞也套用
- 新增 `setReadingPolicyExclusions(vector<{syllable, char}>)`，目前只有 `{ㄧㄢˊ, 癌}`。
- **主詞庫過濾**：在 `getUnigrams` 中，對 `rawGlobalUnigrams` 先過濾，再呼叫 `filterAndTransformUnigrams`，也就是在 phrase replacement、macro、external converter 之前，對原始值過濾（審查第 23 項）。
  - 過濾規則：key 以 `-` 拆成音節，value 拆成 UTF-8 字元，兩者長度相同才逐位置比對，任一位置命中就丟掉。實測 `data.txt` 含「癌」的 93 條，長度全部相同。
  - 先檢查 key 裡有沒有政策音節，沒有就不拆 value。
  - `hasUnigrams` 對「所有 base 條目都被濾掉」的 key 仍可能回傳 true。`getUnigrams` 回傳空清單時，grid 會略過該 span，不建立節點（`reading_grid.cpp:60`、`:685`）。task 2.1 要測這種情況下 walk 與候選字都正常。
- **聯想詞過濾**：`associated-phrases-v2.txt` 的條目自帶逐字讀音（例如 `乳-ㄖㄨˇ-癌-ㄧㄢˊ`，ㄧㄢˊ 版 37 條、ㄞˊ 版 42 條）。在 `findAssociatedPhrasesV2` 的結果上套同一組配對過濾，避免選了之後 `overrideCandidate("癌")` 失敗，結果打出錯字（審查第 5 項）。
- 使用者詞、排除詞、取代表維持原樣。
- 兩個 LM 實例都要設定：`gLanguageModelMcBopomofo` 與 `gLanguageModelPlainBopomofo`（`LanguageModelManager.mm:36-37`）。
- 偏好 key：`AFMAcceptYanReadingForAi`，預設 false，在 activation 的 `syncWithPreferences` 套用到兩個 LM。
- **非目標**：「轉注音」服務（`ServiceProvider.swift` 透過 `getReading`）取分數最高的讀音，會把「癌」轉成 ㄧㄢˊ。本 change 不處理，列為後續工作。
- 替代方案：排除詞機制，或在建置時改 `data.txt`。兩者都不採用，理由同前一版。

### D5. 建置：llama.cpp 預先編成靜態庫
- 新增 `tools/build_llama.sh`，CMake 參數如下（審查第 18 項；vendor 版本 5e03bdd）：
  - `-DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=26.0`
  - `-DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_BLAS=OFF -DGGML_OPENMP=OFF -DGGML_NATIVE=OFF`，`-DGGML_CPU_ARM_ARCH=armv8.4-a+dotprod+fp16`（vendor 會轉成 `-march=`，見 `ggml/src/ggml-cpu/CMakeLists.txt:211`），這是 M1 也支援的指令集。
  - `-DLLAMA_BUILD_COMMON=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_CURL=OFF`（後兩項若已廢除則忽略）
- 產出 `build-llama/` 底下的 `libllama.a` 與 `libggml*.a`。
- **驗收**：
  - 用 `nm` 掃所有 `.a`，不得出現已定義的 `curl_`、`SSL_`、`httplib` 符號，也不得有未定義的 `getaddrinfo`、`socket`、`connect` 參照。
  - 在 App binary 上跑 `otool -L`，不得連結 libomp 或其他非系統 dylib。
- **App target 設定**：連結上述 `.a`，加入 `Metal`、`MetalKit`、`Accelerate`、`Foundation`；header 路徑加入 llama 與 ggml 的 `include/`；`qwen_rescorer.cpp` 加入 target。`precise_walk.cpp` 不依賴 llama，可以先加入 target（task 1.x／5.1）。
- 不弱連結。Metal 初始化失敗時 `ready()` 為 false，狀態記為 failed。

### D6. 模型與校準檔
- 位置：`<app-support>/models/qwen/model.gguf` 與 `<app-support>/models/qwen/precise.conf`。路徑常數 `UserPhraseLocationHelper.qwenModelDirectory`。
- `tools/install_models.sh` 新增 `--with-qwen` 旗標。預設來源為 `../models/Qwen3-0.6B-Base-Q8_0.gguf` 與 `../bench/out/general/precise/precise.conf`，可用 `--qwen-gguf <path>`、`--precise-conf <path>` 改來源。不加旗標時行為與現在相同。
- 記憶體：llama 預設以 mmap 讀取 GGUF，Metal device 是全域單例。「關閉即釋放」指的是釋放 model、context 與 mmap；實際能釋放多少，由 task 7.1 量測後寫進 PLAN.md（審查第 34 項）。

### D7. 偏好設定分頁「選字模型」
- `PreferencesTab` 新增 `.selectionModel`（SF Symbol `brain`），內容為 `SelectionModelPreferencesView`。
- **開關**：三個 Toggle，對應 key 為 `AFMNeuralWalkEnabled`（沿用）、`AFMPreciseModeEnabled`（預設 false）、`AFMAcceptYanReadingForAi`（預設 false）。精準模式的 Toggle 一變動就呼叫 `AFMPreciseController setEnabled:`。
- **狀態推導**：開關關閉時顯示「關閉」。開啟時依模型狀態顯示：
  - R 用 `AFMNeuralSelector` 狀態。這次要新增 `NotFound` 狀態（目錄不存在），與 Failed 分開，並新增 main thread 狀態通知（審查第 13 項）。
  - 精準模式用 controller 狀態。
- view model 訂閱兩邊的通知，分頁開著時即時更新。
- **領域包**：14 列灰色、不可點，標示「尚未提供」，不綁任何 key。
- **說明文字**：註明精準模式約需 1 GB 記憶體。
- **字串**：en 與 zh-Hant。

### D8. 使用者詞不吃模型扣分（2026-10-07 實機測試後加入，使用者決定併入本 change）
- 實機發現：使用者詞「錏錒 ㄧㄚˋ-ㄚ」在 R 關閉時能打出，R 開啟時變成「亞啊」。原因是 R-walk 對每個字加 α×log p，罕見字的 log p 很低，蓋過了使用者詞的高分。
- 做法：
  - `LanguageModel::Unigram` 加一個 `isUserPhrase` 旗標，預設 false，所以上游行為不變。由 `McBopomofoLM` 在合併使用者詞時設定；單音節使用者詞經過分數改寫後仍保留這個旗標。
  - `walkWithCharBonus` 與 N-best 的邊分數規則相同：旗標為 true 的 unigram，每個位置的加分改用「該位置分數表中的最高分」，而不是它自己那個字的分數。
- 保證範圍（verify 後修正 spec 措辭）：只對「涵蓋完全相同位置」的競爭者保證不輸。不同斷詞的競爭者（跨越使用者詞邊界的組合）仍照常計分，可能因為鄰近位置的加分而勝出。
- 理由：使用者詞在模型加分上和最強的競爭者打平，勝負回到原版的詞庫分數，因此原版會選它時，R-walk 也會選它。
  - 不採用「加分為 0」：分數表是 log p，都 ≤ 0，加分 0 等於讓使用者詞壓倒一切。這會重現上游為單音節使用者詞特別處理的「丼」問題（`McBopomofoLM.cpp` 單音節使用者詞會改寫分數）。
- 精準模式：N-best 用同一規則。若 Qwen 合併打分後選了別的候選，仍依 D3 套用。使用者詞在 R 端拿到最高加分，pTop1 通常偏高、閘門多半關閉；這點由實機測試觀察。
- 測試：
  - gtest：旗標 false 時輸出與現行完全相同；旗標 true 時加分等於各位置最高分；N-best 新舊參照的比對沿用並涵蓋旗標。
  - XCTest：真模型加上使用者詞「錏錒」時，R-walk 選錏錒。

### D9. 斷路器門檻 10 ms → 30 ms（2026-10-07 使用者決定）
- 發現：鍵與鍵之間 CPU 閒置時，R 單執行緒推論慢 3–5 倍。連續打字 p95 約 4 ms；每鍵間隔 80 ms 時 p50 14.9 ms、p95 24.1 ms。這與 Qwen 無關，排除了 App Nap，研判是降頻或被排到 E-core。舊規則「連續 3 鍵 > 10 ms」在正常打字節奏下第 14 鍵就跳脫，會讓 R 整個 process 停用。
- 決定：斷路器門檻改為 30 ms，這是使用者開始感覺得到延遲的程度。每鍵 p95 < 10 ms 的預算文字不變，以連續打字的量測為準。
- 已知落差：有停頓的打字節奏下，每鍵 p95 約 24 ms，超過 10 ms 預算，但仍在 30 ms 斷路器之下，屬於人感覺不到的範圍。若要讓預算在任何節奏下都成立，需另開 change 處理暖機或預算定義。
- 精準模式的 QoS 與啟動延遲（D2 補充）照舊保留。

### 審查處理（2026-10-07）
- **P1 全部處理**：
  - 1：D1 改成「按鍵事件結束時送一次請求，並以 key 去重」。
  - 2：D1 第 3 步加入固定位置過濾；task 1.2 改用多音節 fixture。
  - 3：D3 由 IMC 檢查狀態與 `composingBuffer`。
  - 4：D0，加上 MODIFIED「Plain mode」與「Enable switch」兩條 requirement。
  - 5：D4 加入聯想詞過濾，spec 新增一條 requirement。
- **P2 全部處理**：6、7、8、9、10、11、12、13、14、15、16、17、18、19、20、21。
- **P3 處理**：22、23、24（列為非目標）、25、26、27、28、29、30、31、32、33、34。
- **再審（同日）**：N1–N6 全部處理，對應 D1 的 hook 與斷路器、D3 的兩種 key 與 `reloadGeneration`、spec 的 "most recent"、task 2.3 的措辭；部分解決的 #18、#23 已補齊。

## Risks / Trade-offs

- **[整句跳回 R-walk]**：套用合併結果後，任何重新 walk（新的音、選字、UOM）都讓整句回到 R-walk，之前修正過的字可能先跳回去，等新的請求再修正。
  - 處理：照 spec 實作，在實際打字測試觀察（task 7.1）。
  - 若跳字明顯，另開 change 做「前綴沿用」或「選字時保留合併路徑」，這會改 spec。
- **[N-best 拖慢前景]**：只在精準模式 ready 時才算，而且計入斷路器。超過預算就改為只在快取 miss 時計算（task 5.5）。
- **[背景 Qwen 搶 CPU 或 GPU]**：Qwen 用 4 條 CPU thread 加 Metal，可能拖慢 main thread。
  - 處理：task 7.1 用真模型量測閘門持續開啟時的每鍵延遲（審查第 19 項）。
  - 若超過預算，把 Qwen 的 `n_threads` 降到 2 後重測。
- **[1 GB 常駐記憶體]**：預設關閉，關掉就釋放。實際可釋放量由 task 7.1 量測。
- **[llama.cpp 靜態庫讓 App 變大、build 變慢]**：預先編好，只在升級時重跑。
- **[llama.cpp 內含網路功能]**：build 參數關掉相關選項，再用 `nm` 掃符號、`nettop` 實測。
- **[手動選字與 UOM 建議無法區分]**：兩者都當成固定位置，代價是 UOM 建議的位置不會被 Qwen 修正。

## Migration Plan

- 新增的 defaults key 預設值都是「與現行版本行為相同」，唯一例外是讀音政策：預設不接受 ㄧㄢˊ，這是使用者已決定的行為改變。不需要遷移資料。
- 回退：關掉精準模式即回到現行行為；讀音政策可在偏好設定打開 ㄧㄢˊ。整個 change 都在 `phase2-app` 分支，可直接 revert。

## Open Questions

- 「前綴沿用」是否需要：視實際打字測試的跳字情況，另開 change 決定。
- 精準模式的實機記憶體與 Metal 載入時間：在 task 中量測後記入 `../PLAN.md`，不影響設計。

## Implementation notes

- 7.2 實機打字（2026-10-07，文字編輯，computer use；細節見 `../PLAN.md` 的 "real-install test" 兩段）：
  - 15:15 版：讀音政策（抗嚴／肺癌）、「接受 ㄧㄢˊ」開關生效（R 關時出現抗癌）、R 開關、精準模式（太鹹了 vs 只開 R 時的太閒了）、記憶體釋放（986 → 267 MB）、0 條網路連線。
  - 16:51 最終版重測：錏錒（R 開）、太鹹了、抗炎（沒有抗癌）／肺癌、960 MB、0 條連線。
  - 15:15 版之後只改了 D8、D9 與 D2 補充，「接受 ㄧㄢˊ」開關與 R 開關的程式路徑沒有變動，所以這兩項沿用 15:15 版的實機結果。
- 實機無法截圖的部分（輸入法自己的視窗，不在可截圖的 App 清單內）：候選字窗內容、偏好分頁狀態、「乳」之後的聯想詞清單。這些由 XCTest（`ReadingPolicyTests`、`PreferencesTests`、`SelectionModelPreferencesRenderTests`）與 gtest（`ReadingPolicyTest` 聯想詞窮舉）涵蓋。
- 已知行為：「接受 ㄧㄢˊ」開啟且 R 開時，ㄎㄤˋㄧㄢˊ 預設仍組出非「抗癌」的結果，因為 R 是在 ㄞˊ 政策下訓練的。「抗癌」仍會出現在候選字，符合 spec；若要改變需要重新訓練 R。
