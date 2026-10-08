# Sloth-listwise Bopomofo

> **非官方 fork（Unofficial fork）**：本專案是 [McBopomofo 小麥注音](https://github.com/openvanilla/McBopomofo) 的個人分支，與小麥注音官方團隊沒有關聯，也未經其背書。This is an independent fork, not affiliated with or endorsed by the McBopomofo project.

> **English summary.** A macOS Zhuyin (Bopomofo) input method forked from [McBopomofo](https://github.com/openvanilla/McBopomofo) (MIT). A 12M-parameter on-device rescoring model, called **R** in this README (fine-tuned from SlothLM-E on license-clean data), adds per-character scores inside McBopomofo's n-gram walk. On a 1,495-sentence general test set it raises sentence accuracy from 62.2% (stock McBopomofo) to 76.4%, at about 2.9 ms p50 / 3.9 ms p95 per keystroke inside the app process. An optional "precise mode" (off by default) calls Qwen3-0.6B in the background only when the small model is unsure (~40% of sentences) and reaches 82.5% on the same test set. Two on-device personalization layers are included: a persistent memory of your manual picks, and an optional local fine-tune of the small model (LoRA, trained on your own committed sentences by a separate background helper, adopted only if it passes held-out accuracy gates; one click returns to the factory model). Everything runs locally; typed text never leaves the machine. Status: personal pre-release, Apple Silicon + macOS 26 only, not signed or notarized.

---

## 這是什麼

一個 macOS 注音輸入法。基底是小麥注音（McBopomofo），選字時多了一個在本機執行的選字小模型，本文簡稱 **R**（取自 Rescorer，「重新評分器」），幫忙判斷同音字。

- **每一鍵**：小麥原本的詞頻選字（n-gram walk）＋ R 對每個字的上下文打分，一起決定組字區顯示什麼。約 13 MB 的 int8 模型，CPU 執行。
- **精準模式（選配，預設關閉）**：R 沒把握時，在背景請 Qwen3-0.6B 一起評分，合併後更新組字區。打字不會等它。需另外放入約 600 MB 的模型檔。
- **學習我的選字（預設開啟）**：記得你手動改過的字，下次同樣的上下文優先建議。
- **個人化模型（選配）**：用你自己打過的句子，在你的 Mac 上微調 R；通過驗收才換上，隨時可以退回。
- **全部在本機**：不連雲端。打字內容不離開你的電腦。
- 可以和原版小麥注音同時安裝、互不干擾（不同 bundle id、不同使用者資料夾）。

### 名詞速查

| 名詞 | 意思 |
|---|---|
| **R**（Rescorer） | 本專案的選字小模型。它不自己產生候選字，而是替小麥列出的每個同音字依整句上下文重新打分。由公開的注音模型 SlothLM-E（約 1,200 萬參數）微調而來，量化後約 13 MB。 |
| **R-walk** | 把 R 的分數直接加進小麥原本的選字路徑（walk）裡的做法，也就是輸入法每一鍵實際在跑的東西。 |
| **精準模式** | 選配功能：R 沒把握時，在背景另外請較大的 Qwen3-0.6B 一起評分。 |
| **個人化模型** | 用你自己的句子在本機微調過的 R（LoRA 微調），通過驗收才會換上。 |

## 原理（不用 ML 背景也看得懂的版本）

### 1. 小麥注音怎麼選字
注音只記發音，ㄕˋ ㄕˊ 可能是「事實」「是時」「試時」……小麥把注音串切成所有可能的詞，每個詞有一個「常不常見」的分數（詞頻），再找一條總分最高的路徑（walk），就是整句結果。問題是：詞頻不看上下文，所以「說得 / 說的」這類同音字常常選錯。

### 2. R：會看整句上下文的小模型
R 由 SlothLM-E（11.6M 參數的注音模型，作者 Luigi）微調而來。它一次讀完整串注音，對每個位置給出「這裡最可能是哪個字」的機率。

微調的方式：拿乾淨的真實句子轉成注音，讓小麥列出前 50 個候選句，訓練 R 從中挑出正確那句（listwise），同時加上逐字的學習目標。專案名稱的「listwise」來自這裡；但要誠實說明：在對照實驗中，listwise 這一項本身相對於純逐字訓練**未達統計顯著**（72.4 vs 72.0），進步主要來自訓練資料量。

### 3. R-walk：把 R 放進選字路徑裡
我們不是讓小麥先選出 10 句再請 R 挑，而是在小麥找路徑的時候，每個字的分數就加上 α × R 的打分（α = 0.5）。這樣做的好處是候選範圍是整張詞網，不受「前 10 名」限制——在 D 模型的錯誤分析中，有 48.7% 的錯誤是正解根本不在小麥前 10 名裡。

使用者手動選過的字永遠優先，不會被模型改掉。

### 4. 精準模式：只在沒把握時叫大模型
1. R-walk 列出 10 個候選句。
2. 把 R 的分數校準成「R 對第一名有幾成把握」的機率（溫度 T = 1.08）。
3. 把握低於 0.8 才叫 Qwen3-0.6B，約 40% 的句子會觸發。
4. R 與 Qwen 的分數加權合併（log pR + 0.7 · log pQ）取最高者。

為什麼合併而不是讓 Qwen 直接取代：R 和 Qwen 錯的地方不一樣，兩者任一方答對的比例達 83.1%。實測「合併」也比「Qwen 單獨」高（見下表）。

## 學習我的選字

預設開啟，可在偏好設定的「選字模型」分頁關閉。

- **記什麼**：你在組字區手動改選的字，連同前後文，會被記起來；之後遇到相同的上下文，優先建議你上次選的字。記憶容量上限 5,000 個上下文，七天內的選擇權重會逐漸衰減（半衰期 7 天）。你自己的詞組檔優先於這份記憶。
- **句子池**：每次從注音模式送出（Enter、Shift+Space，或切到別的 App 時自動送出）的組字，會把讀音和選到的字存在本機的句子池，供下面的「個人化模型」使用。丟掉的組字（按 Esc）、純注音模式不會記。
- **密碼欄位不學習**：macOS 的安全輸入（secure input）啟用時，不記憶也不記錄。
- **只存在你的電腦**：檔案放在 `~/Library/Application Support/Sloth-listwise-Bopomofo/memory/`，只有你的帳號可讀寫；不會上傳。首次啟動會跳出一次說明（不會搶走你正在打字的視窗焦點）。
- **怎麼關、怎麼清**：關閉「學習我的選字」後，輸入法不再讀寫這些檔案，選字記憶回到小麥原版的行為（只在當次使用期間有效）。按「清除學習資料」（會先確認）會刪除選字記憶、句子池，以及下面的個人化模型，並立即回到出廠模型。
- **打字速度**：我們在測試主機上比較學習開 / 關，整個按鍵的 p95 為 3.66–3.68 ms（關）與 3.66–3.72 ms（開），差異 0.04 ms 以內。

## 個人化模型（本機微調）

這是選配功能：在你的 Mac 上，用你自己累積的句子對 R 做一次小幅度的再訓練（LoRA，只調整約 45 萬個參數，約為 R 全部參數的 4%）。

- **什麼時候會跑**：句子池累積到至少 1,000 句可用句子後，可以在偏好設定按「立即學習」；也會在插電、電腦閒置、距上次有足夠新句子時自動排程（最多一天一次）。訓練在一個獨立的背景程式中進行，不在輸入法本體內，不會擋住打字，也沒有網路權限。
- **要多久、多耗資源**：我們在模擬資料上量到單次約 40–66 秒、記憶體約 1 GB；訓練期間每個按鍵的 p95 為 3.85–4.42 ms（未訓練時 3.67–3.81 ms）。電池供電下單次比閒置多耗約 0.09–0.14 Wh（約 0.15–0.25% 電量）。
- **怎麼確定它沒有變差**：訓練完不會直接換上，會先過兩道驗收——(1) 留下最新的一部分你的句子當驗收題，新版本必須明顯比出廠模型好（至少多答對 2 個百分點，且通過統計檢定）；(2) 在隨附的一般句子上，準確率下降不能超過 1 個百分點。沒通過就維持原本的模型。
- **怎麼退回**：偏好設定的「改用上一版」「改回出廠模型」「清除學習資料」。出廠模型更新時，舊的個人化版本會作廢並在新模型上重新訓練。
- **只在本機**：句子、權重都只放在 `memory/` 資料夾；訓練用的小包資料（5,000 句一般句子與驗收題）隨出廠模型一起安裝，不含 Tatoeba、維基百科或授權不明的文字。
- **打字不會自動變準的情形**：句子池少於 1,000 句時不會訓練；你的選字記憶（上一節）仍會運作。

### 這個功能實際有沒有用？

目前的證據有三層，強度不同，請分開看：

| 證據 | 結果 | 限制 |
|---|---|---|
| 模擬醫學使用者（LLM 合成的醫學句，pool 1,000 句） | 使用者 held-out 句準確率 29.8% → 41.2%（疊在記憶層之上再多 +6.8 個百分點，p < 0.001）；一般測試 76.4% → 75.9% | 訓練句與測試句來自同一生成器，78% 的測試句與訓練句共享至少一個 4 字片段；真實使用者的重複度可能較低，增益可能被高估。另一組手寫醫學 150 句上的增益 +2.7 至 +4.0 個百分點，未達統計顯著 |
| 在實機安裝版上用模擬 pool 跑完整流程 | 採用 v1：個人 +6.10 個百分點（p = 1.7e-5，n = 377）、一般 −0.50；之後的「改用上一版」「改回出廠模型」「清除學習資料」都實測通過 | 驗證的是流程是否運作，資料仍是模擬的 |
| **用作者自己的寫作當替代資料（proxy）** | 個人 held-out 81.75% → 84.50%（+2.75 個百分點，單尾 p = 0.0064，n = 400）；一般測試 −0.90 個百分點，接近 −1.0 的上限 | 這是**替代資料，不是實際打字紀錄**：來源是作者自己的書面文件（不是一個字一個字打出來的）、讀音由自動工具（g2pW）標註、沒有人工挑字的紀錄；單一使用者、單次執行；本機原始文字不公開。出廠模型在這類文字上本來就有約 82%，所以增益比模擬小 |

換句話說：這個機制在流程、安全性（可退回、不外傳、不拖慢打字）上已經驗證過；對真實日常打字的準確率改善，目前只有「替代資料上有小幅進步」的證據，需要等真實句子池累積後再重測。

## 準確率與延遲

**測試集**：general test，1,495 句真實句子（Common Voice zh-TW、中文維基條目開頭、SlothLM-E 作者的同音字難題集）。指標是「整句全對」的比例。調參只用另外 200 句 dev。配對檢定為 McNemar exact 雙尾。

| 系統 | 整句準確率 | 每鍵 / 每句成本 | 狀態 |
|---|---|---|---|
| 小麥注音原版 | 62.2% | — | 上游 |
| SlothLM-E 12M 直接重排前 10 名 | 68.8% | — | 對照 |
| McBopomofoLM（另一個小麥 fork，原碼重跑） | 74.2% | — | 對照 |
| **R-walk（目前輸入法內）** | **76.4%**（C++ int8 76.5%） | app 內每鍵 p50 2.85 / p95 3.86 ms | 已安裝實測 |
| Qwen3-0.6B 單獨重排 | 77.1% | 10 候選 31 / 45 ms（Metal） | 對照 |
| **精準模式** | **82.5%** | 每句平均 16.3 ms，只有 40% 句子叫 Qwen | 已接進輸入法（選配、預設關閉）；上表數字為引擎層 benchmark |

重點（附檢定）：
- R-walk vs SlothLM-E 直接重排：p = 3.6e-18。
- 精準模式 vs Qwen 單獨：p = 1.9e-11；vs R-walk：p = 3.8e-17。
- 精準模式 vs 「每句都叫 Qwen」（82.9%）：**未達統計顯著**（p = 0.07），但 Qwen 呼叫量少了六成。
- R-walk 與 Qwen 單獨之間，在研究版模型上**未達統計顯著**（p = 0.065）；授權乾淨版尚未做直接檢定。
- 測試集中沒有任何「不用 Qwen」的方案顯著勝過 Qwen 單獨。

**延遲細節**（app 測試主機、Release、2,400 次按鍵、組字區 1–40 音節）：p50 / p95 / p99 / max = 2.85 / 3.86 / 4.20 / 5.41 ms，規格門檻是 p95 < 10 ms。模型載入 59 ms（背景載入，載完前用原版選字）。記憶體增加約 77 MB。

**要注意的限制**：
- Common Voice 的其他句子也在 R 的訓練資料裡（同一語料、不同句子，已做完全相符與 8-gram 去重），所以 Common Voice 那部分可能偏高；維基分層才是分布外的檢查（R-walk 48.4%、Qwen 53.3%、精準模式 63.8%）。
- 讀音標註為單一（自動 + LLM）標註者，50 句抽查 0 錯。
- 延遲是在測試主機量的，已安裝輸入法的逐鍵延遲還沒量。
- 尚未與 macOS 內建注音、自然輸入法等常用輸入法在同一測試集上比較。
- 上表的準確率都是出廠模型、不含個人化；個人化的結果見上一節，條件完全不同（個人 held-out，不是 general test）。

## 怎麼做出來的

這個專案大約三天（2026-10-06 → 10-08）從構想做到可以裝進系統打字、並加上本機個人化，主要開發者是 Claude Code，我負責方向與每個關鍵決策。

1. **研究先行**：先查辭典授權、Apple Foundation Models（AFM）能不能用、有沒有人做過。結論：AFM 沒有「對候選打分」的 API、而且太慢（300 句實測 p50 439 ms，還有 17 句被安全過濾拒答），只能放棄當選字引擎——專案資料夾至今還叫「AFM Input」。
2. **先量再選**：做了 300 句可行性測試，再建 1,495 句真實句的 general benchmark 當主要指標，把十幾種方案（大小模型、合併、重排 vs 嵌入）放在同一張表比較。
3. **走過的死路**：25M encoder 沒有比 12M 好（70.7 vs 73.3）；60M decoder 在 Apple 神經引擎上快，但 llama.cpp 上每句約 192 ms，無法用在 C++ 路徑；把原版 SlothLM-E 當備援層，最後也拿掉了。
4. **授權乾淨重訓**：研究版 R 用過 c4（無授權標示），只能研究用。正式版只用 CC0 / CC BY / 政府開放資料 / 本機 Qwen 生成句，共 144,595 句，結果反而略高（76.4 vs 75.5，未達統計顯著）。
5. **C++ 引擎**：在小麥的選字引擎加入 N-best、逐字加分 walk、精準模式，每項都有 gtest，C++ 與 Python 結果逐句一致。
6. **OpenSpec 規格驅動整合**：輸入法本體的改動用 OpenSpec 管理——先寫提案與規格（例如「每鍵 p95 < 10 ms」「模型出問題不能讓你無法打字」「手動選字永遠優先」），實作後跑 verify（完整性 / 正確性 / 一致性）全綠才 archive。
7. **真機安裝測試**：和原版小麥並存安裝、在 TextEdit 實打。過程中發現：App 程式碼沒有任何網址，卻有對外連線——追查後是預編譯的 onnxruntime 內建 Microsoft telemetry，每次載入模型會送出約 1 KB。已關閉並用 `nettop` 確認 0 條外連。
8. **個人化：先查先例再動手**：調查了小麥、vChewing、libchewing、fcitx5、Mozc、librime 等輸入法，發現找到的成熟開源輸入法都沒有在裝置上更新神經網路權重，而是用有上限、可解釋的統計記憶。於是拆成「快迴路」（選字記憶，上面那一節）與「慢迴路」（微調 R），先做快的、慢的另開一個規格。
9. **訓練工具選型**：比較了 MLX、ONNX Runtime training、Core ML updatable、libtorch、ExecuTorch、ggml 等。ONNX Runtime 在 macOS 沒有官方 training 套件、Python 套件停在 2024 年；Core ML 表達不了我們要的訓練目標；libtorch 動態庫 216 MB 對輸入法太大。最後選 MLX（MIT），在獨立背景程式裡用 CPU 跑，約 15–20 MB。訓練完的權重用 ONNX Runtime 的 `AddInitializer` 在載入時覆蓋，出貨的模型檔一個 byte 都不改。
10. **先做會讓整件事失敗的實驗**：在寫產品程式前先驗證——C++ 路徑的覆蓋結果與 Python 逐句一致（0 句不一致）、量化後準確率不變（int8 與 fp32 差 ≤ 0.5 個百分點，皆未達統計顯著）、驗收閘門的功效模擬（原本「留 10%」的驗收題在 1,000 句時只有 55% 機會接受真有 +4 個百分點的進步，改成留 30%、300–400 句後約 84%）。
11. **過夜的多 agent 實作與兩層審查**：個人化的實作在一晚內由多個 agent 分組完成，之後先用內部 verify 修了 5 項（例如被取消的訓練仍被採用），再請另一個模型（Codex）獨立審查，又找到 5 項並全部修正，包括「清除學習資料時，個人化模型有可能復活」「句子池檔案權限過寬」「背景程式沒有自己的逾時保護」，每一項都補了會在舊行為下失敗的測試。

## 安裝（開發者版）

> 目前未簽章、未公證，只適合自己 build 來試。需求：Apple Silicon、macOS 26 以上、Xcode 27。

1. 用 Xcode 以 Release 建置 `McBopomofo` scheme，產物為 `Sloth-listwise-Bopomofo.app`。
2. 複製到 `~/Library/Input Methods/`，執行：
   ```bash
   ~/Library/Input\ Methods/Sloth-listwise-Bopomofo.app/Contents/MacOS/Sloth-listwise-Bopomofo install --all
   ```
3. 放入 R 模型（約 14 MB）：
   ```bash
   tools/install_models.sh   # 預設複製到 ~/Library/Application Support/Sloth-listwise-Bopomofo/models/r-walk
   pkill -x Sloth-listwise-Bopomofo
   ```
   模型檔與個人化資料包**目前尚未公開發佈**；建置與資料夾配置見 [docs/BUILDING.md](docs/BUILDING.md)。
   若要使用個人化模型，還需要隨出廠模型一起安裝「個人化資料包」（`models/r-walk/personalize/`，約 15–20 MB）；沒有資料包時，偏好設定會顯示「缺少個人化資料包」，其他功能不受影響。
4. 到「系統設定 > 鍵盤 > 輸入來源」手動加入。ad-hoc 簽章的版本可能要**重開機**後才會出現在清單裡。
5. 用 Ctrl-Space 切換輸入法。注意：用程式呼叫 `TISSelectInputSource` 切換不會觸發 activate，偏好設定變更要用鍵盤切換才會生效。

開啟精準模式（選配）：偏好設定「選字模型」分頁打開「精準模式」；需另放約 600 MB 的 Qwen 模型檔。

關閉 R（回到原版選字）：`defaults write io.github.fireman333.inputmethod.Sloth-listwise-Bopomofo AFMNeuralWalkEnabled -bool NO`，再切換一次輸入法。

## Roadmap

已完成：
- [x] 精準模式接進輸入法、偏好設定「選字模型」分頁、「癌」讀音政策（預設 ㄞˊ，ㄧㄢˊ 需手動開啟）
- [x] 學習我的選字（持久選字記憶與句子池）
- [x] 個人化模型（本機微調 R，過閘門才換上，可退回）

待做：
- [ ] 把 Qwen 的判斷蒸餾進 R（不增加每鍵延遲）
- [ ] 用真實打字累積的句子池重新評估個人化模型，並重新校準驗收閘門
- [ ] 注音拆成聲母 / 介音 / 韻母 / 聲調輸入（容錯與簡拼）：評估後暫不做，需要先有引擎端的模糊展開
- [ ] 14 個領域包（照 macOS 注音「特殊字彙」分類；西醫詞庫已有雛形）
- [ ] 簽章、公證、公開發佈（含重新評估把訓練程式放進沙盒的 XPC service，取得作業系統層級的無網路保證）

## 致謝與授權

本專案以 MIT 授權釋出，沿用上游（見 [LICENSE.txt](LICENSE.txt)）；第三方元件與資料授權完整列表見 [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md)。使用與參考的元件：

| 元件 | 用途 | 授權 |
|---|---|---|
| [McBopomofo](https://github.com/openvanilla/McBopomofo)（Mengjuei Hsieh et al.） | 輸入法基底、選字引擎、詞庫 | MIT |
| SlothLM-E（Luigi，`Luigi/slothlm-e-12m-zhuyin`） | R 的底模權重 | Apache-2.0（權重）；上游訓練程式未附授權檔（本專案未使用其程式碼） |
| Qwen3-0.6B-Base（Qwen team） | 精準模式背景評分 | Apache-2.0 |
| Qwen3-4B-Instruct-2507 | 本機生成部分訓練句 | Apache-2.0 |
| llama.cpp / ggml | Qwen 推論 | MIT |
| ONNX Runtime（Microsoft） | R 推論（已關閉 telemetry） | MIT |
| MLX（Apple ml-explore） | 本機微調的訓練框架（獨立背景程式，CPU） | MIT |
| g2pW | 訓練資料與測試用替代資料的讀音標註 | Apache-2.0 |
| libchewing 讀音表 | 每個音節可用字集 | LGPL-2.1（只用於模型檔隨附的讀音表，不在本 repo 內；公開發佈模型前會改由小麥詞庫重建） |
| McBopomofoLM（workfunction） | 比較對象與設計參考 | MIT / Apache-2.0 |

訓練資料：
- Mozilla Common Voice zh-TW sentences（CC0-1.0）
- Sentences from Tatoeba (https://tatoeba.org), CC BY 2.0 FR
- 本產品之訓練資料部分取自 行政院、衛生福利部食品藥物管理署、文化部、文化部影視及流行音樂產業局、農業部（含林業及自然保育署、漁業署）、外交部、經濟部國際貿易署、數位發展部 2026 年釋出之開放資料，依政府資料開放授權條款－第1版 (https://data.gov.tw/license) 進行公眾釋出。
- 個人化資料包（隨出廠模型安裝）只含 Common Voice（CC0）、上述政府開放資料與本機生成句，不含 Tatoeba。
- 不使用：c4、維基百科、CC BY-ND / NC 文字、Apple Foundation Models 生成文字、教育部辭典。

以上授權整理非法律意見。
