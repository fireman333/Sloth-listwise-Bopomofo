# Design

## Context

動機見 proposal.md - Why；需求見 `specs/app-identity/spec.md`。現況（2026-10-07 唯讀調查）：

- 身分散在四層，彼此必須一致：
  1. Build settings（`McBopomofo.xcodeproj/project.pbxproj`）：App `PRODUCT_BUNDLE_IDENTIFIER = org.openvanilla.inputmethod.McBopomofo`（:1102、:1156）、`PRODUCT_NAME = $(TARGET_NAME)`（:1103、:1157）；Installer `org.openvanilla.McBopomofo.${PRODUCT_NAME:rfc1034identifier}`（:1262、:1303）；Tests `com.cerence.McBopomofoTests`（:1352、:1399）、`TEST_HOST = .../McBopomofo.app/Contents/MacOS/McBopomofo`（:1360、:1406）；`MACOSX_DEPLOYMENT_TARGET = 13.0` 共 10 處（project :997、:1041；App :1100、:1155；舊的 "Data" target :1190、:1216；Installer :1260、:1302；Tests :1348、:1395）；scheme `McBopomofo.xcscheme` 的 `BuildableName = "McBopomofo.app"`（:18、:59、:76）；Installer 以檔案參照 `McBopomofo.app`（:34、:130、:705）把 App 包進 Resources。
  2. Info.plist（`Source/McBopomofo-Info.plist`）：輸入模式 id（:27、:55、:86–87）、`InputMethodConnectionName = McBopomofo_1_Connection`（:91）、`TISInputSourceID`（:323，寫死而非引用 build setting）、更新端點（:328–331）、Services 選單「Add to McBopomofo User Phrases」（:118）。`CFBundleName = ${PRODUCT_NAME}`、`LSMinimumSystemVersion = ${MACOSX_DEPLOYMENT_TARGET}` 會自動跟著 build setting。
  3. 程式碼寫死：`Source/main.swift:73` connection name 常數（與 plist 重複定義）；`Source/KeyHandler.mm:48–49` 兩個輸入模式 id；`Source/Preferences.swift:528` 預設資料夾名 `"McBopomofo"`（`LanguageModelManager.mm:419` `dataFolderPath` 由此衍生所有使用者檔路徑）；`Source/Installer/AppDelegate.swift:27–32` 安裝路徑與 `killall` 對象。
  4. 顯示字串：`Source/{Base,en,zh-Hant}.lproj/InfoPlist.strings`（名稱與模式名，key 是模式 id）、`Source/{en,zh-Hant}.lproj/Localizable.strings`（:2、:5、:32、:98、:140/142）、`Source/Base.lproj/MainMenu.xib:17–35`、`Source/Installer/{en,zh-Hant}.lproj/InfoPlist.strings`、`Source/Installer/Base.lproj/MainMenu.xib:18–19`、Installer 完成訊息（`Source/Installer/AppDelegate.swift:255、271`）。
- Preferences 一律用 `UserDefaults.standard`（無 suiteName），domain 自動等於 bundle id。
- UserOverrideModel 只存在記憶體（`LanguageModelManager.mm:38`），沒有檔案要搬。
- ObjC 介面標頭以 module 名命名：`#import "McBopomofo-Swift.h"`（`KeyHandler.mm:27`、`LanguageModelManager.mm:26`、`ServiceProviderInputHelper.mm:25`），測試用 `@testable import McBopomofo`。Swift 類別都有明確 `@objc(...)` 名稱，xib 不依賴 module 名。
- 已安裝的原版小麥注音佔用：`~/Library/Input Methods/McBopomofo.app`、bundle id / mode id / connection name 同上、`~/Library/Application Support/McBopomofo/`、defaults domain `org.openvanilla.inputmethod.McBopomofo`、程序名 `McBopomofo`。

## Goals / Non-Goals

**Goals:**
- 一次把所有會和原版衝突的身分切乾淨，且每個 id 只有一個定義來源（build setting），其他地方引用它。
- pbxproj 改動最小化：只改 build settings 與產物檔案參照，不動 target / group 結構。

**Non-Goals:**
- 引擎（C++）與按鍵處理邏輯不動；選字行為與上游完全一致（model 缺席時本來就是原版 walk）。
- 不改類別名、檔名（`McBopomofo-Info.plist`、bridging header 等）、圖示、scheme 名稱。

## Decisions

### D1. 保留 Xcode target 名稱，用 `PRODUCT_NAME` 改產物名，並固定 `PRODUCT_MODULE_NAME = McBopomofo`
- App target `PRODUCT_NAME = Sloth-listwise-Bopomofo` → 產物 `Sloth-listwise-Bopomofo.app`、執行檔 `Sloth-listwise-Bopomofo`。
- 同時明設 `PRODUCT_MODULE_NAME = McBopomofo`：否則 module 名會從含連字號的 PRODUCT_NAME 衍生成 `Sloth_listwise_Bopomofo`，三個 `.mm` 的 `McBopomofo-Swift.h` 與測試的 `@testable import McBopomofo` 全部要改。
- 替代方案 A：改 target 名稱 → pbxproj、scheme、`TEST_HOST`、bridging header 路徑大量 churn，與上游 merge 衝突多，收益只有 Xcode 內的名稱好看，不採用。
- 替代方案 B：維持 `McBopomofo.app` 檔名只改 bundle id → 安裝路徑 `~/Library/Input Methods/McBopomofo.app` 與原版相撞，違反並存需求，不可行。
- Installer target 只改 bundle id 與顯示名，`PRODUCT_NAME` 維持 `McBopomofoInstaller`（開發期幾乎不用 Installer，不值得改）。Installer 對 App 的檔案參照（pbxproj :130 `path = McBopomofo.app`）同步改成新產物名，否則 Installer 的 Copy Resources 會找不到檔案。

### D2. 由 bundle id 推導所有 id，單一來源
- 輸入模式 id：`<bundle id>.Bopomofo` / `.PlainBopomofo`。Xcode **不會**展開 plist dict **key** 中的 build setting（Xcode 27 實測：key 原樣輸出為 `$(PRODUCT_BUNDLE_IDENTIFIER).Bopomofo`），因此 `tsInputModeListKey` 的兩個 key（Info.plist :27、:55）寫新 mode id 字面值；陣列與字串**值**（`tsVisibleInputModeOrderedArrayKey` :86–87、`InputMethodConnectionName` :91、`TISInputSourceID` :323）用 `$(PRODUCT_BUNDLE_IDENTIFIER)` 展開。tasks 以 `plutil -p` 檢查成品無殘留 `$(`，且 key 集合等於陣列值。
- Connection name：`$(PRODUCT_BUNDLE_IDENTIFIER)_Connection`（Apple 範例慣例）。`main.swift` 改為從 `Bundle.main.infoDictionary["InputMethodConnectionName"]` 讀取，移除重複常數；讀不到時以 `<bundleID>_Connection` 組出（不寫死舊值）。
- `KeyHandler.mm` 的兩個 `InputMode` 常數改為字面值新 id。替代方案：執行期由 `NSBundle.mainBundle.bundleIdentifier` 組字串——但它們是全域常數且被 XCTest（host 為 App）使用，執行期組字串會讓常數初始化順序變複雜；兩處字面值 + 一個 plutil 一致性檢查（tasks）較簡單。
- `InfoPlist.strings` 的 key 必須是展開後的實際 mode id（strings 檔不做 build-setting 展開），因此三份 strings 寫新 id 的字面值。

### D3. 顯示名稱
- 全部 localization 的 bundle 名與句子模式名 = `Sloth-listwise Bopomofo`；純注音模式 = `Sloth-listwise Plain Bopomofo`（zh-Hant 也不用「傳統注音」，避免與原版的「傳統注音」在選單中無法區分）。
- 因 `CFBundleName` 從 `${PRODUCT_NAME}` 會變成帶連字號的 `Sloth-listwise-Bopomofo`，以 `InfoPlist.strings` 覆寫為空白版本（`LSHasLocalizedDisplayName = true` 已開）。
- Localizable.strings（Base / en / zh-Hant）與 MainMenu.xib（Base / zh-Hant）只改「產品名」出現的**值**，NSLocalizedString 的 key 不動；更新檢查相關字串在 D5 後不再顯示，但其值也一併改成新名稱，讓殘留掃描乾淨。Services 項目改名時，`ServicesMenu.strings`（en / zh-Hant）的 key 與值一併改。Installer 的 xib（Base / zh-Hant）與 `Source/Installer/{en,zh-Hant}.lproj/Localizable.strings` 也只改值。

### D4. 使用者資料與模型根目錄
- 在 `@objc class UserPhraseLocationHelper`（`Preferences.swift:524`）上新增 `@objc static` 常數 `applicationSupportRoot = <Application Support>/Sloth-listwise-Bopomofo`、`modelsDirectory = root/models`、`rWalkModelDirectory = root/models/r-walk`（名稱固定為 `rWalkModelDirectory`，`selection-pipeline` 依此引用；`@objc` 讓 ObjC++ 也能取用），`defaultUserPhraseLocation` 回傳 root。
- 模型目錄不跟隨「自訂使用者詞庫位置」：使用者把詞庫放 iCloud / Dropbox 時，不應把 ~15 MB 模型也同步出去。
- 不建立 `models/`：由 `selection-pipeline` 的 `tools/install_models.sh` 負責。本 change 只保證根目錄在首次啟動時建立（沿用既有 `ensureFileExists` 流程）。
- 不自動搬移（使用者決定）。替代方案「首次啟動時若舊資料夾存在就複製」被否決：會讓實驗中的 App 汙染判斷基準，也可能誤帶原版的排除詞。

### D5. 移除上游更新端點
- 刪掉 Info.plist 的 `UpdateInfoEndpoint` / `UpdateInfoSite`。`VersionUpdateApi.check` 在讀不到 endpoint 時已回傳 `nil`（`AppDelegate.swift:68–72`），不需改程式；但它此時**不呼叫 callback**，`McBopomofoTests/VersionUpdateTests.swift:32–36` 的 `withCheckedContinuation` 會永遠等待，故該測試改為斷言 `VersionUpdateApi.check(forced: true) { _ in } == nil`。替代方案：指向自己的端點——目前沒有發佈管道，留待簽章/公證階段另開 change。

### D6. `MACOSX_DEPLOYMENT_TARGET = 26.0` 在本 change 處理
- project 層與 App / Installer / Tests 全部改 26.0，避免 target 間不一致；`LSMinimumSystemVersion` 自動跟隨。放在這裡而非 `selection-pipeline`：這是 App 對外宣告的身分屬性（最低系統），且先做可讓 `selection-pipeline` 加入 onnxruntime dylib（需 ≥ 14）時不再碰 build settings。Swift packages（`Packages/*/Package.swift`）的 `platforms` 不在範圍內，只要不高於 26 即可被 App 使用。

### D7. Installer 不再用程序名 `killall`
- 新執行檔名 `Sloth-listwise-Bopomofo` 長 23 字元，macOS 的程序短名（`p_comm`）上限 16 字元，`killall` 以短名比對可能失敗。改用 `NSRunningApplication.runningApplications(withBundleIdentifier:)` 對本 App bundle id `forceTerminate()`，再輪詢 `isTerminated` 最多約 2 秒才複製新檔；只會影響本 App，絕不碰原版小麥注音。`kTargetBin` 同時被用作 Resources 內的資源名（`AppDelegate.swift:184`）與 `ArchiveUtil` 的 appName（:71），三處一致改為 `Sloth-listwise-Bopomofo`。
- 其他需要結束程序的地方（`Source/Data/Makefile:50–55` 的 `_install` / `_deploy`）同理改用 `pkill -x Sloth-listwise-Bopomofo`（精確比對完整名稱）並指向 `Sloth-listwise-Bopomofo.app`，不再用會誤中原版的 `pkill -f McBopomofo`。
- 開發期主要安裝方式仍是：`xcodebuild` → 複製到 `~/Library/Input Methods/` → 執行 `<app>/Contents/MacOS/Sloth-listwise-Bopomofo install`（`main.swift` 既有的註冊流程，以 bundle id 註冊與啟用）。

## Risks / Trade-offs

- [TIS 對舊 bundle id 的快取：開發機若曾安裝本 fork 的舊 build（id 與原版相同），會與原版混在一起] → 測試前確認 `~/Library/Input Methods/McBopomofo.app` 是**原版**（`codesign -dv` 看簽章 / 版本），必要時重新安裝原版；改完後登出再登入一次讓 TIS 重新掃描。
- [Services 選單中兩個 App 會出現同名項目（Add Bopomofo Readings… 等）] → 只改產品名那一項；其餘同名服務為可接受的外觀問題，不影響功能。
- [InfoPlist.strings key 與 Info.plist mode id 不一致會讓選單顯示原始 id] → tasks 加 plutil / grep 一致性檢查。
- [`PRODUCT_MODULE_NAME` 與產物名不同，未來新加的 Swift 檔若手打 `Sloth_listwise_Bopomofo-Swift.h` 會失敗] → 在 design 記錄；module 名維持 `McBopomofo` 與上游一致。
- [deployment target 26.0 讓 macOS < 26 無法安裝] → 專案本來就宣告不支援 macOS < 26。

## Migration Plan

1. 在 `afm-rescoring` 分支實作 tasks 1–4；`xcodebuild build` / `test` 通過。
2. 確認原版小麥注音安裝於 `~/Library/Input Methods/McBopomofo.app` 且可正常使用。
3. 安裝新 App → `install` 註冊 → 登出登入 → 兩者同時出現在輸入來源並各自打字。
4. 回退：刪除 `~/Library/Input Methods/Sloth-listwise-Bopomofo.app`、`~/Library/Application Support/Sloth-listwise-Bopomofo/`、`defaults delete io.github.fireman333.inputmethod.Sloth-listwise-Bopomofo`；原版不受影響。程式層回退為 revert 本 change 的 commit。

## Open Questions

- 日後公開發佈時 Installer 是否改名為 `Sloth-listwise-Bopomofo Installer.app`（目前不影響並存與實測）。
