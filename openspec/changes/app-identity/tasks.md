# Tasks

> 不自動 `git commit`（走 auto-git 或使用者確認）。每個 App task 以 `xcodebuild` 成功為最低驗證。共用指令：BUILD = `xcodebuild -project McBopomofo.xcodeproj -scheme McBopomofo -configuration Debug -derivedDataPath build-xcode build`；TEST = 同上但動作為 `test`；IBUILD = `xcodebuild -project McBopomofo.xcodeproj -scheme McBopomofoInstaller -configuration Debug -derivedDataPath build-xcode build`。產物目錄 `$P = build-xcode/Build/Products/Debug`。

## 1. Build settings（pbxproj / scheme）

- [x] 1.1 App target（Debug/Release）：`PRODUCT_BUNDLE_IDENTIFIER = io.github.fireman333.inputmethod.Sloth-listwise-Bopomofo`、`PRODUCT_NAME = Sloth-listwise-Bopomofo`、`PRODUCT_MODULE_NAME = McBopomofo`；App 產物檔案參照（pbxproj :130 `path`，註解 :34/:315/:581/:705）改為 `Sloth-listwise-Bopomofo.app`；`McBopomofo.xcscheme` 的 `BuildableName`（:18、:59、:76）改為 `Sloth-listwise-Bopomofo.app`。Verify: BUILD 成功，`$P/Sloth-listwise-Bopomofo.app/Contents/MacOS/Sloth-listwise-Bopomofo` 存在。
- [x] 1.2 Installer target bundle id 改為 `io.github.fireman333.inputmethod.Sloth-listwise-Bopomofo.Installer`；Tests target bundle id 改為 `io.github.fireman333.inputmethod.Sloth-listwise-Bopomofo.Tests`、`TEST_HOST = $(BUILT_PRODUCTS_DIR)/Sloth-listwise-Bopomofo.app/Contents/MacOS/Sloth-listwise-Bopomofo`。Verify: IBUILD 成功且 `$P/McBopomofoInstaller.app/Contents/Resources/Sloth-listwise-Bopomofo.app` 存在；TEST 能啟動 test host。
- [x] 1.3 全部 10 處 `MACOSX_DEPLOYMENT_TARGET`（project :997/:1041、App :1100/:1155、舊 "Data" target :1190/:1216、Installer :1260/:1302、Tests :1348/:1395）改 `26.0`。Verify: `grep -c "MACOSX_DEPLOYMENT_TARGET = 13.0" McBopomofo.xcodeproj/project.pbxproj` 為 0；BUILD 後 `plutil -extract LSMinimumSystemVersion raw "$P/Sloth-listwise-Bopomofo.app/Contents/Info.plist"` 輸出 `26.0`。

## 2. Info.plist 與 id 單一來源

- [x] 2.1 `Source/McBopomofo-Info.plist`：`tsInputModeListKey` 的兩個 dict **key**（:27、:55）寫新 mode id 字面值（Xcode 不展開 key 中的 build setting）；`tsVisibleInputModeOrderedArrayKey` 值（:86–87）用 `$(PRODUCT_BUNDLE_IDENTIFIER).Bopomofo` / `.PlainBopomofo`；`InputMethodConnectionName = $(PRODUCT_BUNDLE_IDENTIFIER)_Connection`；`TISInputSourceID = $(PRODUCT_BUNDLE_IDENTIFIER)`；刪除 `UpdateInfoEndpoint` / `UpdateInfoSite`；Services 項目 default 改為 `Add to Sloth-listwise Bopomofo User Phrases`。Verify: BUILD 後對 `$P/Sloth-listwise-Bopomofo.app/Contents/Info.plist`：`plutil -p ... | grep -c '\$('` 為 0；`grep -c "openvanilla\|McBopomofo_1"` 為 0；`CFBundleIdentifier`、`TISInputSourceID` 為新 bundle id；connection name 為 `io.github.fireman333.inputmethod.Sloth-listwise-Bopomofo_Connection`；`tsInputModeListKey` 的 key 集合等於 `tsVisibleInputModeOrderedArrayKey` 陣列值。
- [x] 2.2 `Source/main.swift`：移除寫死的 `kConnectionName`，改讀 Info.plist `InputMethodConnectionName`（讀不到時以 `<bundleID>_Connection` 組出）。Verify: BUILD 成功；`grep -n "McBopomofo_1_Connection" Source/main.swift` 無結果。
- [x] 2.3 `Source/KeyHandler.mm:48–49` 的 `InputModeBopomofo` / `InputModePlainBopomofo` 改為新 mode id 字面值。Verify: TEST 中 KeyHandler 測試通過；`grep -n "InputMode.*= @\"" Source/KeyHandler.mm` 的兩個字串與 2.1 的陣列值相同。
- [x] 2.4 `McBopomofoTests/VersionUpdateTests.swift:32–36`：端點移除後 `check()` 回傳 `nil` 且不呼叫 callback，原 `withCheckedContinuation` 會永遠等待；改為斷言 `VersionUpdateApi.check(forced: true) { _ in } == nil`。Verify: TEST 中該測試通過且不逾時。

## 3. 顯示名稱

- [x] 3.1 `Source/{Base,en,zh-Hant}.lproj/InfoPlist.strings`：`CFBundleName` / `CFBundleDisplayName` = `Sloth-listwise Bopomofo`；key 改為新 mode id，值分別為 `Sloth-listwise Bopomofo` / `Sloth-listwise Plain Bopomofo`。Verify: `grep -rn "openvanilla\|小麥注音\|傳統注音" Source/*.lproj/InfoPlist.strings` 無結果；`plutil -lint` 三檔 OK；BUILD 成功。
- [x] 3.2 App 字串（只改值，key 不動）：`Source/{Base,en,zh-Hant}.lproj/Localizable.strings` 的 About（:2）、Preferences（:5）、up to date（:98）、user file issues（Base/zh-Hant :140、en :142）；`Source/{Base,zh-Hant}.lproj/MainMenu.xib:17–35` 的選單標題；`Source/{en,zh-Hant}.lproj/ServicesMenu.strings:1` 的 key 與值改為新 Services 名稱。Verify: `plutil -lint` 上述 strings 全 OK；`grep -n "小麥注音\|= \"About McBopomofo\|= \"McBopomofo Preferences" Source/*.lproj/Localizable.strings Source/*.lproj/ServicesMenu.strings` 無結果；BUILD 成功。
- [x] 3.3 Installer 字串：`Source/Installer/{en,zh-Hant}.lproj/InfoPlist.strings` 的 `CFBundleName` 改為 `Install Sloth-listwise Bopomofo` / `安裝 Sloth-listwise Bopomofo`；`Source/Installer/Base.lproj/MainMenu.xib`（:18–19、:21、:36、:55、:86、:126）與 `Source/Installer/zh-Hant.lproj/MainMenu.xib`（:18–161）中的產品名；`Source/Installer/AppDelegate.swift:255、271` 的 NSLocalizedString key 保留，改 `Source/Installer/{en,zh-Hant}.lproj/Localizable.strings:29、33` 的值。Verify: `plutil -lint` 四份 strings OK；`grep -rn "小麥\|McBopomofo Installer\|Install McBopomofo" Source/Installer/*.lproj/*.xib Source/Installer/Base.lproj/*.xib Source/Installer/*.lproj/InfoPlist.strings` 無結果；IBUILD 成功。

## 4. 路徑與安裝

- [x] 4.1 `Source/Preferences.swift:524` `@objc class UserPhraseLocationHelper`：新增 `@objc static` 的 `applicationSupportRoot`（`~/Library/Application Support/Sloth-listwise-Bopomofo`）、`modelsDirectory`、`rWalkModelDirectory`（`<root>/models/r-walk`）；`defaultUserPhraseLocation` 回傳 root；模型目錄不受自訂詞庫位置影響。新增 XCTest 斷言三個路徑，以及 `useCustomUserPhraseLocation = true` 時 `rWalkModelDirectory` 不變（測試前後依 `PreferencesTests.swift:33–47` 的方式快照並還原 defaults）。Verify: TEST 通過；`grep -n "\"McBopomofo\"" Source/Preferences.swift` 無結果。
- [x] 4.2 `Source/Installer/AppDelegate.swift`：`kTargetBin` / `kTargetBundle` / 安裝路徑改為 `Sloth-listwise-Bopomofo(.app)`（`kTargetBin` 同時是 :184 的資源名與 :71 `ArchiveUtil` appName，一致更名）；以 `NSRunningApplication.runningApplications(withBundleIdentifier:)` + `forceTerminate()` 取代 `killall -9`，之後輪詢 `isTerminated` 最多約 2 秒再複製。Verify: IBUILD 成功；`grep -n "killall\|McBopomofo.app\|\"McBopomofo\"" Source/Installer/AppDelegate.swift` 無結果。
- [x] 4.3 `Source/Data/Makefile:50–55` 的 `_install` / `_deploy` 改指向 `Sloth-listwise-Bopomofo.app`，並改用 `pkill -HUP -x Sloth-listwise-Bopomofo`（絕不碰原版 McBopomofo）。Verify: `make -n -C Source/Data _install` 的輸出只含 `Sloth-listwise-Bopomofo`，不含 `McBopomofo.app` 或 `pkill -HUP -f`。
- [x] 4.4 全 repo 殘留掃描。Verify: `grep -rn "org.openvanilla.inputmethod\|McBopomofo_1_Connection\|Input Methods/McBopomofo\|killall" Source McBopomofoTests McBopomofo.xcodeproj` 無結果（著作權註解、類別名、module 名不在掃描字串內）；完整 TEST 通過，或失敗項在 base commit 39b6e79 上同樣失敗。

## 5. 並存實測（需已安裝原版小麥注音；只能人工執行）

- [x] 5.1 前置：確認 `~/Library/Input Methods/McBopomofo.app` 為原版（`codesign -dv --verbose=2` 顯示上游簽章、`defaults read ~/Library/Input\ Methods/McBopomofo.app/Contents/Info.plist CFBundleShortVersionString` 為上游版本），並記下 `shasum -a 256` 與 `ls -la ~/Library/Application\ Support/McBopomofo/`。Verify: 結果記錄於 `../PLAN.md`。
  - 2026-10-07 實機：原版 McBopomofo 1.0，Developer ID Application: Lukhnos Liu (PYD935QB58)；App 檔案 hash cd3b9382…；`~/Library/Application Support/McBopomofo/` 為空；defaults md5 c6fd3508…。
- [ ] 5.2 安裝：Release build 複製到 `~/Library/Input Methods/`，執行 `~/Library/Input\ Methods/Sloth-listwise-Bopomofo.app/Contents/MacOS/Sloth-listwise-Bopomofo install --all`，登出再登入。Verify: `codesign --verify --deep --strict` 對新 App 成功；原版 `McBopomofo.app` 的 `shasum` 與 5.1 相同；系統設定 > 鍵盤 > 輸入來源同時列出「小麥注音」與「Sloth-listwise Bopomofo」並可同時啟用。
  - 2026-10-07 實機：Release b5824b5 已安裝，codesign strict OK；須重新開機後才出現在系統設定清單（ad-hoc 簽章仍可用）。原版 App hash 與 5.1 相同。待確認：輸入來源清單同時列出兩者（目前原版未啟用）。
- [ ] 5.3 實際打字：在 TextEdit 以兩者輪流輸入同一句、各自開選單看 About / 偏好設定名稱、在新 App 加一個使用者詞。Verify: 兩者都能正常組字送出；新詞只出現在 `~/Library/Application Support/Sloth-listwise-Bopomofo/data.txt`，`~/Library/Application Support/McBopomofo/` 列表與 5.1 相同；選單名稱為 `Sloth-listwise Bopomofo`。
  - 2026-10-07 實機：新 App 打字正常（R-walk 開／關各一句）；新增使用者詞「與台」只寫入新 root 的 data.txt，原版資料夾仍為空（測後已刪除該詞）。待做：與原版輪流打同一句、檢查選單名稱。
- [x] 5.4 偏好設定隔離：在新 App 偏好設定改一項。Verify: `defaults read io.github.fireman333.inputmethod.Sloth-listwise-Bopomofo` 出現該項，`defaults read org.openvanilla.inputmethod.McBopomofo` 無變化。
  - 2026-10-07 實機：`defaults write` AFMNeuralWalkEnabled 只出現在新 domain；原版 domain md5 與 5.1 相同。
- [ ] 5.5 首次啟動不遷移、不連上游更新：刪除新 root 後重啟新 App。Verify: 新 root 的 `data.txt` 只有 template 內容；Console.app 以程序名過濾無對 `mcbopomofo.openvanilla.org` 的請求、無「有新版本」提示。
  - 2026-10-07 實機：新 root 首次建立時只有 template 檔與 models（無遷移）。網路：App 程式碼無任何 URL、更新端點已移除，原因已查明：預編譯 onnxruntime 1.30 內建 Microsoft 1DS telemetry，載入模型時 POST 到 Microsoft Azure（4.207.44.76、40.79.141.156）。已以 `ORT_DISABLE_TELEMETRY=1` + `Env::DisableTelemetryEvents()` 關閉（`Source/Engine/Rescorer/ort_env.h`），重裝後 nettop 40 秒內 0 條外連、0 個 URL task。剩「刪除新 root 後重啟」未做，未勾。

## 6. 文件同步

- [x] 6.1 `openspec/project.md` Deploy & Distribution 的模型目錄由 `AFMInput` 改為 `~/Library/Application Support/Sloth-listwise-Bopomofo/models/`，並補 bundle id；提醒 `selection-pipeline` 實作時以 4.1 的 `rWalkModelDirectory` 取代其文件中的 `AFMInput` 路徑（不改該 change 的檔案）。Verify: `openspec validate --all` 通過；`grep -n "AFMInput" openspec/project.md` 無結果。
- [ ] 6.2 收尾檢查。Verify: `/opsx:verify` 三維度（completeness / correctness / coherence）全綠後才可 archive。
