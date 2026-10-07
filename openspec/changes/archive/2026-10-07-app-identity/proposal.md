# Proposal

## Why

AFMBopomofo 目前仍沿用小麥注音（McBopomofo）的全部身分：bundle id `org.openvanilla.inputmethod.McBopomofo`、輸入模式 id、IMK connection name、安裝路徑 `~/Library/Input Methods/McBopomofo.app`、UserDefaults domain 與 `~/Library/Application Support/McBopomofo/` 使用者資料夾。因此它與已安裝的原版小麥注音互相覆蓋，無法並存做實際打字測試；`selection-pipeline` change 的安裝實測（第 6 組）也因此被擋住，且它需要一個確定的模型根目錄。這個 change 必須先做。

## What Changes

- **BREAKING** App 的 bundle id 改為 `io.github.fireman333.inputmethod.Sloth-listwise-Bopomofo`；兩個輸入模式 id、`TISInputSourceID`、`InputMethodConnectionName` 全部由此推導（`<bundle id>.Bopomofo`、`<bundle id>.PlainBopomofo`、`<bundle id>_Connection`）。
- **BREAKING** 顯示名稱（輸入法選單、`CFBundleName` / `CFBundleDisplayName`、輸入模式名稱、App 內選單「關於 / 偏好設定」字串）在 Base / en / zh-Hant 全部改為 **Sloth-listwise Bopomofo**；純注音模式顯示為 **Sloth-listwise Plain Bopomofo**。
- **BREAKING** 產物改名為 `Sloth-listwise-Bopomofo.app`（以 `PRODUCT_NAME` 設定，Xcode target 名稱不改），安裝到 `~/Library/Input Methods/Sloth-listwise-Bopomofo.app`；Installer 的目標路徑與 `killall` 對象同步改名。Installer 自身 bundle id 改為 `io.github.fireman333.inputmethod.Sloth-listwise-Bopomofo.Installer`。
- **BREAKING** 預設使用者資料夾改為 `~/Library/Application Support/Sloth-listwise-Bopomofo/`，並定義 `models/r-walk/` 子目錄給 `selection-pipeline` 使用；路徑以單一常數集中定義。
- UserDefaults domain 隨 bundle id 自動換成新 domain；全新開始，不自動搬移小麥注音的使用者詞庫、排除詞、偏好設定（UserOverrideModel 本來就只存在記憶體，不受影響）。
- 移除指向 `mcbopomofo.openvanilla.org` 的自動更新檢查端點（`UpdateInfoEndpoint` / `UpdateInfoSite`），避免本 App 提示使用者去下載原版小麥注音。
- App、Installer、Tests 與 project 層級的 `MACOSX_DEPLOYMENT_TARGET` 一律升到 **26.0**（`LSMinimumSystemVersion` 跟著變；內嵌的 onnxruntime dylib 需要 ≥ 14，專案本來就只支援 macOS 26+）。本項在此 change 完成，`selection-pipeline` 不再處理。
- 測試 target bundle id 改為 `io.github.fireman333.inputmethod.Sloth-listwise-Bopomofo.Tests`，`TEST_HOST` 指向新產物路徑。

**非目標**：不自動遷移小麥注音資料（使用者可自行把舊資料夾的 `data.txt` 等檔案手動複製到新資料夾，本 change 不提供工具）；不改 Xcode target / scheme 名稱、Swift module 名稱、Objective-C 類別名稱；不改圖示；不處理簽章與 notarization。

## Capabilities

### New Capabilities
- `app-identity`：輸入法的對外身分與它擁有的系統資源——bundle id 與由它推導的各種 id、顯示名稱、安裝位置、偏好設定 domain、使用者資料與模型根目錄、最低系統版本，以及與已安裝原版小麥注音並存的保證。

### Modified Capabilities
（無——`openspec/specs/` 目前沒有任何既有 capability。）

## Impact

- `McBopomofo.xcodeproj/project.pbxproj`：App（:1095–1103、:1150–1157）、Installer（:1255–1263、:1297–1304）、Tests（:1342–1360、:1389–1406）的 `PRODUCT_BUNDLE_IDENTIFIER` / `PRODUCT_NAME` / `TEST_HOST` / `MACOSX_DEPLOYMENT_TARGET`；project 層 :997、:1041；產物檔案參照 :34、:130。
- `Source/McBopomofo-Info.plist`：輸入模式 id（:27、:55、:86–87）、`InputMethodConnectionName`（:91）、`TISInputSourceID`（:323）、更新端點（:328–331）、Services 選單字串（:118）。
- `Source/{Base,en,zh-Hant}.lproj/InfoPlist.strings`、`Source/Installer/{en,zh-Hant}.lproj/InfoPlist.strings`、`Source/{en,zh-Hant}.lproj/Localizable.strings`、`Source/Base.lproj/MainMenu.xib`、`Source/Installer/Base.lproj/MainMenu.xib`。
- `Source/main.swift:73`（connection name）、`Source/KeyHandler.mm:48–49`（輸入模式 id）、`Source/Preferences.swift:528`（使用者資料夾）、`Source/Installer/AppDelegate.swift:27–32`（安裝路徑、killall）。
- `openspec/project.md` 的模型目錄（目前寫 `AFMInput`）需更新；`selection-pipeline` 的模型路徑常數改用本 change 定義的根目錄（由該 change 實作時對齊，本 change 不改它的文件）。
- 已在用本 App 舊 build（舊 bundle id）的開發者需手動移除舊的 `~/Library/Input Methods/McBopomofo.app` 開發版；原版小麥注音不受影響。
