// Copyright (c) 2026 and onwards The McBopomofo Authors.
//
// Permission is hereby granted, free of charge, to any person
// obtaining a copy of this software and associated documentation
// files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use,
// copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following
// conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.

import Carbon
import FSEventStreamHelper
import InputMethodKit
import XCTest

@testable import McBopomofo

/// Personal memory (openspec change personal-memory).
class PersonalMemoryTests: XCTestCase {

    // MARK: - Task 2.1: user-phrase folder watcher filter (design D13)

    private var tempRoot: String!

    override func setUpWithError() throws {
        // NSTemporaryDirectory() is under /var/folders, a symlink to
        // /private/var/folders; FSEvents reports the /private form.
        tempRoot = (NSTemporaryDirectory() as NSString).appendingPathComponent(
            "PersonalMemoryTests-\(UUID().uuidString)")
        let fm = FileManager.default
        try fm.createDirectory(
            atPath: (tempRoot as NSString).appendingPathComponent("memory"),
            withIntermediateDirectories: true)
        try fm.createDirectory(
            atPath: (tempRoot as NSString).appendingPathComponent("models/r-walk"),
            withIntermediateDirectories: true)
        try "x".write(
            toFile: (tempRoot as NSString).appendingPathComponent("memory/override-model.txt"),
            atomically: false, encoding: .utf8)
    }

    override func tearDownWithError() throws {
        if let tempRoot {
            try? FileManager.default.removeItem(atPath: tempRoot)
        }
    }

    private var excluded: [String] {
        [
            (tempRoot as NSString).appendingPathComponent("memory"),
            (tempRoot as NSString).appendingPathComponent("models"),
        ]
    }

    private func path(_ relative: String, root: String? = nil) -> String {
        ((root ?? tempRoot) as NSString).appendingPathComponent(relative)
    }

    /// Feeds event batches through the filter and counts the reloads.
    private func reloadCount(_ batches: [[String]], excluded: [String]) -> Int {
        batches.filter {
            UserPhraseFolderEventFilter.shouldReload(eventPaths: $0, excludedDirectories: excluded)
        }.count
    }

    func testMemoryOnlyEventsDoNotReload() {
        let batch = [
            path("memory/override-model.txt"),
            path("memory/override-model.txt.tmp"),  // Renamed away, no longer exists.
            path("memory/sentences.jsonl"),  // Does not exist yet.
            path("memory/override-model.bad-1700000000.txt"),
            path("memory"),
        ]
        XCTAssertEqual(reloadCount([batch], excluded: excluded), 0)
        XCTAssertEqual(reloadCount(batch.map { [$0] }, excluded: excluded), 0)
    }

    func testModelsOnlyEventsDoNotReload() {
        let batch = [
            path("models/r-walk/model.onnx"),
            path("models/qwen/model.gguf"),  // Folder does not exist.
            path("models"),
        ]
        XCTAssertEqual(reloadCount([batch], excluded: excluded), 0)
    }

    func testMixedBatchReloadsOnce() {
        let batches = [
            [path("memory/override-model.txt"), path("memory/sentences.jsonl")],
            [path("memory/sentences.jsonl"), path("data.txt"), path("models/r-walk/x")],
            [path("models/r-walk/model.onnx")],
        ]
        XCTAssertEqual(reloadCount(batches, excluded: excluded), 1)
    }

    func testUserPhraseFilesReload() {
        XCTAssertEqual(reloadCount([[path("data.txt")]], excluded: excluded), 1)
        XCTAssertEqual(reloadCount([[path("phrases-replacement.txt")]], excluded: excluded), 1)
        XCTAssertEqual(reloadCount([[tempRoot]], excluded: excluded), 1)
        // Same prefix but not inside the excluded folders.
        XCTAssertEqual(reloadCount([[path("memory-notes.txt")]], excluded: excluded), 1)
        XCTAssertEqual(reloadCount([[path("modelsx/a.txt")]], excluded: excluded), 1)
        XCTAssertEqual(reloadCount([[]], excluded: excluded), 0)
    }

    func testPrivateVarEventPathMatchesVarRoot() throws {
        guard tempRoot.hasPrefix("/var/") else {
            throw XCTSkip("Temporary directory is not under /var: \(tempRoot!)")
        }
        let privateRoot = "/private" + tempRoot
        // Event paths in the /private form, excluded folders in the /var form.
        let memoryBatch = [
            path("memory/override-model.txt", root: privateRoot),
            path("memory/override-model.txt.tmp", root: privateRoot),
            path("memory/new-folder/new-file", root: privateRoot),
        ]
        XCTAssertEqual(reloadCount([memoryBatch], excluded: excluded), 0)
        XCTAssertEqual(
            reloadCount([memoryBatch + [path("data.txt", root: privateRoot)]], excluded: excluded),
            1)

        // And the other way around.
        let privateExcluded = excluded.map { "/private" + $0 }
        let varBatch = [path("memory/override-model.txt"), path("models/a/b")]
        XCTAssertEqual(reloadCount([varBatch], excluded: privateExcluded), 0)
        XCTAssertEqual(reloadCount([[path("data.txt")]], excluded: privateExcluded), 1)
    }

    func testCanonicalPathOfMissingFileKeepsResolvedParent() throws {
        let existing = UserPhraseFolderEventFilter.canonicalPath(path("memory"))
        let missing = UserPhraseFolderEventFilter.canonicalPath(path("memory/a/b.txt"))
        XCTAssertEqual(missing, existing + "/a/b.txt")
        let viaDots = UserPhraseFolderEventFilter.canonicalPath(path("models/../memory/x"))
        XCTAssertEqual(viaDots, existing + "/x")
    }

    func testDefaultExcludedDirectoriesAreUnderApplicationSupportRoot() {
        let root = UserPhraseLocationHelper.applicationSupportRoot
        XCTAssertEqual(UserPhraseLocationHelper.learningDataDirectory, root + "/memory")
        XCTAssertEqual(
            UserPhraseFolderEventFilter.defaultExcludedDirectories,
            [root + "/memory", root + "/models"])
    }

    /// Events injected into the real delegate (AppDelegate): memory-only and
    /// models-only batches cause no reload; a mixed batch causes exactly one
    /// (the same reloadGeneration delta as a user-phrase-only batch).
    func testDelegateReloadsOnlyForUserPhraseEvents() throws {
        let root = UserPhraseLocationHelper.applicationSupportRoot
        let memory = UserPhraseLocationHelper.learningDataDirectory
        let models = UserPhraseLocationHelper.modelsDirectory
        let delegate = AppDelegate()
        let helper = FSEventStreamHelper(path: root, queue: DispatchQueue.main)
        func event(_ path: String) -> FSEventStreamHelper.Event {
            FSEventStreamHelper.Event(
                path: path,
                flags: FSEventStreamEventFlags(kFSEventStreamEventFlagItemModified), id: 1)
        }
        func deliver(_ paths: [String]) -> UInt {
            let before = LanguageModelManager.reloadGeneration
            delegate.helper(helper, didReceive: paths.map(event))
            // The delegate reloads asynchronously on the main queue.
            RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.3))
            return LanguageModelManager.reloadGeneration - before
        }

        // Event paths in the /private form, as FSEvents reports them.
        let privateMemory = memory.hasPrefix("/var/") ? "/private" + memory : memory

        XCTAssertEqual(
            deliver([memory + "/override-model.txt", privateMemory + "/sentences.jsonl", memory]),
            0)
        XCTAssertEqual(deliver([models + "/r-walk/model.onnx", models]), 0)
        let single = deliver([root + "/data.txt"])
        XCTAssertGreaterThan(single, 0)
        XCTAssertEqual(
            deliver([
                memory + "/override-model.txt", root + "/data.txt",
                root + "/phrases-replacement.txt", models + "/x",
            ]),
            single)
    }
}

// MARK: - Shared test setUp (design D15)

/// Principal class of the test bundle (INFOPLIST_KEY_NSPrincipalClass): before
/// every XCTest, learning is off and the learning store is reset, so existing
/// tests run with the stock session-only user override model and nothing is
/// recorded. Personal-memory tests turn learning on in their own setUp.
@objc(LearningTestObserver)
final class LearningTestObserver: NSObject, XCTestObservation {
    override init() {
        super.init()
        XCTestObservationCenter.shared.addTestObserver(self)
        Self.turnLearningOff()
    }

    func testCaseWillStart(_ testCase: XCTestCase) {
        Self.turnLearningOff()
    }

    static func turnLearningOff() {
        Preferences.learningEnabled = false
        AFMLearningStore.shared.resetForTesting()
        LanguageModelManager.configureLearning(enabled: false)
    }
}

/// A text client that records what the input method commits.
final class FakeTextClient: NSObject, IMKTextInput {
    var inserted: [String] = []
    var marked = ""

    func insertText(_ string: Any!, replacementRange: NSRange) {
        if let s = string as? String {
            inserted.append(s)
        } else if let s = string as? NSAttributedString {
            inserted.append(s.string)
        }
    }
    func setMarkedText(_ string: Any!, selectionRange: NSRange, replacementRange: NSRange) {
        marked = (string as? String) ?? (string as? NSAttributedString)?.string ?? ""
    }
    func selectedRange() -> NSRange { NSRange(location: 0, length: 0) }
    func markedRange() -> NSRange { NSRange(location: NSNotFound, length: 0) }
    func attributedSubstring(from range: NSRange) -> NSAttributedString! { NSAttributedString() }
    func length() -> Int { 0 }
    func characterIndex(
        for point: NSPoint, tracking mappingMode: IMKLocationToOffsetMappingMode,
        inMarkedRange: UnsafeMutablePointer<ObjCBool>!
    ) -> Int { 0 }
    func attributes(forCharacterIndex index: Int, lineHeightRectangle lineRect: UnsafeMutablePointer<NSRect>!)
        -> [AnyHashable: Any]!
    { [:] }
    func validAttributesForMarkedText() -> [Any]! { [] }
    func overrideKeyboard(withKeyboardNamed keyboardUniqueName: String!) {}
    func selectMode(_ modeIdentifier: String!) {}
    func supportsUnicode() -> Bool { true }
    func bundleIdentifier() -> String! { "test.client" }
    func windowLevel() -> CGWindowLevel { 0 }
    func supportsProperty(_ property: TSMDocumentPropertyTag) -> Bool { false }
    func uniqueClientIdentifierString() -> String! { "test.client" }
    func string(from range: NSRange, actualRange: NSRangePointer!) -> String! { "" }
    func firstRect(forCharacterRange aRange: NSRange, actualRange: NSRangePointer!) -> NSRect { .zero }
}

/// Base class for personal-memory tests: learning on, injected clock, secure
/// input check and save delay; the memory folder under the per-PID test root
/// is emptied before and after each test.
class LearningTestCase: XCTestCase {
    var store: AFMLearningStore { AFMLearningStore.shared }
    var memoryDirectory: String { UserPhraseLocationHelper.learningDataDirectory }
    var memoryFile: String { (memoryDirectory as NSString).appendingPathComponent("override-model.txt") }
    var poolFile: String { (memoryDirectory as NSString).appendingPathComponent("sentences.jsonl") }

    /// Injected wall clock (seconds since 1970).
    var now: Double = 1_800_000_000
    var secure = false
    /// Fakes captured unowned by their blocks; kept alive until tearDown.
    var retainedFakes: [AnyObject] = []
    private var savedPreferences: [String: Any] = [:]
    private static let preferenceKeys = [
        "AssociatedPhrasesEnabled", "AFMNeuralWalkEnabled", "AFMPreciseModeEnabled", "KeyboardLayout",
        "ChineseConversionEnabled", "ChineseConversionStyle", "BopomofoFontAnnotationSupportEnabled",
        "PhraseReplacementEnabled", "LetterBehavior", "ControlEnterOutput", "ChooseCandidateUsingSpaceKey",
        "EscToCleanInputBuffer", "SwitchInputSourceUponShiftLetterKeyComboEnabled",
        "SelectPhraseAfterCursorAsCandidate", "MoveCursorAfterSelectingCandidate", "AFMLearningEnabled",
    ]

    static let day: Double = 86400

    override func setUpWithError() throws {
        for key in Self.preferenceKeys {
            savedPreferences[key] = Preferences.defaults.object(forKey: key) ?? NSNull()
        }
        Preferences.associatedPhrasesEnabled = false
        Preferences.neuralWalkEnabled = true
        Preferences.preciseModeEnabled = false
        Preferences.keyboardLayout = .standard
        Preferences.chineseConversionEnabled = false
        Preferences.chineseConversionStyle = .output
        Preferences.bopomofoFontAnnotationSupportEnabled = false
        Preferences.phraseReplacementEnabled = false
        Preferences.letterBehavior = 0
        Preferences.controlEnterOutput = .off
        Preferences.chooseCandidateUsingSpace = true
        Preferences.escToCleanInputBuffer = false
        Preferences.switchInputSourceUponShiftLetterKeyComboEnabled = false
        Preferences.selectPhraseAfterCursorAsCandidate = false
        Preferences.moveCursorAfterSelectingCandidate = false
        LanguageModelManager.loadDataModels()
        LanguageModelManager.clearUserPhrasesForTesting()
        LanguageModelManager.phraseReplacementEnabled = false
        AFMNeuralSelector.resetForTesting()
        AFMPreciseController.resetForTesting()

        try? FileManager.default.removeItem(atPath: memoryDirectory)
        store.resetForTesting()
        store.clock = { [unowned self] in self.now }
        store.secureInputCheck = { [unowned self] in self.secure }
        store.saveDelay = 0.2
        Preferences.learningEnabled = true
        // As at process start: stock model, then the launch configuration.
        LanguageModelManager.resetLearningConfigurationForTesting()
        LanguageModelManager.configureLearning(enabled: true)
        XCTAssertTrue(LanguageModelManager.learningConfigurationActive)
        XCTAssertTrue(store.recordingEnabled)
    }

    override func tearDownWithError() throws {
        store.waitUntilIdleForTesting()
        store.resetForTesting()
        LanguageModelManager.configureLearning(enabled: false)
        AFMPreciseController.shared.enabled = false
        AFMPreciseController.resetForTesting()
        AFMNeuralSelector.resetForTesting()
        for (key, value) in savedPreferences {
            if value is NSNull {
                Preferences.defaults.removeObject(forKey: key)
            } else {
                Preferences.defaults.set(value, forKey: key)
            }
        }
        Preferences.learningEnabled = false
        LanguageModelManager.phraseReplacementEnabled = false
        LanguageModelManager.loadUserPhrases(enableForPlainBopomofo: Preferences.enableUserPhrasesInPlainBopomofo)
        try? FileManager.default.removeItem(atPath: memoryDirectory)
    }

    // MARK: Helpers

    func makeHandler(_ mode: InputMode = .bopomofo) -> KeyHandler {
        let handler = KeyHandler()
        handler.inputMode = mode
        handler.syncWithPreferences()
        return handler
    }

    func key(_ text: String, keyCode: UInt16 = 0, charCode code: UInt16? = nil, flags: NSEvent.ModifierFlags = [])
        -> KeyHandlerInput
    {
        KeyHandlerInput(
            inputText: text, keyCode: keyCode, charCode: code ?? charCode(text), flags: flags,
            isVerticalMode: false)
    }

    var enterKey: KeyHandlerInput { key(" ", keyCode: KeyCode.enter.rawValue, charCode: 13) }

    /// Types keys into a bare KeyHandler; returns the last state.
    @discardableResult
    func type(_ keys: String, into handler: KeyHandler, from initial: InputState = InputState.Empty()) -> InputState {
        var state = initial
        for k in keys.map({ String($0) }) {
            handler.handle(input: key(k), state: state) { state = $0 } errorCallback: {}
        }
        return state
    }

    /// Sends one key to a bare KeyHandler and reports committed texts to it,
    /// as the input method controller does.
    @discardableResult
    func send(_ input: KeyHandlerInput, to handler: KeyHandler, state: inout InputState) -> [String] {
        var committed: [String] = []
        let previous = state
        var last = state
        handler.handle(input: input, state: previous) { newState in
            if let c = newState as? InputState.Committing {
                committed.append(c.poppedText)
                handler.recordCommittedText(c.poppedText)
            } else if newState is InputState.Empty, let p = last as? InputState.NotEmpty {
                committed.append(p.composingBuffer)
                handler.recordCommittedText(p.composingBuffer)
            }
            last = newState
        } errorCallback: {}
        state = last
        return committed
    }

    /// Picks a candidate at the candidate cursor, like the candidate window.
    @discardableResult
    func pick(_ reading: String, _ value: String, in handler: KeyHandler) -> InputState {
        handler.fixNode(
            reading: reading, value: value, originalCursorIndex: handler.cursorIndex,
            useMoveCursorAfterSelectionSetting: false)
        return handler.buildInputtingState()
    }

    /// Types a composition into a fresh handler and commits it with Enter.
    @discardableResult
    func typeAndCommit(_ keys: String, picks: [(String, String)] = [], handler: KeyHandler? = nil) -> String? {
        let h = handler ?? makeHandler()
        var state = type(keys, into: h)
        for (reading, value) in picks {
            state = pick(reading, value, in: h)
        }
        return send(enterKey, to: h, state: &state).first
    }

    /// The sentence pool records, after the store queue drained.
    func poolRecords() -> [[String: Any]] {
        store.waitUntilIdleForTesting()
        guard let data = FileManager.default.contents(atPath: poolFile),
            let text = String(data: data, encoding: .utf8)
        else {
            return []
        }
        return text.split(separator: "\n").compactMap {
            (try? JSONSerialization.jsonObject(with: Data($0.utf8))) as? [String: Any]
        }
    }

    func nodes(_ record: [String: Any]) -> [[String]] {
        (record["nodes"] as? [[String]]) ?? []
    }

    func composing(_ state: InputState) -> String? {
        (state as? InputState.Inputting)?.composingBuffer
    }

    func waitUntil(timeout: TimeInterval = 10, _ condition: () -> Bool) {
        let deadline = Date().addingTimeInterval(timeout)
        while !condition() && Date() < deadline {
            RunLoop.current.run(until: Date().addingTimeInterval(0.01))
        }
    }

    func spin(_ seconds: TimeInterval) {
        RunLoop.current.run(until: Date().addingTimeInterval(seconds))
    }

    @discardableResult
    func installFakeSelector(_ fake: FakeNeuralScorer) -> AFMNeuralSelector {
        retainedFakes.append(fake)
        let selector = AFMNeuralSelector(scoreBlockForTesting: fake.block, loadDelay: 0)
        selector.enabled = true
        selector.alpha = 0.5
        selector.startLoadingIfNeeded()
        waitUntil { selector.state != .loading }
        XCTAssertEqual(selector.state, .ready)
        AFMNeuralSelector.shared = selector
        return selector
    }

    func fileAttributes(_ path: String) -> [FileAttributeKey: Any]? {
        try? FileManager.default.attributesOfItem(atPath: path)
    }

    func posixMode(_ path: String) -> Int? {
        (fileAttributes(path)?[.posixPermissions] as? NSNumber)?.intValue
    }
}

/// ㄓㄨㄥ ㄏㄨㄚˊ (中華) on the standard layout.
let kZhongHuaKeys = "5j/ cj86"
/// ㄓㄨㄥ (中).
let kZhongKeys = "5j/ "

// MARK: - Task 2.2: preferences and storage path (design D10 / D14)

class LearningPreferencesTests: XCTestCase {
    func testLearningPreferenceDefaultsAndKeys() {
        let saved = Preferences.defaults.object(forKey: "AFMLearningEnabled")
        let savedNotice = Preferences.defaults.object(forKey: "AFMLearningNoticeShown")
        defer {
            Preferences.defaults.set(saved, forKey: "AFMLearningEnabled")
            Preferences.defaults.set(savedNotice, forKey: "AFMLearningNoticeShown")
        }
        Preferences.defaults.removeObject(forKey: "AFMLearningEnabled")
        Preferences.defaults.removeObject(forKey: "AFMLearningNoticeShown")
        XCTAssertTrue(Preferences.learningEnabled)
        XCTAssertFalse(Preferences.learningNoticeShown)
        XCTAssertTrue(Preferences.allKeys.contains("AFMLearningEnabled"))
        XCTAssertTrue(Preferences.allKeys.contains("AFMLearningNoticeShown"))
        Preferences.learningNoticeShown = true
        XCTAssertEqual(Preferences.defaults.object(forKey: "AFMLearningNoticeShown") as? Bool, true)
    }

    /// The shared setUp (LearningTestObserver) turned learning off.
    func testSharedSetUpTurnsLearningOff() {
        XCTAssertFalse(Preferences.learningEnabled)
        XCTAssertFalse(AFMLearningStore.shared.recordingEnabled)
        XCTAssertFalse(LanguageModelManager.learningConfigurationActive)
    }

    func testLearningDataDirectoryIgnoresCustomUserPhraseLocation() {
        let savedUse = Preferences.useCustomUserPhraseLocation
        let savedLocation = Preferences.customUserPhraseLocation
        defer {
            Preferences.useCustomUserPhraseLocation = savedUse
            Preferences.customUserPhraseLocation = savedLocation
        }
        let root = UserPhraseLocationHelper.applicationSupportRoot
        XCTAssertTrue(root.contains("-tests-\(ProcessInfo.processInfo.processIdentifier)"), root)
        let expected = root + "/memory"
        XCTAssertEqual(UserPhraseLocationHelper.learningDataDirectory, expected)
        let custom = (NSTemporaryDirectory() as NSString).appendingPathComponent("custom-\(UUID().uuidString)")
        Preferences.customUserPhraseLocation = custom
        Preferences.useCustomUserPhraseLocation = true
        XCTAssertEqual(LanguageModelManager.dataFolderPath, custom)
        XCTAssertEqual(UserPhraseLocationHelper.learningDataDirectory, expected)
        AFMLearningStore.shared.resetForTesting()
        XCTAssertEqual(AFMLearningStore.shared.directory, expected)
        XCTAssertEqual(AFMLearningStore.shared.saveDelay, 3.0)
    }
}

// MARK: - Task 2.3: store, configuration lifecycle, clearing (design D2 / D3 / D11)

class LearningStoreTests: LearningTestCase {

    /// Picks 鍾 for ㄓㄨㄥ (a context with no preceding nodes).
    func pickZhong(_ value: String = "鍾") {
        let handler = makeHandler()
        type(kZhongKeys, into: handler)
        pick("ㄓㄨㄥ", value, in: handler)
    }

    /// What a fresh composition of ㄓㄨㄥ shows.
    func zhongShown() -> String? {
        composing(type(kZhongKeys, into: makeHandler()))
    }

    func testPickFlushRestartRestoresSuggestion() {
        XCTAssertEqual(zhongShown(), "中")
        pickZhong()
        XCTAssertEqual(zhongShown(), "鍾")
        XCTAssertTrue(store.hasUnsavedChanges)
        LanguageModelManager.flushLearningData()
        store.waitUntilIdleForTesting()
        XCTAssertTrue(FileManager.default.fileExists(atPath: memoryFile))

        // Process restart: stock model, nothing saved, then the launch configuration.
        LanguageModelManager.resetLearningConfigurationForTesting()
        XCTAssertEqual(zhongShown(), "中")
        LanguageModelManager.configureLearning(enabled: true)
        XCTAssertEqual(LanguageModelManager.userOverrideModelSizeForTesting, 1)
        XCTAssertEqual(zhongShown(), "鍾")
        // 3 days later (stock 1.5 h half-life would have forgotten it).
        now += 3 * Self.day
        XCTAssertEqual(zhongShown(), "鍾")
    }

    func testDebouncedSaveWritesAfterDelay() {
        pickZhong()
        store.waitUntilIdleForTesting()
        XCTAssertFalse(FileManager.default.fileExists(atPath: memoryFile), "not before the delay")
        waitUntil(timeout: 3) { FileManager.default.fileExists(atPath: memoryFile) }
        XCTAssertTrue(FileManager.default.fileExists(atPath: memoryFile))
        XCTAssertFalse(store.hasUnsavedChanges)
        let text = try? String(contentsOfFile: memoryFile, encoding: .utf8)
        XCTAssertEqual(text?.hasPrefix("AFM-UOM 1\n"), true)
        XCTAssertEqual(text?.contains("\t鍾\t"), true)
    }

    /// Learning off with a dirty model: saved exactly once at the switch, then
    /// nothing is written (real 3 s save delay).
    func testTurningOffSavesOnceThenNothing() throws {
        store.saveDelay = 3.0
        pickZhong()
        XCTAssertFalse(FileManager.default.fileExists(atPath: memoryFile))
        let saves = store.saveCountForTesting
        Preferences.setLearningEnabled(false)
        XCTAssertFalse(Preferences.learningEnabled)
        XCTAssertFalse(store.recordingEnabled)
        XCTAssertEqual(store.saveCountForTesting, saves + 1)
        let saved = try String(contentsOfFile: memoryFile, encoding: .utf8)
        XCTAssertTrue(saved.contains("\t鍾\t"))
        let mtime = fileAttributes(memoryFile)?[.modificationDate] as? Date

        // Before the next activation (long-term model still in memory) ...
        pickZhong("忠")
        typeAndCommit(kZhongHuaKeys)
        // ... and after it (stock model).
        LanguageModelManager.configureLearning(enabled: Preferences.learningEnabled)
        XCTAssertFalse(LanguageModelManager.learningConfigurationActive)
        pickZhong("終")
        typeAndCommit(kZhongHuaKeys)
        LanguageModelManager.flushLearningData()
        spin(3.5)
        store.waitUntilIdleForTesting()
        XCTAssertEqual(try String(contentsOfFile: memoryFile, encoding: .utf8), saved)
        XCTAssertEqual(fileAttributes(memoryFile)?[.modificationDate] as? Date, mtime)
        XCTAssertEqual(store.saveCountForTesting, saves + 1)
        XCTAssertFalse(FileManager.default.fileExists(atPath: poolFile))
        let names = try FileManager.default.contentsOfDirectory(atPath: memoryDirectory)
        XCTAssertEqual(names, ["override-model.txt"])
    }

    /// Recording on while the stock model is still active (preference turned
    /// on, no activation yet): the stock model never writes the memory file.
    func testStockSessionModelNeverWritesMemoryFile() {
        LanguageModelManager.configureLearning(enabled: false)
        XCTAssertFalse(LanguageModelManager.learningConfigurationActive)
        Preferences.setLearningEnabled(true)
        XCTAssertTrue(store.recordingEnabled)
        pickZhong()
        store.markDirty()
        spin(0.5)
        LanguageModelManager.flushLearningData()
        store.waitUntilIdleForTesting()
        XCTAssertFalse(FileManager.default.fileExists(atPath: memoryFile))
    }

    func testLearningBackOnRestoresSavedPick() {
        pickZhong()
        LanguageModelManager.flushLearningData()
        Preferences.setLearningEnabled(false)
        LanguageModelManager.configureLearning(enabled: false)
        XCTAssertEqual(zhongShown(), "中", "stock session model is empty")
        Preferences.setLearningEnabled(true)
        LanguageModelManager.configureLearning(enabled: true)
        XCTAssertEqual(zhongShown(), "鍾")
    }

    func testMalformedFileIsSetAsideAndTypingWorks() throws {
        try FileManager.default.createDirectory(atPath: memoryDirectory, withIntermediateDirectories: true)
        let garbage = "AFM-UOM 1\nkey\tonly-two-fields\n"
        try garbage.write(toFile: memoryFile, atomically: true, encoding: .utf8)
        LanguageModelManager.resetLearningConfigurationForTesting()
        LanguageModelManager.configureLearning(enabled: true)
        XCTAssertTrue(LanguageModelManager.learningConfigurationActive)
        XCTAssertEqual(LanguageModelManager.userOverrideModelSizeForTesting, 0)
        XCTAssertFalse(FileManager.default.fileExists(atPath: memoryFile))
        let names = try FileManager.default.contentsOfDirectory(atPath: memoryDirectory)
        let bad = names.filter { $0.hasPrefix("override-model.bad-") && $0.hasSuffix(".txt") }
        XCTAssertEqual(bad, ["override-model.bad-\(Int(now)).txt"])
        let badPath = (memoryDirectory as NSString).appendingPathComponent(bad[0])
        XCTAssertEqual(try String(contentsOfFile: badPath, encoding: .utf8), garbage)
        XCTAssertEqual(zhongShown(), "中")
        pickZhong()
        XCTAssertEqual(zhongShown(), "鍾")
    }

    func testUnreadableFileIsSetAside() throws {
        try FileManager.default.createDirectory(atPath: memoryDirectory, withIntermediateDirectories: true)
        try "AFM-UOM 1\n".write(toFile: memoryFile, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o000], ofItemAtPath: memoryFile)
        LanguageModelManager.resetLearningConfigurationForTesting()
        LanguageModelManager.configureLearning(enabled: true)
        XCTAssertEqual(LanguageModelManager.userOverrideModelSizeForTesting, 0)
        XCTAssertFalse(FileManager.default.fileExists(atPath: memoryFile))
        let names = try FileManager.default.contentsOfDirectory(atPath: memoryDirectory)
        XCTAssertEqual(names.filter { $0.hasPrefix("override-model.bad-") }.count, 1)
        XCTAssertEqual(zhongShown(), "中")
    }

    func testSecureInputHeldByClientDecision() {
        // (enabled, holder pid, client pid) -> paused
        XCTAssertTrue(AFMSecureInputHeldByClient(true, 500, 500))
        XCTAssertFalse(AFMSecureInputHeldByClient(true, 413, 500))  // e.g. loginwindow holds it
        XCTAssertFalse(AFMSecureInputHeldByClient(false, 500, 500))
        XCTAssertFalse(AFMSecureInputHeldByClient(true, 0, 500))    // holder unknown
        XCTAssertFalse(AFMSecureInputHeldByClient(true, 500, 0))    // client unknown
        XCTAssertFalse(AFMSecureInputHeldByClient(true, 0, 0))
    }

    func testFileAndFolderModes() throws {
        pickZhong()
        typeAndCommit(kZhongHuaKeys)
        LanguageModelManager.flushLearningData()
        store.waitUntilIdleForTesting()
        XCTAssertEqual(posixMode(memoryDirectory), 0o700)
        XCTAssertEqual(posixMode(memoryFile), 0o600)
        XCTAssertEqual(posixMode(poolFile), 0o600)
        XCTAssertFalse(FileManager.default.fileExists(atPath: memoryFile + ".tmp"))
    }

    func testClearRemovesDataAndEmptiesMemory() throws {
        let fm = FileManager.default
        let root = UserPhraseLocationHelper.applicationSupportRoot
        XCTAssertTrue(LanguageModelManager.checkIfUserLanguageModelFilesExist())
        let modelsFile = (UserPhraseLocationHelper.modelsDirectory as NSString).appendingPathComponent("keep.txt")
        try fm.createDirectory(atPath: UserPhraseLocationHelper.modelsDirectory, withIntermediateDirectories: true)
        try "model".write(toFile: modelsFile, atomically: true, encoding: .utf8)
        defer { try? fm.removeItem(atPath: modelsFile) }
        let others = [
            LanguageModelManager.userPhrasesDataPathMcBopomofo,
            LanguageModelManager.excludedPhrasesDataPathMcBopomofo,
            LanguageModelManager.phraseReplacementDataPathMcBopomofo,
            LanguageModelManager.userPhrasesDataPathPlainBopomofo,
            modelsFile,
        ]
        XCTAssertTrue(others.allSatisfy { $0.hasPrefix(root) })
        let before = others.map { fm.contents(atPath: $0) }
        let beforeDates = others.map { fileAttributes($0)?[.modificationDate] as? Date }

        pickZhong()
        typeAndCommit(kZhongHuaKeys)
        LanguageModelManager.flushLearningData()
        store.waitUntilIdleForTesting()
        let badFile = (memoryDirectory as NSString).appendingPathComponent("override-model.bad-1.txt")
        try "bad".write(toFile: badFile, atomically: true, encoding: .utf8)
        let unrelated = (memoryDirectory as NSString).appendingPathComponent("notes.txt")
        try "keep".write(toFile: unrelated, atomically: true, encoding: .utf8)
        XCTAssertTrue(fm.fileExists(atPath: memoryFile))
        XCTAssertTrue(fm.fileExists(atPath: poolFile))
        let prefs = Preferences.defaults.dictionaryRepresentation() as NSDictionary

        // A composition open across the clear, and a pending save timer.
        let open = makeHandler()
        var openState = type(kZhongHuaKeys, into: open)
        pickZhong("忠")
        XCTAssertTrue(store.hasUnsavedChanges)

        LanguageModelManager.clearLearningData()
        XCTAssertEqual(LanguageModelManager.userOverrideModelSizeForTesting, 0, "emptied immediately")
        XCTAssertTrue(LanguageModelManager.learningConfigurationActive)
        XCTAssertEqual(zhongShown(), "中")
        spin(0.5)  // Past the save delay: the cancelled timer writes nothing.
        store.waitUntilIdleForTesting()
        XCTAssertFalse(fm.fileExists(atPath: memoryFile))
        XCTAssertFalse(fm.fileExists(atPath: poolFile))
        XCTAssertFalse(fm.fileExists(atPath: badFile))
        XCTAssertTrue(fm.fileExists(atPath: unrelated))
        XCTAssertEqual(others.map { fm.contents(atPath: $0) }, before)
        XCTAssertEqual(others.map { fileAttributes($0)?[.modificationDate] as? Date }, beforeDates)
        XCTAssertEqual(Preferences.defaults.dictionaryRepresentation() as NSDictionary, prefs)
        XCTAssertTrue(Preferences.learningEnabled)

        // The open composition is committed after the clear: new data.
        send(enterKey, to: open, state: &openState)
        XCTAssertEqual(poolRecords().count, 1)
    }

    /// A save and a pool record queued before a clear do not recreate data.
    func testQueuedWritesBeforeClearDoNotRecreateFiles() {
        store.suspendQueueForTesting()
        pickZhong()
        LanguageModelManager.flushLearningData()  // Queued save.
        typeAndCommit(kZhongHuaKeys)  // Queued record.
        LanguageModelManager.clearLearningData()
        store.resumeQueueForTesting()
        store.waitUntilIdleForTesting()
        XCTAssertFalse(FileManager.default.fileExists(atPath: memoryFile))
        XCTAssertFalse(FileManager.default.fileExists(atPath: poolFile))
        // Data after the clear is written again.
        typeAndCommit(kZhongHuaKeys)
        XCTAssertEqual(poolRecords().count, 1)
    }

    /// The files a save and a pool append write, delivered as watcher events
    /// to the real delegate, do not reload user phrases.
    func testSaveAndPoolAppendDoNotReloadUserPhrases() throws {
        // Barrier (2026-10-08): the test host's FSEvents stream delivers its first batch several seconds
        // after launch, including the launch-time user-phrase template writes in the root, which do reload.
        // Write a file in the root and wait for its reload, so no earlier event lands in the measured window.
        let barrier = (UserPhraseLocationHelper.applicationSupportRoot as NSString)
            .appendingPathComponent("afm-fsevents-barrier.txt")
        let g0 = LanguageModelManager.reloadGeneration
        try "barrier".write(toFile: barrier, atomically: false, encoding: .utf8)
        addTeardownBlock { try? FileManager.default.removeItem(atPath: barrier) }
        waitUntil(timeout: 10) { LanguageModelManager.reloadGeneration != g0 }
        spin(1.2)
        let before = LanguageModelManager.reloadGeneration
        pickZhong()
        typeAndCommit(kZhongHuaKeys)
        LanguageModelManager.flushLearningData()
        store.waitUntilIdleForTesting()
        spin(1.5)  // FSEvents latency is 1 s (if the watcher runs at all).
        XCTAssertEqual(LanguageModelManager.reloadGeneration, before)

        let written = try FileManager.default.contentsOfDirectory(atPath: memoryDirectory).map {
            (memoryDirectory as NSString).appendingPathComponent($0)
        }
        XCTAssertEqual(Set(written), [memoryFile, poolFile])
        let delegate = AppDelegate()
        let helper = FSEventStreamHelper(
            path: UserPhraseLocationHelper.applicationSupportRoot, queue: DispatchQueue.main)
        let events = (written + [memoryFile + ".tmp", memoryDirectory]).flatMap { path in
            [path, path.hasPrefix("/var/") ? "/private" + path : path]
        }.map {
            FSEventStreamHelper.Event(
                path: $0, flags: FSEventStreamEventFlags(kFSEventStreamEventFlagItemModified), id: 1)
        }
        delegate.helper(helper, didReceive: events)
        spin(0.3)
        XCTAssertEqual(LanguageModelManager.reloadGeneration, before)
    }

    /// Load of 5,000 contexts < 50 ms; serialization time recorded.
    func testLoadAndSerializeTimingFor5000Contexts() throws {
        var text = "AFM-UOM 1\n"
        for i in 0..<5000 {
            // Realistic key: (reading,value)-(reading,value)-(reading,value).
            let key = "(ㄋㄧˇ,你\(i))-(ㄏㄠˇ,好)-(ㄓㄨㄥ,中)"
            text += "\(key)\t鍾\t2\t\(now - Double(i))\t0\n"
            if i % 3 == 0 {
                text += "\(key)\t忠\t1\t\(now - Double(i) - 1)\t0\n"
            }
        }
        try FileManager.default.createDirectory(atPath: memoryDirectory, withIntermediateDirectories: true)
        try text.write(toFile: memoryFile, atomically: true, encoding: .utf8)
        var loads: [Double] = []
        for _ in 0..<5 {
            LanguageModelManager.resetLearningConfigurationForTesting()
            LanguageModelManager.configureLearning(enabled: true)
            loads.append(store.lastLoadMilliseconds)
        }
        XCTAssertEqual(LanguageModelManager.userOverrideModelSizeForTesting, 5000)
        var serializes: [Double] = []
        for _ in 0..<5 {
            pickZhong()
            LanguageModelManager.flushLearningData()
            serializes.append(store.lastSerializeMilliseconds)
        }
        store.waitUntilIdleForTesting()
        let maxLoad = loads.max() ?? 0
        let size = (fileAttributes(memoryFile)?[.size] as? NSNumber)?.intValue ?? 0
        print(
            String(
                format: "AFM learning timing: load 5000 contexts min %.2f / max %.2f ms; serialize min %.2f / max %.2f ms; file %d bytes",
                loads.min() ?? 0, maxLoad, serializes.min() ?? 0, serializes.max() ?? 0, size))
        XCTAssertLessThan(maxLoad, 50)
    }
}

// MARK: - Task 4.1: learning switch and clear button (design D10)

class LearningPreferencesViewModelTests: LearningTestCase {

    @MainActor
    func testToggleWritesKeyAndStopsRecording() {
        let model = PreferencesViewModel()
        XCTAssertTrue(model.learningEnabled)

        model.learningEnabled = false
        XCTAssertEqual(Preferences.defaults.object(forKey: "AFMLearningEnabled") as? Bool, false)
        XCTAssertFalse(Preferences.learningEnabled)
        XCTAssertFalse(store.recordingEnabled, "recording stops at once, before any reactivation")
        typeAndCommit(kZhongHuaKeys, picks: [("ㄓㄨㄥ", "鍾")])
        XCTAssertTrue(poolRecords().isEmpty)
        XCTAssertFalse(FileManager.default.fileExists(atPath: poolFile))

        model.learningEnabled = true
        XCTAssertEqual(Preferences.defaults.object(forKey: "AFMLearningEnabled") as? Bool, true)
        XCTAssertTrue(store.recordingEnabled)
        typeAndCommit(kZhongHuaKeys)
        XCTAssertEqual(poolRecords().count, 1)
    }

    /// Writes a memory file and a pool record.
    func makeLearningData() {
        let handler = makeHandler()
        type(kZhongKeys, into: handler)
        pick("ㄓㄨㄥ", "鍾", in: handler)
        typeAndCommit(kZhongHuaKeys)
        LanguageModelManager.flushLearningData()
        store.waitUntilIdleForTesting()
        XCTAssertTrue(FileManager.default.fileExists(atPath: memoryFile))
        XCTAssertTrue(FileManager.default.fileExists(atPath: poolFile))
    }

    @MainActor
    func testConfirmClears() {
        makeLearningData()
        let model = PreferencesViewModel()
        var asked = 0
        model.confirmClearLearningData = {
            asked += 1
            return true
        }
        XCTAssertTrue(model.clearLearningData())
        XCTAssertEqual(asked, 1)
        XCTAssertEqual(LanguageModelManager.userOverrideModelSizeForTesting, 0)
        store.waitUntilIdleForTesting()
        XCTAssertFalse(FileManager.default.fileExists(atPath: memoryFile))
        XCTAssertFalse(FileManager.default.fileExists(atPath: poolFile))
        XCTAssertTrue(model.learningEnabled, "clearing keeps the learning switch")
        XCTAssertEqual(composing(type(kZhongKeys, into: makeHandler())), "中")
    }

    @MainActor
    func testCancelDeletesNothing() {
        makeLearningData()
        let memoryBefore = FileManager.default.contents(atPath: memoryFile)
        let poolBefore = FileManager.default.contents(atPath: poolFile)
        let sizeBefore = LanguageModelManager.userOverrideModelSizeForTesting
        let model = PreferencesViewModel()
        var asked = 0
        model.confirmClearLearningData = {
            asked += 1
            return false
        }
        XCTAssertFalse(model.clearLearningData())
        XCTAssertEqual(asked, 1)
        store.waitUntilIdleForTesting()
        XCTAssertEqual(FileManager.default.contents(atPath: memoryFile), memoryBefore)
        XCTAssertEqual(FileManager.default.contents(atPath: poolFile), poolBefore)
        XCTAssertEqual(LanguageModelManager.userOverrideModelSizeForTesting, sizeBefore)
        XCTAssertEqual(composing(type(kZhongKeys, into: makeHandler())), "鍾")
    }

    func testControlTitlesLocalized() throws {
        for (localization, toggle, button, clear, cancel) in [
            ("zh-Hant", "學習我的選字", "清除學習資料…", "清除", "取消"),
            ("en", "Learn from my typing", "Clear Learning Data…", "Clear", "Cancel"),
        ] {
            guard let path = Bundle.main.path(forResource: localization, ofType: "lproj"),
                let bundle = Bundle(path: path)
            else {
                throw XCTSkip("no \(localization).lproj in the app bundle")
            }
            let l = { bundle.localizedString(forKey: $0, value: nil, table: nil) }
            XCTAssertEqual(l("Learn from my typing"), toggle)
            XCTAssertEqual(l("Clear Learning Data…"), button)
            XCTAssertEqual(l("Clear"), clear)
            XCTAssertEqual(l("Cancel"), cancel)
        }
    }
}

// MARK: - Task 4.2: first-run notice (design D12)

class LearningNoticeTests: XCTestCase {
    private var savedEnabled: Any?
    private var savedShown: Any?
    var controller: LearningNoticeController { MainActor.assumeIsolated { LearningNoticeController.shared } }

    override func setUpWithError() throws {
        savedEnabled = Preferences.defaults.object(forKey: "AFMLearningEnabled")
        savedShown = Preferences.defaults.object(forKey: "AFMLearningNoticeShown")
        Preferences.defaults.removeObject(forKey: "AFMLearningNoticeShown")
        MainActor.assumeIsolated { LearningNoticeController.shared.resetForTesting() }
    }

    override func tearDownWithError() throws {
        MainActor.assumeIsolated { LearningNoticeController.shared.resetForTesting() }
        for (key, value) in [("AFMLearningEnabled", savedEnabled), ("AFMLearningNoticeShown", savedShown)] {
            if let value {
                Preferences.defaults.set(value, forKey: key)
            } else {
                Preferences.defaults.removeObject(forKey: key)
            }
        }
        Preferences.learningEnabled = false
    }

    func spin(_ seconds: TimeInterval) {
        RunLoop.current.run(until: Date().addingTimeInterval(seconds))
    }

    func testDecisionFunction() {
        XCTAssertTrue(shouldShowLearningNotice(learningEnabled: true, noticeShown: false))
        XCTAssertFalse(shouldShowLearningNotice(learningEnabled: true, noticeShown: true))
        XCTAssertFalse(shouldShowLearningNotice(learningEnabled: false, noticeShown: false))
        XCTAssertFalse(shouldShowLearningNotice(learningEnabled: false, noticeShown: true))
    }

    func testFlagDefaultsAndNoticeTextMentionsRequiredPoints() {
        XCTAssertFalse(Preferences.learningNoticeShown)
        let body = LearningNoticeText.body
        for phrase in ["打字", "手動選字", "這台 Mac", "不會透過網路傳送", "「選字模型」", "關閉", "清除"] {
            XCTAssertTrue(body.contains(phrase), phrase)
        }
        XCTAssertEqual(LearningNoticeText.okButton, "好")
        XCTAssertEqual(LearningNoticeText.openSettingsButton, "打開設定")
    }

    @MainActor
    func testSuppressedUnderXCTest() throws {
        Preferences.learningEnabled = true
        XCTAssertTrue(controller.suppressed)
        controller.delay = 0.05
        controller.scheduleIfNeeded()
        XCTAssertFalse(controller.isScheduled)
        spin(0.2)
        XCTAssertNil(controller.panel)
        XCTAssertFalse(Preferences.learningNoticeShown)
    }

    @MainActor
    func testShownAfterDelayOnceAndFlagSet() throws {
        Preferences.learningEnabled = true
        controller.suppressed = false
        controller.delay = 0.3
        controller.scheduleIfNeeded()
        XCTAssertTrue(controller.isScheduled)
        controller.scheduleIfNeeded()  // no second schedule
        spin(0.1)
        XCTAssertNil(controller.panel, "not before the delay")
        XCTAssertFalse(Preferences.learningNoticeShown)
        let deadline = Date().addingTimeInterval(3)
        while controller.panel == nil && Date() < deadline {
            spin(0.02)
        }
        let panel = try XCTUnwrap(controller.panel)
        XCTAssertTrue(panel.isVisible)
        XCTAssertTrue(Preferences.learningNoticeShown)

        // Panel properties (design D12).
        XCTAssertTrue(panel.isFloatingPanel)
        XCTAssertTrue(panel.styleMask.contains(.nonactivatingPanel))
        XCTAssertFalse(panel.canBecomeKey)
        XCTAssertFalse(panel.canBecomeMain)
        XCTAssertTrue(panel.becomesKeyOnlyIfNeeded)
        XCTAssertEqual(panel.level, .floating)
        XCTAssertTrue(panel.collectionBehavior.contains(.canJoinAllSpaces))
        XCTAssertTrue(panel.collectionBehavior.contains(.fullScreenAuxiliary))
        XCTAssertFalse(panel.hidesOnDeactivate)
        XCTAssertFalse(panel.isKeyWindow)
        if let dir = ProcessInfo.processInfo.environment["AFM_PREFS_SCREENSHOT_DIR"],
            let view = panel.contentView,
            let rep = view.bitmapImageRepForCachingDisplay(in: view.bounds)
        {
            panel.appearance = NSAppearance(named: .aqua)
            view.layoutSubtreeIfNeeded()
            view.cacheDisplay(in: view.bounds, to: rep)
            try rep.representation(using: .png, properties: [:])?
                .write(to: URL(fileURLWithPath: dir).appendingPathComponent("learning-notice.png"))
        }

        // 「好」 closes; the notice is not shown again.
        try XCTUnwrap(button(LearningNoticeText.okButton, in: panel)).performClick(nil)
        XCTAssertNil(controller.panel)
        XCTAssertFalse(panel.isVisible)
        controller.scheduleIfNeeded()
        XCTAssertFalse(controller.isScheduled)
        controller.showIfNeeded()
        XCTAssertNil(controller.panel)
    }

    @MainActor
    func testFlagNotSetWhenNotOrderedFront() {
        Preferences.learningEnabled = true
        controller.orderFront = { _ in }
        controller.showIfNeeded()
        XCTAssertNil(controller.panel)
        XCTAssertFalse(Preferences.learningNoticeShown)
    }

    @MainActor
    func testNotShownWhenLearningOff() {
        Preferences.learningEnabled = false
        controller.suppressed = false
        controller.scheduleIfNeeded()
        XCTAssertFalse(controller.isScheduled)
        controller.showIfNeeded()
        XCTAssertNil(controller.panel)
        XCTAssertFalse(Preferences.learningNoticeShown)
    }

    @MainActor
    func testOpenSettingsButtonOpensPreferencesAndCloses() throws {
        Preferences.learningEnabled = true
        var opened = 0
        controller.openPreferences = { opened += 1 }
        controller.showIfNeeded()
        let panel = try XCTUnwrap(controller.panel)
        try XCTUnwrap(button(LearningNoticeText.openSettingsButton, in: panel)).performClick(nil)
        XCTAssertEqual(opened, 1)
        XCTAssertNil(controller.panel)
        XCTAssertTrue(Preferences.learningNoticeShown)
        XCTAssertTrue(Preferences.learningEnabled, "the notice changes no setting")
    }

    @MainActor
    func testPreferencesWindowSelectsTab() {
        let model = PreferencesViewModel()
        let windowController = PreferencesWindowController(preferences: model)
        XCTAssertEqual(model.selectedTab, .basic)
        windowController.select(tab: .selectionModel)
        XCTAssertEqual(model.selectedTab, .selectionModel)
    }

    func button(_ title: String, in panel: NSPanel) -> NSButton? {
        func find(_ view: NSView) -> NSButton? {
            if let b = view as? NSButton, b.title == title { return b }
            for sub in view.subviews {
                if let b = find(sub) { return b }
            }
            return nil
        }
        return panel.contentView.flatMap(find)
    }
}
