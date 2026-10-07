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

import XCTest

@testable import McBopomofo

/// Standard-layout keys for each Bopomofo symbol and tone (first tone = space).
private let kStandardLayoutKeys: [Character: Character] = [
    "ㄅ": "1", "ㄆ": "q", "ㄇ": "a", "ㄈ": "z", "ㄉ": "2", "ㄊ": "w", "ㄋ": "s", "ㄌ": "x",
    "ㄍ": "e", "ㄎ": "d", "ㄏ": "c", "ㄐ": "r", "ㄑ": "f", "ㄒ": "v", "ㄓ": "5", "ㄔ": "t",
    "ㄕ": "g", "ㄖ": "b", "ㄗ": "y", "ㄘ": "h", "ㄙ": "n", "ㄧ": "u", "ㄨ": "j", "ㄩ": "m",
    "ㄚ": "8", "ㄛ": "i", "ㄜ": "k", "ㄝ": ",", "ㄞ": "9", "ㄟ": "o", "ㄠ": "l", "ㄡ": ".",
    "ㄢ": "0", "ㄣ": "p", "ㄤ": ";", "ㄥ": "/", "ㄦ": "-",
    "ˊ": "6", "ˇ": "3", "ˋ": "4", "˙": "7",
]

/// Keys typing one syllable on the standard layout; unmarked first tone ends with space.
func standardLayoutKeys(forReading reading: String) -> String {
    var keys = String(reading.compactMap { kStandardLayoutKeys[$0] })
    if let last = reading.last, !"ˊˇˋ˙".contains(last) {
        keys.append(" ")
    }
    return keys
}

/// Fake R model: every character the grid can place gets log p = -10, except
/// favored ones which get 0. Counts its calls.
final class FakeNeuralScorer {
    var favored: Set<String>
    var delay: TimeInterval = 0
    var fail = false
    private(set) var calls = 0

    init(favored: Set<String> = []) {
        self.favored = favored
    }

    var block: AFMNeuralScoreBlock {
        return { [unowned self] readings, charsAtPositions in
            self.calls += 1
            if self.delay > 0 {
                Thread.sleep(forTimeInterval: self.delay)
            }
            if self.fail {
                return nil
            }
            return readings.indices.map { i in
                var position: [String: NSNumber] = [:]
                let chars = i < charsAtPositions.count ? charsAtPositions[i] : []
                for c in chars {
                    position[c] = NSNumber(value: self.favored.contains(c) ? 0.0 : -10.0)
                }
                return position
            }
        }
    }
}

class NeuralSelectorTests: XCTestCase {

    var handler = KeyHandler()
    var savedAssociatedPhrasesEnabled = false
    var savedNeuralWalkEnabled = true
    var savedNeuralWalkAlpha = 0.5
    var savedKeyboardLayout: KeyboardLayout = .standard
    var savedChineseConversionEnabled = false

    override func setUpWithError() throws {
        savedAssociatedPhrasesEnabled = Preferences.associatedPhrasesEnabled
        savedNeuralWalkEnabled = Preferences.neuralWalkEnabled
        savedNeuralWalkAlpha = Preferences.neuralWalkAlpha
        savedKeyboardLayout = Preferences.keyboardLayout
        savedChineseConversionEnabled = Preferences.chineseConversionEnabled
        Preferences.associatedPhrasesEnabled = false
        Preferences.neuralWalkEnabled = true
        Preferences.neuralWalkAlpha = 0.5
        Preferences.keyboardLayout = .standard
        Preferences.chineseConversionEnabled = false
        LanguageModelManager.loadDataModels()
        LanguageModelManager.resetUserOverrideModelForTesting()
        AFMNeuralSelector.resetForTesting()
        handler = KeyHandler()
        handler.inputMode = .bopomofo
    }

    override func tearDownWithError() throws {
        AFMNeuralSelector.resetForTesting()
        LanguageModelManager.resetUserOverrideModelForTesting()
        Preferences.associatedPhrasesEnabled = savedAssociatedPhrasesEnabled
        Preferences.neuralWalkEnabled = savedNeuralWalkEnabled
        Preferences.neuralWalkAlpha = savedNeuralWalkAlpha
        Preferences.keyboardLayout = savedKeyboardLayout
        Preferences.chineseConversionEnabled = savedChineseConversionEnabled
    }

    // MARK: - Helpers

    /// Types keys one by one; returns every state emitted (last = current).
    @discardableResult
    func type(_ keys: String, into handler: KeyHandler, from initial: InputState = InputState.Empty()) -> [InputState] {
        var state = initial
        var all: [InputState] = []
        for key in keys.map({ String($0) }) {
            let input = KeyHandlerInput(
                inputText: key, keyCode: 0, charCode: charCode(key), flags: [],
                isVerticalMode: false)
            handler.handle(input: input, state: state) { newState in
                state = newState
                all.append(newState)
            } errorCallback: {
            }
        }
        return all
    }

    func composingBuffer(_ states: [InputState]) -> String? {
        (states.last as? InputState.Inputting)?.composingBuffer
    }

    /// The composing buffer with the stock walk (no model) for the same keys.
    func stockBuffer(_ keys: String, mode: InputMode = .bopomofo) -> String? {
        let current = AFMNeuralSelector.shared
        AFMNeuralSelector.resetForTesting()
        defer { AFMNeuralSelector.shared = current }
        let stockHandler = KeyHandler()
        stockHandler.inputMode = mode
        return composingBuffer(type(keys, into: stockHandler))
    }

    func waitUntilLoaded(_ selector: AFMNeuralSelector, timeout: TimeInterval = 10) {
        let deadline = Date().addingTimeInterval(timeout)
        while selector.state == .loading && Date() < deadline {
            RunLoop.current.run(until: Date().addingTimeInterval(0.01))
        }
    }

    /// A ready fake selector installed as the shared selector.
    func installFakeSelector(_ fake: FakeNeuralScorer) -> AFMNeuralSelector {
        let selector = AFMNeuralSelector(scoreBlockForTesting: fake.block, loadDelay: 0)
        selector.enabled = true
        selector.alpha = 0.5
        selector.startLoadingIfNeeded()
        waitUntilLoaded(selector)
        XCTAssertEqual(selector.state, .ready)
        AFMNeuralSelector.shared = selector
        return selector
    }

    /// The "AFM Input" directory that holds this repo, train/ and bench/.
    private var afmInputRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // McBopomofoTests
            .deletingLastPathComponent()  // AFMBopomofo
            .deletingLastPathComponent()  // AFM Input
    }

    /// R model directory for real-model tests: $AFM_RWALK_MODEL_DIR, the
    /// installed model directory, or the clean_v1 export next to the repo.
    func realModelDirectory() throws -> String {
        var candidates: [String] = []
        if let env = ProcessInfo.processInfo.environment["AFM_RWALK_MODEL_DIR"], !env.isEmpty {
            candidates.append(env)
        }
        candidates.append(UserPhraseLocationHelper.rWalkModelDirectory)
        candidates.append(afmInputRoot.appendingPathComponent("train/ranker/export/clean_v1").path)
        for dir in candidates
        where FileManager.default.fileExists(atPath: (dir as NSString).appendingPathComponent("onnx/model_quantized.onnx")) {
            return dir
        }
        throw XCTSkip("R model directory not found (tried \(candidates))")
    }

    // MARK: - Model loading (task 3.2)

    func testNonexistentModelDirectoryFails() {
        let selector = AFMNeuralSelector(modelDirectory: NSTemporaryDirectory() + "/no-such-r-walk-\(UUID().uuidString)")
        selector.enabled = true
        selector.startLoadingIfNeeded()
        XCTAssertEqual(selector.state, .loading)
        waitUntilLoaded(selector)
        XCTAssertEqual(selector.state, .failed)

        AFMNeuralSelector.shared = selector
        XCTAssertEqual(composingBuffer(type("su3cl3", into: handler)), "你好")
    }

    func testTruncatedModelFails() throws {
        let fm = FileManager.default
        let dir = (NSTemporaryDirectory() as NSString).appendingPathComponent("truncated-r-walk-\(UUID().uuidString)")
        try fm.createDirectory(atPath: dir + "/onnx", withIntermediateDirectories: true)
        try fm.createDirectory(atPath: dir + "/tokenizer", withIntermediateDirectories: true)
        defer { try? fm.removeItem(atPath: dir) }

        let source = try? realModelDirectory()
        for file in ["syl_vocab.json", "phonetic_table.tsv", "tokenizer/tokenizer.json"] {
            if let source = source {
                try fm.copyItem(atPath: source + "/" + file, toPath: dir + "/" + file)
            } else {
                fm.createFile(atPath: dir + "/" + file, contents: Data())
            }
        }
        var onnx = Data(repeating: 0x42, count: 4096)
        if let source = source, let full = fm.contents(atPath: source + "/onnx/model_quantized.onnx") {
            onnx = full.prefix(full.count / 3)
        }
        fm.createFile(atPath: dir + "/onnx/model_quantized.onnx", contents: onnx)

        let selector = AFMNeuralSelector(modelDirectory: dir)
        selector.enabled = true
        selector.startLoadingIfNeeded()
        waitUntilLoaded(selector)
        XCTAssertEqual(selector.state, .failed)

        AFMNeuralSelector.shared = selector
        XCTAssertEqual(composingBuffer(type("su3cl3", into: handler)), "你好")
    }

    func testRealModelBecomesReady() throws {
        let dir = try realModelDirectory()
        let selector = AFMNeuralSelector(modelDirectory: dir)
        selector.enabled = true
        selector.startLoadingIfNeeded()
        waitUntilLoaded(selector, timeout: 20)
        XCTAssertEqual(selector.state, .ready)
        print("R-walk model load time: \(String(format: "%.0f", selector.lastLoadMilliseconds)) ms (\(dir))")
    }

    // MARK: - Loading does not block activation (task 3.3)

    func testActivationReturnsBeforeSlowLoadCompletes() {
        let fake = FakeNeuralScorer(favored: ["妳"])
        let selector = AFMNeuralSelector(scoreBlockForTesting: fake.block, loadDelay: 1.0)
        AFMNeuralSelector.shared = selector

        let start = Date()
        handler.syncWithPreferences()  // the activation path
        let elapsed = Date().timeIntervalSince(start)
        XCTAssertLessThan(elapsed, 0.1)
        XCTAssertEqual(selector.state, .loading)

        // Typing during the load uses the stock walk.
        XCTAssertEqual(composingBuffer(type("su3", into: handler)), "你")
        XCTAssertEqual(selector.neuralWalkCount, 0)

        waitUntilLoaded(selector)
        XCTAssertEqual(selector.state, .ready)
        handler.clear()
        XCTAssertEqual(composingBuffer(type("su3", into: handler)), "妳")
        XCTAssertGreaterThan(selector.neuralWalkCount, 0)
    }

    func testSwitchTurnedOnAfterStartingOff() {
        let fake = FakeNeuralScorer(favored: ["妳"])
        let selector = AFMNeuralSelector(scoreBlockForTesting: fake.block, loadDelay: 0)
        AFMNeuralSelector.shared = selector

        Preferences.neuralWalkEnabled = false
        handler.syncWithPreferences()
        XCTAssertEqual(selector.state, .disabled)
        XCTAssertFalse(selector.enabled)

        Preferences.neuralWalkEnabled = true
        handler.syncWithPreferences()
        XCTAssertEqual(selector.state, .loading)
        waitUntilLoaded(selector)
        XCTAssertEqual(selector.state, .ready)
        XCTAssertEqual(composingBuffer(type("su3", into: handler)), "妳")
    }

    // MARK: - Cache and circuit breaker (task 5.2)

    func testOneForwardPerReadingAndNoneForCursorMoves() {
        let fake = FakeNeuralScorer()
        let selector = installFakeSelector(fake)

        let states = type("su3cl3", into: handler)
        XCTAssertEqual(fake.calls, 2)
        XCTAssertEqual(selector.forwardCount, 2)

        // Same readings re-walked (e.g. after a candidate pick) reuse the scores.
        let before = selector.neuralWalkCount
        handler.fixNode(reading: "ㄏㄠˇ", value: "好", originalCursorIndex: 2, useMoveCursorAfterSelectionSetting: false)
        XCTAssertEqual(fake.calls, 2)
        XCTAssertGreaterThan(selector.neuralWalkCount, before)

        var state = handler.buildInputtingState()
        let left = KeyHandlerInput(
            inputText: " ", keyCode: KeyCode.left.rawValue, charCode: 0, flags: [],
            isVerticalMode: false)
        handler.handle(input: left, state: state) { state = $0 } errorCallback: {}
        handler.handle(input: left, state: state) { state = $0 } errorCallback: {}
        XCTAssertTrue(state is InputState.Inputting, "\(state)")
        XCTAssertEqual(fake.calls, 2)
        XCTAssertEqual(states.isEmpty, false)
    }

    func testSlowScorerTripsToStockWalk() {
        let fake = FakeNeuralScorer(favored: ["妳"])
        fake.delay = 0.012
        let selector = installFakeSelector(fake)

        type("su3su3su3", into: handler)
        XCTAssertEqual(selector.state, .tripped)
        XCTAssertEqual(fake.calls, 3)

        handler.clear()
        let buffer = composingBuffer(type("su3", into: handler))
        XCTAssertEqual(fake.calls, 3)
        XCTAssertEqual(buffer, stockBuffer("su3"))
        XCTAssertEqual(buffer, "你")
    }

    func testEmptyScoresFallBackAndRetry() {
        let fake = FakeNeuralScorer(favored: ["妳"])
        fake.fail = true
        let selector = installFakeSelector(fake)

        XCTAssertEqual(composingBuffer(type("su3", into: handler)), "你")
        XCTAssertEqual(fake.calls, 1)
        XCTAssertEqual(selector.neuralWalkCount, 0)

        // Not remembered as valid scores: the next keystroke tries again.
        fake.fail = false
        handler.clear()
        XCTAssertEqual(composingBuffer(type("su3", into: handler)), "妳")
        XCTAssertEqual(fake.calls, 2)
        XCTAssertEqual(selector.state, .ready)
    }

    // MARK: - Behavior with a loaded model (task 5.3)

    func testManualPickSurvivesFurtherTyping() {
        let fake = FakeNeuralScorer(favored: ["妳"])
        _ = installFakeSelector(fake)

        XCTAssertEqual(composingBuffer(type("su3", into: handler)), "妳")
        handler.fixNode(reading: "ㄋㄧˇ", value: "你", originalCursorIndex: 1, useMoveCursorAfterSelectionSetting: false)
        let buffer = composingBuffer(type("cl3", into: handler, from: handler.buildInputtingState()))
        XCTAssertEqual(buffer?.first, "你")
    }

    func testUserOverrideSuggestionAppliedOverModel() {
        let fake = FakeNeuralScorer(favored: ["妳"])
        let selector = installFakeSelector(fake)

        XCTAssertEqual(composingBuffer(type("su3", into: handler)), "妳")
        handler.fixNode(reading: "ㄋㄧˇ", value: "你", originalCursorIndex: 1, useMoveCursorAfterSelectionSetting: false)
        handler.clear()

        let calls = fake.calls
        let neural = selector.neuralWalkCount
        XCTAssertEqual(composingBuffer(type("su3", into: handler)), "你")
        // Both walks of this keystroke (R-walk, then the re-walk after the UOM
        // override) used neural scores; the readings are unchanged since the
        // pick, so the cached table is reused and no forward pass runs.
        XCTAssertEqual(fake.calls, calls)
        XCTAssertEqual(selector.neuralWalkCount, neural + 2)
    }

    func testCommitThenContinueTyping() {
        let fake = FakeNeuralScorer(favored: ["妳"])
        _ = installFakeSelector(fake)

        var states = type("su3cl3", into: handler)
        let enter = KeyHandlerInput(
            inputText: " ", keyCode: KeyCode.enter.rawValue, charCode: 13, flags: [],
            isVerticalMode: false)
        var state = states.last!
        var committed: [String] = []
        handler.handle(input: enter, state: state) { newState in
            state = newState
            if let c = newState as? InputState.Committing {
                committed.append(c.poppedText)
            }
        } errorCallback: {
        }
        XCTAssertEqual(committed, ["妳好"])

        states = type("su3", into: handler, from: state)
        for s in states {
            XCTAssertFalse(s is InputState.Committing, "\(s)")
        }
        XCTAssertEqual(composingBuffer(states), "妳")
        XCTAssertEqual(committed, ["妳好"])
    }

    func testPlainBopomofoWithModelEqualsStock() {
        let fake = FakeNeuralScorer(favored: ["妳"])
        _ = installFakeSelector(fake)

        func candidates(_ handler: KeyHandler) -> [String] {
            let states = type("su3", into: handler)
            return (states.last as? InputState.ChoosingCandidate)?.candidates.map { $0.value } ?? []
        }
        let plain = KeyHandler()
        plain.inputMode = .plainBopomofo
        let withModel = candidates(plain)
        XCTAssertEqual(fake.calls, 0)

        AFMNeuralSelector.resetForTesting()
        let stock = KeyHandler()
        stock.inputMode = .plainBopomofo
        XCTAssertFalse(withModel.isEmpty)
        XCTAssertEqual(withModel, candidates(stock))
    }

    func testDisabledAndAlphaZeroEqualStock() {
        let fake = FakeNeuralScorer(favored: ["妳"])
        let selector = installFakeSelector(fake)

        Preferences.neuralWalkEnabled = false
        handler.syncWithPreferences()
        XCTAssertEqual(composingBuffer(type("su3cl3", into: handler)), stockBuffer("su3cl3"))
        XCTAssertEqual(fake.calls, 0)

        Preferences.neuralWalkEnabled = true
        Preferences.neuralWalkAlpha = 0
        handler.syncWithPreferences()
        XCTAssertEqual(selector.alpha, 0)
        handler.clear()
        XCTAssertEqual(composingBuffer(type("su3cl3", into: handler)), stockBuffer("su3cl3"))
        XCTAssertEqual(fake.calls, 0)
    }

    func testLongerThanNeuralLimitUsesStock() {
        let fake = FakeNeuralScorer(favored: ["妳"])
        let selector = installFakeSelector(fake)

        let limit = Int(kAFMNeuralMaxSyllables)
        let keys = String(repeating: "su3", count: limit)
        XCTAssertEqual(composingBuffer(type(keys, into: handler)), String(repeating: "妳", count: limit))
        XCTAssertEqual(fake.calls, limit)
        XCTAssertNotEqual(selector.state, .tripped)

        let buffer = composingBuffer(type("su3", into: handler, from: handler.buildInputtingState()))
        XCTAssertEqual(fake.calls, limit)
        XCTAssertEqual(buffer, stockBuffer(keys + "su3"))
    }

    // MARK: - Parity with the C++ R-walk (task 5.4)

    func testParityWithRescorerBenchFixture() throws {
        let dir = try realModelDirectory()
        guard let url = Bundle(for: NeuralSelectorTests.self).url(forResource: "rwalk_app_parity", withExtension: "tsv")
        else {
            XCTFail("rwalk_app_parity.tsv missing from the test bundle")
            return
        }
        let selector = AFMNeuralSelector(modelDirectory: dir)
        selector.enabled = true
        selector.alpha = 0.5
        selector.startLoadingIfNeeded()
        waitUntilLoaded(selector, timeout: 20)
        XCTAssertEqual(selector.state, .ready)
        AFMNeuralSelector.shared = selector

        LanguageModelManager.clearUserPhrasesForTesting()
        defer { LanguageModelManager.loadUserPhrases(enableForPlainBopomofo: Preferences.enableUserPhrasesInPlainBopomofo) }

        let rows = try String(contentsOf: url, encoding: .utf8)
            .split(separator: "\n")
            .filter { !$0.hasPrefix("#") }
        XCTAssertGreaterThanOrEqual(rows.count, 5)
        for row in rows {
            let cols = row.split(separator: "\t").map(String.init)
            XCTAssertEqual(cols.count, 4, "\(row)")
            guard cols.count == 4 else { continue }
            let keys = cols[1].split(separator: " ").map { standardLayoutKeys(forReading: String($0)) }.joined()
            LanguageModelManager.resetUserOverrideModelForTesting()
            let typing = KeyHandler()
            typing.inputMode = .bopomofo
            XCTAssertEqual(composingBuffer(type(keys, into: typing)), cols[2], "\(cols[0]) (stock: \(cols[3]))")
        }
        XCTAssertNotEqual(selector.state, .tripped)
    }

    // MARK: - In-process latency (task 6.2, test-host interim evidence)

    /// Process physical footprint in MB (task_vm_info.phys_footprint).
    private func physFootprintMB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? Double(info.phys_footprint) / 1_048_576.0 : -1
    }

    /// Drives the real KeyHandler (Bopomofo mode) with the real R model and
    /// measures the wall time of every syllable-completing keystroke while the
    /// composing buffer grows from 1 to 40 syllables.
    func testInProcessLatencyUpTo40Syllables() throws {
        let dir = try realModelDirectory()
        let jsonl = afmInputRoot.appendingPathComponent("bench/data/general/test.jsonl")
        guard let text = try? String(contentsOf: jsonl, encoding: .utf8) else {
            throw XCTSkip("bench test.jsonl not found at \(jsonl.path)")
        }
        var syllables: [String] = []
        var sentences = 0
        for line in text.split(separator: "\n") {
            guard let obj = try? JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any],
                let reading = obj["reading"] as? [String]
            else { continue }
            syllables.append(contentsOf: reading)
            sentences += 1
        }
        XCTAssertGreaterThan(syllables.count, 2000)

        let rssBefore = physFootprintMB()
        let selector = AFMNeuralSelector(modelDirectory: dir)
        selector.enabled = true
        selector.alpha = 0.5
        selector.startLoadingIfNeeded()
        waitUntilLoaded(selector, timeout: 30)
        XCTAssertEqual(selector.state, .ready)
        AFMNeuralSelector.shared = selector
        let rssLoaded = physFootprintMB()

        LanguageModelManager.clearUserPhrasesForTesting()
        defer { LanguageModelManager.loadUserPhrases(enableForPlainBopomofo: Preferences.enableUserPhrasesInPlainBopomofo) }

        let maxLen = Int(kAFMNeuralMaxSyllables)
        let targetSamples = 2400
        var samples: [(len: Int, ms: Double)] = []
        var state: InputState = InputState.Empty()
        var len = 0
        var errors = 0
        var skippedRuns = 0
        var cursor = 0
        let enter = KeyHandlerInput(
            inputText: " ", keyCode: KeyCode.enter.rawValue, charCode: 13, flags: [], isVerticalMode: false)

        func resetRun() {
            if len > 0 {
                handler.handle(input: enter, state: state) { state = $0 } errorCallback: {}
            }
            handler = KeyHandler()
            handler.inputMode = .bopomofo
            state = InputState.Empty()
            len = 0
        }

        // Warm-up: one short run, not recorded.
        for syl in syllables.prefix(5) {
            if let last = type(standardLayoutKeys(forReading: syl), into: handler, from: state).last {
                state = last
            }
        }
        resetRun()

        var runSamples: [(len: Int, ms: Double)] = []
        while samples.count < targetSamples && cursor < syllables.count {
            let syl = syllables[cursor]
            cursor += 1
            let keys = standardLayoutKeys(forReading: syl).map { String($0) }
            var failed = false
            for (i, key) in keys.enumerated() {
                let input = KeyHandlerInput(inputText: key, keyCode: 0, charCode: charCode(key), flags: [], isVerticalMode: false)
                let completing = i == keys.count - 1
                let t0 = DispatchTime.now().uptimeNanoseconds
                handler.handle(input: input, state: state) { state = $0 } errorCallback: { failed = true }
                let ms = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
                if completing && !failed {
                    runSamples.append((len + 1, ms))
                }
            }
            if failed {
                errors += 1
                skippedRuns += 1
                runSamples.removeAll()
                resetRun()
                continue
            }
            len += 1
            if len >= maxLen {
                samples.append(contentsOf: runSamples)
                runSamples.removeAll()
                resetRun()
            }
        }
        XCTAssertGreaterThanOrEqual(samples.count, 1500, "not enough samples")
        XCTAssertEqual(selector.state, .ready, "circuit breaker must not trip")
        let rssAfter = physFootprintMB()

        func pct(_ v: [Double], _ p: Double) -> Double {
            guard !v.isEmpty else { return 0 }
            let s = v.sorted()
            let idx = max(0, Int(ceil(p / 100.0 * Double(s.count))) - 1)
            return s[idx]
        }
        func stats(_ v: [Double]) -> [String: Any] {
            [
                "n": v.count, "p50": pct(v, 50), "p95": pct(v, 95), "p99": pct(v, 99), "max": v.max() ?? 0,
            ]
        }
        let all = samples.map { $0.ms }
        var buckets: [String: Any] = [:]
        for (name, range) in [("1-10", 1...10), ("11-20", 11...20), ("21-30", 21...30), ("31-40", 31...40)] {
            buckets[name] = stats(samples.filter { range.contains($0.len) }.map { $0.ms })
        }
        #if DEBUG
        let config = "Debug"
        #else
        let config = "Release"
        #endif
        let report: [String: Any] = [
            "config": config,
            "samples": samples.count,
            "sentences_pool": sentences,
            "errors": errors,
            "overall": stats(all),
            "buckets": buckets,
            "forward_count": selector.forwardCount,
            "neural_walk_count": selector.neuralWalkCount,
            "breaker_state": selector.state == .ready ? "ready" : "NOT_READY(\(selector.state.rawValue))",
            "load_ms": selector.lastLoadMilliseconds,
            "rss_mb_before_load": rssBefore,
            "rss_mb_after_load": rssLoaded,
            "rss_mb_after_test": rssAfter,
            "rss_delta_mb": rssAfter - rssBefore,
        ]
        let data = try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys])
        print("LATENCY_JSON " + String(decoding: data, as: UTF8.self))
        XCTAssertLessThan(pct(all, 95), 10.0, "p95 full-keystroke latency must be < 10 ms")
    }
}
