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

import CandidateUI
import InputMethodKit
import XCTest

@testable import McBopomofo

// MARK: - Task 3.1: sentence pool capture (design D4 / D5)

class SentencePoolCaptureTests: LearningTestCase {

    var client = FakeTextClient()

    /// An input method controller with a text client that records commits.
    func makeController() throws -> McBopomofoInputMethodController {
        guard let controller = McBopomofoInputMethodController(server: nil, delegate: nil, client: nil) else {
            throw XCTSkip("cannot create an input method controller without a server")
        }
        client = FakeTextClient()
        controller.currentClient = client
        controller.keyHandler.inputMode = .bopomofo
        controller.keyHandler.syncWithPreferences()
        return controller
    }

    func send(_ input: KeyHandlerInput, to imc: McBopomofoInputMethodController) {
        _ = imc.keyHandler.handle(input: input, state: imc.state) { newState in
            imc.handle(state: newState, client: imc.currentClient)
        } errorCallback: {}
        imc.requestPreciseIfNeeded()
    }

    func type(_ keys: String, into imc: McBopomofoInputMethodController) {
        for k in keys.map({ String($0) }) {
            send(key(k), to: imc)
        }
    }

    func pickInController(_ reading: String, _ value: String, _ imc: McBopomofoInputMethodController) {
        imc.keyHandler.fixNode(
            reading: reading, value: value, originalCursorIndex: imc.keyHandler.cursorIndex,
            useMoveCursorAfterSelectionSetting: false)
        imc.handle(state: imc.keyHandler.buildInputtingState(), client: imc.currentClient)
    }

    // Records

    func testEnterWithManualPick() throws {
        let imc = try makeController()
        type(kZhongHuaKeys, into: imc)
        XCTAssertEqual(client.marked, "中華")
        pickInController("ㄏㄨㄚˊ", "滑", imc)
        send(enterKey, to: imc)
        XCTAssertEqual(client.inserted, ["中滑"])
        let records = poolRecords()
        XCTAssertEqual(records.count, 1)
        let record = try XCTUnwrap(records.first)
        XCTAssertEqual(record["v"] as? Int, 1)
        XCTAssertEqual(record["manual"] as? Bool, true)
        XCTAssertEqual(record["t"] as? Double, now)
        XCTAssertEqual(record["walk"] as? String, "stock")
        XCTAssertEqual(record["model"] as? String, "")
        XCTAssertEqual((record["app"] as? String)?.isEmpty, false)
        XCTAssertEqual(nodes(record), [["ㄓㄨㄥ", "中", "w"], ["ㄏㄨㄚˊ", "滑", "p"]])
        let readings = nodes(record).flatMap { $0[0].split(separator: "-") }
        XCTAssertEqual(readings.count, 2)
    }

    func testCommitWithoutPick() throws {
        let imc = try makeController()
        type(kZhongHuaKeys, into: imc)
        send(enterKey, to: imc)
        let record = try XCTUnwrap(poolRecords().first)
        XCTAssertEqual(record["manual"] as? Bool, false)
        XCTAssertEqual(nodes(record).map { $0[2] }, Array(repeating: "w", count: nodes(record).count))
        XCTAssertEqual(nodes(record).map { $0[1] }.joined(), "中華")
    }

    /// Every commit path gives exactly one record.
    func testEachCommitPathRecordsOnce() throws {
        let paths: [(String, (McBopomofoInputMethodController) -> Void)] = [
            ("Enter", { self.send(self.enterKey, to: $0) }),
            ("Shift+Space", { self.send(self.key(" ", keyCode: 49, charCode: 32, flags: .shift), to: $0) }),
            ("deactivateServer", { $0.deactivateServer(self.client) }),
            ("commitComposition", { $0.commitComposition(self.client) }),
            ("Caps Lock", { self.send(self.key("a", charCode: charCode("a"), flags: .capsLock), to: $0) }),
            (
                "input mode switch",
                { imc in
                    imc.setValue(InputMode.plainBopomofo.rawValue, forTag: 0, client: self.client)
                }
            ),
            ("Shift+letter", { self.send(self.key("A", flags: .shift), to: $0) }),
        ]
        for (name, commit) in paths {
            try? FileManager.default.removeItem(atPath: poolFile)
            let imc = try makeController()
            type(kZhongHuaKeys, into: imc)
            commit(imc)
            XCTAssertEqual(client.inserted.first, "中華", name)
            let records = poolRecords()
            XCTAssertEqual(records.count, 1, name)
            XCTAssertEqual(records.first.map { nodes($0).map { $0[1] }.joined() }, "中華", name)
            // Nothing more is recorded afterwards for the same composition.
            imc.deactivateServer(client)
            XCTAssertEqual(poolRecords().count, 1, name)
        }
    }

    func testBopomofoFontAnnotationStillRecorded() throws {
        Preferences.bopomofoFontAnnotationSupportEnabled = true
        let imc = try makeController()
        type("up6c;6", into: imc)  // ㄧㄣˊ ㄏㄤˊ 銀行
        send(enterKey, to: imc)
        let committed = try XCTUnwrap(client.inserted.first)
        let record = try XCTUnwrap(poolRecords().first)
        XCTAssertEqual(nodes(record).map { $0[1] }.joined(), "銀行")
        // The annotated text carries variant selectors / PUA code points.
        XCTAssertNotEqual(committed, "銀行", "annotation expected for 行 (ㄏㄤˊ)")
    }

    func testImplicitCommitWithUnfinishedSyllable() throws {
        let imc = try makeController()
        type(kZhongKeys + "c", into: imc)  // 中 + ㄏ (unfinished)
        XCTAssertEqual(client.marked, "中ㄏ")
        imc.deactivateServer(client)
        XCTAssertEqual(client.inserted, ["中ㄏ"])
        let record = try XCTUnwrap(poolRecords().first)
        XCTAssertEqual(nodes(record), [["ㄓㄨㄥ", "中", "w"]])
    }

    func testClearWithEmptyBufferKeepsPendingSnapshot() throws {
        let handler = makeHandler()
        let state = type(kZhongHuaKeys, into: handler)
        let text = try XCTUnwrap(composing(state))
        handler.clear()
        handler.clear()  // Nothing composed: the snapshot stays.
        handler.recordCommittedText(text)
        XCTAssertEqual(poolRecords().count, 1)
        handler.recordCommittedText(text)  // Consumed.
        XCTAssertEqual(poolRecords().count, 1)
    }

    func testProvenanceSurvivesInsertionBeforePickedNode() throws {
        let handler = makeHandler()
        var state = type(kZhongHuaKeys, into: handler)
        state = pick("ㄏㄨㄚˊ", "滑", in: handler)
        let home = key(" ", keyCode: KeyCode.home.rawValue, charCode: 0)
        send(home, to: handler, state: &state)
        XCTAssertEqual(handler.cursorIndex, 0)
        state = type("su3", into: handler, from: state)  // ㄋㄧˇ before 中
        XCTAssertEqual(composing(state)?.hasSuffix("滑"), true)
        send(enterKey, to: handler, state: &state)
        let record = try XCTUnwrap(poolRecords().first)
        XCTAssertEqual(nodes(record).last, ["ㄏㄨㄚˊ", "滑", "p"])
        XCTAssertEqual(nodes(record).flatMap { $0[0].split(separator: "-") }.count, 3)
        XCTAssertEqual(record["manual"] as? Bool, true)
    }

    func testDiscardAndNonCompositionOutputsAreNotRecorded() throws {
        Preferences.escToCleanInputBuffer = true
        let handler = makeHandler()
        // Esc discard.
        var state = type(kZhongHuaKeys, into: handler)
        send(key("\u{1b}", charCode: 27), to: handler, state: &state)
        XCTAssertEqual(poolRecords().count, 0)
        // Ctrl+Enter reading output.
        Preferences.controlEnterOutput = .bpmfReading
        state = type(kZhongHuaKeys, into: handler, from: state)
        let ctrlEnter = key(" ", keyCode: KeyCode.enter.rawValue, charCode: 13, flags: .control)
        let output = send(ctrlEnter, to: handler, state: &state)
        XCTAssertEqual(output, ["ㄓㄨㄥ-ㄏㄨㄚˊ"])
        XCTAssertEqual(poolRecords().count, 0)
        // Punctuation only.
        state = type("!", into: handler, from: state)
        XCTAssertNotNil(composing(state))
        let punctuation = send(enterKey, to: handler, state: &state)
        XCTAssertEqual(punctuation.count, 1)
        XCTAssertEqual(poolRecords().count, 0)
        // The stale snapshot from Esc is not recorded by a later commit.
        state = type(kZhongKeys, into: handler, from: state)
        send(enterKey, to: handler, state: &state)
        let records = poolRecords()
        XCTAssertEqual(records.count, 1)
        XCTAssertEqual(records.first.map { nodes($0) }, [["ㄓㄨㄥ", "中", "w"]])
    }

    func testPlainBopomofoNotRecorded() throws {
        let imc = try makeController()
        imc.keyHandler.inputMode = .plainBopomofo
        type(kZhongKeys, into: imc)
        XCTAssertTrue(imc.state is InputState.ChoosingCandidate)
        imc.candidateController(CandidateController.vertical, didSelectCandidateAtIndex: 1)
        XCTAssertEqual(client.inserted.count, 1)
        XCTAssertEqual(poolRecords().count, 0)
    }

    func testSimplifiedOutputRecordsTraditionalCharacters() throws {
        Preferences.chineseConversionEnabled = true
        for style in [ChineseConversionStyle.output, .model] {
            Preferences.chineseConversionStyle = style
            try? FileManager.default.removeItem(atPath: poolFile)
            let imc = try makeController()
            imc.keyHandler.syncWithPreferences()
            type("5k4xu3", into: imc)  // ㄓㄜˋ ㄌㄧˇ
            send(enterKey, to: imc)
            XCTAssertEqual(client.inserted, ["这里"], "\(style)")
            let record = try XCTUnwrap(poolRecords().first, "\(style)")
            XCTAssertEqual(nodes(record).map { $0[1] }.joined(), "這裡", "\(style)")
        }
    }

    func testPhraseReplacementAndMacroRecordRawValues() throws {
        // Phrase replacement 中華 -> 中国.
        let replacementPath = LanguageModelManager.phraseReplacementDataPathMcBopomofo
        XCTAssertTrue(LanguageModelManager.checkIfUserLanguageModelFilesExist())
        let savedReplacement = FileManager.default.contents(atPath: replacementPath)
        defer {
            FileManager.default.createFile(atPath: replacementPath, contents: savedReplacement)
            LanguageModelManager.loadUserPhraseReplacement()
        }
        try "中華 中国\n".write(toFile: replacementPath, atomically: true, encoding: .utf8)
        LanguageModelManager.loadUserPhraseReplacement()
        Preferences.phraseReplacementEnabled = true
        LanguageModelManager.phraseReplacementEnabled = true
        let handler = makeHandler()
        var state = type(kZhongHuaKeys, into: handler)
        XCTAssertEqual(composing(state), "中国")
        send(enterKey, to: handler, state: &state)
        var record = try XCTUnwrap(poolRecords().last)
        XCTAssertEqual(nodes(record).map { $0[1] }, ["中華"])

        // A date macro (ㄇㄧㄥˊ ㄊㄧㄢ, 明天): the raw MACRO@ value.
        state = type("au/6wu0 ", into: handler, from: state)
        let candidates = handler.handleCandidates()
        let macro = try XCTUnwrap(candidates.first { $0.rawValue.hasPrefix("MACRO@") && $0.value != $0.rawValue })
        state = pick(macro.reading, macro.value, in: handler)
        XCTAssertEqual(composing(state), macro.value)
        send(enterKey, to: handler, state: &state)
        record = try XCTUnwrap(poolRecords().last)
        XCTAssertEqual(nodes(record), [[macro.reading, macro.rawValue, "p"]])
    }

    func testMemorySuggestionPreciseAndUserPhraseSources() throws {
        // m: a remembered pick applied in a later composition.
        typeAndCommit(kZhongKeys, picks: [("ㄓㄨㄥ", "鍾")])
        typeAndCommit(kZhongKeys)
        var records = poolRecords()
        XCTAssertEqual(records.count, 2)
        XCTAssertEqual(nodes(records[0]), [["ㄓㄨㄥ", "鍾", "p"]])
        XCTAssertEqual(nodes(records[1]), [["ㄓㄨㄥ", "鍾", "m"]])
        XCTAssertEqual(records[1]["manual"] as? Bool, false)

        // u: a user phrase.
        LanguageModelManager.loadUserPhrases(forTesting: "測試 ㄘㄜˋ-ㄕˋ\n")
        typeAndCommit("hk4g4")
        records = poolRecords()
        XCTAssertEqual(records.last.map { nodes($0) }, [["ㄘㄜˋ-ㄕˋ", "測試", "u"]])

        // r: R-walk records the walk and the model identifier.
        installFakeSelector(FakeNeuralScorer(favored: ["妳"]))
        XCTAssertEqual(typeAndCommit("su3" + kZhongKeys), "妳中")
        let rRecord = try XCTUnwrap(poolRecords().last)
        XCTAssertEqual(rRecord["walk"] as? String, "r")
        XCTAssertEqual(rRecord["model"] as? String, "test")
        XCTAssertEqual(nodes(rRecord), [["ㄋㄧˇ", "妳", "w"], ["ㄓㄨㄥ", "中", "w"]])
    }

    func testPreciseResultRecordedAsQ() throws {
        Preferences.preciseModeEnabled = true
        installFakeSelector(FakeNeuralScorer())
        let heavy = FakeHeavyScorer()
        retainedFakes.append(heavy)
        let dir = (NSTemporaryDirectory() as NSString).appendingPathComponent("qwen-\(UUID().uuidString)")
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(atPath: dir) }
        FileManager.default.createFile(
            atPath: (dir as NSString).appendingPathComponent("precise.conf"), contents: Data(kForcedOpenPreciseConf.utf8))
        let controller = AFMPreciseController(modelDirectory: dir, heavyScoreBlockForTesting: heavy.block, loadDelay: 0)
        controller.startDelay = 0
        AFMPreciseController.shared = controller
        controller.enabled = true
        waitUntil { controller.state != .loading }
        XCTAssertEqual(controller.state, .ready)

        let handler = makeHandler()
        let delegate = PreciseTestDelegate()
        handler.delegate = delegate
        for k in "su3cl3".map({ String($0) }) {
            handler.handle(input: key(k), state: delegate.state) { delegate.state = $0 } errorCallback: {}
            handler.requestPreciseIfNeeded(composingBuffer: (delegate.state as? InputState.Inputting)?.composingBuffer)
        }
        waitUntil { controller.isIdle && handler.preciseAppliedCount > 0 }
        XCTAssertTrue(handler.showingPreciseResult)
        let text = try XCTUnwrap(composing(delegate.state))
        handler.clear()
        handler.recordCommittedText(text)
        let record = try XCTUnwrap(poolRecords().first)
        XCTAssertEqual(record["walk"] as? String, "precise")
        XCTAssertEqual(nodes(record).map { $0[1] }.joined(), text)
        XCTAssertEqual(Set(nodes(record).map { $0[2] }), ["q"])
    }

    func testLearningTurnedOffStopsRecordingWithoutReactivation() throws {
        let imc = try makeController()
        type(kZhongHuaKeys, into: imc)
        send(enterKey, to: imc)
        XCTAssertEqual(poolRecords().count, 1)
        Preferences.setLearningEnabled(false)
        type(kZhongHuaKeys, into: imc)
        send(enterKey, to: imc)
        XCTAssertEqual(client.inserted.count, 2)
        XCTAssertEqual(poolRecords().count, 1)
    }
}

extension KeyHandler {
    /// The candidates the candidate window would show at the cursor.
    func handleCandidates() -> [InputState.Candidate] {
        var result: [InputState.Candidate] = []
        let down = KeyHandlerInput(
            inputText: " ", keyCode: KeyCode.down.rawValue, charCode: 0, flags: [], isVerticalMode: false)
        handle(input: down, state: buildInputtingState()) { state in
            if let choosing = state as? InputState.ChoosingCandidate {
                result = choosing.candidates
            }
        } errorCallback: {}
        return result
    }
}

// MARK: - Task 3.2: secure input (design D6)

class SecureInputLearningTests: LearningTestCase {
    func testNothingLearnedUnderSecureInput() throws {
        // A remembered pick made before secure input, for the refresh check.
        typeAndCommit(kZhongKeys, picks: [("ㄓㄨㄥ", "鍾")])
        LanguageModelManager.flushLearningData()
        XCTAssertEqual(poolRecords().count, 1)
        XCTAssertEqual(LanguageModelManager.userOverrideModelSizeForTesting, 1)

        secure = true
        // A pick does not change the memory: no suggestion afterwards.
        let huaKeys = "cj86"  // ㄏㄨㄚˊ
        XCTAssertEqual(composing(type(huaKeys, into: makeHandler())), "華")
        typeAndCommit(huaKeys, picks: [("ㄏㄨㄚˊ", "滑")])
        XCTAssertEqual(LanguageModelManager.userOverrideModelSizeForTesting, 1)
        XCTAssertFalse(store.hasUnsavedChanges)
        XCTAssertEqual(composing(type(huaKeys, into: makeHandler())), "華")
        // A commit adds no record and does not refresh: the remembered
        // suggestion committed unchanged 100 days later ...
        now += 100 * Self.day
        XCTAssertEqual(typeAndCommit(kZhongKeys), "鍾")
        XCTAssertEqual(poolRecords().count, 1)
        XCTAssertFalse(store.hasUnsavedChanges)
        // ... is gone 200 days after the pick.
        secure = false
        now += 100 * Self.day
        XCTAssertEqual(composing(type(kZhongKeys, into: makeHandler())), "中")
    }
}

// MARK: - Task 3.3: user-phrase guard (design D7)

class UserPhraseGuardTests: LearningTestCase {

    func run(neural: Bool, _ body: () throws -> Void) rethrows {
        LanguageModelManager.resetUserOverrideModelForTesting()
        LanguageModelManager.clearUserPhrasesForTesting()
        if neural {
            installFakeSelector(FakeNeuralScorer(favored: []))
        } else {
            AFMNeuralSelector.resetForTesting()
        }
        try body()
    }

    /// (a) Remembered 鍾, then a user phrase 中 equal to the stock top value.
    func testUserPhraseWinsOverRememberedPick() throws {
        for neural in [false, true] {
            try run(neural: neural) {
                let first = makeHandler()
                type(kZhongKeys, into: first)
                pick("ㄓㄨㄥ", "鍾", in: first)
                XCTAssertEqual(composing(type(kZhongKeys, into: makeHandler())), "鍾", "neural \(neural)")

                LanguageModelManager.loadUserPhrases(forTesting: "中 ㄓㄨㄥ\n")
                // Precondition: the key still matches, so suggest() is non-empty.
                XCTAssertEqual(try suggestionForWalk(kZhongKeys), "鍾", "precondition, neural \(neural)")
                XCTAssertEqual(composing(type(kZhongKeys, into: makeHandler())), "中", "neural \(neural)")
            }
        }
    }

    /// The suggestion the memory makes for the walk of keys (the composition
    /// itself shows the guarded result).
    func suggestionForWalk(_ keys: String) throws -> String? {
        let handler = makeHandler()
        type(keys, into: handler)
        return handler.userOverrideSuggestionForTesting()
    }

    /// (b) A 2-character forced suggestion next to a 1-character user phrase.
    /// ㄒㄧㄣ ㄅㄨˋ walks as 心 + 不 (single characters outscore 心部, also
    /// after a user phrase 心 demotes the mono-syllable 心 to the second score,
    /// as upstream rewrites mono-syllable user phrases). Picking 心部 is
    /// remembered as a forced (high-score) override keyed on 不.
    func testForcedTwoCharacterSuggestionNotAppliedOverUserPhrase() throws {
        for neural in [false, true] {
            try run(neural: neural) {
                let keys = "vup 1j4"  // ㄒㄧㄣ ㄅㄨˋ
                let handler = makeHandler()
                XCTAssertEqual(composing(type(keys, into: handler)), "心不", "fixture, neural \(neural)")
                pick("ㄒㄧㄣ-ㄅㄨˋ", "心部", in: handler)
                XCTAssertEqual(composing(handler.buildInputtingState()), "心部")
                XCTAssertEqual(composing(type(keys, into: makeHandler())), "心部", "remembered, neural \(neural)")

                LanguageModelManager.loadUserPhrases(forTesting: "心 ㄒㄧㄣ\n")
                // Precondition: the user phrase alone keeps 心 + 不 (no memory) ...
                let saved = LanguageModelManager.userOverrideModelSizeForTesting
                XCTAssertEqual(saved, 1)
                // ... and the memory still suggests 心部 for that walk.
                XCTAssertEqual(try suggestionForWalk(keys), "心部", "precondition, neural \(neural)")
                XCTAssertEqual(composing(type(keys, into: makeHandler())), "心不", "neural \(neural)")
            }
        }
    }

    /// Without user phrases the remembered pick is still applied (guard
    /// does not over-trigger), with learning off too.
    func testSuggestionAppliedWithoutUserPhrase() throws {
        for learning in [true, false] {
            LanguageModelManager.configureLearning(enabled: learning)
            LanguageModelManager.resetUserOverrideModelForTesting()
            let handler = makeHandler()
            type(kZhongKeys, into: handler)
            pick("ㄓㄨㄥ", "鍾", in: handler)
            XCTAssertEqual(composing(type(kZhongKeys, into: makeHandler())), "鍾", "learning \(learning)")
        }
    }
}

// MARK: - Task 3.4: refresh on commit (D8) and plain-mode picks (D9)

class RefreshAndPlainModeTests: LearningTestCase {

    /// Remembers 鍾 for ㄓㄨㄥ at the injected time.
    func rememberZhong() {
        typeAndCommit(kZhongKeys, picks: [("ㄓㄨㄥ", "鍾")])
        LanguageModelManager.flushLearningData()
    }

    func shownForZhong() -> String? {
        composing(type(kZhongKeys, into: makeHandler()))
    }

    func testCommittedSuggestionIsRefreshed() {
        rememberZhong()
        now += 100 * Self.day
        XCTAssertFalse(store.hasUnsavedChanges)
        XCTAssertEqual(typeAndCommit(kZhongKeys), "鍾")
        XCTAssertTrue(store.hasUnsavedChanges, "refresh marks the memory dirty")
        now += 100 * Self.day
        XCTAssertEqual(shownForZhong(), "鍾", "200 days after the pick, 100 after the commit")
    }

    func testChangedSuggestionIsNotRefreshed() {
        rememberZhong()
        now += 100 * Self.day
        let handler = makeHandler()
        var state = type(kZhongKeys, into: handler)
        XCTAssertEqual(composing(state), "鍾")
        // The suggested syllable is deleted and another one typed before the
        // commit (no pick, so the memory is not otherwise touched).
        send(key("\u{8}", charCode: 8), to: handler, state: &state)
        state = type("5j/4", into: handler, from: state)  // ㄓㄨㄥˋ
        let committed = send(enterKey, to: handler, state: &state)
        XCTAssertEqual(committed.count, 1)
        XCTAssertNotEqual(committed.first, "鍾")
        XCTAssertEqual(poolRecords().count, 2)
        XCTAssertFalse(store.hasUnsavedChanges, "nothing refreshed")
        now += 100 * Self.day
        XCTAssertEqual(shownForZhong(), "中")
    }

    func testUncommittedSuggestionExpires() {
        rememberZhong()
        now += 100 * Self.day
        XCTAssertEqual(shownForZhong(), "鍾", "shown, never committed")
        now += 100 * Self.day
        XCTAssertEqual(shownForZhong(), "中")
    }

    func testReadingOutputDoesNotRefresh() {
        Preferences.controlEnterOutput = .bpmfReading
        rememberZhong()
        now += 100 * Self.day
        let handler = makeHandler()
        var state = type(kZhongKeys, into: handler)
        XCTAssertEqual(composing(state), "鍾")
        let ctrlEnter = key(" ", keyCode: KeyCode.enter.rawValue, charCode: 13, flags: .control)
        XCTAssertEqual(send(ctrlEnter, to: handler, state: &state), ["ㄓㄨㄥ"])
        XCTAssertFalse(store.hasUnsavedChanges)
        now += 100 * Self.day
        XCTAssertEqual(shownForZhong(), "中")
    }

    func plainPick() {
        let handler = makeHandler(.plainBopomofo)
        let state = type(kZhongKeys, into: handler)
        XCTAssertTrue(state is InputState.ChoosingCandidate)
        handler.fixNode(reading: "ㄓㄨㄥ", value: "鍾", originalCursorIndex: 1, useMoveCursorAfterSelectionSetting: true)
        handler.clear()
    }

    func testPlainBopomofoPickNotObservedWithLearningOn() {
        plainPick()
        XCTAssertEqual(LanguageModelManager.userOverrideModelSizeForTesting, 0)
        XCTAssertFalse(store.hasUnsavedChanges)
    }

    func testPlainBopomofoPickObservedWithLearningOff() {
        LanguageModelManager.configureLearning(enabled: false)
        plainPick()
        XCTAssertEqual(LanguageModelManager.userOverrideModelSizeForTesting, 1)
    }
}

// MARK: - Task 3.5: context keys across walks (design Risks, review F10)

class ContextKeyAcrossWalksTests: LearningTestCase {
    /// A pick made with R-walk on, typed again with R-walk off. ㄋㄧˇ ㄓㄨㄥ:
    /// R-walk (fake model favoring 妳) shows 妳 + 中, the stock walk 你 + 中.
    /// The context key of the pick at ㄓㄨㄥ contains the preceding node's
    /// current value (妳), so it differs between the two walks.
    func testPickWithRWalkOnIsMissedWithRWalkOff() throws {
        let keys = "su3" + kZhongKeys
        installFakeSelector(FakeNeuralScorer(favored: ["妳"]))
        let handler = makeHandler()
        XCTAssertEqual(composing(type(keys, into: handler)), "妳中")
        pick("ㄓㄨㄥ", "鍾", in: handler)
        XCTAssertEqual(composing(handler.buildInputtingState()), "妳鍾")
        XCTAssertEqual(composing(type(keys, into: makeHandler())), "妳鍾", "R-walk on: remembered")

        AFMNeuralSelector.resetForTesting()
        let stock = makeHandler()
        let shown = composing(type(keys, into: stock))
        // Observed (matches design Risks): the pick is not suggested with the
        // stock walk.
        XCTAssertEqual(shown, "你中")
        XCTAssertNil(stock.userOverrideSuggestionForTesting())

        // And the other way around: a pick made with the stock walk is missed
        // once R-walk changes the preceding node.
        pick("ㄓㄨㄥ", "忠", in: stock)
        XCTAssertEqual(composing(type(keys, into: makeHandler())), "你忠")
        installFakeSelector(FakeNeuralScorer(favored: ["妳"]))
        XCTAssertEqual(composing(type(keys, into: makeHandler())), "妳鍾")
    }
}

// MARK: - personal-finetune task 1.6 (E6): retraction feasibility (design D7)

/// Records every handleInput call so the tests can see which keys reach KeyHandler.
final class SpyKeyHandler: KeyHandler {
    struct Call {
        var charCode: UInt16
        var keyCode: UInt16
        var stateClass: String
        var handled: Bool
    }
    var calls: [Call] = []

    override func handle(
        input: KeyHandlerInput, state: InputState, stateCallback: @escaping (InputState) -> Void,
        errorCallback: @escaping () -> Void
    ) -> Bool {
        let stateClass = String(describing: Swift.type(of: state))
        let handled = super.handle(
            input: input, state: state, stateCallback: stateCallback, errorCallback: errorCallback)
        calls.append(Call(charCode: input.charCode, keyCode: input.keyCode, stateClass: stateClass, handled: handled))
        return handled
    }
}

/// The retraction marker (D7) relies on: after a commit, the next Backspace / Delete with an empty buffer goes
/// through IMK `handle(_:client:)` → `KeyHandler handleInput` and is passed through to the application
/// (handled == NO). These tests drive real key-down NSEvents through the controller.
class RetractionFeasibilityTests: LearningTestCase {

    var client = FakeTextClient()

    func makeSpyController() throws -> (McBopomofoInputMethodController, SpyKeyHandler) {
        guard let imc = McBopomofoInputMethodController(server: nil, delegate: nil, client: nil) else {
            throw XCTSkip("cannot create an input method controller without a server")
        }
        client = FakeTextClient()
        imc.currentClient = client
        let spy = SpyKeyHandler()
        spy.delegate = imc
        spy.inputMode = .bopomofo
        spy.syncWithPreferences()
        imc.keyHandler = spy
        return (imc, spy)
    }

    func event(_ chars: String, keyCode: UInt16, flags: NSEvent.ModifierFlags = []) throws -> NSEvent {
        try XCTUnwrap(
            NSEvent.keyEvent(
                with: .keyDown, location: .zero, modifierFlags: flags, timestamp: 0, windowNumber: 0,
                context: nil, characters: chars, charactersIgnoringModifiers: chars, isARepeat: false,
                keyCode: keyCode))
    }

    /// Types 中華 through KeyHandler events and commits it with a Return key-down event.
    func commitZhongHua(_ imc: McBopomofoInputMethodController) throws {
        for k in kZhongHuaKeys.map({ String($0) }) {
            _ = imc.handle(try event(k, keyCode: 0), client: client)
        }
        XCTAssertEqual(client.marked, "中華")
        XCTAssertTrue(imc.handle(try event("\r", keyCode: 36), client: client))
        XCTAssertEqual(client.inserted, ["中華"])
        XCTAssertEqual(poolRecords().count, 1)
    }

    func testBackspaceAfterCommitReachesKeyHandlerAndPassesThrough() throws {
        let (imc, spy) = try makeSpyController()
        try commitZhongHua(imc)
        XCTAssertTrue(imc.state is InputState.Empty)
        spy.calls.removeAll()
        let passed = imc.handle(try event("\u{8}", keyCode: 51), client: client)
        XCTAssertFalse(passed, "Backspace with an empty buffer must reach the application")
        XCTAssertEqual(spy.calls.count, 1)
        XCTAssertEqual(spy.calls.first?.charCode, 8)
        XCTAssertEqual(spy.calls.first?.stateClass, "Empty")
        XCTAssertEqual(spy.calls.first?.handled, false)
        XCTAssertEqual(client.inserted, ["中華"])
        XCTAssertTrue(imc.state is InputState.Empty)
    }

    func testDeleteVariantsAfterCommitPassThrough() throws {
        let (imc, spy) = try makeSpyController()
        try commitZhongHua(imc)
        // Forward Delete (NSDeleteFunctionKey), Option+Backspace (delete word), Command+Backspace (delete line).
        let keys: [(String, UInt16, NSEvent.ModifierFlags)] = [
            ("\u{F728}", 117, [.function]), ("\u{8}", 51, [.option]), ("\u{8}", 51, [.command]),
        ]
        for (chars, code, flags) in keys {
            spy.calls.removeAll()
            XCTAssertFalse(imc.handle(try event(chars, keyCode: code, flags: flags), client: client), "\(code) \(flags)")
            XCTAssertEqual(spy.calls.count, 1)
            XCTAssertEqual(spy.calls.first?.handled, false)
        }
        XCTAssertEqual(client.inserted, ["中華"])
    }

    /// With associated phrases on, the Bopomofo-mode commit still ends in the empty state (post-commit
    /// associated-phrase panels only exist in plain Bopomofo, which is never recorded), so the Backspace that
    /// follows a recorded commit is passed through too.
    func testAssociatedPhrasesOnBackspaceAfterCommitStillPassesThrough() throws {
        Preferences.associatedPhrasesEnabled = true
        let (imc, spy) = try makeSpyController()
        try commitZhongHua(imc)
        XCTAssertTrue(imc.state is InputState.Empty)
        spy.calls.removeAll()
        XCTAssertFalse(imc.handle(try event("\u{8}", keyCode: 51), client: client))
        XCTAssertEqual(spy.calls.count, 1)
        XCTAssertEqual(spy.calls.first?.stateClass, "Empty")
    }
}
