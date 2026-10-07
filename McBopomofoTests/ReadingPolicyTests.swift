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

/// Reading policy (task 4.2): 癌 is read ㄞˊ unless AFMAcceptYanReadingForAi
/// is on. Uses the stock walk (no R model) so expectations are upstream data.
class ReadingPolicyTests: XCTestCase {

    var savedAcceptYan = false
    var savedAssociatedPhrasesEnabled = false
    var savedChineseConversionEnabled = false
    var savedKeyboardLayout: KeyboardLayout = .standard

    override func setUpWithError() throws {
        savedAcceptYan = Preferences.acceptYanReadingForAi
        savedAssociatedPhrasesEnabled = Preferences.associatedPhrasesEnabled
        savedChineseConversionEnabled = Preferences.chineseConversionEnabled
        savedKeyboardLayout = Preferences.keyboardLayout
        Preferences.associatedPhrasesEnabled = false
        Preferences.chineseConversionEnabled = false
        Preferences.keyboardLayout = .standard
        LanguageModelManager.loadDataModels()
        LanguageModelManager.resetUserOverrideModelForTesting()
        LanguageModelManager.clearUserPhrasesForTesting()
        AFMNeuralSelector.resetForTesting()
    }

    override func tearDownWithError() throws {
        Preferences.acceptYanReadingForAi = savedAcceptYan
        Preferences.associatedPhrasesEnabled = savedAssociatedPhrasesEnabled
        Preferences.chineseConversionEnabled = savedChineseConversionEnabled
        Preferences.keyboardLayout = savedKeyboardLayout
        LanguageModelManager.applyReadingPolicy(acceptYanReadingForAi: savedAcceptYan)
        LanguageModelManager.resetUserOverrideModelForTesting()
        LanguageModelManager.loadUserPhrases(enableForPlainBopomofo: Preferences.enableUserPhrasesInPlainBopomofo)
        AFMNeuralSelector.resetForTesting()
    }

    // MARK: - Helpers

    /// A handler in the given mode after the activation path applied the
    /// policy (AFMAcceptYanReadingForAi = accept).
    func makeHandler(_ mode: InputMode, accept: Bool) -> KeyHandler {
        Preferences.acceptYanReadingForAi = accept
        let handler = KeyHandler()
        handler.inputMode = mode
        handler.syncWithPreferences()
        return handler
    }

    @discardableResult
    func type(_ keys: String, into handler: KeyHandler, from initial: InputState = InputState.Empty()) -> InputState {
        var state = initial
        for key in keys.map({ String($0) }) {
            let input = KeyHandlerInput(
                inputText: key, keyCode: 0, charCode: charCode(key), flags: [],
                isVerticalMode: false)
            handler.handle(input: input, state: state) { state = $0 } errorCallback: {}
        }
        return state
    }

    func buffer(_ state: InputState) -> String? {
        (state as? InputState.Inputting)?.composingBuffer
    }

    /// Candidate values after typing keys: Bopomofo mode opens the candidate
    /// window with the down arrow; Plain Bopomofo shows it after the syllable.
    func candidates(_ keys: String, mode: InputMode, accept: Bool) -> [String] {
        let handler = makeHandler(mode, accept: accept)
        var state = type(keys, into: handler)
        if mode == .bopomofo {
            let down = KeyHandlerInput(
                inputText: " ", keyCode: KeyCode.down.rawValue, charCode: 0, flags: [],
                isVerticalMode: false)
            handler.handle(input: down, state: state) { state = $0 } errorCallback: {}
        }
        guard let choosing = state as? InputState.ChoosingCandidate else {
            XCTFail("no candidate window for \(keys) in \(mode): \(state)")
            return []
        }
        return choosing.candidates.map { $0.value }
    }

    /// True if the candidate places 癌 at a position whose syllable is ㄧㄢˊ
    /// (reading syllables joined by "-", one per character).
    func readsAiAsYan(value: String, reading: String) -> Bool {
        let chars = value.map { String($0) }
        let syllables = reading.split(separator: "-").map(String.init)
        guard chars.count == syllables.count else { return false }
        return zip(chars, syllables).contains { $0 == "癌" && $1 == "ㄧㄢˊ" }
    }

    // MARK: - Tests

    func testYanOffersNoAiInBothModes() {
        for mode in [InputMode.bopomofo, InputMode.plainBopomofo] {
            let off = candidates("u06", mode: mode, accept: false)
            let on = candidates("u06", mode: mode, accept: true)
            XCTAssertTrue(on.contains("癌"), "\(mode): \(on)")
            XCTAssertFalse(off.contains("癌"), "\(mode): \(off)")
            // Every other character for ㄧㄢˊ is offered as upstream, in order.
            XCTAssertEqual(off, on.filter { $0 != "癌" }, "\(mode)")
            XCTAssertGreaterThan(off.count, 5)
        }
    }

    func testKangYanNeverShowsKangAi() {
        let handler = makeHandler(.bopomofo, accept: false)
        let composed = buffer(type("d;4u06", into: handler))
        XCTAssertNotNil(composed)
        XCTAssertNotEqual(composed, "抗癌")
        XCTAssertFalse(composed?.contains("癌") ?? true, "\(String(describing: composed))")
        let off = candidates("d;4u06", mode: .bopomofo, accept: false)
        XCTAssertFalse(off.contains("抗癌"), "\(off)")
        XCTAssertFalse(off.contains("癌"), "\(off)")
    }

    func testKangYanOfferedWhenAccepted() {
        let on = candidates("d;4u06", mode: .bopomofo, accept: true)
        XCTAssertTrue(on.contains("抗癌"), "\(on)")
        let handler = makeHandler(.bopomofo, accept: true)
        XCTAssertEqual(buffer(type("d;4u06", into: handler)), "抗癌")
    }

    func testFeiAiUnaffected() {
        for accept in [false, true] {
            let handler = makeHandler(.bopomofo, accept: accept)
            XCTAssertEqual(buffer(type("zo496", into: handler)), "肺癌", "accept=\(accept)")
            XCTAssertTrue(candidates("zo496", mode: .bopomofo, accept: accept).contains("肺癌"))
        }
        XCTAssertEqual(
            candidates("zo496", mode: .bopomofo, accept: false),
            candidates("zo496", mode: .bopomofo, accept: true))
    }

    func testAssociatedPhrasesAfterRuUseAi() {
        // Bopomofo mode.
        let handler = makeHandler(.bopomofo, accept: false)
        let state = type("bj3", into: handler)
        XCTAssertEqual(buffer(state), "乳")
        let params = BuildAssociatedPhraseParams()
        params.previousState = state
        params.prefixCursorIndex = 1
        params.reading = "ㄖㄨˇ"
        params.value = "乳"
        params.candidateIndex = 0
        params.useVerticalMode = false
        params.autoTriggered = false
        guard let associated = handler.buildAssociatedPhraseState(with: params) as? InputState.AssociatedPhrases
        else {
            XCTFail("no associated phrases after 乳")
            return
        }
        XCTAssertFalse(associated.candidates.contains { readsAiAsYan(value: $0.value, reading: $0.reading) },
                       "\(associated.candidates.map { $0.reading })")
        guard let ruAi = associated.candidates.first(where: { $0.value == "乳癌" }) else {
            XCTFail("乳癌 not offered: \(associated.candidates.map { $0.value })")
            return
        }
        XCTAssertEqual(ruAi.reading, "ㄖㄨˇ-ㄞˊ")
        handler.fixNodeForAssociatedPhraseWithPrefix(
            at: associated.prefixCursorIndex, prefixReading: associated.prefixReading,
            prefixValue: associated.prefixValue, associatedPhraseReading: ruAi.reading,
            associatedPhraseValue: ruAi.value)
        XCTAssertEqual(buffer(handler.buildInputtingState()), "乳癌")

        // Plain Bopomofo mode.
        let plain = makeHandler(.plainBopomofo, accept: false)
        guard let plainAssociated = plain.buildAssociatedPhrasePlainState(
            withReading: "ㄖㄨˇ", value: "乳", useVerticalMode: false) as? InputState.AssociatedPhrasesPlain
        else {
            XCTFail("no plain associated phrases after 乳")
            return
        }
        XCTAssertFalse(plainAssociated.candidates.contains { readsAiAsYan(value: $0.value, reading: $0.reading) })
        // Plain candidates carry only the part after the prefix.
        XCTAssertTrue(plainAssociated.candidates.contains { $0.value == "癌" && $0.reading == "ㄞˊ" },
                      "\(plainAssociated.candidates.map { $0.value + " " + $0.reading })")
    }

    func testAssociatedPhrasesWithYanWhenAccepted() {
        let handler = makeHandler(.bopomofo, accept: true)
        let state = type("bj3", into: handler)
        let params = BuildAssociatedPhraseParams()
        params.previousState = state
        params.prefixCursorIndex = 1
        params.reading = "ㄖㄨˇ"
        params.value = "乳"
        params.candidateIndex = 0
        params.useVerticalMode = false
        params.autoTriggered = false
        let associated = handler.buildAssociatedPhraseState(with: params) as? InputState.AssociatedPhrases
        XCTAssertTrue(associated?.candidates.contains { $0.value == "乳癌" } ?? false)
        // Accepted: the ㄧㄢˊ-aligned entries are offered as upstream (the
        // upstream dedup keeps one 乳癌; it may be either reading).

    }

    func testRememberedAiForYanNotShownWhenPolicyOff() {
        // Learn 癌 for ㄧㄢˊ while acceptance is on.
        var handler = makeHandler(.bopomofo, accept: true)
        type("u06", into: handler)
        handler.fixNode(reading: "ㄧㄢˊ", value: "癌", originalCursorIndex: 1, useMoveCursorAfterSelectionSetting: false)
        XCTAssertEqual(buffer(handler.buildInputtingState()), "癌")
        handler = makeHandler(.bopomofo, accept: true)
        XCTAssertEqual(buffer(type("u06", into: handler)), "癌", "UOM should remember 癌 while accepted")

        // Acceptance off from the next activation: 癌 is not shown for ㄧㄢˊ.
        handler = makeHandler(.bopomofo, accept: false)
        let composed = buffer(type("u06", into: handler))
        XCTAssertNotNil(composed)
        XCTAssertNotEqual(composed, "癌")
    }

    func testUserPhraseWithYanStillOffered() {
        LanguageModelManager.loadUserPhrases(forTesting: "癌 ㄧㄢˊ\n")
        let off = candidates("u06", mode: .bopomofo, accept: false)
        XCTAssertTrue(off.contains("癌"), "\(off)")
    }

    func testPolicyChangeBumpsReloadGeneration() {
        LanguageModelManager.applyReadingPolicy(acceptYanReadingForAi: false)
        let g0 = LanguageModelManager.reloadGeneration
        LanguageModelManager.applyReadingPolicy(acceptYanReadingForAi: false)
        XCTAssertEqual(LanguageModelManager.reloadGeneration, g0)
        LanguageModelManager.applyReadingPolicy(acceptYanReadingForAi: true)
        XCTAssertGreaterThan(LanguageModelManager.reloadGeneration, g0)
    }
}
