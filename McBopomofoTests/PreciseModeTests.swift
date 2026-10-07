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
import SwiftUI
import XCTest

@testable import McBopomofo

/// Gate always open (tau >= 1); fusion = heavy model only (r_weight 0), so the
/// fake heavy scorer decides which candidate wins.
let kForcedOpenPreciseConf = """
    alpha = 0.5
    k = 10
    tau = 1.0
    r_weight = 0
    beta = 1
    qwen_timeout_ms = 1000

    """

/// Fake heavy model: scores one candidate 0 and the others -20 (by default the
/// last one, so the fused choice differs from the displayed rank 0). Runs on
/// the controller's queue.
final class FakeHeavyScorer {
    private let lock = NSLock()
    private var _delay: TimeInterval = 0
    private var _fail = false
    private var _preferredIndex: Int?
    private var _calls = 0
    private var _seen: [[String]] = []

    var delay: TimeInterval {
        get { lock.withLock { _delay } }
        set { lock.withLock { _delay = newValue } }
    }
    /// Busy-wait for `delay` instead of sleeping (latency harnesses only).
    var spin = false
    var fail: Bool {
        get { lock.withLock { _fail } }
        set { lock.withLock { _fail = newValue } }
    }
    /// Candidate index scored best; nil = the last candidate.
    var preferredIndex: Int? {
        get { lock.withLock { _preferredIndex } }
        set { lock.withLock { _preferredIndex = newValue } }
    }
    var calls: Int { lock.withLock { _calls } }
    var seenCandidates: [[String]] { lock.withLock { _seen } }

    var block: AFMPreciseHeavyScoreBlock {
        return { [unowned self] _, candidates in
            let (delay, fail, preferred) = self.lock.withLock { () -> (TimeInterval, Bool, Int?) in
                self._calls += 1
                self._seen.append(candidates)
                return (self._delay, self._fail, self._preferredIndex)
            }
            if delay > 0 && self.spin {
                let end = DispatchTime.now().uptimeNanoseconds + UInt64(delay * 1e9)
                while DispatchTime.now().uptimeNanoseconds < end {}
            } else if delay > 0 {
                Thread.sleep(forTimeInterval: delay)
            }
            if fail {
                return []
            }
            let best = preferred ?? candidates.count - 1
            return candidates.indices.map { NSNumber(value: $0 == best ? 0.0 : -20.0) }
        }
    }
}

/// Stands in for the input method controller: keeps the current state and
/// applies precise results with the controller's own check.
final class PreciseTestDelegate: NSObject, KeyHandlerDelegate {
    var state: InputState = InputState.Empty()
    var hasClient = true
    var appliedStates: [InputState] = []
    var canApplyCalls = 0

    func candidateController(for keyHandler: KeyHandler) -> Any { CandidateController.vertical }
    func keyHandler(_ keyHandler: KeyHandler, didSelectCandidateAt index: Int, candidateController controller: Any) {}
    func keyHandler(_ keyHandler: KeyHandler, didRequestWriteUserPhraseWith state: InputState) -> Bool { false }
    func keyHandler(_ keyHandler: KeyHandler, didRequestBoostScoreForPhrase phrase: String, reading: String) -> Bool { false }
    func keyHandler(_ keyHandler: KeyHandler, didRequestExcludePhrase phrase: String, reading: String) -> Bool { false }
    func keyHandlerDidRequestReloadLanguageModel(_ keyHandler: KeyHandler) -> Bool { false }

    func keyHandler(_ keyHandler: KeyHandler, canApplyPreciseResultForComposingBuffer composingBuffer: String) -> Bool {
        canApplyCalls += 1
        return McBopomofoInputMethodController.canApplyPreciseResult(
            state: state, composingBuffer: composingBuffer, hasClient: hasClient)
    }

    func keyHandler(_ keyHandler: KeyHandler, didUpdatePreciseState state: InputState) {
        self.state = state
        appliedStates.append(state)
    }
}

/// A requester for controller-level tests.
final class RecordingRequester: NSObject, AFMPreciseRequester {
    var finished: [(key: String, chosen: Int)] = []
    func preciseRequestDidFinish(_ request: AFMPreciseRequest, chosenIndex: UInt) {
        finished.append((request.key, Int(chosenIndex)))
    }
}

class PreciseModeTests: XCTestCase {

    var saved: [String: Any?] = [:]
    var tempDirectories: [String] = []
    /// Fakes captured unowned by their blocks; kept alive for the test.
    var retainedFakes: [AnyObject] = []

    override func setUpWithError() throws {
        saved = [
            "assoc": Preferences.associatedPhrasesEnabled,
            "rwalk": Preferences.neuralWalkEnabled,
            "alpha": Preferences.neuralWalkAlpha,
            "precise": Preferences.preciseModeEnabled,
            "layout": Preferences.keyboardLayout.rawValue,
            "conv": Preferences.chineseConversionEnabled,
            "yan": Preferences.acceptYanReadingForAi,
        ]
        Preferences.associatedPhrasesEnabled = false
        Preferences.neuralWalkEnabled = true
        Preferences.neuralWalkAlpha = 0.5
        Preferences.preciseModeEnabled = true
        Preferences.keyboardLayout = .standard
        Preferences.chineseConversionEnabled = false
        Preferences.acceptYanReadingForAi = false
        LanguageModelManager.loadDataModels()
        LanguageModelManager.resetUserOverrideModelForTesting()
        LanguageModelManager.clearUserPhrasesForTesting()
        AFMNeuralSelector.resetForTesting()
        AFMPreciseController.resetForTesting()
    }

    override func tearDownWithError() throws {
        AFMPreciseController.shared.enabled = false
        AFMPreciseController.resetForTesting()
        AFMNeuralSelector.resetForTesting()
        LanguageModelManager.resetUserOverrideModelForTesting()
        Preferences.associatedPhrasesEnabled = saved["assoc"] as? Bool ?? false
        Preferences.neuralWalkEnabled = saved["rwalk"] as? Bool ?? true
        Preferences.neuralWalkAlpha = saved["alpha"] as? Double ?? 0.5
        Preferences.preciseModeEnabled = saved["precise"] as? Bool ?? false
        Preferences.keyboardLayout = KeyboardLayout(rawValue: saved["layout"] as? Int ?? 0) ?? .standard
        Preferences.chineseConversionEnabled = saved["conv"] as? Bool ?? false
        Preferences.acceptYanReadingForAi = saved["yan"] as? Bool ?? false
        LanguageModelManager.loadUserPhrases(enableForPlainBopomofo: Preferences.enableUserPhrasesInPlainBopomofo)
        for dir in tempDirectories {
            try? FileManager.default.removeItem(atPath: dir)
        }
    }

    // MARK: - Helpers

    func waitUntil(timeout: TimeInterval = 10, _ condition: () -> Bool) {
        let deadline = Date().addingTimeInterval(timeout)
        while !condition() && Date() < deadline {
            RunLoop.current.run(until: Date().addingTimeInterval(0.005))
        }
    }

    /// A temporary model directory, with precise.conf when conf is given.
    func makeModelDirectory(conf: String?) -> String {
        let dir = (NSTemporaryDirectory() as NSString).appendingPathComponent("qwen-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        if let conf = conf {
            FileManager.default.createFile(
                atPath: (dir as NSString).appendingPathComponent("precise.conf"), contents: Data(conf.utf8))
        }
        tempDirectories.append(dir)
        return dir
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

    func makePreciseController(
        _ heavy: FakeHeavyScorer, conf: String? = kForcedOpenPreciseConf, loadDelay: TimeInterval = 0
    ) -> AFMPreciseController {
        retainedFakes.append(heavy)
        let controller = AFMPreciseController(
            modelDirectory: makeModelDirectory(conf: conf), heavyScoreBlockForTesting: heavy.block,
            loadDelay: loadDelay)
        // Most tests check request ordering and staleness; the start delay
        // (D2 addendum) is tested on its own below.
        controller.startDelay = 0
        return controller
    }

    @discardableResult
    func installPrecise(_ heavy: FakeHeavyScorer, conf: String = kForcedOpenPreciseConf) -> AFMPreciseController {
        let controller = makePreciseController(heavy, conf: conf)
        AFMPreciseController.shared = controller
        controller.enabled = true
        waitUntil { controller.state == .loading ? false : true }
        XCTAssertEqual(controller.state, .ready)
        return controller
    }

    func waitIdle(_ controller: AFMPreciseController = AFMPreciseController.shared) {
        waitUntil { controller.isIdle }
    }

    func makeHandler(_ mode: InputMode = .bopomofo) -> (KeyHandler, PreciseTestDelegate) {
        let handler = KeyHandler()
        let delegate = PreciseTestDelegate()
        handler.delegate = delegate
        handler.inputMode = mode
        return (handler, delegate)
    }

    func hook(_ handler: KeyHandler, _ delegate: PreciseTestDelegate) {
        handler.requestPreciseIfNeeded(composingBuffer: (delegate.state as? InputState.Inputting)?.composingBuffer)
    }

    /// Types keys like the input method controller: each key event, then
    /// (optionally) the end-of-key-event hook.
    func type(_ keys: String, _ handler: KeyHandler, _ delegate: PreciseTestDelegate, hook doHook: Bool = true) {
        for key in keys.map({ String($0) }) {
            let input = KeyHandlerInput(
                inputText: key, keyCode: 0, charCode: charCode(key), flags: [], isVerticalMode: false)
            handler.handle(input: input, state: delegate.state) { delegate.state = $0 } errorCallback: {}
            if doHook {
                hook(handler, delegate)
            }
        }
    }

    func press(_ keyCode: KeyCode, charCode: UInt16 = 0, _ handler: KeyHandler, _ delegate: PreciseTestDelegate, hook doHook: Bool = true) {
        let input = KeyHandlerInput(
            inputText: " ", keyCode: keyCode.rawValue, charCode: charCode, flags: [], isVerticalMode: false)
        handler.handle(input: input, state: delegate.state) { delegate.state = $0 } errorCallback: {}
        if doHook {
            hook(handler, delegate)
        }
    }

    /// A candidate pick as the input method controller does it (fixNode,
    /// inputting state, hook).
    func pick(reading: String, value: String, cursor: Int, _ handler: KeyHandler, _ delegate: PreciseTestDelegate, hook doHook: Bool = true) {
        handler.fixNode(reading: reading, value: value, originalCursorIndex: cursor, useMoveCursorAfterSelectionSetting: false)
        delegate.state = handler.buildInputtingState()
        if doHook {
            hook(handler, delegate)
        }
    }

    func buffer(_ delegate: PreciseTestDelegate) -> String? {
        (delegate.state as? InputState.Inputting)?.composingBuffer
    }

    /// Runs body with a disabled shared controller, keeping the current one
    /// (and its loaded model) untouched.
    func withPreciseOff<T>(_ body: () -> T) -> T {
        let current = AFMPreciseController.shared
        AFMPreciseController.shared = AFMPreciseController(modelDirectory: nil)
        defer { AFMPreciseController.shared = current }
        return body()
    }

    /// Composing buffer for keys with R-walk only (same fake R, precise off).
    func rWalkBuffer(_ keys: String) -> String? {
        withPreciseOff {
            let (handler, delegate) = makeHandler()
            type(keys, handler, delegate)
            return buffer(delegate)
        }
    }

    // MARK: - 5.1 Selector: payload, deferred breaker, NotFound

    func testOneForwardPerReadingWithPreciseReady() {
        let fake = FakeNeuralScorer()
        let selector = installFakeSelector(fake)
        let heavy = FakeHeavyScorer()
        heavy.delay = 0.05
        installPrecise(heavy)

        let (handler, delegate) = makeHandler()
        type("su3cl3", handler, delegate)
        XCTAssertEqual(fake.calls, 2)
        XCTAssertEqual(selector.forwardCount, 2)
        XCTAssertGreaterThan(selector.precisePayloadCount, 0)
        XCTAssertEqual(selector.breakerRecordCount, 2)
        waitIdle()
        XCTAssertEqual(fake.calls, 2, "N-best and gate reuse the cached table")
    }

    func testPreciseNotReadyIsIdenticalToPreciseOff() {
        let keys = "su3cl3u06"
        func run(_ configure: () -> Void) -> (String?, UInt, UInt, UInt, UInt) {
            let fake = FakeNeuralScorer()
            let selector = installFakeSelector(fake)
            configure()
            let (handler, delegate) = makeHandler()
            type(keys, handler, delegate)
            return (buffer(delegate), selector.forwardCount, selector.breakerRecordCount, selector.precisePayloadCount, handler.preciseRequestCount)
        }
        let off = run { AFMPreciseController.resetForTesting() }
        // Enabled but still loading.
        let heavy = FakeHeavyScorer()
        let loading = run {
            let controller = makePreciseController(heavy, loadDelay: 2)
            AFMPreciseController.shared = controller
            controller.enabled = true
            XCTAssertEqual(controller.state, .loading)
        }
        // Enabled but the model directory is missing.
        let notFound = run {
            let controller = AFMPreciseController(modelDirectory: NSTemporaryDirectory() + "/no-qwen-\(UUID().uuidString)")
            AFMPreciseController.shared = controller
            controller.enabled = true
            waitUntil { controller.state != .loading }
            XCTAssertEqual(controller.state, .notFound)
        }
        for other in [loading, notFound] {
            XCTAssertEqual(other.0, off.0)
            XCTAssertEqual(other.1, off.1)
            XCTAssertEqual(other.2, off.2, "breaker recorded immediately, as with precise off")
            XCTAssertEqual(other.3, 0, "no enumeration or gate")
            XCTAssertEqual(other.4, 0)
        }
        XCTAssertEqual(off.1, off.2)
        XCTAssertEqual(heavy.calls, 0)
    }

    func testBreakerTripStopsPreciseRequests() {
        let fake = FakeNeuralScorer()
        fake.delay = 0.035  // over the 30 ms breaker threshold (D9)
        let selector = installFakeSelector(fake)
        let heavy = FakeHeavyScorer()
        heavy.delay = 0.5
        let controller = installPrecise(heavy)

        let (handler, delegate) = makeHandler()
        type("su3cl3u06", handler, delegate)
        XCTAssertEqual(selector.state, .tripped)
        let requests = handler.preciseRequestCount
        let submitted = controller.submittedCount
        type("d;4", handler, delegate)
        XCTAssertEqual(handler.preciseRequestCount, requests, "no request once R-walk is tripped")
        XCTAssertEqual(controller.submittedCount, submitted)
        XCTAssertLessThan(requests, 3, "the tripping keystroke makes no request")
    }

    func testEveryCacheMissRecordedOnceWithOrWithoutHook() {
        let fake = FakeNeuralScorer()
        let selector = installFakeSelector(fake)
        installPrecise(FakeHeavyScorer())

        let (handler, delegate) = makeHandler()
        type("su3cl3", handler, delegate, hook: true)
        XCTAssertEqual(selector.breakerRecordCount, selector.forwardCount)

        // Without the hook (e.g. a non-inputting state): each measurement is
        // recorded by the next walk; the last one stays pending until then.
        type("u06d;4", handler, delegate, hook: false)
        XCTAssertEqual(selector.forwardCount, 4)
        XCTAssertEqual(selector.breakerRecordCount, 3)
        handler.requestPreciseIfNeeded(composingBuffer: nil)
        XCTAssertEqual(selector.breakerRecordCount, 4)
        handler.requestPreciseIfNeeded(composingBuffer: nil)
        XCTAssertEqual(selector.breakerRecordCount, 4, "recorded exactly once")
        waitIdle()
    }

    // MARK: - 5.2 Controller

    func testMissingDirectoryIsNotFound() {
        let controller = AFMPreciseController(modelDirectory: NSTemporaryDirectory() + "/no-qwen-\(UUID().uuidString)")
        controller.enabled = true
        XCTAssertEqual(controller.state, .loading)
        waitUntil { controller.state != .loading }
        XCTAssertEqual(controller.state, .notFound)
    }

    func testMissingConfOrModelIsNotFound() {
        let heavy = FakeHeavyScorer()
        let noConf = makePreciseController(heavy, conf: nil)
        noConf.enabled = true
        waitUntil { noConf.state != .loading }
        XCTAssertEqual(noConf.state, .notFound)

        // Production loader: conf present, model.gguf missing.
        let noModel = AFMPreciseController(modelDirectory: makeModelDirectory(conf: kForcedOpenPreciseConf))
        noModel.enabled = true
        waitUntil { noModel.state != .loading }
        XCTAssertEqual(noModel.state, .notFound)
        XCTAssertEqual(heavy.calls, 0)
    }

    func testBadConfFails() {
        let controller = makePreciseController(FakeHeavyScorer(), conf: kForcedOpenPreciseConf + "unknown_key = 1\n")
        controller.enabled = true
        waitUntil { controller.state != .loading }
        XCTAssertEqual(controller.state, .failed)
    }

    func testAlphaMismatchFails() {
        Preferences.neuralWalkAlpha = 0.3
        let controller = makePreciseController(FakeHeavyScorer())
        controller.enabled = true
        waitUntil { controller.state != .loading }
        XCTAssertEqual(controller.state, .failed)

        // Matching alpha after a preference change + activation: reloads.
        Preferences.neuralWalkAlpha = 0.5
        controller.syncWithPreferences()
        waitUntil { controller.state != .loading }
        XCTAssertEqual(controller.state, .ready)
    }

    func testStateNotificationsOnMainThread() {
        let controller = makePreciseController(FakeHeavyScorer())
        var states: [AFMPreciseState] = []
        let observer = NotificationCenter.default.addObserver(
            forName: .AFMPreciseControllerStateDidChange, object: controller, queue: nil
        ) { _ in
            XCTAssertTrue(Thread.isMainThread)
            states.append(controller.state)
        }
        defer { NotificationCenter.default.removeObserver(observer) }
        controller.enabled = true
        waitUntil { controller.state != .loading }
        controller.enabled = false
        XCTAssertEqual(states, [.loading, .ready, .off])
    }

    func request(_ candidates: [String], key: String, requester: RecordingRequester) -> AFMPreciseRequest {
        AFMPreciseRequest(
            forTestingWithReadings: ["ㄋㄧˇ"], candidates: candidates,
            rScores: candidates.indices.map { NSNumber(value: -Double($0)) }, key: key,
            composingText: candidates[0], requester: requester)
    }

    func testRapidRequestsLatestWins() {
        let heavy = FakeHeavyScorer()
        heavy.delay = 0.2
        let controller = installPrecise(heavy)
        let requester = RecordingRequester()
        controller.submitRequest(request(["a1", "a2"], key: "1", requester: requester))
        controller.submitRequest(request(["b1", "b2"], key: "2", requester: requester))
        controller.submitRequest(request(["c1", "c2"], key: "3", requester: requester))
        waitIdle(controller)
        XCTAssertEqual(controller.submittedCount, 3)
        XCTAssertEqual(controller.scoredCount, 2)
        XCTAssertEqual(heavy.seenCandidates.map { $0[0] }, ["a1", "c1"])
        XCTAssertEqual(requester.finished.map { $0.key }, ["1", "3"])
        XCTAssertEqual(requester.finished.map { $0.chosen }, [1, 1])
    }

    func testThreeTimeoutsTripStickily() {
        let heavy = FakeHeavyScorer()
        heavy.delay = 0.06
        let conf = kForcedOpenPreciseConf.replacingOccurrences(of: "qwen_timeout_ms = 1000", with: "qwen_timeout_ms = 20")
        let controller = installPrecise(heavy, conf: conf)
        let requester = RecordingRequester()
        for i in 0..<3 {
            XCTAssertEqual(controller.state, .ready)
            controller.submitRequest(request(["x\(i)", "y\(i)"], key: "\(i)", requester: requester))
            waitIdle(controller)
        }
        XCTAssertEqual(controller.state, .tripped)
        XCTAssertTrue(requester.finished.isEmpty, "timed-out results are discarded")

        controller.enabled = false
        controller.enabled = true
        waitUntil(timeout: 0.3) { false }
        XCTAssertEqual(controller.state, .tripped)
        let calls = heavy.calls
        controller.submitRequest(request(["z1", "z2"], key: "z", requester: requester))
        waitIdle(controller)
        XCTAssertEqual(heavy.calls, calls, "no request after tripping")
    }

    func testEmptyScoresFailWithoutStreak() {
        let heavy = FakeHeavyScorer()
        heavy.fail = true
        let controller = installPrecise(heavy)
        let requester = RecordingRequester()
        controller.submitRequest(request(["a1", "a2"], key: "1", requester: requester))
        waitIdle(controller)
        XCTAssertEqual(controller.scoredCount, 1)
        XCTAssertEqual(controller.timeoutStreak, 0)
        XCTAssertTrue(requester.finished.isEmpty)
        XCTAssertEqual(controller.state, .ready)

        heavy.fail = false
        controller.submitRequest(request(["b1", "b2"], key: "2", requester: requester))
        waitIdle(controller)
        XCTAssertEqual(requester.finished.map { $0.key }, ["2"])
    }

    func testDisableDuringInferenceDiscardsResult() {
        let heavy = FakeHeavyScorer()
        heavy.delay = 0.2
        let controller = installPrecise(heavy)
        let requester = RecordingRequester()
        controller.submitRequest(request(["a1", "a2"], key: "1", requester: requester))
        waitUntil { heavy.calls == 1 }
        controller.enabled = false
        XCTAssertEqual(controller.state, .off)
        waitIdle(controller)
        waitUntil(timeout: 0.3) { false }
        XCTAssertTrue(requester.finished.isEmpty)
        XCTAssertEqual(controller.deliveredCount, 0)
    }

    func testRapidOffOnDoesNotUseEarlierLoad() {
        let heavy = FakeHeavyScorer()
        let controller = makePreciseController(heavy, loadDelay: 0.3)
        var states: [AFMPreciseState] = []
        let observer = NotificationCenter.default.addObserver(
            forName: .AFMPreciseControllerStateDidChange, object: controller, queue: nil
        ) { _ in states.append(controller.state) }
        defer { NotificationCenter.default.removeObserver(observer) }
        controller.enabled = true
        waitUntil(timeout: 0.1) { false }  // first load is sleeping on the queue
        controller.enabled = false
        controller.enabled = true
        waitUntil { controller.state == .ready }
        XCTAssertEqual(controller.state, .ready)
        XCTAssertEqual(states, [.loading, .off, .loading, .ready])
        XCTAssertEqual(controller.discardedLoadCount, 1)

        let requester = RecordingRequester()
        controller.submitRequest(request(["a1", "a2"], key: "1", requester: requester))
        waitIdle(controller)
        XCTAssertEqual(requester.finished.map { $0.key }, ["1"])
    }

    // MARK: - D2 addendum: low-QoS queue, start delay after key events

    func testDefaultStartDelayAndQoS() {
        let heavy = FakeHeavyScorer()
        retainedFakes.append(heavy)
        let controller = AFMPreciseController(
            modelDirectory: makeModelDirectory(conf: kForcedOpenPreciseConf), heavyScoreBlockForTesting: heavy.block,
            loadDelay: 0)
        XCTAssertEqual(controller.startDelay, 0.04, accuracy: 1e-9)
        controller.enabled = true
        waitUntil { controller.state == .ready }
        let requester = RecordingRequester()
        controller.submitRequest(request(["a1", "a2"], key: "1", requester: requester))
        waitIdle(controller)
        XCTAssertEqual(controller.lastHeavyQoSClass, QOS_CLASS_UTILITY, "heavy inference runs at utility QoS")
        XCTAssertEqual(requester.finished.map { $0.key }, ["1"])
        controller.enabled = false
    }

    func testHeavyDoesNotStartWithinDelayAfterKeyEvent() {
        let heavy = FakeHeavyScorer()
        let controller = installPrecise(heavy)
        controller.startDelay = 0.3
        let requester = RecordingRequester()
        controller.noteKeyEvent()
        controller.submitRequest(request(["a1", "a2"], key: "1", requester: requester))
        waitUntil(timeout: 0.1) { false }
        XCTAssertEqual(heavy.calls, 0, "no start within the delay")
        XCTAssertFalse(controller.isHeavyRunning)
        XCTAssertFalse(controller.isIdle, "the request is queued")
        // A newer request during the delay replaces it (latest wins).
        controller.submitRequest(request(["b1", "b2"], key: "2", requester: requester))
        waitIdle(controller)
        XCTAssertEqual(heavy.calls, 1)
        XCTAssertEqual(requester.finished.map { $0.key }, ["2"])
        XCTAssertGreaterThanOrEqual(controller.deferredStartCount, 1)
    }

    func testFurtherKeyEventsPushTheStartBack() {
        let heavy = FakeHeavyScorer()
        let controller = installPrecise(heavy)
        controller.startDelay = 0.15
        let requester = RecordingRequester()
        controller.noteKeyEvent()
        controller.submitRequest(request(["a1", "a2"], key: "1", requester: requester))
        // Keep typing every 50 ms for ~400 ms: nothing starts meanwhile.
        for _ in 0..<8 {
            waitUntil(timeout: 0.05) { false }
            controller.noteKeyEvent()
            XCTAssertEqual(heavy.calls, 0)
        }
        waitIdle(controller)
        XCTAssertEqual(heavy.calls, 1, "starts once typing pauses")
        XCTAssertEqual(requester.finished.map { $0.key }, ["1"])
    }

    func testKeyHandlerNotesEveryKeyEventAndResultStillApplies() {
        installFakeSelector(FakeNeuralScorer())
        let heavy = FakeHeavyScorer()
        let controller = installPrecise(heavy)
        controller.startDelay = 0.05
        let (handler, delegate) = makeHandler()
        type("su3cl3", handler, delegate)
        let shown = buffer(delegate)
        XCTAssertEqual(handler.preciseRequestCount, 2)
        // Both requests were made within the delay: only the latest runs.
        waitIdle()
        XCTAssertEqual(heavy.calls, 1)
        XCTAssertEqual(handler.preciseAppliedCount, 1)
        XCTAssertNotEqual(buffer(delegate), shown)
    }

    // MARK: - 5.3 KeyHandler integration

    func testResultAppliedWhenNothingChanged() {
        installFakeSelector(FakeNeuralScorer())
        let heavy = FakeHeavyScorer()
        let controller = installPrecise(heavy)

        let (handler, delegate) = makeHandler()
        type("su3cl3", handler, delegate)
        let shown = buffer(delegate)
        let cursor = (delegate.state as? InputState.Inputting)?.cursorIndex
        XCTAssertEqual(shown, rWalkBuffer("su3cl3"))
        // One request per syllable; the first result arrives after the second
        // syllable and is not applied.
        XCTAssertEqual(handler.preciseRequestCount, 2)
        waitIdle()
        XCTAssertEqual(controller.deliveredCount, 2)
        XCTAssertEqual(handler.preciseAppliedCount, 1)
        XCTAssertEqual(delegate.appliedStates.count, 1)
        let fused = buffer(delegate)
        XCTAssertNotNil(fused)
        XCTAssertNotEqual(fused, shown)
        XCTAssertEqual(fused, heavy.seenCandidates.last?.last, "the heavy model's choice is shown")
        XCTAssertEqual((delegate.state as? InputState.Inputting)?.cursorIndex, cursor)
        XCTAssertTrue(handler.showingPreciseResult)

        // Zero new requests after applying, also after a no-op re-walk.
        hook(handler, delegate)
        handler.rewalkForTesting()
        XCTAssertEqual((handler.buildInputtingState() as? InputState.Inputting)?.composingBuffer, fused)
        XCTAssertTrue(handler.showingPreciseResult)
        hook(handler, delegate)
        waitIdle()
        XCTAssertEqual(handler.preciseRequestCount, 2)
        XCTAssertEqual(controller.submittedCount, 2)
        XCTAssertEqual(handler.preciseAppliedCount, 1)

        // Not learned as a user choice: the same keys show R-walk again.
        let (other, otherDelegate) = makeHandler()
        AFMPreciseController.shared.enabled = false
        type("su3cl3", other, otherDelegate)
        XCTAssertEqual(buffer(otherDelegate), shown)
    }

    func testCandidatePickSendsOneRequest() {
        installFakeSelector(FakeNeuralScorer())
        let heavy = FakeHeavyScorer()
        heavy.delay = 0.3
        installPrecise(heavy)
        let (handler, delegate) = makeHandler()
        type("su3cl3u06", handler, delegate)
        let typed = handler.preciseRequestCount
        XCTAssertEqual(typed, 3, "one per syllable with the gate forced open")

        // Open the candidate window (no request: not inputting), then pick.
        press(.down, handler, delegate)
        XCTAssertTrue(delegate.state is InputState.ChoosingCandidate)
        XCTAssertEqual(handler.preciseRequestCount, typed)
        guard let choosing = delegate.state as? InputState.ChoosingCandidate,
            let candidate = choosing.candidates.first(where: { $0.value.count == 1 })
        else {
            XCTFail("no candidates")
            return
        }
        pick(reading: candidate.reading, value: candidate.value, cursor: 3, handler, delegate)
        XCTAssertEqual(handler.preciseRequestCount, typed + 1)
        hook(handler, delegate)
        XCTAssertEqual(handler.preciseRequestCount, typed + 1)
        waitIdle()
        XCTAssertTrue(buffer(delegate)?.hasSuffix(candidate.value) ?? false, "\(String(describing: buffer(delegate)))")
    }

    func testUserOverrideSuggestionMakesOneRequestThatKeepsIt() {
        installFakeSelector(FakeNeuralScorer())
        // Teach the user override model 妳 after 好, with precise off.
        do {
            let (handler, delegate) = makeHandler()
            type("cl3su3", handler, delegate)
            pick(reading: "ㄋㄧˇ", value: "妳", cursor: 2, handler, delegate)
            XCTAssertEqual(buffer(delegate)?.last, "妳")
        }
        let heavy = FakeHeavyScorer()
        installPrecise(heavy)
        let (handler, delegate) = makeHandler()
        type("cl3", handler, delegate)
        waitIdle()
        let requestsBefore = handler.preciseRequestCount
        let appliedBefore = handler.preciseAppliedCount
        type("su3", handler, delegate)
        XCTAssertEqual(buffer(delegate)?.last, "妳", "UOM suggestion applied in this key event")
        XCTAssertEqual(handler.preciseRequestCount, requestsBefore + 1)
        waitIdle()
        let last = heavy.seenCandidates.last ?? []
        XCTAssertGreaterThanOrEqual(last.count, 2)
        XCTAssertTrue(last.allSatisfy { $0.hasSuffix("妳") }, "\(last)")
        XCTAssertEqual(handler.preciseAppliedCount, appliedBefore + 1)
        XCTAssertEqual(buffer(delegate)?.last, "妳")
        XCTAssertTrue(handler.showingPreciseResult)
    }

    func testDiscardedAfterAnotherSyllable() {
        installFakeSelector(FakeNeuralScorer())
        let heavy = FakeHeavyScorer()
        heavy.delay = 0.2
        let controller = installPrecise(heavy)
        let (handler, delegate) = makeHandler()
        type("su3cl3", handler, delegate)
        type("u06", handler, delegate)
        waitIdle()
        XCTAssertEqual(controller.deliveredCount, 2)
        XCTAssertEqual(handler.preciseAppliedCount, 1, "only the result for the latest readings")
        XCTAssertEqual(delegate.appliedStates.count, 1)
        XCTAssertEqual((delegate.appliedStates.first as? InputState.Inputting)?.composingBuffer.count, 3)
    }

    func testDiscardedAfterCommit() {
        installFakeSelector(FakeNeuralScorer())
        let heavy = FakeHeavyScorer()
        heavy.delay = 0.2
        installPrecise(heavy)
        let (handler, delegate) = makeHandler()
        type("su3cl3", handler, delegate)
        var committed: [String] = []
        let enter = KeyHandlerInput(inputText: " ", keyCode: KeyCode.enter.rawValue, charCode: 13, flags: [], isVerticalMode: false)
        handler.handle(input: enter, state: delegate.state) { newState in
            delegate.state = newState
            if let c = newState as? InputState.Committing { committed.append(c.poppedText) }
        } errorCallback: {}
        hook(handler, delegate)
        waitIdle()
        XCTAssertEqual(committed.count, 1)
        XCTAssertEqual(handler.preciseAppliedCount, 0)
        XCTAssertTrue(delegate.appliedStates.isEmpty)
        XCTAssertFalse(delegate.state is InputState.Inputting)
    }

    func testDiscardedAfterManualPick() {
        installFakeSelector(FakeNeuralScorer())
        let heavy = FakeHeavyScorer()
        heavy.delay = 0.2
        installPrecise(heavy)
        let (handler, delegate) = makeHandler()
        type("su3cl3", handler, delegate)
        pick(reading: "ㄏㄠˇ", value: "郝", cursor: 2, handler, delegate, hook: false)
        waitIdle()
        XCTAssertEqual(handler.preciseAppliedCount, 0)
        XCTAssertEqual(buffer(delegate)?.last, "郝")
    }

    func testDiscardedAfterUserPhraseReload() {
        installFakeSelector(FakeNeuralScorer())
        let heavy = FakeHeavyScorer()
        heavy.delay = 0.2
        installPrecise(heavy)
        let (handler, delegate) = makeHandler()
        type("su3cl3", handler, delegate)
        LanguageModelManager.loadUserPhrases(forTesting: "測試 ㄘㄜˋ-ㄕˋ\n")
        waitIdle()
        XCTAssertEqual(handler.preciseAppliedCount, 0)
    }

    func testEarlierManualPickPreserved() {
        installFakeSelector(FakeNeuralScorer())
        let heavy = FakeHeavyScorer()
        installPrecise(heavy)
        let (handler, delegate) = makeHandler()
        type("su3", handler, delegate, hook: false)
        pick(reading: "ㄋㄧˇ", value: "妳", cursor: 1, handler, delegate, hook: false)
        type("cl3u06", handler, delegate)
        waitIdle()
        XCTAssertGreaterThan(handler.preciseAppliedCount, 0)
        XCTAssertEqual(buffer(delegate)?.first, "妳")
        XCTAssertTrue(heavy.seenCandidates.allSatisfy { $0.allSatisfy { $0.hasPrefix("妳") } })
    }

    func testPickAfterAppliedResultReevaluatesWithRWalk() {
        let fake = FakeNeuralScorer()
        installFakeSelector(fake)
        let heavy = FakeHeavyScorer()
        installPrecise(heavy)
        let (handler, delegate) = makeHandler()
        type("su3cl3", handler, delegate)
        waitIdle()
        XCTAssertTrue(handler.showingPreciseResult)
        guard let fused = buffer(delegate), let lastChar = fused.last else {
            XCTFail("no fused result")
            return
        }
        pick(reading: "ㄏㄠˇ", value: String(lastChar), cursor: 2, handler, delegate, hook: false)
        XCTAssertFalse(handler.showingPreciseResult)

        // Same pick with R-walk only.
        let expected: String? = withPreciseOff {
            let (reference, referenceDelegate) = makeHandler()
            type("su3cl3", reference, referenceDelegate)
            pick(reading: "ㄏㄠˇ", value: String(lastChar), cursor: 2, reference, referenceDelegate)
            return buffer(referenceDelegate)
        }
        XCTAssertEqual(buffer(delegate), expected)
    }

    func testPlainBopomofoMakesNoRequest() {
        let fake = FakeNeuralScorer()
        installFakeSelector(fake)
        let heavy = FakeHeavyScorer()
        let controller = installPrecise(heavy)
        let (handler, delegate) = makeHandler(.plainBopomofo)
        type("s", handler, delegate)
        XCTAssertTrue(delegate.state is InputState.Inputting)
        type("u3", handler, delegate)
        let (other, otherDelegate) = makeHandler(.plainBopomofo)
        type("cl3", other, otherDelegate)
        waitIdle()
        XCTAssertEqual(handler.preciseRequestCount, 0)
        XCTAssertEqual(controller.submittedCount, 0)
        XCTAssertEqual(fake.calls, 0)
    }

    func testRNotReadyMakesNoRequest() {
        let heavy = FakeHeavyScorer()
        let controller = installPrecise(heavy)
        let (handler, delegate) = makeHandler()  // shared selector has no model
        type("su3cl3", handler, delegate)
        waitIdle()
        XCTAssertEqual(handler.preciseRequestCount, 0)
        XCTAssertEqual(controller.submittedCount, 0)
        XCTAssertEqual(buffer(delegate), "你好")
    }

    func testPreciseOffIdenticalToCurrentBehavior() {
        let fake = FakeNeuralScorer(favored: ["妳"])
        let selector = installFakeSelector(fake)
        let (handler, delegate) = makeHandler()
        type("su3cl3u06", handler, delegate)
        XCTAssertEqual(handler.preciseRequestCount, 0)
        XCTAssertEqual(selector.precisePayloadCount, 0)
        XCTAssertEqual(selector.breakerRecordCount, selector.forwardCount)

        // Same keys through the plain KeyHandler path (no hook, no delegate).
        let plain = KeyHandler()
        plain.inputMode = .bopomofo
        var state: InputState = InputState.Empty()
        for key in "su3cl3u06".map({ String($0) }) {
            let input = KeyHandlerInput(inputText: key, keyCode: 0, charCode: charCode(key), flags: [], isVerticalMode: false)
            plain.handle(input: input, state: state) { state = $0 } errorCallback: {}
        }
        XCTAssertEqual((state as? InputState.Inputting)?.composingBuffer, buffer(delegate))
    }

    func testPreferenceOnAtActivationStartsLoading() {
        let heavy = FakeHeavyScorer()
        let controller = makePreciseController(heavy, loadDelay: 0.2)
        AFMPreciseController.shared = controller
        Preferences.preciseModeEnabled = true
        let (handler, _) = makeHandler()
        handler.syncWithPreferences()  // the activation path
        XCTAssertEqual(controller.state, .loading)
        waitUntil { controller.state != .loading }
        XCTAssertEqual(controller.state, .ready)

        Preferences.preciseModeEnabled = false
        handler.syncWithPreferences()
        XCTAssertEqual(controller.state, .off)
    }

    /// Apply step on the key-handling thread (checks, re-walk without
    /// inference, inputting state, delegate): p95 < 10 ms up to 40 syllables.
    func testApplyStepLatency() throws {
        let fake = FakeNeuralScorer()
        installFakeSelector(fake)
        installPrecise(FakeHeavyScorer())
        // Buffers of every length 1...40 cut from the concatenated sentences.
        let stream = try benchSentences().flatMap { $0 }
        var samples: [(len: Int, ms: Double)] = []
        var offset = 0
        for i in 0..<400 {
            let length = i % Int(kAFMNeuralMaxSyllables) + 1
            guard offset + length <= stream.count else { break }
            let syllables = Array(stream[offset..<offset + length])
            offset += length
            let (handler, delegate) = makeHandler()
            type(syllables.map { standardLayoutKeys(forReading: $0) }.joined(), handler, delegate, hook: false)
            guard delegate.state is InputState.Inputting else { continue }
            hook(handler, delegate)
            waitIdle()
            if handler.preciseAppliedCount == 1 {
                samples.append((syllables.count, handler.lastPreciseApplyMilliseconds))
            }
        }
        XCTAssertGreaterThan(samples.count, 300)
        let report = latencyReport(samples)
        print("PRECISE_APPLY_JSON " + report.json)
        XCTAssertLessThan(report.p95, 10.0)
    }

    // MARK: - 5.4 Input method controller

    func testCanApplyOnlyInSameInputtingStateWithClient() {
        let text = "你好"
        let inputting = InputState.Inputting(composingBuffer: text, cursorIndex: 2)
        XCTAssertTrue(McBopomofoInputMethodController.canApplyPreciseResult(state: inputting, composingBuffer: text, hasClient: true))
        XCTAssertFalse(McBopomofoInputMethodController.canApplyPreciseResult(state: inputting, composingBuffer: "妳好", hasClient: true))
        XCTAssertFalse(McBopomofoInputMethodController.canApplyPreciseResult(state: inputting, composingBuffer: text, hasClient: false))
        let marking = InputState.Marking(composingBuffer: text, cursorIndex: 2, markerIndex: 0, readings: ["ㄋㄧˇ", "ㄏㄠˇ"])
        let candidate = InputState.Candidate(reading: "ㄋㄧˇ", value: "你", displayText: "你", rawValue: "你")
        let choosing = InputState.ChoosingCandidate(composingBuffer: text, cursorIndex: 2, candidates: [candidate], useVerticalMode: false)
        let associated = InputState.AssociatedPhrases(
            previousState: inputting, prefixCursorIndex: 1, prefixReading: "ㄋㄧˇ", prefixValue: "你",
            selectedIndex: 0, candidates: [candidate], useVerticalMode: false, autoTriggered: true)
        let menu = InputState.CustomMenu(
            composingBuffer: text, cursorIndex: 2, title: "t",
            entries: [InputState.CustomMenuEntry(title: "x", callback: {})], previousState: inputting, selectedIndex: 0)
        for state in [marking, choosing, associated, menu] as [InputState] {
            XCTAssertFalse(
                McBopomofoInputMethodController.canApplyPreciseResult(state: state, composingBuffer: text, hasClient: true),
                "\(state)")
        }
    }

    /// A controller as the input method server would create it, with a
    /// non-text client (state changes are tracked, nothing is drawn).
    func makeInputMethodController() throws -> McBopomofoInputMethodController {
        guard let controller = McBopomofoInputMethodController(server: nil, delegate: nil, client: nil) else {
            throw XCTSkip("cannot create an input method controller without a server")
        }
        controller.currentClient = NSObject()
        controller.keyHandler.inputMode = .bopomofo
        return controller
    }

    func type(_ keys: String, into controller: McBopomofoInputMethodController) {
        for key in keys.map({ String($0) }) {
            let input = KeyHandlerInput(inputText: key, keyCode: 0, charCode: charCode(key), flags: [], isVerticalMode: false)
            _ = controller.keyHandler.handle(input: input, state: controller.state) { newState in
                controller.handle(state: newState, client: controller.currentClient)
            } errorCallback: {}
            controller.requestPreciseIfNeeded()
        }
    }

    func testInputMethodControllerAppliesWhenUnchanged() throws {
        installFakeSelector(FakeNeuralScorer())
        installPrecise(FakeHeavyScorer())
        let imc = try makeInputMethodController()
        type("su3cl3", into: imc)
        let shown = (imc.state as? InputState.Inputting)?.composingBuffer
        waitIdle()
        let fused = (imc.state as? InputState.Inputting)?.composingBuffer
        XCTAssertNotNil(fused)
        XCTAssertNotEqual(fused, shown)
        XCTAssertEqual(imc.keyHandler.preciseAppliedCount, 1)
    }

    func testInputMethodControllerDropsInCandidateWindowAndAfterDeactivate() throws {
        installFakeSelector(FakeNeuralScorer())
        let heavy = FakeHeavyScorer()
        heavy.delay = 0.2
        installPrecise(heavy)

        let imc = try makeInputMethodController()
        type("su3cl3", into: imc)
        let down = KeyHandlerInput(inputText: " ", keyCode: KeyCode.down.rawValue, charCode: 0, flags: [], isVerticalMode: false)
        _ = imc.keyHandler.handle(input: down, state: imc.state) { imc.handle(state: $0, client: imc.currentClient) } errorCallback: {}
        imc.requestPreciseIfNeeded()
        XCTAssertTrue(imc.state is InputState.ChoosingCandidate)
        waitIdle()
        XCTAssertTrue(imc.state is InputState.ChoosingCandidate, "candidate window unaffected")
        XCTAssertEqual(imc.keyHandler.preciseAppliedCount, 0)

        let other = try makeInputMethodController()
        type("su3cl3", into: other)
        other.deactivateServer(nil)
        waitIdle()
        XCTAssertEqual(other.keyHandler.preciseAppliedCount, 0)
        XCTAssertFalse(other.state is InputState.Inputting)
    }

    func testResultForOneKeyHandlerNeverReachesAnother() throws {
        installFakeSelector(FakeNeuralScorer())
        let heavy = FakeHeavyScorer()
        heavy.delay = 0.1
        installPrecise(heavy)
        let a = try makeInputMethodController()
        let b = try makeInputMethodController()
        type("su3cl3", into: a)
        // B shows the same text but makes no request of its own.
        for key in "su3cl3".map({ String($0) }) {
            let input = KeyHandlerInput(inputText: key, keyCode: 0, charCode: charCode(key), flags: [], isVerticalMode: false)
            _ = b.keyHandler.handle(input: input, state: b.state) { b.handle(state: $0, client: b.currentClient) } errorCallback: {}
        }
        let bShown = (b.state as? InputState.Inputting)?.composingBuffer
        XCTAssertEqual(bShown, (a.state as? InputState.Inputting)?.composingBuffer)
        waitIdle()
        XCTAssertEqual(a.keyHandler.preciseAppliedCount, 1)
        XCTAssertEqual(b.keyHandler.preciseAppliedCount, 0)
        XCTAssertEqual((b.state as? InputState.Inputting)?.composingBuffer, bShown)
        XCTAssertNotEqual((a.state as? InputState.Inputting)?.composingBuffer, bShown)
    }

    // MARK: - Latency helpers

    /// Readings of the general benchmark test sentences.
    func benchSentences() throws -> [[String]] {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
        let jsonl = root.appendingPathComponent("bench/data/general/test.jsonl")
        guard let text = try? String(contentsOf: jsonl, encoding: .utf8) else {
            throw XCTSkip("bench test.jsonl not found at \(jsonl.path)")
        }
        return text.split(separator: "\n").compactMap { line in
            (try? JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any])?["reading"] as? [String]
        }
    }
}

struct LatencyReport {
    var p95: Double
    var json: String
    var jsonObject: [String: Any]
}

func percentile(_ values: [Double], _ p: Double) -> Double {
    guard !values.isEmpty else { return 0 }
    let s = values.sorted()
    return s[max(0, Int(ceil(p / 100.0 * Double(s.count))) - 1)]
}

func latencyStats(_ v: [Double]) -> [String: Any] {
    ["n": v.count, "p50": percentile(v, 50), "p95": percentile(v, 95), "p99": percentile(v, 99), "max": v.max() ?? 0]
}

/// Overall and per-length-bucket (1-10, 11-20, 21-30, 31-40) statistics.
func latencyReport(_ samples: [(len: Int, ms: Double)], extra: [String: Any] = [:]) -> LatencyReport {
    var buckets: [String: Any] = [:]
    for (name, range) in [("1-10", 1...10), ("11-20", 11...20), ("21-30", 21...30), ("31-40", 31...40)] {
        buckets[name] = latencyStats(samples.filter { range.contains($0.len) }.map { $0.ms })
    }
    #if DEBUG
    let config = "Debug"
    #else
    let config = "Release"
    #endif
    var report: [String: Any] = ["config": config, "overall": latencyStats(samples.map { $0.ms }), "buckets": buckets]
    report.merge(extra) { $1 }
    let data = (try? JSONSerialization.data(withJSONObject: report, options: [.sortedKeys])) ?? Data()
    return LatencyReport(
        p95: percentile(samples.map { $0.ms }, 95), json: String(decoding: data, as: UTF8.self), jsonObject: report)
}

/// Renders the selection-model preferences tab (task 6.2). Always renders as a
/// smoke test; writes PNGs only when AFM_PREFS_SCREENSHOT_DIR is set (via
/// TEST_RUNNER_AFM_PREFS_SCREENSHOT_DIR for xcodebuild).
class SelectionModelPreferencesRenderTests: XCTestCase {
    var savedPrecise = false
    var savedNeural = true
    var fakes: [AnyObject] = []
    var tempDirectories: [String] = []

    override func setUpWithError() throws {
        savedPrecise = Preferences.preciseModeEnabled
        savedNeural = Preferences.neuralWalkEnabled
        Preferences.neuralWalkEnabled = true
        // Shown as on (its default); the shared test setUp turns it off. Only
        // the key is written, recording stays off.
        Preferences.learningEnabled = true
        let fake = FakeNeuralScorer()
        fakes.append(fake)
        let selector = AFMNeuralSelector(scoreBlockForTesting: fake.block, loadDelay: 0)
        selector.enabled = true
        selector.startLoadingIfNeeded()
        let deadline = Date().addingTimeInterval(5)
        while selector.state == .loading && Date() < deadline {
            RunLoop.current.run(until: Date().addingTimeInterval(0.01))
        }
        AFMNeuralSelector.shared = selector
    }

    override func tearDownWithError() throws {
        AFMPreciseController.shared.enabled = false
        AFMPreciseController.resetForTesting()
        AFMNeuralSelector.resetForTesting()
        Preferences.preciseModeEnabled = savedPrecise
        Preferences.neuralWalkEnabled = savedNeural
        Preferences.learningEnabled = false
        for dir in tempDirectories {
            try? FileManager.default.removeItem(atPath: dir)
        }
    }

    @MainActor
    func render(localization: String) throws -> Data {
        guard let path = Bundle.main.path(forResource: localization, ofType: "lproj"),
            let bundle = Bundle(path: path)
        else {
            throw XCTSkip("no \(localization).lproj in the app bundle")
        }
        let localizer: (String) -> String = { bundle.localizedString(forKey: $0, value: nil, table: nil) }
        let model = PreferencesViewModel()
        let root = SelectionModelPreferencesView()
            .environmentObject(model)
            .environment(\.preferencesLocalizer, localizer)
            .frame(width: preferencesWindowWidth, alignment: .topLeading)
            .fixedSize(horizontal: false, vertical: true)
            .background(Color(nsColor: .windowBackgroundColor))
        let hosting = NSHostingView(rootView: root)
        hosting.appearance = NSAppearance(named: .aqua)
        hosting.frame = NSRect(origin: .zero, size: hosting.fittingSize)
        let window = NSWindow(
            contentRect: hosting.frame, styleMask: [.borderless], backing: .buffered, defer: false)
        window.appearance = NSAppearance(named: .aqua)
        window.contentView = hosting
        hosting.layoutSubtreeIfNeeded()
        RunLoop.current.run(until: Date().addingTimeInterval(0.2))
        let bounds = hosting.bounds
        XCTAssertGreaterThan(bounds.height, 200)
        guard let rep = hosting.bitmapImageRepForCachingDisplay(in: bounds) else {
            XCTFail("no bitmap")
            return Data()
        }
        hosting.cacheDisplay(in: bounds, to: rep)
        return rep.representation(using: .png, properties: [:]) ?? Data()
    }

    func configure(_ precise: String) {
        AFMPreciseController.shared.enabled = false
        switch precise {
        case "loading":
            let heavy = FakeHeavyScorer()
            fakes.append(heavy)
            let dir = (NSTemporaryDirectory() as NSString).appendingPathComponent("qwen-shot-\(UUID().uuidString)")
            try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
            FileManager.default.createFile(
                atPath: (dir as NSString).appendingPathComponent("precise.conf"), contents: Data(kForcedOpenPreciseConf.utf8))
            tempDirectories.append(dir)
            let controller = AFMPreciseController(modelDirectory: dir, heavyScoreBlockForTesting: heavy.block, loadDelay: 60)
            AFMPreciseController.shared = controller
            Preferences.preciseModeEnabled = true
            controller.enabled = true
            XCTAssertEqual(controller.state, .loading)
        case "notfound":
            let controller = AFMPreciseController(modelDirectory: NSTemporaryDirectory() + "/no-qwen-\(UUID().uuidString)")
            AFMPreciseController.shared = controller
            Preferences.preciseModeEnabled = true
            controller.enabled = true
            let deadline = Date().addingTimeInterval(5)
            while controller.state == .loading && Date() < deadline {
                RunLoop.current.run(until: Date().addingTimeInterval(0.01))
            }
            XCTAssertEqual(controller.state, .notFound)
        default:
            AFMPreciseController.resetForTesting()
            Preferences.preciseModeEnabled = false
        }
    }

    @MainActor
    func testRenderSelectionModelTab() throws {
        let outputDirectory = ProcessInfo.processInfo.environment["AFM_PREFS_SCREENSHOT_DIR"]
        for precise in ["off", "loading", "notfound"] {
            configure(precise)
            for localization in ["zh-Hant", "en"] {
                let png = try render(localization: localization)
                XCTAssertGreaterThan(png.count, 1000)
                if let outputDirectory {
                    try FileManager.default.createDirectory(atPath: outputDirectory, withIntermediateDirectories: true)
                    let url = URL(fileURLWithPath: outputDirectory)
                        .appendingPathComponent("selection-model-\(localization)-precise-\(precise).png")
                    try png.write(to: url)
                }
            }
        }
    }
}
