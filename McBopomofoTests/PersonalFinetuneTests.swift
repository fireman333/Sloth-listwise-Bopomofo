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

import SwiftUI
import CryptoKit
import XCTest

@testable import McBopomofo

// MARK: - Task 5.6: retraction marker (personal-finetune design D7)

class RetractionMarkerTests: LearningTestCase {

    var client = FakeTextClient()

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

    @discardableResult
    func send(_ input: KeyHandlerInput, to imc: McBopomofoInputMethodController) -> Bool {
        let handled = imc.keyHandler.handle(input: input, state: imc.state) { newState in
            imc.handle(state: newState, client: imc.currentClient)
        } errorCallback: {}
        imc.requestPreciseIfNeeded()
        return handled
    }

    func type(_ keys: String, into imc: McBopomofoInputMethodController) {
        for k in keys.map({ String($0) }) {
            send(key(k), to: imc)
        }
    }

    var backspace: KeyHandlerInput { key("\u{8}", keyCode: 51, charCode: 8) }
    var forwardDelete: KeyHandlerInput { key("\u{F728}", keyCode: KeyCode.delete.rawValue, charCode: 0xF728) }

    /// Commits 中華 at `now`; returns the controller.
    func commitZhongHua() throws -> McBopomofoInputMethodController {
        let imc = try makeController()
        type(kZhongHuaKeys, into: imc)
        send(enterKey, to: imc)
        XCTAssertEqual(client.inserted, ["中華"])
        return imc
    }

    func retractLines() -> [Double] {
        poolRecords().compactMap { $0["retract"] as? Double }
    }

    func sentenceRecords() -> [[String: Any]] {
        poolRecords().filter { $0["retract"] == nil }
    }

    func testBackspaceWithinWindowMarksRecord() throws {
        let imc = try commitZhongHua()
        let t = try XCTUnwrap(sentenceRecords().first?["t"] as? Double)
        now += 2
        XCTAssertFalse(send(backspace, to: imc), "Backspace is still passed through to the application")
        XCTAssertEqual(retractLines(), [t])
        XCTAssertEqual(sentenceRecords().count, 1)
        // A second Backspace does not mark again.
        XCTAssertFalse(send(backspace, to: imc))
        XCTAssertEqual(retractLines(), [t])
        XCTAssertEqual(client.inserted, ["中華"], "committed text unchanged")
    }

    func testForwardDeleteMarksRecord() throws {
        let imc = try commitZhongHua()
        now += 9.5
        XCTAssertFalse(send(forwardDelete, to: imc))
        XCTAssertEqual(retractLines().count, 1)
    }

    func testNewCompositionFirstMarksNothing() throws {
        let imc = try commitZhongHua()
        type("5j/", into: imc)  // ㄓㄨㄥ
        XCTAssertTrue(send(backspace, to: imc), "Backspace edits the composing buffer")
        // Buffer empty again, but the window ended with the first key.
        send(backspace, to: imc)
        XCTAssertEqual(retractLines(), [])
    }

    func testOtherKeyFirstMarksNothing() throws {
        let imc = try commitZhongHua()
        send(key("\u{F702}", keyCode: KeyCode.left.rawValue, charCode: 0xF702), to: imc)
        send(backspace, to: imc)
        XCTAssertEqual(retractLines(), [])
    }

    func testAfterWindowMarksNothing() throws {
        let imc = try commitZhongHua()
        now += 15
        XCTAssertFalse(send(backspace, to: imc))
        XCTAssertEqual(retractLines(), [])
    }

    func testOtherClientMarksNothing() throws {
        store.clientProcessIdentifier = 4242
        let imc = try commitZhongHua()
        store.clientProcessIdentifier = 4343
        now += 1
        XCTAssertFalse(send(backspace, to: imc))
        XCTAssertEqual(retractLines(), [])
    }

    func testSecureInputMarksNothing() throws {
        let imc = try commitZhongHua()
        secure = true
        now += 1
        XCTAssertFalse(send(backspace, to: imc))
        secure = false
        XCTAssertEqual(retractLines(), [])
    }

    func testLearningOffMarksNothing() throws {
        let imc = try commitZhongHua()
        Preferences.setLearningEnabled(false)
        now += 1
        XCTAssertFalse(send(backspace, to: imc))
        XCTAssertEqual(retractLines(), [])
        Preferences.setLearningEnabled(true)
    }

    func testNotRecordedCommitMarksNothing() throws {
        Preferences.setLearningEnabled(false)
        let imc = try commitZhongHua()
        Preferences.setLearningEnabled(true)
        now += 1
        XCTAssertFalse(send(backspace, to: imc))
        XCTAssertEqual(poolRecords().count, 0)
    }

    /// The marker adds an O(1) check to the key path; Backspace right after a commit (including the queued append)
    /// stays far below 1 ms at p95.
    func testKeyPathCostOfMarker() throws {
        let imc = try makeController()
        var withMarker: [Double] = []
        var plain: [Double] = []
        for i in 0..<200 {
            type(kZhongHuaKeys, into: imc)
            send(enterKey, to: imc)
            now += 1
            let t0 = DispatchTime.now().uptimeNanoseconds
            let handled = send(backspace, to: imc)
            let dt = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
            XCTAssertFalse(handled)
            withMarker.append(dt)
            let t1 = DispatchTime.now().uptimeNanoseconds
            send(backspace, to: imc)  // window already ended: plain pass-through
            plain.append(Double(DispatchTime.now().uptimeNanoseconds - t1) / 1e6)
            if i % 50 == 0 { store.waitUntilIdleForTesting() }
        }
        XCTAssertEqual(retractLines().count, 200)
        func p95(_ a: [Double]) -> Double { a.sorted()[Int(Double(a.count) * 0.95)] }
        print("retraction marker: Backspace p95 \(p95(withMarker)) ms (plain \(p95(plain)) ms)")
        XCTAssertLessThan(p95(withMarker), 1.0)
        XCTAssertLessThan(p95(withMarker) - p95(plain), 1.0)
    }
}

// MARK: - Shared fixtures for the personalization controller (design D16)

/// A fake AFMPersonalTrainer: a shell script whose behavior is read from files next to it, writing the same
/// files as the real helper (status.json, staging/<run>/{weights.bin, precise-r.conf, report.json}).
let kFakeHelperScript = #"""
    #!/bin/sh
    DIR=$(cd "$(dirname "$0")" && pwd)
    echo "$@" > "$DIR/args.txt"
    while [ $# -gt 0 ]; do
      case "$1" in
        --memory-dir) MEM="$2"; shift 2;;
        --run-id) RUN="$2"; shift 2;;
        --generation) GEN="$2"; shift 2;;
        --alpha) ALPHA="$2"; shift 2;;
        --accept-yan-for-ai) YAN="$2"; shift 2;;
        --mode) MODE="$2"; shift 2;;
        *) shift;;
      esac
    done
    BEHAVIOR=$(cat "$DIR/behavior")
    DELAY=$(cat "$DIR/delay" 2>/dev/null || echo 0)
    RP="$MEM/r-personal"
    ST="$RP/staging/$RUN"
    /bin/mkdir -p "$ST"
    OFFSET=$(/usr/bin/wc -c < "$MEM/sentences.jsonl" 2>/dev/null | /usr/bin/tr -d ' ')
    [ -z "$OFFSET" ] && OFFSET=0
    if [ "$YAN" = "1" ]; then YANJ=true; else YANJ=false; fi
    status() {
      printf '{"run":"%s","phase":"%s","step":%s,"steps":10,"eligible":1200,"started":"x","pid":%s,"generation":%s}\n' "$RUN" "$1" "$2" "$$" "$GEN" > "$RP/status.json.tmp"
      /bin/mv "$RP/status.json.tmp" "$RP/status.json"
    }
    report() {
      printf '{"run":"%s","generation":%s,"result":"%s","exit_code":%s,"scan_offset":%s,"eligible":1200,"weights_sha256":"%s","eval_config":{"factory_sha256":"%s","alpha":%s,"accept_yan_for_ai":%s,"pack_version":"%s","calibration_version":"%s"},"gate":{"delta_personal_pt":5.0,"delta_general_pt":-0.4,"p":0.01,"factory_personal_pct":30.0,"candidate_personal_pct":35.0}}\n' \
        "$RUN" "$GEN" "$1" "$2" "$OFFSET" "$(cat "$DIR/weights_sha")" "$(cat "$DIR/factory_sha")" "$ALPHA" "$YANJ" "$(cat "$DIR/pack_version")" "$(cat "$DIR/pack_version")" > "$ST/report.json"
    }
    status train 3
    case "$BEHAVIOR" in
      adopt)
        /bin/cp "$DIR/weights.bin" "$ST/weights.bin"
        printf 'run = %s\nbucket_upper =\ntemperatures = 2.5\n' "$RUN" > "$ST/precise-r.conf"
        /bin/sleep "$DELAY"
        report adopted 0; status done 0; exit 0;;
      adoptlate)
        trap '' TERM
        /bin/cp "$DIR/weights.bin" "$ST/weights.bin"
        printf 'run = %s\nbucket_upper =\ntemperatures = 2.5\n' "$RUN" > "$ST/precise-r.conf"
        /bin/sleep "$DELAY"
        report adopted 0; status done 0; exit 0;;
      reject) report not_significant 0; status done 0; exit 0;;
      regress) report general_regression 0; exit 0;;
      crash) kill -SEGV $$;;
      hang) trap '' TERM; /bin/sleep 600 & wait $!; /bin/sleep 600;;
      slow) trap 'report cancelled 5; exit 5' TERM; /bin/sleep 600 & wait $!; exit 0;;
      latewrite) trap '( /bin/sleep 1; echo late > "$ST/late.txt"; echo late > "$RP/late.txt" ) & exit 5' TERM; /bin/sleep 600 & wait $!; exit 0;;
      fail) report failed:train 7; exit 7;;
      deferred) report deferred 4; exit 4;;
      *) exit 2;;
    esac
    """#

class PersonalTrainingTestCase: LearningTestCase {
    var root = ""
    var modelDirectory = ""
    var helperDirectory = ""
    var controller: AFMPersonalTrainingController!
    var finished: [String] = []
    var ac = true
    var lowPower = false
    var thermal = false
    var idle: Double = 3600
    var learning = true
    var poolCounter = 0

    override func setUpWithError() throws {
        try super.setUpWithError()
        root = (NSTemporaryDirectory() as NSString).appendingPathComponent("afm-pf-\(UUID().uuidString)")
        modelDirectory = (root as NSString).appendingPathComponent("r-walk")
        helperDirectory = (root as NSString).appendingPathComponent("helper")
        let fm = FileManager.default
        try fm.createDirectory(atPath: modelDirectory + "/onnx", withIntermediateDirectories: true)
        try fm.createDirectory(atPath: modelDirectory + "/tokenizer", withIntermediateDirectories: true)
        try fm.createDirectory(atPath: modelDirectory + "/personalize", withIntermediateDirectories: true)
        try fm.createDirectory(atPath: helperDirectory, withIntermediateDirectories: true)
        try writeFactoryModel(seed: 1)
        // Vocabulary of the fake pool: 40 single-letter syllables / characters.
        var syllables: [String: Int] = [:]
        var chars: [String: Int] = [:]
        for (i, pair) in Self.pairs.enumerated() {
            syllables[pair.0] = i
            chars[pair.1] = i
        }
        try JSONSerialization.data(withJSONObject: syllables).write(
            to: URL(fileURLWithPath: modelDirectory + "/syl_vocab.json"))
        try JSONSerialization.data(withJSONObject: ["model": ["vocab": chars]]).write(
            to: URL(fileURLWithPath: modelDirectory + "/tokenizer/tokenizer.json"))
        try writePack()
        let script = helperDirectory + "/AFMPersonalTrainer"
        try kFakeHelperScript.write(toFile: script, atomically: true, encoding: .utf8)
        try fm.setAttributes([.posixPermissions: 0o755], ofItemAtPath: script)
        try setBehavior("adopt")
        try "test-pack-1".write(toFile: helperDirectory + "/pack_version", atomically: true, encoding: .utf8)
        try prepareWeights()

        let c = AFMPersonalTrainingController(
            memoryDirectory: memoryDirectory, modelDirectory: modelDirectory, helperPath: script)
        c.clock = { [unowned self] in self.now }
        c.learningEnabledCheck = { [unowned self] in self.learning }
        let probes = AFMPersonalSystemProbes()
        probes.onACPower = { [unowned self] in self.ac }
        probes.lowPowerMode = { [unowned self] in self.lowPower }
        probes.thermalPressure = { [unowned self] in self.thermal }
        probes.idleSeconds = { [unowned self] in self.idle }
        c.probes = probes
        c.runDidFinishHandler = { [unowned self] in self.finished.append($0) }
        controller = c
        AFMPersonalTrainingController.shared = c
    }

    override func tearDownWithError() throws {
        if let c = controller, c.isRunning {
            kill(c.helperProcessIdentifierForTesting, SIGKILL)
            waitUntil { !c.isRunning }
        }
        controller?.waitUntilIdleForTesting()
        AFMPersonalTrainingController.resetForTesting()
        try super.tearDownWithError()
        try? FileManager.default.removeItem(atPath: root)
    }

    static let pairs: [(String, String)] = {
        let letters = Array("ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄐㄑㄒㄓㄔㄕㄖㄗㄘㄙㄚㄛㄜㄝㄞㄟㄠㄡㄢㄣㄤㄥㄦㄧㄨㄩ")
        var result: [(String, String)] = []
        for (i, l) in letters.enumerated() {
            result.append((String(l), String(UnicodeScalar(0x4E00 + i * 7)!)))
        }
        return result
    }()

    func writeFactoryModel(seed: UInt8) throws {
        var bytes = [UInt8](repeating: 0, count: 4096)
        for i in 0..<bytes.count { bytes[i] = UInt8((i * 31 + Int(seed) * 17) & 0xFF) }
        try Data(bytes).write(to: URL(fileURLWithPath: modelDirectory + "/onnx/model_quantized.onnx"))
    }

    func writePack(version: String = "test-pack-1") throws {
        try JSONSerialization.data(withJSONObject: ["pack_version": version]).write(
            to: URL(fileURLWithPath: modelDirectory + "/personalize/pack.json"))
    }

    var factorySHA: String { AFMFactoryModelSHA256(modelDirectory) ?? "" }

    func setBehavior(_ behavior: String, delay: Double = 0) throws {
        try behavior.write(toFile: helperDirectory + "/behavior", atomically: true, encoding: .utf8)
        try String(delay).write(toFile: helperDirectory + "/delay", atomically: true, encoding: .utf8)
    }

    /// weights.bin bound to the current factory model, and the hashes the fake report carries.
    func prepareWeights(corrupt: Bool = false) throws {
        let path = helperDirectory + "/weights.bin"
        XCTAssertTrue(AFMWriteFakeWeightsFileForTesting(path, factorySHA, 84))
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        try factorySHA.write(toFile: helperDirectory + "/factory_sha", atomically: true, encoding: .utf8)
        try sha256Hex(data).write(toFile: helperDirectory + "/weights_sha", atomically: true, encoding: .utf8)
        if corrupt {
            try data.prefix(data.count - 40).write(to: URL(fileURLWithPath: path))
        }
    }

    func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// A model folder under the test root with the clean_v1 export's files (never the installed model) and a pack.
    func makeRealModelDirectory() throws -> String {
        let source = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("train/ranker/export/clean_v1").path
        guard FileManager.default.fileExists(atPath: source + "/onnx/model_quantized.onnx") else {
            throw XCTSkip("R model not found at \(source)")
        }
        let dir = (root as NSString).appendingPathComponent("real-r-walk")
        let fm = FileManager.default
        try fm.createDirectory(atPath: dir + "/onnx", withIntermediateDirectories: true)
        try fm.createDirectory(atPath: dir + "/personalize", withIntermediateDirectories: true)
        for item in ["config.json", "phonetic_table.tsv", "syl_vocab.json", "tokenizer"] {
            try fm.createSymbolicLink(atPath: dir + "/" + item, withDestinationPath: source + "/" + item)
        }
        try fm.createSymbolicLink(
            atPath: dir + "/onnx/model_quantized.onnx", withDestinationPath: source + "/onnx/model_quantized.onnx")
        try JSONSerialization.data(withJSONObject: ["pack_version": "test-pack-1"]).write(
            to: URL(fileURLWithPath: dir + "/personalize/pack.json"))
        return dir
    }

    /// Writes `count` distinct eligible 2-syllable records (times now - count ... now - 1) to the pool.
    @discardableResult
    func writePool(_ count: Int, startTime: Double? = nil) -> [Double] {
        var lines = ""
        var times: [Double] = []
        let n = Self.pairs.count
        let t0 = startTime ?? (now - Double(count))
        for k in 0..<count {
            // Distinct 3-syllable segments: the digits of a running counter in base n.
            let id = poolCounter
            poolCounter += 1
            let a = Self.pairs[id % n]
            let b = Self.pairs[(id / n) % n]
            let c = Self.pairs[(id / n / n) % n]
            let t = t0 + Double(k)
            times.append(t)
            lines += String(
                format: "{\"v\":1,\"t\":%.3f,\"manual\":false,\"walk\":\"r\",\"app\":\"t\",\"model\":\"m\",\"nodes\":[[\"%@\",\"%@\",\"w\"],[\"%@-%@\",\"%@%@\",\"w\"]]}\n",
                t, a.0, a.1, b.0, c.0, b.1, c.1)
        }
        try? FileManager.default.createDirectory(atPath: memoryDirectory, withIntermediateDirectories: true)
        if let handle = FileHandle(forWritingAtPath: poolFile) {
            handle.seekToEndOfFile()
            handle.write(Data(lines.utf8))
            handle.closeFile()
        } else {
            FileManager.default.createFile(atPath: poolFile, contents: Data(lines.utf8))
        }
        return times
    }

    func refresh() {
        var done = false
        controller.refresh { done = true }
        waitUntil { done }
    }

    func waitForRun(timeout: TimeInterval = 20) -> String? {
        let before = finished.count
        waitUntil(timeout: timeout) { self.finished.count > before }
        controller.waitUntilIdleForTesting()
        return finished.count > before ? finished.last : nil
    }

    func startAndWait(_ behavior: String = "adopt") throws -> String? {
        try setBehavior(behavior)
        XCTAssertEqual(controller.startManualRun(), .started)
        return waitForRun()
    }

    var personal: String { (memoryDirectory as NSString).appendingPathComponent("r-personal") }
    func exists(_ relative: String) -> Bool {
        FileManager.default.fileExists(atPath: (personal as NSString).appendingPathComponent(relative))
    }
    func stagingEntries() -> [String] {
        (try? FileManager.default.contentsOfDirectory(atPath: personal + "/staging")) ?? []
    }
    func manifestOnDisk() -> [String: Any]? {
        controller.waitUntilIdleForTesting()
        guard let data = FileManager.default.contents(atPath: personal + "/manifest.json") else { return nil }
        return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    }
}

// MARK: - Task 5.1: controller (design D6, D16)

class PersonalTrainingControllerTests: PersonalTrainingTestCase {

    func testEligibleCountFromPoolScan() {
        writePool(150)
        refresh()
        XCTAssertEqual(controller.eligibleCount, 150)
        XCTAssertTrue(controller.packAvailable)
        XCTAssertEqual(controller.packVersion, "test-pack-1")
        XCTAssertEqual(controller.factorySHA, factorySHA)
        // Appended records (after queued appends) and a retraction are seen by the next scan.
        let times = writePool(10, startTime: now)
        store.appendRetraction(forTimestamp: times[0])
        refresh()
        XCTAssertEqual(controller.eligibleCount, 159)
    }

    func testManualRunAdoptedEndToEnd() throws {
        writePool(1100)
        refresh()
        XCTAssertEqual(controller.eligibleCount, 1100)
        var notifications = 0
        let token = NotificationCenter.default.addObserver(
            forName: .AFMPersonalTrainingControllerDidChange, object: controller, queue: nil
        ) { _ in notifications += 1 }
        defer { NotificationCenter.default.removeObserver(token) }
        try setBehavior("adopt", delay: 0.5)
        XCTAssertEqual(controller.startManualRun(), .started)
        XCTAssertTrue(controller.isRunning)
        XCTAssertFalse(controller.currentRunIsAutomatic)
        XCTAssertEqual(controller.startManualRun(), .alreadyRunning)
        // Progress from status.json while running.
        var status: [AnyHashable: Any]?
        waitUntil { self.exists("status.json") }
        controller.readStatus { status = $0 }
        waitUntil { status != nil }
        XCTAssertEqual(status?["phase"] as? String, "train")
        XCTAssertEqual(status?["run"] as? String, controller.currentRunIdentifier)
        XCTAssertEqual(waitForRun(), "adopted")
        XCTAssertFalse(controller.isRunning)
        XCTAssertGreaterThanOrEqual(notifications, 2)
        // Arguments of the spawn (design D5 note: helper CLI).
        let args = try String(contentsOfFile: helperDirectory + "/args.txt", encoding: .utf8)
        XCTAssertTrue(args.contains("--mode manual"))
        XCTAssertTrue(args.contains("--generation 0"))
        XCTAssertTrue(args.contains("--accept-yan-for-ai 0"))
        XCTAssertTrue(args.contains("--alpha 0.5"))
        XCTAssertTrue(args.contains("--memory-dir \(memoryDirectory)"))
        XCTAssertFalse(args.contains("--active-weights"))
        // Adoption (design D13).
        XCTAssertEqual(controller.activeVersion, 1)
        XCTAssertNil(controller.previousVersion)
        XCTAssertTrue(exists("v1/weights.bin"))
        XCTAssertTrue(exists("v1/precise-r.conf"))
        XCTAssertTrue(exists("v1/report.json"))
        XCTAssertEqual(stagingEntries(), [])
        let manifest = try XCTUnwrap(manifestOnDisk())
        XCTAssertEqual(manifest["active"] as? Int, 1)
        XCTAssertEqual(controller.lastRun?["result"] as? String, "adopted")
        XCTAssertEqual(controller.lastRun?["delta_personal_pt"] as? Double, 5.0)
        XCTAssertEqual(controller.lastRun?["delta_general_pt"] as? Double, -0.4)
        // Modes: dirs 0700, files 0600.
        let attrs = try FileManager.default.attributesOfItem(atPath: personal + "/manifest.json")
        XCTAssertEqual((attrs[.posixPermissions] as? NSNumber)?.intValue, 0o600)
        let dirAttrs = try FileManager.default.attributesOfItem(atPath: personal + "/v1")
        XCTAssertEqual((dirAttrs[.posixPermissions] as? NSNumber)?.intValue, 0o700)
        let request = try XCTUnwrap(controller.activeVersionRequest())
        XCTAssertEqual(request.version, 1)
        XCTAssertEqual(request.evalConfig["factory_sha256"] as? String, factorySHA)
        // The next run gets the active version for gate rule 3.
        XCTAssertEqual(try startAndWait("reject"), "not_significant")
        XCTAssertTrue(try String(contentsOfFile: helperDirectory + "/args.txt", encoding: .utf8).contains("--active-weights"))
    }

    func testRejectedRunKeepsModel() throws {
        writePool(1100)
        refresh()
        XCTAssertEqual(try startAndWait("reject"), "not_significant")
        XCTAssertNil(controller.activeVersion)
        XCTAssertFalse(exists("v1"))
        XCTAssertEqual(stagingEntries(), [])
        XCTAssertEqual(controller.lastRun?["result"] as? String, "not_significant")
        XCTAssertEqual(controller.consecutiveFailures, 0)
    }

    func testCrashReportedAsFailure() throws {
        writePool(1100)
        refresh()
        XCTAssertEqual(try startAndWait("crash"), "failed:crash")
        XCTAssertNil(controller.activeVersion)
        XCTAssertEqual(stagingEntries(), [])
        XCTAssertEqual(controller.consecutiveFailures, 1)
    }

    func testHangKilledAtTheLimit() throws {
        writePool(1100)
        refresh()
        controller.runTimeLimit = 0.5
        controller.killGracePeriod = 0.5
        try setBehavior("hang")
        XCTAssertEqual(controller.startManualRun(), .started)
        let pid = controller.helperProcessIdentifierForTesting
        let t0 = Date()
        XCTAssertEqual(waitForRun(), "failed:timeout")
        XCTAssertLessThan(Date().timeIntervalSince(t0), 5)
        XCTAssertNotEqual(kill(pid, 0), 0, "SIGTERM ignored -> SIGKILL")
        XCTAssertEqual(stagingEntries(), [])
    }

    func testCancel() throws {
        writePool(1100)
        refresh()
        try setBehavior("slow")
        XCTAssertEqual(controller.startManualRun(), .started)
        waitUntil { self.exists("status.json") }
        controller.cancelRun()
        XCTAssertEqual(waitForRun(), "cancelled")
        XCTAssertNil(controller.activeVersion)
        XCTAssertEqual(stagingEntries(), [])
        XCTAssertEqual(controller.consecutiveFailures, 0, "a cancel is not a failure")
    }

    func testNothingRunsWithLearningOffOrTooFewSegments() throws {
        writePool(600)
        refresh()
        XCTAssertEqual(controller.eligibleCount, 600)
        XCTAssertEqual(controller.startManualRun(), .notEnoughData)
        XCTAssertEqual(controller.automaticBlockReason(), .notEnoughData)
        writePool(500, startTime: now)
        refresh()
        learning = false
        XCTAssertEqual(controller.startManualRun(), .learningOff)
        XCTAssertEqual(controller.automaticBlockReason(), .learningOff)
        XCTAssertFalse(controller.runAutomaticAttemptIfPossible())
        XCTAssertEqual(controller.spawnCountForTesting, 0)
    }

    func testMissingPackReported() throws {
        try FileManager.default.removeItem(atPath: modelDirectory + "/personalize/pack.json")
        writePool(1100)
        refresh()
        XCTAssertFalse(controller.packAvailable)
        XCTAssertEqual(controller.startManualRun(), .packMissing)
        XCTAssertEqual(controller.automaticBlockReason(), .packMissing)
        XCTAssertEqual(controller.spawnCountForTesting, 0)
    }

    func testThreeFailuresPauseAutomaticRuns() throws {
        writePool(1100)
        refresh()
        for i in 1...3 {
            XCTAssertEqual(try startAndWait("fail"), "failed:train")
            XCTAssertEqual(controller.consecutiveFailures, i)
        }
        XCTAssertTrue(controller.failurePaused)
        XCTAssertEqual(controller.automaticBlockReason(), .failurePaused)
        // 「立即學習」 resumes.
        XCTAssertEqual(try startAndWait("reject"), "not_significant")
        XCTAssertFalse(controller.failurePaused)
        XCTAssertEqual(controller.consecutiveFailures, 0)
    }

    func testHelperMissing() {
        writePool(1100)
        refresh()
        controller.helperPath = root + "/no-such-helper"
        XCTAssertEqual(controller.startManualRun(), .helperMissing)
    }
}

// MARK: - Task 5.2: scheduler and conditions (design D6)

class PersonalTrainingSchedulerTests: PersonalTrainingTestCase {

    override func setUpWithError() throws {
        try super.setUpWithError()
        writePool(1500)
        refresh()
        try setBehavior("reject")
    }

    func testEachConditionBlocks() {
        XCTAssertEqual(controller.automaticBlockReason(), .none, "no attempt yet on this factory model")
        ac = false
        XCTAssertEqual(controller.automaticBlockReason(), .onBattery)
        ac = true
        lowPower = true
        XCTAssertEqual(controller.automaticBlockReason(), .lowPowerMode)
        lowPower = false
        thermal = true
        XCTAssertEqual(controller.automaticBlockReason(), .thermal)
        thermal = false
        idle = 599
        XCTAssertEqual(controller.automaticBlockReason(), .notIdle)
        idle = 600
        XCTAssertEqual(controller.automaticBlockReason(), .none)
        learning = false
        XCTAssertEqual(controller.automaticBlockReason(), .learningOff)
        learning = true
        XCTAssertEqual(controller.spawnCountForTesting, 0)
    }

    func testAllMetSpawnsOnceThen24hAnd200NewRules() throws {
        XCTAssertTrue(controller.runAutomaticAttemptIfPossible())
        XCTAssertEqual(controller.spawnCountForTesting, 1)
        XCTAssertTrue(controller.currentRunIsAutomatic)
        XCTAssertFalse(controller.runAutomaticAttemptIfPossible(), "running")
        XCTAssertEqual(waitForRun(), "not_significant")
        XCTAssertTrue(try String(contentsOfFile: helperDirectory + "/args.txt", encoding: .utf8).contains("--mode auto"))
        XCTAssertEqual(controller.spawnCountForTesting, 1)
        // 24 h rule.
        now += 23 * 3600
        XCTAssertEqual(controller.automaticBlockReason(), .tooSoon)
        now += 2 * 3600
        // >= 200 new eligible segments since the last attempt.
        XCTAssertEqual(controller.automaticBlockReason(), .notEnoughNew)
        writePool(199, startTime: now)
        refresh()
        XCTAssertEqual(controller.automaticBlockReason(), .notEnoughNew)
        writePool(1, startTime: now + 500)
        refresh()
        XCTAssertEqual(controller.automaticBlockReason(), .none)
        XCTAssertTrue(controller.runAutomaticAttemptIfPossible())
        XCTAssertEqual(waitForRun(), "not_significant")
        XCTAssertEqual(controller.spawnCountForTesting, 2)
    }

    func testManualRunIgnoresPowerIdleAndTimeRules() throws {
        XCTAssertTrue(controller.runAutomaticAttemptIfPossible())
        XCTAssertEqual(waitForRun(), "not_significant")
        ac = false
        lowPower = true
        thermal = true
        idle = 0
        XCTAssertNotEqual(controller.automaticBlockReason(), .none)
        XCTAssertEqual(controller.startManualRun(), .started, "power, idle, 24 h and new-segment rules ignored")
        XCTAssertEqual(waitForRun(), "not_significant")
        learning = false
        XCTAssertEqual(controller.startManualRun(), .learningOff)
    }

    func testConfigurationChangeWaivesNewSegmentRule() throws {
        XCTAssertTrue(controller.runAutomaticAttemptIfPossible())
        XCTAssertEqual(waitForRun(), "not_significant")
        now += 25 * 3600
        XCTAssertEqual(controller.automaticBlockReason(), .notEnoughNew)
        Preferences.acceptYanReadingForAi = true
        defer { Preferences.acceptYanReadingForAi = false }
        XCTAssertEqual(controller.automaticBlockReason(), .none, "configuration not attempted yet")
    }

    func testSchedulerWiring() throws {
        controller.startScheduler()
        defer { controller.stopScheduler() }
        let scheduler = try XCTUnwrap(controller.scheduler)
        XCTAssertTrue(scheduler.identifier.hasSuffix(".personal-finetune"))
        XCTAssertEqual(scheduler.interval, 24 * 3600)
        XCTAssertEqual(scheduler.tolerance, 6 * 3600)
        XCTAssertTrue(scheduler.repeats)
        XCTAssertEqual(scheduler.qualityOfService, .background)
        // The scheduled block: blocked -> deferred, all met -> one spawn, finished.
        ac = false
        var results: [NSBackgroundActivityScheduler.Result] = []
        controller.handleScheduledActivity { results.append($0) }
        waitUntil { results.count == 1 }
        XCTAssertEqual(results, [.deferred])
        XCTAssertEqual(controller.spawnCountForTesting, 0)
        ac = true
        controller.handleScheduledActivity { results.append($0) }
        waitUntil { results.count == 2 }
        XCTAssertEqual(results, [.deferred, .finished])
        XCTAssertEqual(controller.spawnCountForTesting, 1)
        XCTAssertEqual(waitForRun(), "not_significant")
    }
}

// MARK: - Task 5.3: adoption and versions (design D13)

class PersonalVersionTests: PersonalTrainingTestCase {

    override func setUpWithError() throws {
        try super.setUpWithError()
        writePool(1100)
        refresh()
    }

    func versionDirectories() -> [String] {
        ((try? FileManager.default.contentsOfDirectory(atPath: personal)) ?? []).filter { $0.hasPrefix("v") }.sorted()
    }

    func testAdoptionsKeepActiveAndPrevious() throws {
        XCTAssertEqual(try startAndWait(), "adopted")
        XCTAssertEqual(try startAndWait(), "adopted")
        XCTAssertEqual(controller.activeVersion, 2)
        XCTAssertEqual(controller.previousVersion, 1)
        XCTAssertEqual(versionDirectories(), ["v1", "v2"])
        XCTAssertEqual(try startAndWait(), "adopted")
        XCTAssertEqual(controller.activeVersion, 3)
        XCTAssertEqual(controller.previousVersion, 2)
        XCTAssertEqual(versionDirectories(), ["v2", "v3"], "older versions are deleted")
        let manifest = try XCTUnwrap(manifestOnDisk())
        XCTAssertEqual(manifest["active"] as? Int, 3)
        XCTAssertEqual(manifest["previous"] as? Int, 2)
        XCTAssertEqual(Set(((manifest["versions"] as? [String: Any]) ?? [:]).keys), ["2", "3"])
        XCTAssertGreaterThan(controller.createdTime(ofVersion: 3), 0)
    }

    func testCorruptStagingNotAdopted() throws {
        try prepareWeights(corrupt: true)
        XCTAssertEqual(try startAndWait(), "failed:adopt")
        XCTAssertNil(controller.activeVersion)
        XCTAssertEqual(versionDirectories(), [])
        XCTAssertEqual(stagingEntries(), [])
    }

    func testWeightsForAnotherFactoryModelNotAdopted() throws {
        try writeFactoryModel(seed: 9)  // the staged file still names the old factory model
        XCTAssertEqual(try startAndWait(), "failed:adopt")
        XCTAssertEqual(versionDirectories(), [])
    }

    func testFactoryModelChange() throws {
        XCTAssertEqual(try startAndWait(), "adopted")
        XCTAssertEqual(try startAndWait(), "adopted")
        let poolBefore = FileManager.default.contents(atPath: poolFile)
        XCTAssertEqual(controller.automaticBlockReason(), .notEnoughNew)
        try writeFactoryModel(seed: 2)
        refresh()
        XCTAssertEqual(controller.factorySHA, factorySHA)
        XCTAssertNil(controller.activeVersion)
        XCTAssertNil(controller.previousVersion)
        XCTAssertEqual(versionDirectories(), [], "versions deleted")
        XCTAssertEqual(FileManager.default.contents(atPath: poolFile), poolBefore, "pool untouched")
        XCTAssertEqual(manifestOnDisk()?["factory_sha"] as? String, factorySHA)
        XCTAssertEqual(controller.automaticBlockReason(), .none, "next automatic run without 200 new segments")
    }

    /// Verify C4 (task 9.2): the version kept as `reverted_previous` after 「改回出廠模型」 is deleted too
    /// when the factory model changes.
    func testFactoryModelChangeAfterRevertDeletesAllVersions() throws {
        XCTAssertEqual(try startAndWait(), "adopted")
        XCTAssertEqual(try startAndWait(), "adopted")
        controller.revertToFactory()
        controller.waitUntilIdleForTesting()
        XCTAssertEqual(versionDirectories(), ["v1", "v2"])
        try writeFactoryModel(seed: 2)
        refresh()
        controller.waitUntilIdleForTesting()
        XCTAssertEqual(versionDirectories(), [], "no version of the old factory model remains")
        XCTAssertNil(manifestOnDisk()?["reverted_previous"])
        XCTAssertNil(controller.previousVersion)
    }

    /// Verify C5 (task 9.2): a run that finishes after learning was switched off is not adopted.
    func testRunFinishingAfterLearningOffNotAdopted() throws {
        try setBehavior("adopt", delay: 1.5)
        XCTAssertEqual(controller.startManualRun(), .started)
        waitUntil { self.exists("status.json") }
        learning = false
        XCTAssertEqual(waitForRun(), "cancelled")
        XCTAssertNil(controller.activeVersion)
        XCTAssertEqual(versionDirectories(), [])
        XCTAssertEqual(stagingEntries(), [])
    }

    /// A cancelled run whose helper still exits 0 with an adoptable result (SIGTERM ignored / too late)
    /// is not adopted (spec "Cancel").
    func testCancelledRunExitingZeroNotAdopted() throws {
        try setBehavior("adoptlate", delay: 1.5)
        XCTAssertEqual(controller.startManualRun(), .started)
        waitUntil { self.exists("status.json") }
        controller.cancelRun()
        XCTAssertEqual(waitForRun(), "cancelled")
        XCTAssertNil(controller.activeVersion)
        XCTAssertEqual(versionDirectories(), [])
    }

    /// The background-activity completion is called even when a clear lands during the refresh.
    func testRefreshCompletesWhenClearedMeanwhile() throws {
        var done = false
        controller.refresh { done = true }
        controller.clearAll(completion: nil)
        waitUntil { done }
        XCTAssertTrue(done)
    }

    func testRevertToFactoryAndUsePrevious() throws {
        XCTAssertEqual(try startAndWait(), "adopted")
        XCTAssertEqual(try startAndWait(), "adopted")
        controller.revertToFactory()
        XCTAssertNil(controller.activeVersion)
        XCTAssertNil(controller.activeVersionRequest())
        XCTAssertTrue(controller.paused)
        XCTAssertEqual(controller.automaticBlockReason(), .paused)
        XCTAssertFalse(controller.runAutomaticAttemptIfPossible())
        controller.waitUntilIdleForTesting()
        XCTAssertEqual(versionDirectories(), ["v1", "v2"], "both versions kept on disk")
        // 「改用上一版」 after reverting: the version in use before.
        controller.usePreviousVersion()
        XCTAssertEqual(controller.activeVersion, 2)
        XCTAssertEqual(controller.previousVersion, 1)
        XCTAssertTrue(controller.paused, "still paused until 「立即學習」")
        controller.usePreviousVersion()
        XCTAssertEqual(controller.activeVersion, 1)
        XCTAssertEqual(controller.previousVersion, 2)
        XCTAssertEqual(controller.activeVersionRequest()?.version, 1)
        XCTAssertEqual(manifestOnDisk()?["active"] as? Int, 1)
        // 「立即學習」 resumes automatic runs.
        XCTAssertEqual(try startAndWait("reject"), "not_significant")
        XCTAssertFalse(controller.paused)
    }

    func testRetractionAfterGateNotAdopted() throws {
        let times = writePool(5, startTime: now)
        refresh()
        try setBehavior("adopt", delay: 1.0)
        XCTAssertEqual(controller.startManualRun(), .started)
        waitUntil { self.exists("status.json") }
        // After the helper read the pool: a retraction of a used (newest) record.
        store.appendRetraction(forTimestamp: times[4])
        XCTAssertEqual(waitForRun(), "discarded:retracted")
        XCTAssertNil(controller.activeVersion)
        XCTAssertEqual(versionDirectories(), [])
        XCTAssertEqual(stagingEntries(), [])
    }

    func testRetractionOfUnusedRecordStillAdopts() throws {
        try setBehavior("adopt", delay: 1.0)
        XCTAssertEqual(controller.startManualRun(), .started)
        waitUntil { self.exists("status.json") }
        store.appendRetraction(forTimestamp: now + 1234)  // no such record
        XCTAssertEqual(waitForRun(), "adopted")
    }
}

// MARK: - Task 5.4: personalized load and hot swap (design D11)

class PersonalSelectorTests: PersonalTrainingTestCase {
    let factoryScorer = FakeNeuralScorer(favored: ["中"])
    let personalScorer = FakeNeuralScorer(favored: ["鐘"])
    var personalDelay: TimeInterval = 0
    var personalFails = false
    var selector: AFMNeuralSelector!
    private var savedYan: Any?
    private var savedAlpha: Any?

    override func setUpWithError() throws {
        try super.setUpWithError()
        savedYan = Preferences.defaults.object(forKey: "AFMAcceptYanReadingForAi")
        savedAlpha = Preferences.defaults.object(forKey: "AFMNeuralWalkAlpha")
        Preferences.acceptYanReadingForAi = false
        Preferences.neuralWalkAlpha = 0.5
        retainedFakes.append(factoryScorer)
        retainedFakes.append(personalScorer)
        let s = AFMNeuralSelector(
            scoreBlockForTesting: factoryScorer.block,
            personalBlockProvider: { [unowned self] _, error in
                if self.personalDelay > 0 { Thread.sleep(forTimeInterval: self.personalDelay) }
                if self.personalFails {
                    error.pointee = "weights.bin truncated"
                    return nil
                }
                return self.personalScorer.block
            }, loadDelay: 0)
        let sha = factorySHA
        s.currentEvalConfigProvider = { alpha, yan in AFMCurrentEvalConfig(sha, "test-pack-1", alpha, yan) }
        s.circuitBreakerDisabledForTesting = true
        selector = s
        AFMNeuralSelector.shared = s
        s.syncWithPreferences()
        waitUntil { s.state == .ready }
        XCTAssertEqual(s.personalStatus, .factory)
        writePool(1100)
        refresh()
    }

    override func tearDownWithError() throws {
        selector?.waitForLoadQueueForTesting()
        for (key, value) in [("AFMAcceptYanReadingForAi", savedYan), ("AFMNeuralWalkAlpha", savedAlpha)] {
            if let value { Preferences.defaults.set(value, forKey: key) } else { Preferences.defaults.removeObject(forKey: key) }
        }
        try super.tearDownWithError()
    }

    func activate() {
        selector.syncWithPreferences()
    }

    func waitLoaded() {
        waitUntil { !self.selector.isReloading }
        RunLoop.current.run(until: Date().addingTimeInterval(0.05))
    }

    func composeZhong() -> String? {
        composing(type(kZhongKeys, into: makeHandler()))
    }

    func testTypingDuringReloadUsesOldScorerThenNewVersionAfterActivation() throws {
        XCTAssertEqual(composeZhong(), "中")
        XCTAssertEqual(try startAndWait(), "adopted")
        // Adoption alone changes nothing until the next activation.
        XCTAssertEqual(composeZhong(), "中")
        XCTAssertEqual(selector.loadedPersonalVersion, 0)
        personalDelay = 1.0
        activate()
        XCTAssertTrue(selector.isReloading)
        XCTAssertEqual(selector.personalStatus, .loading)
        XCTAssertEqual(selector.state, .ready, "the current scorer stays ready")
        // (a) keystrokes keep being handled with the old scorer while the new one loads.
        let handler = makeHandler()
        var state: InputState = InputState.Empty()
        var maxMs = 0.0
        for _ in 0..<5 {
            for k in kZhongKeys.map({ String($0) }) {
                let t0 = DispatchTime.now().uptimeNanoseconds
                XCTAssertTrue(handler.handle(input: key(k), state: state) { state = $0 } errorCallback: {})
                maxMs = max(maxMs, Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6)
            }
            XCTAssertTrue((composing(state) ?? "").hasSuffix("中"))
        }
        XCTAssertTrue(selector.isReloading, "typing finished before the load")
        XCTAssertLessThan(maxMs, 50)
        // (b) after the load: the new version.
        waitLoaded()
        XCTAssertEqual(selector.personalStatus, .active)
        XCTAssertEqual(selector.loadedPersonalVersion, 1)
        XCTAssertEqual(selector.modelIdentifier, "test+p1")
        XCTAssertEqual(selector.freedScorerCount, 1, "old scorer freed on the load queue")
        XCTAssertEqual(composeZhong(), "鐘")
        // Pool records name the version.
        XCTAssertNotNil(typeAndCommit(kZhongKeys))
        let record = try XCTUnwrap(poolRecords().last { $0["retract"] == nil && ($0["walk"] as? String) == "r" })
        XCTAssertEqual(record["model"] as? String, "test+p1")
        // Same version at the next activation: no reload.
        let loads = selector.loadCount
        activate()
        XCTAssertEqual(selector.loadCount, loads)
    }

    func testUnusableVersionFallsBackToFactoryOnce() throws {
        XCTAssertEqual(try startAndWait(), "adopted")
        personalFails = true
        activate()
        waitLoaded()
        XCTAssertEqual(selector.personalStatus, .unusable)
        XCTAssertEqual(selector.personalError, "weights.bin truncated")
        XCTAssertEqual(selector.loadedPersonalVersion, 0)
        XCTAssertEqual(selector.modelIdentifier, "test")
        XCTAssertEqual(selector.state, .ready)
        XCTAssertEqual(composeZhong(), "中")
        // Not retried (one log line) at later activations.
        let loads = selector.loadCount
        activate()
        activate()
        XCTAssertEqual(selector.loadCount, loads)
    }

    func testReadingPolicyAndAlphaBinding() throws {
        XCTAssertEqual(try startAndWait(), "adopted")
        activate()
        waitLoaded()
        XCTAssertEqual(selector.personalStatus, .active)
        // Reading policy toggled -> factory weights at the next activation, version kept.
        Preferences.acceptYanReadingForAi = true
        activate()
        waitLoaded()
        XCTAssertEqual(selector.personalStatus, .mismatch)
        XCTAssertEqual(selector.modelIdentifier, "test")
        XCTAssertEqual(composeZhong(), "中")
        XCTAssertEqual(controller.activeVersion, 1, "version kept")
        XCTAssertEqual(controller.automaticBlockReason(), .none, "re-run allowed without new segments")
        Preferences.acceptYanReadingForAi = false
        activate()
        waitLoaded()
        XCTAssertEqual(selector.personalStatus, .active)
        XCTAssertEqual(composeZhong(), "鐘")
        // Alpha changed with defaults -> same.
        Preferences.neuralWalkAlpha = 0.6
        activate()
        waitLoaded()
        XCTAssertEqual(selector.personalStatus, .mismatch)
        Preferences.neuralWalkAlpha = 0.5
        activate()
        waitLoaded()
        XCTAssertEqual(selector.personalStatus, .active)
    }

    func testRevertTakesEffectAtActivation() throws {
        XCTAssertEqual(try startAndWait(), "adopted")
        activate()
        waitLoaded()
        XCTAssertEqual(composeZhong(), "鐘")
        controller.revertToFactory()
        XCTAssertEqual(composeZhong(), "鐘", "until the next activation")
        activate()
        waitLoaded()
        XCTAssertEqual(selector.personalStatus, .factory)
        XCTAssertEqual(composeZhong(), "中")
        XCTAssertEqual(selector.freedScorerCount, 2)
    }

    /// Many swaps while walking: run under Address Sanitizer (task 5.4 d) to show no freed scorer is used.
    func testRepeatedSwapsWhileWalking() throws {
        XCTAssertEqual(try startAndWait(), "adopted")
        for i in 0..<20 {
            if i % 2 == 0 { controller.usePreviousVersion() }
            if i % 4 == 0 { controller.revertToFactory() } else if controller.activeVersion == nil { controller.usePreviousVersion() }
            activate()
            for _ in 0..<3 {
                _ = composeZhong()
                RunLoop.current.run(until: Date().addingTimeInterval(0.002))
            }
            waitLoaded()
            _ = composeZhong()
        }
        selector.waitForLoadQueueForTesting()
        XCTAssertGreaterThan(selector.freedScorerCount, 5)
        XCTAssertEqual(selector.state, .ready)
    }
}

// MARK: - Task 5.4 (c): real model, corrupt personalized weights

class PersonalRealModelTests: PersonalTrainingTestCase {

    func installVersion(modelDirectory dir: String, scale: Double, truncate: Bool) throws -> AFMPersonalVersionRequest {
        let versionDir = (personal as NSString).appendingPathComponent("v1")
        try FileManager.default.createDirectory(atPath: versionDir, withIntermediateDirectories: true)
        let weights = versionDir + "/weights.bin"
        XCTAssertTrue(AFMWriteFactoryWeightsFileForTesting(dir, weights, scale))
        if truncate {
            let data = try Data(contentsOf: URL(fileURLWithPath: weights))
            try data.prefix(data.count / 2).write(to: URL(fileURLWithPath: weights))
        }
        let sha = try XCTUnwrap(AFMFactoryModelSHA256(dir))
        return AFMPersonalVersionRequest(
            version: 1, directory: versionDir, runIdentifier: "run-real",
            evalConfig: AFMCurrentEvalConfig(sha, "test-pack-1", 0.5, false), generation: controller.generation,
            created: now)
    }

    func loadSelector(_ dir: String, request: AFMPersonalVersionRequest?) -> AFMNeuralSelector {
        let selector = AFMNeuralSelector(modelDirectory: dir)
        selector.personalVersionProvider = { request }
        AFMNeuralSelector.shared = selector
        selector.syncWithPreferences()
        waitUntil(timeout: 30) { selector.state != .loading }
        return selector
    }

    func testCorruptWeightsUseFactory() throws {
        let dir = try makeRealModelDirectory()
        let request = try installVersion(modelDirectory: dir, scale: 1.0, truncate: true)
        let selector = loadSelector(dir, request: request)
        XCTAssertEqual(selector.state, .ready)
        XCTAssertEqual(selector.personalStatus, .unusable)
        XCTAssertNotNil(selector.personalError)
        XCTAssertFalse(selector.modelIdentifier?.contains("+p") ?? true)
        XCTAssertEqual(composing(type(kZhongKeys, into: makeHandler())), "中")
    }

    func testPersonalizedWeightsLoaded() throws {
        let dir = try makeRealModelDirectory()
        let request = try installVersion(modelDirectory: dir, scale: 1.0, truncate: false)
        let selector = loadSelector(dir, request: request)
        XCTAssertEqual(selector.state, .ready)
        XCTAssertEqual(selector.personalStatus, .active)
        XCTAssertTrue(selector.modelIdentifier?.hasSuffix("+p1") ?? false)
        XCTAssertGreaterThan(selector.neuralWalkCount + 1, 0)
        XCTAssertNotNil(composing(type("5j/ cj86", into: makeHandler())))
    }
}

// MARK: - Task 5.5: precise-mode overlay (design D12)

let kOverlayTestPreciseConf = """
    alpha = 0.5
    k = 10
    tau = 0.5
    r_weight = 0
    beta = 1
    qwen_timeout_ms = 1000

    """

class PersonalPreciseOverlayTests: LearningTestCase {
    var dirs: [String] = []

    override func tearDownWithError() throws {
        try super.tearDownWithError()
        for d in dirs { try? FileManager.default.removeItem(atPath: d) }
    }

    func tempDir() throws -> String {
        let d = (NSTemporaryDirectory() as NSString).appendingPathComponent("afm-overlay-\(UUID().uuidString)")
        try FileManager.default.createDirectory(atPath: d, withIntermediateDirectories: true)
        dirs.append(d)
        return d
    }

    func makeController() throws -> AFMPreciseController {
        Preferences.preciseModeEnabled = true
        let heavy = FakeHeavyScorer()
        retainedFakes.append(heavy)
        let dir = try tempDir()
        FileManager.default.createFile(
            atPath: (dir as NSString).appendingPathComponent("precise.conf"), contents: Data(kOverlayTestPreciseConf.utf8))
        let controller = AFMPreciseController(modelDirectory: dir, heavyScoreBlockForTesting: heavy.block, loadDelay: 0)
        controller.startDelay = 0
        AFMPreciseController.shared = controller
        controller.enabled = true
        waitUntil { controller.state != .loading }
        XCTAssertEqual(controller.state, .ready)
        return controller
    }

    func version(_ run: String, overlay: String?) throws -> AFMPersonalVersionRequest {
        let dir = try tempDir()
        if let overlay {
            try overlay.write(toFile: dir + "/precise-r.conf", atomically: true, encoding: .utf8)
        }
        return AFMPersonalVersionRequest(
            version: 1, directory: dir, runIdentifier: run, evalConfig: [:], generation: 0, created: 0)
    }

    func select(_ request: AFMPersonalVersionRequest?, on controller: AFMPreciseController) {
        let before = controller.overlayUpdateCount
        controller.setPersonalVersion(request)
        waitUntil { controller.overlayUpdateCount > before }
    }

    /// Types 我是誰 (su3cl3) with the precise hook after every key; returns requests submitted.
    func requestsWhileTyping(_ controller: AFMPreciseController) -> Int {
        let before = controller.submittedCount
        let handler = KeyHandler()
        let delegate = PreciseTestDelegate()
        handler.delegate = delegate
        handler.inputMode = .bopomofo
        handler.syncWithPreferences()
        for k in "su3cl3".map({ String($0) }) {
            handler.handle(input: key(k), state: delegate.state) { delegate.state = $0 } errorCallback: {}
            handler.requestPreciseIfNeeded(composingBuffer: (delegate.state as? InputState.Inputting)?.composingBuffer)
        }
        waitUntil { controller.isIdle }
        return Int(controller.submittedCount) - Int(before)
    }

    func testOverlayTemperaturesDriveTheGate() throws {
        let selector = installFakeSelector(FakeNeuralScorer())
        let controller = try makeController()
        XCTAssertEqual(controller.configTemperaturesForTesting, [1.0])
        // Sharp temperature: p_R[0] ~ 1 -> gate closed.
        select(try version("run-a", overlay: "run = run-a\nbucket_upper =\ntemperatures = 0.02\n"), on: controller)
        XCTAssertTrue(controller.isPersonalOverlayApplied)
        XCTAssertEqual(controller.configTemperaturesForTesting, [0.02])
        XCTAssertEqual(controller.state, .ready)
        XCTAssertEqual(requestsWhileTyping(controller), 0)
        // Flat temperature: p_R[0] small -> gate open.
        select(try version("run-b", overlay: "run = run-b\nbucket_upper =\ntemperatures = 50\n"), on: controller)
        XCTAssertEqual(controller.configTemperaturesForTesting, [50.0])
        XCTAssertGreaterThan(requestsWhileTyping(controller), 0)
        XCTAssertGreaterThan(selector.neuralWalkCount, 0)
    }

    func testMissingOrForeignOverlayFailsPreciseOnly() throws {
        let selector = installFakeSelector(FakeNeuralScorer())
        let controller = try makeController()
        select(try version("run-a", overlay: nil), on: controller)
        XCTAssertEqual(controller.state, .failed)
        XCTAssertFalse(controller.isPersonalOverlayApplied)
        // R-walk continues.
        let walks = selector.neuralWalkCount
        XCTAssertNotNil(composing(type(kZhongHuaKeys, into: makeHandler())))
        XCTAssertGreaterThan(selector.neuralWalkCount, walks)
        XCTAssertEqual(requestsWhileTyping(controller), 0)
        // Overlay of another run (version) -> failed too.
        select(try version("run-a", overlay: "run = run-z\nbucket_upper =\ntemperatures = 2\n"), on: controller)
        XCTAssertEqual(controller.state, .failed)
        // Bucket layout differing from precise.conf -> failed.
        select(try version("run-c", overlay: "run = run-c\nbucket_upper = 7 10\ntemperatures = 1 1 1\n"), on: controller)
        XCTAssertEqual(controller.state, .failed)
        // Back to the factory R: factory calibration, ready again.
        select(nil, on: controller)
        XCTAssertEqual(controller.state, .ready)
        XCTAssertEqual(controller.configTemperaturesForTesting, [1.0])
        XCTAssertFalse(controller.isPersonalOverlayApplied)
    }

    func testFactoryRUsesQwenCalibrationUnchanged() throws {
        installFakeSelector(FakeNeuralScorer())
        let controller = try makeController()
        select(nil, on: controller)
        XCTAssertEqual(controller.state, .ready)
        XCTAssertEqual(controller.configTemperaturesForTesting, [1.0])
        // Overlay selected before the model finished loading is applied when it becomes ready.
        controller.enabled = false
        select(try version("run-a", overlay: "run = run-a\nbucket_upper =\ntemperatures = 3\n"), on: controller)
        controller.enabled = true
        waitUntil { controller.state != .loading }
        XCTAssertEqual(controller.state, .ready)
        XCTAssertEqual(controller.configTemperaturesForTesting, [3.0])
    }
}

// MARK: - Task 5.7: clearing (design D13)

class PersonalClearTests: PersonalTrainingTestCase {

    func testClearDuringRunKillsHelperAndLeavesNothing() throws {
        writePool(1100)
        refresh()
        try setBehavior("latewrite")
        XCTAssertEqual(controller.startManualRun(), .started)
        waitUntil { self.exists("status.json") }
        let pid = controller.helperProcessIdentifierForTesting
        LanguageModelManager.clearLearningData()
        XCTAssertEqual(controller.generation, 1)
        XCTAssertNil(controller.activeVersionRequest())
        waitUntil { !self.controller.isRunning && !FileManager.default.fileExists(atPath: self.personal) }
        XCTAssertEqual(finished.last, "discarded:generation")
        XCTAssertNotEqual(kill(pid, 0), 0, "helper killed")
        XCTAssertFalse(FileManager.default.fileExists(atPath: personal))
        // The helper's late write (into the deleted staging dir) fails: nothing is recreated.
        spin(1.5)
        XCTAssertFalse(FileManager.default.fileExists(atPath: personal))
        XCTAssertFalse(FileManager.default.fileExists(atPath: poolFile))
        XCTAssertEqual(controller.eligibleCount, 0)
    }

    func testAdoptableResultDuringClearIsNotAdopted() throws {
        writePool(1100)
        refresh()
        try setBehavior("adopt", delay: 0.3)
        XCTAssertEqual(controller.startManualRun(), .started)
        store.suspendQueueForTesting()
        // The helper exits 0 with an adoptable result; its adoption waits on the store queue.
        waitUntil { !self.controller.isRunning }
        LanguageModelManager.clearLearningData()
        store.resumeQueueForTesting()
        waitUntil { self.finished.count == 1 && !FileManager.default.fileExists(atPath: self.personal) }
        XCTAssertEqual(finished, ["discarded:generation"])
        XCTAssertNil(controller.activeVersion)
        XCTAssertFalse(FileManager.default.fileExists(atPath: personal), "no file of the run remains")
        spin(0.3)
        XCTAssertFalse(FileManager.default.fileExists(atPath: personal))
    }

    func testPersonalizedLoadInFlightDuringClearIsDiscarded() throws {
        writePool(1100)
        refresh()
        let factoryScorer = FakeNeuralScorer(favored: ["中"])
        let personalScorer = FakeNeuralScorer(favored: ["鐘"])
        retainedFakes.append(factoryScorer)
        retainedFakes.append(personalScorer)
        let selector = AFMNeuralSelector(
            scoreBlockForTesting: factoryScorer.block,
            personalBlockProvider: { _, _ in
                Thread.sleep(forTimeInterval: 0.8)
                return personalScorer.block
            }, loadDelay: 0)
        let sha = factorySHA
        selector.currentEvalConfigProvider = { alpha, yan in AFMCurrentEvalConfig(sha, "test-pack-1", alpha, yan) }
        AFMNeuralSelector.shared = selector
        selector.syncWithPreferences()
        waitUntil { selector.state == .ready }
        XCTAssertEqual(try startAndWait(), "adopted")
        selector.syncWithPreferences()
        XCTAssertTrue(selector.isReloading)
        LanguageModelManager.clearLearningData()
        waitUntil { !selector.isReloading }
        selector.waitForLoadQueueForTesting()
        spin(0.1)
        XCTAssertEqual(selector.discardedLoadCount, 1, "the personalized load completing after the clear is freed")
        XCTAssertEqual(selector.personalStatus, .factory)
        XCTAssertEqual(selector.loadedPersonalVersion, 0)
        XCTAssertEqual(composing(type(kZhongKeys, into: makeHandler())), "中")
        selector.syncWithPreferences()
        XCTAssertFalse(selector.isReloading)
    }

    /// Integration on the test root with the real R model: an adopted, loaded version; clear -> r-personal gone,
    /// factory weights after reactivation, memory and pool gone.
    func testClearIntegrationWithRealModel() throws {
        let dir = try makeRealModelDirectory()
        let fm = FileManager.default
        // Fake helper output bound to the real factory model.
        let sha = try XCTUnwrap(AFMFactoryModelSHA256(dir))
        XCTAssertTrue(AFMWriteFactoryWeightsFileForTesting(dir, helperDirectory + "/weights.bin", 1.05))
        try sha.write(toFile: helperDirectory + "/factory_sha", atomically: true, encoding: .utf8)
        try sha256Hex(try Data(contentsOf: URL(fileURLWithPath: helperDirectory + "/weights.bin")))
            .write(toFile: helperDirectory + "/weights_sha", atomically: true, encoding: .utf8)
        let c = AFMPersonalTrainingController(
            memoryDirectory: memoryDirectory, modelDirectory: dir, helperPath: helperDirectory + "/AFMPersonalTrainer")
        c.clock = { [unowned self] in self.now }
        c.learningEnabledCheck = { true }
        c.minimumEligible = 0
        c.runDidFinishHandler = { [unowned self] in self.finished.append($0) }
        controller = c
        AFMPersonalTrainingController.shared = c
        refresh()
        XCTAssertEqual(try startAndWait(), "adopted")

        let selector = AFMNeuralSelector(modelDirectory: dir)
        AFMNeuralSelector.shared = selector
        selector.syncWithPreferences()
        waitUntil(timeout: 30) { selector.state != .loading }
        XCTAssertEqual(selector.state, .ready)
        XCTAssertEqual(selector.personalStatus, .active)
        XCTAssertTrue(selector.modelIdentifier?.hasSuffix("+p1") ?? false)
        // Some learning data: a commit with a manual pick (memory + pool).
        XCTAssertNotNil(typeAndCommit(kZhongHuaKeys, picks: [("ㄏㄨㄚˊ", "滑")]))
        store.flush()
        store.waitUntilIdleForTesting()
        XCTAssertTrue(fm.fileExists(atPath: poolFile))
        XCTAssertTrue(fm.fileExists(atPath: personal + "/v1/weights.bin"))

        LanguageModelManager.clearLearningData()
        waitUntil(timeout: 30) { !fm.fileExists(atPath: self.personal) && !selector.isReloading }
        selector.waitForLoadQueueForTesting()
        spin(0.1)
        XCTAssertFalse(fm.fileExists(atPath: personal))
        XCTAssertFalse(fm.fileExists(atPath: poolFile))
        XCTAssertFalse(fm.fileExists(atPath: memoryFile))
        XCTAssertEqual(selector.personalStatus, .factory)
        // Factory weights after reactivation.
        selector.syncWithPreferences()
        XCTAssertFalse(selector.isReloading)
        XCTAssertFalse(selector.modelIdentifier?.contains("+p") ?? true)
        XCTAssertEqual(selector.loadedPersonalVersion, 0)
        XCTAssertEqual(composing(type(kZhongKeys, into: makeHandler())), "中")
    }
}

// MARK: - Task 6.1: 「個人化模型」 preferences section (design D14)

/// Records the buttons' calls and reports injected state.
final class SpyPersonalTrainingController: AFMPersonalTrainingController {
    var calls: [String] = []
    var spyRunning = false
    var spyActive: NSNumber?
    var spyPrevious: NSNumber?
    var spyStatus: [AnyHashable: Any]?
    var refreshCount = 0

    override var isRunning: Bool { spyRunning }
    override var activeVersion: NSNumber? { spyActive }
    override var previousVersion: NSNumber? { spyPrevious }
    override func startManualRun() -> AFMPersonalStartResult {
        calls.append("start")
        return .started
    }
    override func cancelRun() { calls.append("cancel") }
    override func revertToFactory() { calls.append("revert") }
    override func usePreviousVersion() { calls.append("previous") }
    override func refreshEligibleCount(completion: (() -> Void)?) {
        refreshCount += 1
        completion?()
    }
    override func readStatus(completion: @escaping ([AnyHashable: Any]?) -> Void) {
        completion(spyStatus)
    }
}

@MainActor
class PersonalizationPreferencesTests: PersonalTrainingTestCase {

    func localizer(_ localization: String) throws -> (String) -> String {
        guard let path = Bundle.main.path(forResource: localization, ofType: "lproj"),
            let bundle = Bundle(path: path)
        else {
            throw XCTSkip("no \(localization).lproj in the app bundle")
        }
        return { bundle.localizedString(forKey: $0, value: nil, table: nil) }
    }

    func makeSpy() -> SpyPersonalTrainingController {
        let spy = SpyPersonalTrainingController(
            memoryDirectory: memoryDirectory, modelDirectory: modelDirectory, helperPath: nil)
        retainedFakes.append(spy)
        return spy
    }

    func testEnablementRules() {
        let model = PreferencesViewModel()
        let spy = makeSpy()
        model.personalTraining = spy
        // Fresh: factory model in use, 「改回出廠模型」 and 「改用上一版」 disabled.
        Preferences.setLearningEnabled(true)
        XCTAssertTrue(model.learnNowEnabled)
        XCTAssertFalse(model.useFactoryModelEnabled)
        XCTAssertFalse(model.usePreviousVersionEnabled)
        XCTAssertFalse(model.personalRunning)
        // Learning off -> 「立即學習」 disabled.
        model.learningEnabled = false
        XCTAssertFalse(model.learnNowEnabled)
        model.learningEnabled = true
        // Running -> 「取消」 replaces 「立即學習」.
        spy.spyRunning = true
        XCTAssertTrue(model.personalRunning)
        XCTAssertFalse(model.learnNowEnabled)
        spy.spyRunning = false
        // A personalized version in use -> 「改回出廠模型」 enabled.
        spy.spyActive = 2
        XCTAssertTrue(model.useFactoryModelEnabled)
        XCTAssertFalse(model.usePreviousVersionEnabled)
        // A previous version exists -> 「改用上一版」 enabled.
        spy.spyPrevious = 1
        XCTAssertTrue(model.usePreviousVersionEnabled)
        spy.spyActive = nil
        XCTAssertFalse(model.useFactoryModelEnabled)
        XCTAssertTrue(model.usePreviousVersionEnabled)
    }

    func testButtonsCallTheController() {
        let model = PreferencesViewModel()
        let spy = makeSpy()
        model.personalTraining = spy
        XCTAssertEqual(model.learnNow(), .started)
        model.cancelLearning()
        model.useFactoryModel()
        model.usePreviousVersion()
        XCTAssertEqual(spy.calls, ["start", "cancel", "revert", "previous"])
    }

    func testPollingOnlyWhileVisible() throws {
        let model = PreferencesViewModel()
        let spy = makeSpy()
        spy.spyRunning = true
        spy.spyStatus = ["phase": "train", "step": 3, "steps": 10]
        model.personalTraining = spy
        model.personalPollInterval = 0.1
        // Shown in a window -> polling; removed -> stopped.
        let root = PersonalizationSection().environmentObject(model)
        let hosting = NSHostingView(rootView: root)
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 500, height: 300), styleMask: [.borderless], backing: .buffered,
            defer: false)
        window.contentView = hosting
        window.orderFront(nil)
        spin(0.35)
        XCTAssertTrue(model.isPollingPersonalization)
        XCTAssertGreaterThanOrEqual(spy.refreshCount, 2)
        XCTAssertEqual(model.personalRunStatus?["phase"] as? String, "train")
        let lines = model.personalizationLines(localize: try localizer("zh-Hant"))
        XCTAssertEqual(lines.progress, "學習中：訓練 30%")
        window.contentView = nil
        window.orderOut(nil)
        spin(0.2)
        XCTAssertFalse(model.isPollingPersonalization)
        let count = spy.refreshCount
        spin(0.35)
        XCTAssertEqual(spy.refreshCount, count, "no polling while hidden")
        // Explicit start / stop.
        model.startPersonalizationPolling()
        XCTAssertTrue(model.isPollingPersonalization)
        model.stopPersonalizationPolling()
        XCTAssertFalse(model.isPollingPersonalization)
    }

    func testLinesWithTheRealController() throws {
        let zh = try localizer("zh-Hant")
        let en = try localizer("en")
        writePool(600)
        refresh()
        let model = PreferencesViewModel()
        model.personalTraining = controller
        var lines = model.personalizationLines(localize: zh)
        XCTAssertEqual(lines.status, "使用中：出廠模型")
        XCTAssertEqual(lines.eligible, "可用句段：600（至少 1,000 才會學習）")
        XCTAssertNil(lines.lastRun)
        // Too little data: the tab says how many are needed and available.
        XCTAssertEqual(model.learnNow(), .notEnoughData)
        lines = model.personalizationLines(localize: zh)
        XCTAssertEqual(lines.message, "資料不足：需要 1,000 個句段，目前 600 個")
        XCTAssertEqual(
            model.personalizationLines(localize: en).message, "Not enough data: 1,000 segments needed, 600 available")
        // Adopted run.
        writePool(500, startTime: now)
        refresh()
        XCTAssertEqual(model.learnNow(), .started)
        XCTAssertEqual(waitForRun(), "adopted")
        lines = model.personalizationLines(localize: zh)
        XCTAssertNil(lines.message)
        let day = { () -> String in
            let f = DateFormatter()
            f.locale = Locale(identifier: "en_US_POSIX")
            f.dateFormat = "yyyy-MM-dd"
            return f.string(from: Date(timeIntervalSince1970: self.now))
        }()
        XCTAssertEqual(lines.lastRun, "上次學習：\(day) — 已採用（個人測試 +5.0 分、一般測試 −0.4 分）")
        XCTAssertEqual(lines.pending, "下次切換到本輸入法時改用第 1 版")
        XCTAssertTrue(model.useFactoryModelEnabled)
        // Rejected run.
        try setBehavior("regress")
        XCTAssertEqual(model.learnNow(), .started)
        XCTAssertEqual(waitForRun(), "general_regression")
        XCTAssertEqual(
            model.personalizationLines(localize: zh).lastRun,
            "上次學習：\(day) — 未採用：一般準確率下降過多（個人測試 +5.0 分、一般測試 −0.4 分）")
        // Revert: paused note.
        model.useFactoryModel()
        lines = model.personalizationLines(localize: zh)
        XCTAssertEqual(lines.paused, "自動學習已暫停")
        XCTAssertFalse(model.useFactoryModelEnabled)
        XCTAssertTrue(model.usePreviousVersionEnabled)
        // Missing pack.
        try FileManager.default.removeItem(atPath: modelDirectory + "/personalize/pack.json")
        refresh()
        XCTAssertEqual(model.personalizationLines(localize: zh).status, "缺少個人化資料包")
    }

    func testSelectorStatusLines() throws {
        let zh = try localizer("zh-Hant")
        writePool(1100)
        refresh()
        XCTAssertEqual(try startAndWait(), "adopted")
        let scorer = FakeNeuralScorer()
        retainedFakes.append(scorer)
        var fails = false
        let selector = AFMNeuralSelector(
            scoreBlockForTesting: scorer.block,
            personalBlockProvider: { _, error in
                if fails {
                    error.pointee = "bad"
                    return nil
                }
                return scorer.block
            }, loadDelay: 0)
        let sha = factorySHA
        selector.currentEvalConfigProvider = { alpha, yan in AFMCurrentEvalConfig(sha, "test-pack-1", alpha, yan) }
        AFMNeuralSelector.shared = selector
        selector.syncWithPreferences()
        waitUntil { selector.state == .ready }
        let model = PreferencesViewModel()
        model.personalTraining = controller
        model.personalSelector = { selector }
        let day = { () -> String in
            let f = DateFormatter()
            f.locale = Locale(identifier: "en_US_POSIX")
            f.dateFormat = "yyyy-MM-dd"
            return f.string(from: Date(timeIntervalSince1970: self.now))
        }()
        XCTAssertEqual(model.personalizationLines(localize: zh).status, "使用中：個人化模型 第 1 版（\(day)）")
        XCTAssertNil(model.personalizationLines(localize: zh).pending)
        _ = fails
        controller.revertToFactory()
        XCTAssertEqual(model.personalizationLines(localize: zh).pending, "下次切換到本輸入法時改用出廠模型")
    }

    func testRenderSectionZhHant() throws {
        let zh = try localizer("zh-Hant")
        writePool(1100)
        refresh()
        XCTAssertEqual(try startAndWait(), "adopted")
        XCTAssertEqual(try startAndWait(), "adopted")
        let model = PreferencesViewModel()
        model.personalTraining = controller
        let root = SelectionModelPreferencesView()
            .environmentObject(model)
            .environment(\.preferencesLocalizer, zh)
            .frame(width: 560, alignment: .topLeading)
            .fixedSize(horizontal: false, vertical: true)
            .background(Color(nsColor: .windowBackgroundColor))
        let hosting = NSHostingView(rootView: root)
        hosting.appearance = NSAppearance(named: .aqua)
        hosting.frame = NSRect(origin: .zero, size: hosting.fittingSize)
        let window = NSWindow(contentRect: hosting.frame, styleMask: [.borderless], backing: .buffered, defer: false)
        window.appearance = NSAppearance(named: .aqua)
        window.contentView = hosting
        hosting.layoutSubtreeIfNeeded()
        spin(0.3)
        let rep = try XCTUnwrap(hosting.bitmapImageRepForCachingDisplay(in: hosting.bounds))
        hosting.cacheDisplay(in: hosting.bounds, to: rep)
        let png = try XCTUnwrap(rep.representation(using: .png, properties: [:]))
        XCTAssertGreaterThan(png.count, 1000)
        let outputDirectory =
            ProcessInfo.processInfo.environment["AFM_PREFS_SCREENSHOT_DIR"] ?? NSTemporaryDirectory()
        let url = URL(fileURLWithPath: outputDirectory).appendingPathComponent("selection-model-personalized-zh-Hant.png")
        try png.write(to: url)
        print("personalized section render: \(url.path)")
        XCTAssertTrue(model.isPollingPersonalization, "visible section polls")
        window.contentView = nil
    }
}

// MARK: - Codex review 2026-10-08 (VERIFY.md "Codex review 2026-10-08")

class CodexReviewFixTests: PersonalTrainingTestCase {

    func mode(_ path: String) -> Int {
        var st = stat()
        return lstat(path, &st) == 0 ? Int(st.st_mode & 0o7777) : -1
    }

    func versionDirectories() -> [String] {
        ((try? FileManager.default.contentsOfDirectory(atPath: personal)) ?? []).filter { $0.hasPrefix("v") }.sorted()
    }

    var clearMarker: String { (memoryDirectory as NSString).appendingPathComponent("r-personal.clearing") }

    /// A link target outside the memory folder. It lives in the (watcher-excluded) models folder: a link
    /// from memory/ to a target elsewhere in the watched root would make the user-phrase watcher, which
    /// resolves links, reload user phrases in later tests.
    func outsideFile(_ name: String) throws -> String {
        let dir = UserPhraseLocationHelper.modelsDirectory
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        let path = (dir as NSString).appendingPathComponent("codex-review-outside-\(name)")
        addTeardownBlock { try? FileManager.default.removeItem(atPath: path) }
        return path
    }

    /// Finding 1: a clear whose deletion fails never reports success, keeps the factory model, and a relaunch
    /// does not read the surviving manifest back; the retry finishes once the folder is deletable.
    func testFailedClearStaysFactoryAcrossRelaunchAndRetries() throws {
        writePool(1100)
        refresh()
        XCTAssertEqual(try startAndWait(), "adopted")
        XCTAssertEqual(controller.activeVersion, 1)
        controller.waitUntilIdleForTesting()
        controller.clearRetryInterval = 0.3
        XCTAssertEqual(chmod(personal, 0o500), 0)  // r-personal/ and its files cannot be deleted
        defer { chmod(personal, 0o700) }
        var cleared = false
        controller.clearAll { cleared = true }
        waitUntil { self.controller.clearIncomplete }
        XCTAssertTrue(controller.clearIncomplete)
        XCTAssertFalse(cleared, "an incomplete clear is never reported as done")
        XCTAssertTrue(exists("manifest.json"), "precondition: the deletion really failed")
        XCTAssertNil(controller.activeVersionRequest())
        XCTAssertEqual(controller.startManualRun(), .clearing)
        XCTAssertTrue(FileManager.default.fileExists(atPath: clearMarker))
        // Relaunch on the same folders: the old version is not resurrected.
        let relaunched = AFMPersonalTrainingController(
            memoryDirectory: memoryDirectory, modelDirectory: modelDirectory, helperPath: nil)
        relaunched.clearRetryInterval = 0.3
        XCTAssertNil(relaunched.activeVersion)
        XCTAssertNil(relaunched.activeVersionRequest())
        XCTAssertTrue(relaunched.clearIncomplete)
        // Deletable again: the retries finish the clear.
        chmod(personal, 0o700)
        waitUntil { cleared && !relaunched.clearIncomplete }
        XCTAssertTrue(cleared)
        XCTAssertFalse(controller.clearIncomplete)
        XCTAssertFalse(relaunched.clearIncomplete)
        relaunched.waitUntilIdleForTesting()
        XCTAssertFalse(FileManager.default.fileExists(atPath: personal))
        XCTAssertFalse(FileManager.default.fileExists(atPath: clearMarker))
        let again = AFMPersonalTrainingController(
            memoryDirectory: memoryDirectory, modelDirectory: modelDirectory, helperPath: nil)
        XCTAssertFalse(again.clearIncomplete)
    }

    /// Finding 2: an existing 0644 pool is narrowed to 0600 and the memory folder to 0700 on the next append.
    func testPoolFilePermissionsNarrowed() throws {
        let fm = FileManager.default
        try fm.createDirectory(atPath: memoryDirectory, withIntermediateDirectories: true)
        chmod(memoryDirectory, 0o755)
        fm.createFile(atPath: poolFile, contents: Data("{\"v\":1,\"retract\":1.000}\n".utf8))
        chmod(poolFile, 0o644)
        store.appendRetraction(forTimestamp: now)
        store.waitUntilIdleForTesting()
        XCTAssertEqual(mode(poolFile), 0o600)
        XCTAssertEqual(mode(memoryDirectory), 0o700)
        let text = try String(contentsOfFile: poolFile, encoding: .utf8)
        XCTAssertEqual(text.split(separator: "\n").count, 2, "the record was still appended")
    }

    /// Finding 2: a symlinked pool is refused; nothing is written outside the memory folder.
    func testPoolSymlinkRefused() throws {
        let fm = FileManager.default
        try fm.createDirectory(atPath: memoryDirectory, withIntermediateDirectories: true)
        let outside = try outsideFile("pool.jsonl")
        addTeardownBlock { [poolFile] in try? FileManager.default.removeItem(atPath: poolFile) }
        fm.createFile(atPath: outside, contents: Data())
        chmod(outside, 0o644)
        try fm.createSymbolicLink(atPath: poolFile, withDestinationPath: outside)
        store.appendRetraction(forTimestamp: now)
        store.waitUntilIdleForTesting()
        XCTAssertEqual(try Data(contentsOf: URL(fileURLWithPath: outside)).count, 0, "nothing written through the link")
        XCTAssertEqual(mode(outside), 0o644, "the link target's mode is untouched")
        XCTAssertEqual(try fm.destinationOfSymbolicLink(atPath: poolFile), outside)
    }

    /// Finding 2: the memory file and the personalization manifest are never written through a planted
    /// `.tmp` symlink, and end up 0600 in 0700 folders.
    func testMemoryAndManifestWritesDoNotFollowSymlinks() throws {
        let fm = FileManager.default
        let outside = try outsideFile("memory.txt")
        addTeardownBlock { [memoryFile, personal] in
            try? FileManager.default.removeItem(atPath: memoryFile + ".tmp")
            try? FileManager.default.removeItem(atPath: personal + "/manifest.json.tmp")
        }
        try "keep".write(toFile: outside, atomically: false, encoding: .utf8)
        try fm.createDirectory(atPath: memoryDirectory, withIntermediateDirectories: true)
        try fm.createSymbolicLink(atPath: memoryFile + ".tmp", withDestinationPath: outside)
        store.markDirty()
        store.flush()
        store.waitUntilIdleForTesting()
        XCTAssertEqual(try String(contentsOfFile: outside, encoding: .utf8), "keep")
        XCTAssertEqual(mode(memoryFile), 0o600)
        try fm.createDirectory(atPath: personal, withIntermediateDirectories: true)
        chmod(personal, 0o755)
        try fm.createSymbolicLink(atPath: personal + "/manifest.json.tmp", withDestinationPath: outside)
        controller.revertToFactory()  // writes manifest.json
        controller.waitUntilIdleForTesting()
        XCTAssertEqual(try String(contentsOfFile: outside, encoding: .utf8), "keep")
        XCTAssertEqual(mode(personal + "/manifest.json"), 0o600)
        XCTAssertEqual(mode(personal), 0o700)
    }

    /// Finding 3 (IME side): the helper gets its own deadline and the IME's pid.
    func testHelperGetsDeadlineAndParentPid() throws {
        writePool(1100)
        refresh()
        XCTAssertEqual(try startAndWait("reject"), "not_significant")
        let args = try String(contentsOfFile: helperDirectory + "/args.txt", encoding: .utf8)
        XCTAssertTrue(args.contains("--deadline-seconds 615"), args)
        XCTAssertTrue(args.contains("--parent-pid \(getpid())"), args)
    }

    /// Finding 4: a factory model rewritten with the same size and mtime is re-hashed and old versions
    /// are invalidated.
    func testFactoryReplacedKeepingSizeAndMtimeDetected() throws {
        writePool(1100)
        refresh()
        XCTAssertEqual(try startAndWait(), "adopted")
        let oldSHA = try XCTUnwrap(controller.factorySHA)
        let path = modelDirectory + "/onnx/model_quantized.onnx"
        var before = stat()
        XCTAssertEqual(stat(path, &before), 0)
        try writeFactoryModel(seed: 3)
        var times = [before.st_atimespec, before.st_mtimespec]
        XCTAssertEqual(utimensat(AT_FDCWD, path, &times, 0), 0)
        var after = stat()
        XCTAssertEqual(stat(path, &after), 0)
        XCTAssertEqual(after.st_size, before.st_size)
        XCTAssertEqual(after.st_mtimespec.tv_sec, before.st_mtimespec.tv_sec)
        XCTAssertEqual(after.st_mtimespec.tv_nsec, before.st_mtimespec.tv_nsec)
        refresh()
        let newSHA = sha256Hex(try Data(contentsOf: URL(fileURLWithPath: path)))
        XCTAssertNotEqual(newSHA, oldSHA)
        XCTAssertEqual(controller.factorySHA, newSHA)
        XCTAssertNil(controller.activeVersion, "versions of the old factory model are invalidated")
        controller.waitUntilIdleForTesting()
        XCTAssertEqual(versionDirectories(), [])
    }

    /// Finding 5: 「改回出廠模型」 during a run cancels it; even a helper that ignores SIGTERM and exits 0 with
    /// an adoptable result is not adopted.
    func testRevertDuringRunCancelsAndNeverAdopts() throws {
        writePool(1100)
        refresh()
        try setBehavior("adoptlate", delay: 1.5)
        XCTAssertEqual(controller.startManualRun(), .started)
        waitUntil { self.exists("status.json") }
        controller.revertToFactory()
        XCTAssertEqual(waitForRun(), "cancelled")
        XCTAssertNil(controller.activeVersion)
        XCTAssertTrue(controller.paused)
        controller.waitUntilIdleForTesting()
        XCTAssertEqual(versionDirectories(), [])
        XCTAssertEqual(stagingEntries(), [])
    }

    /// Finding 5: a revert while an exited run's result is being moved into v<N> is not adopted either.
    func testRevertDuringAdoptionNotAdopted() throws {
        writePool(1100)
        refresh()
        XCTAssertEqual(try startAndWait(), "adopted")
        try setBehavior("adopt", delay: 0.3)
        XCTAssertEqual(controller.startManualRun(), .started)
        store.suspendQueueForTesting()
        waitUntil { !self.controller.isRunning }
        controller.revertToFactory()
        store.resumeQueueForTesting()
        XCTAssertEqual(waitForRun(), "cancelled")
        XCTAssertNil(controller.activeVersion)
        XCTAssertEqual(controller.previousVersion, 1)
        controller.waitUntilIdleForTesting()
        XCTAssertEqual(versionDirectories(), ["v1"], "the renamed v2 is deleted")
        XCTAssertNil(manifestOnDisk()?["active"])
    }
}
