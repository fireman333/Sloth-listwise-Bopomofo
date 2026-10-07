// Copyright (c) 2022 and onwards The McBopomofo Authors.
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
import Cocoa
import InfoCollector
import InputSourceHelper

struct KeyboardLayoutOption: Identifiable, Hashable {
    let id: String
    let localizedName: String
}

/// Status of a selection model as shown next to its switch (preferences spec).
enum SelectionModelStatus: Equatable {
    case off
    case loading
    case ready
    case notFound
    case failed
    case tooSlow

    /// Localizable.strings key.
    var titleKey: String {
        switch self {
        case .off: "Off"
        case .loading: "Loading"
        case .ready: "Ready"
        case .notFound: "Model not found"
        case .failed: "Failed to load"
        case .tooSlow: "Stopped (too slow)"
        }
    }

    /// R-walk: off while the switch is off, else the selector's state.
    static func neural(enabled: Bool, state: AFMNeuralSelectorState) -> SelectionModelStatus {
        guard enabled else { return .off }
        switch state {
        case .disabled: return .off
        case .loading: return .loading
        case .ready: return .ready
        case .notFound: return .notFound
        case .failed: return .failed
        case .tripped: return .tooSlow
        @unknown default: return .failed
        }
    }

    /// Precise mode: off while its switch is off, else the controller's state;
    /// stopped too slow also when R-walk's breaker tripped (that stops precise
    /// mode for the process too).
    static func precise(
        enabled: Bool, state: AFMPreciseState, neuralState: AFMNeuralSelectorState
    ) -> SelectionModelStatus {
        guard enabled else { return .off }
        if neuralState == .tripped {
            return .tooSlow
        }
        switch state {
        case .off: return .off
        case .loading: return .loading
        case .ready: return .ready
        case .notFound: return .notFound
        case .failed: return .failed
        case .tripped: return .tooSlow
        @unknown default: return .failed
        }
    }
}

/// Lines of the 「個人化模型」 section (personal-finetune D14).
struct PersonalizationLines: Equatable {
    var status = ""
    var pending: String?
    var eligible = ""
    var progress: String?
    var lastRun: String?
    var message: String?
    var paused: String?
}

@MainActor
final class PreferencesViewModel: NSObject, ObservableObject {
    private var defaultsObserver: NSObjectProtocol?
    private var modelStateObservers: [NSObjectProtocol] = []
    @Published private(set) var neuralSelectorState: AFMNeuralSelectorState = .disabled
    @Published private(set) var preciseState: AFMPreciseState = .off
    @Published private(set) var basisKeyboardLayoutOptions: [KeyboardLayoutOption] = []
    /// The tab shown in the preferences window; set by
    /// PreferencesWindowController.select(tab:) (e.g. from the learning notice).
    @Published var selectedTab: PreferencesTab = .basic

    /// Asks the user to confirm clearing learning data (design D10). Replaced
    /// in tests; the default shows a modal NSAlert.
    var confirmClearLearningData: () -> Bool = PreferencesViewModel.runClearLearningDataAlert

    var keyboardLayout: KeyboardLayout {
        get { Preferences.keyboardLayout }
        set {
            objectWillChange.send()
            Preferences.keyboardLayout = newValue
        }
    }

    var basisKeyboardLayout: String {
        get { Preferences.basisKeyboardLayout }
        set {
            objectWillChange.send()
            Preferences.basisKeyboardLayout = newValue
        }
    }

    var candidateKeys: String {
        get {
            let keys = Preferences.candidateKeys
            if keys.isEmpty {
                return Preferences.defaultCandidateKeys
            }
            return keys
        }
        set {
            let keys =
                newValue
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased()
            objectWillChange.send()
            do {
                try Preferences.validate(candidateKeys: keys)
                Preferences.candidateKeys = keys
                candidateKeysError = nil
            } catch Preferences.CandidateKeyError.empty {
                candidateKeysError = nil
            } catch {
                candidateKeysError = error.localizedDescription
                NSAlert(error: error).runModal()
            }
        }
    }

    @Published var candidateKeysError: String?

    var chooseCandidateUsingSpace: Bool {
        get { Preferences.chooseCandidateUsingSpace }
        set {
            objectWillChange.send()
            Preferences.chooseCandidateUsingSpace = newValue
        }
    }

    var selectPhraseAfterCursorAsCandidate: Bool {
        get { Preferences.selectPhraseAfterCursorAsCandidate }
        set {
            objectWillChange.send()
            Preferences.selectPhraseAfterCursorAsCandidate = newValue
        }
    }

    var moveCursorAfterSelectingCandidate: Bool {
        get { Preferences.moveCursorAfterSelectingCandidate }
        set {
            objectWillChange.send()
            Preferences.moveCursorAfterSelectingCandidate = newValue
        }
    }

    var allowMovingCursorWhenChoosingCandidates: MovingCursorKey {
        get { Preferences.allowMovingCursorWhenChoosingCandidates }
        set {
            objectWillChange.send()
            Preferences.allowMovingCursorWhenChoosingCandidates = newValue
        }
    }

    var useHorizontalCandidateList: Bool {
        get { Preferences.useHorizontalCandidateList }
        set {
            objectWillChange.send()
            Preferences.useHorizontalCandidateList = newValue
        }
    }

    var candidateListTextSize: CGFloat {
        get { Preferences.candidateListTextSize }
        set {
            objectWillChange.send()
            Preferences.candidateListTextSize = newValue
        }
    }

    var candidateListTextSizeOptions: [CGFloat] {
        let defaultSizes: [CGFloat] = [12, 14, 16, 18, 24, 32, 64, 96]
        let currentSize = CGFloat(Int(Preferences.candidateListTextSize))
        if defaultSizes.contains(currentSize) {
            return defaultSizes
        }
        return (defaultSizes + [currentSize]).sorted()
    }

    var letterBehavior: Int {
        get { Preferences.letterBehavior }
        set {
            objectWillChange.send()
            Preferences.letterBehavior = newValue
        }
    }

    var shiftEnterEnabled: Bool {
        get { Preferences.shiftEnterEnabled }
        set {
            objectWillChange.send()
            Preferences.shiftEnterEnabled = newValue
        }
    }

    var escToCleanInputBuffer: Bool {
        get { Preferences.escToCleanInputBuffer }
        set {
            objectWillChange.send()
            Preferences.escToCleanInputBuffer = newValue
        }
    }

    var beepUponInputError: Bool {
        get { Preferences.beepUponInputError }
        set {
            objectWillChange.send()
            Preferences.beepUponInputError = newValue
        }
    }

    var checkForUpdatesAutomatically: Bool {
        get {
            Preferences.defaults.object(forKey: "CheckUpdateAutomatically") as? Bool ?? true
        }
        set {
            objectWillChange.send()
            Preferences.defaults.set(newValue, forKey: "CheckUpdateAutomatically")
        }
    }

    var useCustomUserPhraseLocation: Bool {
        get { Preferences.useCustomUserPhraseLocation }
        set {
            objectWillChange.send()
            Preferences.useCustomUserPhraseLocation = newValue
            if newValue && Preferences.customUserPhraseLocation.isEmpty {
                Preferences.customUserPhraseLocation =
                    UserPhraseLocationHelper.defaultUserPhraseLocation
            }
        }
    }

    var customUserPhraseLocation: String {
        get { Preferences.customUserPhraseLocation }
        set {
            let path = newValue.trimmingCharacters(in: .whitespaces)
            if !path.isEmpty && !FileManager.default.fileExists(atPath: path) {
                try? FileManager.default.createDirectory(
                    atPath: path, withIntermediateDirectories: true)
            }
            objectWillChange.send()
            Preferences.customUserPhraseLocation = path
        }
    }

    var addPhraseHookEnabled: Bool {
        get { Preferences.addPhraseHookEnabled }
        set {
            objectWillChange.send()
            Preferences.addPhraseHookEnabled = newValue
        }
    }

    var addPhraseHookPath: String {
        get { Preferences.addPhraseHookPath }
        set {
            objectWillChange.send()
            Preferences.addPhraseHookPath = newValue
        }
    }

    var chineseConversionStyle: ChineseConversionStyle {
        get { Preferences.chineseConversionStyle }
        set {
            objectWillChange.send()
            Preferences.chineseConversionStyle = newValue
        }
    }

    var controlEnterOutput: ControlEnterOutput {
        get { Preferences.controlEnterOutput }
        set {
            objectWillChange.send()
            Preferences.controlEnterOutput = newValue
        }
    }

    var big5InputEnabled: Bool {
        get { Preferences.big5InputEnabled }
        set {
            objectWillChange.send()
            Preferences.big5InputEnabled = newValue
        }
    }

    var repeatedPunctuationToSelectCandidateEnabled: Bool {
        get { Preferences.repeatedPunctuationToSelectCandidateEnabled }
        set {
            objectWillChange.send()
            Preferences.repeatedPunctuationToSelectCandidateEnabled = newValue
        }
    }

    // MARK: Selection model tab

    /// R-walk switch (AFMNeuralWalkEnabled). Also synced to the selector right
    /// away, so the status updates while the window is open; typing follows
    /// no later than the next activation.
    var neuralWalkEnabled: Bool {
        get { Preferences.neuralWalkEnabled }
        set {
            objectWillChange.send()
            Preferences.neuralWalkEnabled = newValue
            AFMNeuralSelector.shared.syncWithPreferences()
        }
    }

    /// Precise mode switch (AFMPreciseModeEnabled): loads or releases the
    /// heavy model immediately, in the background.
    var preciseModeEnabled: Bool {
        get { Preferences.preciseModeEnabled }
        set {
            objectWillChange.send()
            Preferences.preciseModeEnabled = newValue
            AFMPreciseController.shared.enabled = newValue
        }
    }

    /// Reading policy switch (AFMAcceptYanReadingForAi); applied on activation.
    var acceptYanReadingForAi: Bool {
        get { Preferences.acceptYanReadingForAi }
        set {
            objectWillChange.send()
            Preferences.acceptYanReadingForAi = newValue
        }
    }

    /// Learning switch (AFMLearningEnabled, personal-memory design D10).
    /// Stops or starts recording immediately; the memory configuration
    /// follows no later than the next activation.
    var learningEnabled: Bool {
        get { Preferences.learningEnabled }
        set {
            objectWillChange.send()
            Preferences.setLearningEnabled(newValue)
        }
    }

    /// The 「清除學習資料…」 button: clears learning data after confirmation
    /// (design D10 / D11). Returns whether the data was cleared.
    @discardableResult
    func clearLearningData() -> Bool {
        guard confirmClearLearningData() else {
            return false
        }
        LanguageModelManager.clearLearningData()
        return true
    }

    static func runClearLearningDataAlert() -> Bool {
        let alert = NSAlert()
        alert.alertStyle = .warning
        alert.messageText = NSLocalizedString("Clear learning data?", comment: "")
        alert.informativeText = NSLocalizedString(
            "This deletes the selections the input method has learned and the saved typing records on this Mac. User phrases and other settings are not changed.",
            comment: "")
        alert.addButton(withTitle: NSLocalizedString("Clear", comment: ""))
        alert.addButton(withTitle: NSLocalizedString("Cancel", comment: ""))
        return alert.runModal() == .alertFirstButtonReturn
    }

    // MARK: Personalized model section (personal-finetune D14)

    /// The controller behind the 「個人化模型」 section (tests inject one).
    var personalTraining: AFMPersonalTrainingController = .shared {
        didSet { observePersonalTraining() }
    }
    /// The selector whose loaded version is shown (tests inject one).
    var personalSelector: () -> AFMNeuralSelector = { AFMNeuralSelector.shared }
    /// status.json of the running helper, refreshed every second while the
    /// section is visible.
    @Published private(set) var personalRunStatus: [AnyHashable: Any]?
    /// Message after 「立即學習」 was refused (e.g. too little data).
    @Published private(set) var personalStartMessage: PersonalStartMessage?
    private var personalPollTimer: Timer?
    private var personalObserver: NSObjectProtocol?
    /// Polling interval while the tab is visible.
    var personalPollInterval: TimeInterval = 1

    enum PersonalStartMessage: Equatable {
        case notEnoughData(available: Int, needed: Int)
        case packMissing
        case helperMissing
        case failedToStart
    }

    var isPollingPersonalization: Bool { personalPollTimer != nil }

    /// Starts the 1 s status polling (tab visible). Never blocks: status.json
    /// is read off main and the pool scan runs on the store queue.
    func startPersonalizationPolling() {
        guard personalPollTimer == nil else { return }
        pollPersonalization()
        let timer = Timer(timeInterval: personalPollInterval, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated { self?.pollPersonalization() }
        }
        RunLoop.main.add(timer, forMode: .common)
        personalPollTimer = timer
    }

    /// Stops polling (tab hidden / window closed).
    func stopPersonalizationPolling() {
        personalPollTimer?.invalidate()
        personalPollTimer = nil
    }

    func pollPersonalization() {
        let controller = personalTraining
        controller.refreshEligibleCount(completion: nil)
        if controller.isRunning {
            controller.readStatus { [weak self] status in
                MainActor.assumeIsolated {
                    guard let self, self.personalTraining === controller else { return }
                    self.personalRunStatus = controller.isRunning ? status : nil
                }
            }
        } else if personalRunStatus != nil {
            personalRunStatus = nil
        }
    }

    private func observePersonalTraining() {
        if let personalObserver {
            NotificationCenter.default.removeObserver(personalObserver)
        }
        personalObserver = NotificationCenter.default.addObserver(
            forName: .AFMPersonalTrainingControllerDidChange, object: personalTraining, queue: .main
        ) { [weak self] _ in
            Task { @MainActor in
                self?.objectWillChange.send()
            }
        }
    }

    var personalRunning: Bool { personalTraining.isRunning }
    /// 「立即學習」: disabled while learning is off (replaced by 「取消」 while running).
    var learnNowEnabled: Bool { learningEnabled && !personalTraining.isRunning }
    /// 「改回出廠模型」: while a personalized version is selected for use.
    var useFactoryModelEnabled: Bool { personalTraining.activeVersion != nil }
    /// 「改用上一版」: only when a previous version exists.
    var usePreviousVersionEnabled: Bool { personalTraining.previousVersion != nil }
    var personalPaused: Bool { personalTraining.paused || personalTraining.failurePaused }

    @discardableResult
    func learnNow() -> AFMPersonalStartResult {
        objectWillChange.send()
        let controller = personalTraining
        let result = controller.startManualRun()
        switch result {
        case .started, .alreadyRunning: personalStartMessage = nil
        case .notEnoughData:
            personalStartMessage = .notEnoughData(
                available: max(controller.eligibleCount, 0), needed: controller.minimumEligible)
        case .packMissing: personalStartMessage = .packMissing
        case .helperMissing: personalStartMessage = .helperMissing
        default: personalStartMessage = .failedToStart
        }
        pollPersonalization()
        return result
    }

    func cancelLearning() {
        objectWillChange.send()
        personalTraining.cancelRun()
    }

    func useFactoryModel() {
        objectWillChange.send()
        personalTraining.revertToFactory()
    }

    func usePreviousVersion() {
        objectWillChange.send()
        personalTraining.usePreviousVersion()
    }

    /// The text lines of the section (localized with `localize`).
    func personalizationLines(localize: (String) -> String) -> PersonalizationLines {
        let controller = personalTraining
        let selector = personalSelector()
        var lines = PersonalizationLines()
        func number(_ n: Int) -> String {
            let formatter = NumberFormatter()
            formatter.numberStyle = .decimal
            return formatter.string(from: NSNumber(value: n)) ?? String(n)
        }
        func date(_ t: Double) -> String {
            let formatter = DateFormatter()
            formatter.locale = Locale(identifier: "en_US_POSIX")
            formatter.dateFormat = "yyyy-MM-dd"
            return formatter.string(from: Date(timeIntervalSince1970: t))
        }
        func points(_ v: Double) -> String {
            let sign = v > 0 ? "+" : (v < 0 ? "−" : "±")
            return String(format: localize("%@ pt"), sign + String(format: "%.1f", abs(v)))
        }
        if !controller.packAvailable {
            lines.status = localize("Personalization data pack missing")
        } else {
            switch selector.personalStatus {
            case .active:
                let version = selector.loadedPersonalVersion
                lines.status = String(
                    format: localize("In use: personalized model version %ld (%@)"), version,
                    date(controller.createdTime(ofVersion: version)))
            case .unusable:
                lines.status = localize("Personalized model could not be loaded; using the factory model")
            case .mismatch:
                lines.status = localize(
                    "Personalized model does not match the current settings; using the factory model")
            case .loading:
                lines.status = localize("Loading personalized model…")
            default:
                lines.status = localize("In use: factory model")
            }
            // A change that applies at the next activation.
            let selected = controller.activeVersion?.intValue ?? 0
            let loaded = selector.personalStatus == .active ? selector.loadedPersonalVersion : 0
            let attempted = selector.personalStatus == .unusable || selector.personalStatus == .mismatch
            if selected != loaded && !(attempted && selected != 0) && selector.personalStatus != .loading {
                lines.pending =
                    selected == 0
                    ? localize("The factory model will be used from the next activation")
                    : String(format: localize("Version %ld will be used from the next activation"), selected)
            }
        }
        let eligible = max(controller.eligibleCount, 0)
        lines.eligible = String(
            format: localize("Eligible segments: %@ (at least %@ needed to learn)"), number(eligible),
            number(controller.minimumEligible))
        if controller.isRunning {
            let phase = (personalRunStatus?["phase"] as? String) ?? "scan"
            let phaseTitle = localize(Self.phaseTitleKey(phase))
            let step = (personalRunStatus?["step"] as? NSNumber)?.intValue ?? 0
            let steps = (personalRunStatus?["steps"] as? NSNumber)?.intValue ?? 0
            if steps > 0 {
                lines.progress = String(
                    format: localize("Learning: %@ %ld%%"), phaseTitle, Int((Double(step) / Double(steps) * 100).rounded()))
            } else {
                lines.progress = String(format: localize("Learning: %@"), phaseTitle)
            }
        }
        if let last = controller.lastRun, let result = last["result"] as? String {
            let time = (last["time"] as? NSNumber)?.doubleValue ?? 0
            var outcome = localize(Self.resultTitleKey(result))
            if let dp = (last["delta_personal_pt"] as? NSNumber)?.doubleValue,
                let dg = (last["delta_general_pt"] as? NSNumber)?.doubleValue
            {
                outcome += String(format: localize("(personal test %@, general test %@)"), points(dp), points(dg))
            }
            lines.lastRun = String(format: localize("Last run: %@ — %@"), date(time), outcome)
        }
        switch personalStartMessage {
        case .notEnoughData(let available, let needed):
            lines.message = String(
                format: localize("Not enough data: %@ segments needed, %@ available"), number(needed), number(available))
        case .packMissing: lines.message = localize("Personalization data pack missing")
        case .helperMissing, .failedToStart: lines.message = localize("Learning could not be started")
        case nil: break
        }
        if personalPaused {
            lines.paused = localize("Automatic learning is paused")
        }
        return lines
    }

    static func phaseTitleKey(_ phase: String) -> String {
        switch phase {
        case "scan": "Reading sentences"
        case "nbest": "Preparing candidates"
        case "train": "Training"
        case "merge": "Merging weights"
        case "gate": "Evaluating"
        case "calib": "Calibrating"
        default: "Finishing"
        }
    }

    static func resultTitleKey(_ result: String) -> String {
        switch result {
        case "adopted": "Adopted"
        case "not_significant": "Not adopted: improvement not statistically significant"
        case "too_small_gain": "Not adopted: improvement too small"
        case "general_regression": "Not adopted: general accuracy dropped too much"
        case "worse_than_active": "Not adopted: not better than the version in use"
        case "cancelled": "Cancelled"
        case "deferred": "Deferred (not on AC power)"
        case "not_enough_data": "Not enough data"
        case "discarded:retracted": "Discarded (a sentence used for learning was deleted)"
        case "discarded:generation": "Cancelled"
        default: "Failed"
        }
    }

    var neuralStatus: SelectionModelStatus {
        .neural(enabled: neuralWalkEnabled, state: neuralSelectorState)
    }

    var preciseStatus: SelectionModelStatus {
        .precise(enabled: preciseModeEnabled, state: preciseState, neuralState: neuralSelectorState)
    }

    private func observeModelStates() {
        neuralSelectorState = AFMNeuralSelector.shared.state
        preciseState = AFMPreciseController.shared.state
        let center = NotificationCenter.default
        modelStateObservers = [
            center.addObserver(forName: .AFMNeuralSelectorStateDidChange, object: nil, queue: .main) {
                [weak self] _ in
                Task { @MainActor in
                    self?.neuralSelectorState = AFMNeuralSelector.shared.state
                }
            },
            center.addObserver(forName: .AFMPreciseControllerStateDidChange, object: nil, queue: .main) {
                [weak self] _ in
                Task { @MainActor in
                    self?.preciseState = AFMPreciseController.shared.state
                }
            },
        ]
    }

    var showBopomofoFontAnnotationSupportItemInInputMenu: Bool {
        get { Preferences.showBopomofoFontAnnotationSupportItemInInputMenu }
        set {
            objectWillChange.send()
            Preferences.showBopomofoFontAnnotationSupportItemInInputMenu = newValue
        }
    }

    @Published private(set) var latestSystemReport = ""

    var effectiveUserPhraseLocation: String {
        if !useCustomUserPhraseLocation || customUserPhraseLocation.isEmpty {
            return UserPhraseLocationHelper.defaultUserPhraseLocation
        }
        return customUserPhraseLocation
    }

    var customUserPhraseLocationText: String {
        if useCustomUserPhraseLocation {
            return customUserPhraseLocation
        }
        return ""
    }

    override init() {
        super.init()
        Preferences.populateDefaults()
        loadBasisKeyboardLayoutOptions()
        observeModelStates()
        observePersonalTraining()
        defaultsObserver = NotificationCenter.default.addObserver(
            forName: UserDefaults.didChangeNotification,
            object: Preferences.defaults,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor in
                self?.objectWillChange.send()
            }
        }
    }

    deinit {
        if let defaultsObserver {
            NotificationCenter.default.removeObserver(defaultsObserver)
        }
        for observer in modelStateObservers {
            NotificationCenter.default.removeObserver(observer)
        }
        if let personalObserver {
            NotificationCenter.default.removeObserver(personalObserver)
        }
        personalPollTimer?.invalidate()
    }

    func chooseUserPhraseFolder() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.directoryURL = URL(fileURLWithPath: effectiveUserPhraseLocation)

        if panel.runModal() == .OK, let url = panel.url {
            customUserPhraseLocation = url.path
        }
    }

    func openUserPhraseFolder() {
        NSWorkspace.shared.open(URL(fileURLWithPath: effectiveUserPhraseLocation))
    }

    func openSystemInfoReport() {
        Task { @MainActor in
            await openSystemInfoReportAsync()
        }
    }

    func openSystemInfoReportAsync() async {
        var report = ""
        report += await InfoCollector.generate()
        report += Preferences.createReport()
        latestSystemReport = report

        let tempDir = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
        let randomName = "SystemInfoReport-\(UUID().uuidString).txt"
        let fileURL = tempDir.appendingPathComponent(randomName)
        do {
            try report.write(to: fileURL, atomically: true, encoding: .utf8)
            NSWorkspace.shared.open(fileURL)
        } catch {
            NSLog("Failed to write report to temporary file: \(error)")
        }
    }

    private func loadBasisKeyboardLayoutOptions() {
        let list = InputSourceHelper.allInstalledInputSources()
        var options: [KeyboardLayoutOption] = []

        for source in list {
            func getString(_ key: CFString) -> String? {
                if let ptr = TISGetInputSourceProperty(source, key) {
                    return String(Unmanaged<CFString>.fromOpaque(ptr).takeUnretainedValue())
                }
                return nil
            }

            func getBool(_ key: CFString) -> Bool? {
                if let ptr = TISGetInputSourceProperty(source, key) {
                    return Unmanaged<CFBoolean>.fromOpaque(ptr).takeUnretainedValue()
                        == kCFBooleanTrue
                }
                return nil
            }

            guard
                getString(kTISPropertyInputSourceCategory)
                    == String(kTISCategoryKeyboardInputSource),
                getBool(kTISPropertyInputSourceIsASCIICapable) == true,
                getString(kTISPropertyInputSourceType) == String(kTISTypeKeyboardLayout),
                let sourceID = getString(kTISPropertyInputSourceID),
                let localizedName = getString(kTISPropertyLocalizedName)
            else {
                continue
            }

            options.append(KeyboardLayoutOption(id: sourceID, localizedName: localizedName))
        }
        basisKeyboardLayoutOptions = options
    }
}
