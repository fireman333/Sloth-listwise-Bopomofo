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

// First-run learning notice (openspec change personal-memory, design D12).

import Cocoa

/// Whether the one-time learning notice should be shown on activation.
func shouldShowLearningNotice(learningEnabled: Bool, noticeShown: Bool) -> Bool {
    learningEnabled && !noticeShown
}

/// The notice text (Traditional Chinese, as required by the spec).
enum LearningNoticeText {
    static let title = "輸入法會學習你的選字"
    static let body =
        "本輸入法會從你的打字內容與手動選字中學習，讓之後的選字更貼近你的用法。"
        + "這些資料只保存在這台 Mac 上，輸入法不會透過網路傳送。"
        + "你可以隨時在偏好設定的「選字模型」分頁關閉學習或清除學習資料。"
    static let okButton = "好"
    static let openSettingsButton = "打開設定"
}

/// A floating panel that never takes keyboard focus nor activates the input
/// method process, so keystrokes keep going to the client application.
final class LearningNoticePanel: NSPanel {
    override var canBecomeKey: Bool { false }
    override var canBecomeMain: Bool { false }

    init(onOK: @escaping () -> Void, onOpenSettings: @escaping () -> Void) {
        super.init(
            contentRect: NSRect(x: 0, y: 0, width: 380, height: 160),
            styleMask: [.nonactivatingPanel, .titled, .fullSizeContentView],
            backing: .buffered, defer: false)
        isFloatingPanel = true
        becomesKeyOnlyIfNeeded = true
        level = .floating
        collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        hidesOnDeactivate = false
        isReleasedWhenClosed = false
        titleVisibility = .hidden
        titlebarAppearsTransparent = true
        isMovableByWindowBackground = true
        contentView = LearningNoticeContentView(onOK: onOK, onOpenSettings: onOpenSettings)
        setContentSize(contentView?.fittingSize ?? frame.size)
    }

    /// Top-right of the visible frame of the screen under the mouse.
    func positionOnScreen() {
        let mouse = NSEvent.mouseLocation
        guard
            let screen = NSScreen.screens.first(where: { $0.frame.contains(mouse) }) ?? NSScreen.main
                ?? NSScreen.screens.first
        else {
            return
        }
        let visible = screen.visibleFrame
        let padding: CGFloat = 24
        setFrameOrigin(
            NSPoint(x: visible.maxX - frame.width - padding, y: visible.maxY - frame.height - padding))
    }
}

/// Mouse-only content: buttons have no key equivalents and refuse first
/// responder.
private final class LearningNoticeContentView: NSView {
    private let onOK: () -> Void
    private let onOpenSettings: () -> Void

    init(onOK: @escaping () -> Void, onOpenSettings: @escaping () -> Void) {
        self.onOK = onOK
        self.onOpenSettings = onOpenSettings
        super.init(frame: .zero)

        let title = NSTextField(labelWithString: LearningNoticeText.title)
        title.font = .boldSystemFont(ofSize: NSFont.systemFontSize)
        let body = NSTextField(wrappingLabelWithString: LearningNoticeText.body)
        body.preferredMaxLayoutWidth = 340
        body.isSelectable = false

        let openSettings = NSButton(
            title: LearningNoticeText.openSettingsButton, target: self, action: #selector(openSettingsClicked))
        let ok = NSButton(title: LearningNoticeText.okButton, target: self, action: #selector(okClicked))
        for button in [openSettings, ok] {
            button.bezelStyle = .push
            button.keyEquivalent = ""
            button.refusesFirstResponder = true
        }
        let buttons = NSStackView(views: [openSettings, ok])
        buttons.orientation = .horizontal
        buttons.spacing = 8

        let stack = NSStackView(views: [title, body, buttons])
        stack.orientation = .vertical
        stack.alignment = .leading
        stack.spacing = 10
        stack.setCustomSpacing(14, after: body)
        stack.translatesAutoresizingMaskIntoConstraints = false
        addSubview(stack)
        NSLayoutConstraint.activate([
            stack.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 20),
            stack.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -20),
            stack.topAnchor.constraint(equalTo: topAnchor, constant: 28),
            stack.bottomAnchor.constraint(equalTo: bottomAnchor, constant: -16),
            body.widthAnchor.constraint(equalToConstant: 340),
        ])
        buttons.trailingAnchor.constraint(equalTo: stack.trailingAnchor).isActive = true
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }

    @objc private func okClicked() { onOK() }
    @objc private func openSettingsClicked() { onOpenSettings() }
}

/// Schedules and shows the notice (design D12): on activation, if learning is
/// on and the notice was never shown, show it 1 s later on main and set
/// AFMLearningNoticeShown once it is ordered front. Suppressed under XCTest
/// unless a test opts in.
@MainActor
final class LearningNoticeController {
    static let shared = LearningNoticeController()

    /// Delay between activateServer and showing the notice.
    var delay: TimeInterval = 1
    /// True under XCTest so test activations never show the notice.
    var suppressed = isRunningUnderXCTest
    /// Orders the panel front; replaceable in tests.
    var orderFront: (NSPanel) -> Void = defaultOrderFront
    /// 「打開設定」: opens preferences on the 「選字模型」 tab (may activate the
    /// app; the click is user-initiated).
    var openPreferences: () -> Void = defaultOpenPreferences

    private static let defaultOrderFront: (NSPanel) -> Void = { $0.orderFrontRegardless() }
    private static let defaultOpenPreferences: () -> Void = {
        (NSApp.delegate as? AppDelegate)?.showPreferences(tab: .selectionModel)
    }

    private(set) var panel: LearningNoticePanel?
    private(set) var isScheduled = false

    private var noticeDue: Bool {
        shouldShowLearningNotice(
            learningEnabled: Preferences.learningEnabled, noticeShown: Preferences.learningNoticeShown)
    }

    /// Called from activateServer.
    func scheduleIfNeeded() {
        guard !suppressed, !isScheduled, panel == nil, noticeDue
        else {
            return
        }
        isScheduled = true
        DispatchQueue.main.asyncAfter(deadline: .now() + delay) { [weak self] in
            MainActor.assumeIsolated {
                guard let self, self.isScheduled else { return }
                self.isScheduled = false
                self.showIfNeeded()
            }
        }
    }

    /// Shows the panel if the decision still holds; sets the flag only when
    /// the panel actually is on screen.
    func showIfNeeded() {
        guard panel == nil, noticeDue
        else {
            return
        }
        let panel = LearningNoticePanel(
            onOK: { [weak self] in self?.close() },
            onOpenSettings: { [weak self] in
                self?.close()
                self?.openPreferences()
            })
        panel.positionOnScreen()
        orderFront(panel)
        if panel.isVisible {
            self.panel = panel
            Preferences.learningNoticeShown = true
        }
    }

    func close() {
        panel?.orderOut(nil)
        panel = nil
    }

    /// Testing only.
    func resetForTesting() {
        close()
        isScheduled = false
        delay = 1
        suppressed = isRunningUnderXCTest
        orderFront = Self.defaultOrderFront
        openPreferences = Self.defaultOpenPreferences
    }
}
