# Proposal: personal-memory

## Why

The input method forgets the user: McBopomofo's user override model (UOM) keeps at most 500 manual picks with a 1.5-hour half-life, only in memory, so every restart or logout starts over, and nothing the user types is kept for later personalization. This change is the "fast loop" of the on-device personalization plan (grill 2026-10-07; prior art `../research/13_on_device_personalization_prior_art.md`): make the user's manual picks a durable, long-term memory and start collecting a local sentence pool that a later change will use to fine-tune R on the device. It is useful on its own even if the later fine-tuning change is never built.

## What Changes

- The UOM becomes long-term and persistent: capacity 5,000 contexts (LRU eviction), half-life 7 days, saved to a file under the application-support root and restored on launch. Suggestions keep the stock override behavior (context-keyed, applied as a node override), plus: the latest manual pick in a context wins immediately, and committing a remembered suggestion unchanged keeps it fresh. Plain Bopomofo picks are not remembered while learning is on.
- A memory suggestion is not applied over any node from the user's own phrase file (priority: user phrase > memory > R-walk).
- New local sentence pool: every composition from Bopomofo mode that is committed (explicitly or implicitly, e.g. on app switch) is appended with readings, dictionary characters per node (before replacement/macros/Simplified conversion), per-node source (manual pick, memory, precise, user phrase, walk), a manual-pick flag and the selection method. Kept permanently until the user clears it. Plain Bopomofo mode and discarded compositions are not recorded.
- Nothing is learned or recorded while macOS secure event input is enabled (password fields).
- New preference 「學習我的選字」 (default on) in the 「選字模型」 tab, plus a 「清除學習資料」 button (with confirmation) that deletes the memory file, the sentence pool and the in-memory UOM. With learning off, picks are remembered as in stock McBopomofo (session-only, stock capacity/half-life/scoring) and nothing is read from or written to the memory files.
- One-time first-run notice in a floating panel that never takes keyboard focus, explaining that learning happens, data stays on this Mac, and where to turn it off or clear it.
- No neural model is trained in this change; Qwen / precise mode are unchanged.

## Capabilities

### New Capabilities
- `personal-memory`: long-term persistent selection memory, the local sentence pool, the learning switch and clearing, secure-input exclusion, first-run notice, storage location and on-device guarantees.

### Modified Capabilities
- `selection-pipeline`: "User choices take precedence over model scores" — a remembered (UOM) suggestion is not applied over a node that comes from the user's phrase file.
- `preferences`: "Selection-model preferences tab" gains the learning switch and the clear-data button; "When preference changes take effect" covers the learning switch.

## Impact

- C++ engine: `Source/Engine/UserOverrideModel.{h,cpp}` (policy flags, refresh, serialize / deserialize), `gramambular2/reading_grid` (const override-target lookup), gtests.
- App: `Source/LanguageModelManager.mm` (UOM construction, load/save, reset), `Source/KeyHandler.mm` (manual-pick flag, commit capture, user-phrase guard for suggestions, secure-input check), new memory store (pool writer + file paths under `<root>/memory/`), `Source/Preferences.swift`, `Source/PreferencesUI/*` (switch, clear button, localization), first-run notice as a non-activating panel; `Packages/FSEventStreamHelper` (public event path) and `AppDelegate` watcher filter.
- Tests: new XCTests for persistence, pool capture, secure-input skip, clear, learning off; existing tests keep using `resetUserOverrideModelForTesting` and the per-PID test root.
- Data: new files `~/Library/Application Support/Sloth-listwise-Bopomofo/memory/{override-model.txt, sentences.jsonl}` (mode 0600). The user-phrase folder watcher is filtered so writes under `memory/` (and `models/`) do not reload user phrases.
- Latency: no new work on the per-keystroke path beyond a flag set and an O(1) node check; file I/O happens off the key-handling path.
