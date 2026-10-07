# Design: personal-memory

## Context

- `UserOverrideModel` (UOM, `Source/Engine/UserOverrideModel.{h,cpp}`) is an LRU map `key -> Observation{count, map<candidate, Override{count, timestamp, force}>}`. Key = `anterior-prev-head`: `(reading,value)` of the two preceding walk nodes (current values) and `(reading, unigrams()[0].value())` of the head node (`.cpp:255-300`). Score = `overrideCount/observationCount * exp(dt * ln(0.5)/halfLife)`, zero below 2^-20 (`.cpp:39,190-242`). `suggest` does not touch LRU order or timestamps. In-memory only.
- One global `gUserOverrideModel(500, 5400)` in `LanguageModelManager.mm:33-38`; KeyHandler keeps a pointer (`KeyHandler.mm:83,171`). `resetUserOverrideModelForTesting` move-assigns in place, so the address is stable. All UOM access is on the main thread. `observe()` at `KeyHandler.mm:248` (inside `fixNodeWithReading:`, also used by Plain Bopomofo picks); `suggest()` + `overrideCandidate` at `KeyHandler.mm:595-602` (Bopomofo only). Timestamps are `NSDate.timeIntervalSince1970`.
- Commit paths (review F2): several KeyHandler sites call `[self clear]` before emitting `InputStateCommitting` (`KeyHandler.mm:366,784,798,1300`); implicit commits come from `InputMethodController.handle(state:)` for `.Empty`/`.Deactivated` and simple candidate handlers (`InputMethodController.swift:494-533,789-808`); `deactivateServer` clears before committing (`:185-189`). The only place text reaches the client is `InputMethodController.commit(text:client:)` (`:474-492`), which applies OpenCC unless the conversion style is `model` (then the LM already converted, `McBopomofoLM.cpp:392-397`).
- The default user-phrase folder **is** the application-support root (`Preferences.swift:615-618`); `FSEventStreamHelper` watches it recursively and `AppDelegate.swift:326-333` reloads user phrases on any event (review F1). `Event.path` is internal in the package (`Packages/FSEventStreamHelper/.../FSEventStreamHelper.swift:31-35`).
- `NonModalAlertWindowController` activates the IME process and takes key focus (`NonModalAlertWindowController.swift:86-90`) — unusable for an unsolicited notice (review F3).
- Product decisions (grill 2026-10-07 + follow-ups): extend/persist the stock UOM; 5,000 contexts / 7-day half-life; latest manual pick wins immediately; committing a remembered suggestion refreshes it; pool of every Bopomofo-mode commit with a manual flag; skip secure input; learning on by default with a one-time non-focus-stealing notice; priority user phrase > memory > R.

## Goals / Non-Goals

**Goals**: durable long-term pick memory; local sentence pool with enough provenance for a later on-device fine-tune; learning switch, clear, notice; no new latency on the keystroke path beyond O(nodes ≤ 40) checks.

**Non-Goals**: training any model; using the pool for selection; personalizing Qwen; syncing between Macs; encryption beyond file permissions / FileVault; detecting edits made after commit; pool compaction.

## Deviation from upstream defaults

`config.yaml` requires the default behavior to stay identical to upstream McBopomofo. This change deliberately deviates, by user decision: with learning on (the default) the UOM uses 5,000 / 7 d, the latest-pick-wins rule, refresh-on-commit and persistence, and plain-mode picks are not observed. With learning off, the UOM is the stock one. The user-phrase guard (D7) applies in both states.

## Decisions

### D1. UOM persistence and policy flags in C++
- `UserOverrideModel(capacity, halfLife, Policy{latestPickWins, observeRefresh})`; the stock constructor keeps both false.
- `latestPickWins`: in `Observation::update`, when a candidate is observed, remove all other candidates' overrides and set `observation.count` to the candidate's count. Result: the latest manual pick has score = decay only.
- `refresh(key, candidate, timestamp)`: if the key exists and its best candidate equals `candidate`, set that override's timestamp and move the key to LRU front; counts unchanged.
- `suggestWithKey(walk, cursor, ts) -> {Suggestion, key}` (or an out-param) so KeyHandler can remember which key produced an applied suggestion.
- `serialize(std::ostream&) const` / `deserialize(std::istream&) -> bool`: header `AFM-UOM 1`, then per (key, candidate) in LRU order (most recent first): `key \t candidate \t count \t timestamp \t force`. Observation count is recomputed as the sum. Doubles written with `std::to_chars` (shortest round-trip) and read with `std::from_chars` (locale-independent). Entries containing `\t`/`\n` skipped on write. All-or-nothing on read (any malformed line → empty model, false). Capacity enforced on load (keep the most recent).
Alternatives: Foundation JSON (needs bridging private C++ structures), binary (not inspectable).

### D2. Configuration lifecycle
`LanguageModelManager.configureLearning(enabled:)`, main thread, idempotent by comparing the **actual** current configuration (not a cached bool, so test resets are handled — review F13):
- enabled → flush nothing, move-assign `UserOverrideModel(5000, 7*86400, {true, true})`, then load `memory/override-model.txt` (D3).
- disabled → move-assign the stock model (the final save already happened in `setRecordingEnabled(false)`, below).
Called from `applicationDidFinishLaunching` and from `activateServer` (cheap no-op when unchanged). The pref setter in PreferencesModel also calls `AFMLearningStore.setRecordingEnabled(_:)` immediately. `setRecordingEnabled(false)` (main): cancel the debounce timer; if the active UOM is the long-term one and dirty, serialize it and write it with `dispatch_sync` on the serial queue (one-time, at a preference change, never in a keystroke); then set `recordingEnabled = false`. From then on no pool record or save is produced (spec "Learning switch": one final save, then nothing). `setRecordingEnabled(true)` only sets the flag; the long-term model is loaded at the next `configureLearning`.

### D3. LearningStore: load, debounced save, generation
New ObjC++ class `AFMLearningStore` (singleton, main-thread API, private serial queue).
- Load: synchronous parse on the main thread inside `configureLearning` (launch / setting change only). Target < 50 ms for 5,000 contexts (measured). Failure → rename to `override-model.bad-<epoch>.txt`, empty model.
- Save: `markDirty` (called after `observe` and `refresh`) schedules a 3 s debounce timer on the main run loop; when it fires and recording is enabled **and the active UOM is the long-term configuration** (so a stock session model never overwrites the saved file, review R2-8), serialize on main (measured, expected ≲ 3 ms, never inside a keystroke) and enqueue an atomic write (`.tmp`, mode 0600, `rename`). `flush` (deactivateServer) does the same immediately. `generation` is a `std::atomic<uint64_t>` bumped only by clear (review R2-9); every queued block captures it at enqueue time and no-ops if it changed, so nothing queued before a clear recreates data.
- No `applicationWillTerminate` today; the guarantee is the 3 s debounce + deactivate flush (review F17).

### D4. Sentence pool via a single commit choke point (review F2)
- `-[KeyHandler clear]` moves (O(1), `std::move`) `_latestWalk`, the composition's provenance map (D5), `_manualPickInComposition`, the input mode, `_latestWalkSelection` (stock / r / precise) and `_lastComposingText` into `_pendingCommitSnapshot` before resetting — only if the walk is non-empty, so a `clear` with nothing composed never overwrites an unconsumed snapshot (review R2-10). `_lastComposingText` is the exact composing-buffer string of the last Inputting state built by KeyHandler (including bopomofo-font annotation variant selectors / PUA, review R2-2), stored when that state is built. A new reading inserted (next composition) drops any stale snapshot.
- `InputMethodController.commit(text:client:)` calls `keyHandler.recordCommittedText(text)` with the **pre-conversion** text before OpenCC. KeyHandler accepts the snapshot only if: learning recording is enabled, secure input is off (D6), mode is Bopomofo, and `text` equals the snapshot's `_lastComposingText` (so Esc-discard, reading/ruby/braille output, letters, Big5 never match). A syllable still in the reading buffer is part of the composing text but not of the walk, so it is simply absent from the record's nodes. The Shift+Space / space-not-selecting commit site (`KeyHandler.mm:658-666`) currently commits before `clear`; it is reordered to `clear` first like the other sites (review R2-3). Then it builds a record and passes it to the store; the snapshot is dropped either way.
- Record (JSON line, encoded on the serial queue): `{"v":1,"t":<epoch>,"manual":<bool>,"walk":"stock|r|precise","app":"<version>","model":"<R model id or empty>","nodes":[["<reading>","<raw>","<src>"],...]}` with `raw = currentUnigram().rawValue()` falling back to `value()` when empty (review F4), `src` ∈ `p` manual pick, `a` associated phrase, `m` memory suggestion, `q` precise result, `u` user phrase, `w` walk (model/dictionary).
- Drop rule (review F20): a record is dropped if no node reading is a Bopomofo syllable (readings starting with `_`, and `" "`, do not count).
- File `memory/sentences.jsonl`, opened `O_APPEND`, mode 0600; write errors logged once per process.

### D5. Provenance and manual flag
`_manualPickInComposition` set in `fixNodeWithReading:` and in the associated-phrase override paths; per-node provenance tracked in a small map keyed by node pointer (`NodePtr`, stable across insertions/deletions elsewhere in the buffer, review R2-4), looked up from the snapshot walk at record time, updated at: manual pick (`p`), associated phrase (`a`), applied memory suggestion (`m`, also stores the UOM key for D8), applied precise result (`q`: every node of the applied fused walk that is not a fixed position — manual pick, associated phrase or memory suggestion keep their own tag). User-phrase nodes are detected at record time via `currentUnigram().isUserPhrase()`. Cleared with the composition.

### D6. Secure input
Revised during apply (user decision 2026-10-07): on this Mac `IsSecureEventInputEnabled()` stays true because loginwindow keeps secure input after a screen lock, which would pause learning indefinitely. The check now requires that secure input is enabled **and** `kCGSSessionSecureInputPID` (from `CGSessionCopyCurrentDictionary`) equals the client's pid, set on `activateServer` from the client bundle id (`AFMLearningStore.clientProcessIdentifier`; unknown → not secure). Original text: `IsSecureEventInputEnabled()` (system-wide, review F12) behind an injectable block for tests; checked before `observe`, before `refresh`, and in `recordCommittedText`. On each transition to enabled, one `os_log` line (no text content).

### D7. User-phrase guard (review F5)
Before applying a suggestion: resolve the target exactly as `overrideCandidate` would, via a new const helper `ReadingGrid::findOverrideTarget(loc, value) -> std::optional<NodeInSpan>` (gtest-covered, shares the search with `overrideCandidate`). Skip the suggestion if any node of `_latestWalk` whose span intersects the target span has `currentUnigram().isUserPhrase()`. O(nodes).

### D8. Refresh on commit
Only when `recordCommittedText` accepts the snapshot (text matched; so Ctrl+Enter reading output, Esc, etc. never refresh, review R2-5), for each provenance entry `m` whose node value still equals the suggested candidate, call `uom.refresh(key, candidate, now)` and `markDirty`. Not under secure input.

### D9. Plain Bopomofo picks
With learning on, `fixNodeWithReading:` skips `observe` in Plain Bopomofo mode (review F15). With learning off, stock behavior.

### D10. Preferences
Keys `AFMLearningEnabled` (Bool, default true) and `AFMLearningNoticeShown` (Bool, default false) in `allKeys`. Model binding + Toggle 「學習我的選字」 and button 「清除學習資料…」 (NSAlert confirm 「清除」 / cancel 「取消」, injectable confirmer) in the 「選字模型」 tab; zh-Hant + en.

### D11. Clearing
`LanguageModelManager.clearLearningData()` (main): bump generation, cancel timer, move-assign an empty UOM of the current configuration (pending commit snapshots in live KeyHandlers need no clearing: records built after the clear are new data, and blocks queued before it no-op on the generation check), then enqueue deletion of `override-model.txt`, `override-model.bad-*.txt`, `sentences.jsonl` (after earlier queued blocks; those already no-op due to the generation).

### D12. First-run notice (review F3)
`shouldShowLearningNotice(learningEnabled:noticeShown:)` decides. On `activateServer`, if true: after a 1 s delay on main show a `LearningNoticePanel` and set `AFMLearningNoticeShown = true` when it is ordered front (review R2-7): an `NSPanel` subclass overriding `canBecomeKey`/`canBecomeMain` to return false, style `.nonactivatingPanel`, `becomesKeyOnlyIfNeeded = true`, level `.floating`, `collectionBehavior` `[.canJoinAllSpaces, .fullScreenAuxiliary]` (visible on the current Space, also over full-screen apps), `orderFrontRegardless()`, no `NSApp.activate`; it stays until a button is clicked; buttons mouse-only (「好」 closes; 「打開設定」 opens preferences on the 「選字模型」 tab — opening preferences may activate the app, which is user-initiated). Suppressed under XCTest.

### D13. Watcher filter (review F1)
Implemented with `realpath` on the longest existing prefix instead of `URL.resolvingSymlinksInPath()` (the latter only strips `/private` for existing paths, so events for an already-renamed `.tmp` would not match). Make `FSEventStreamHelper.Event.path` public (package change) and create the stream with `kFSEventStreamCreateFlagFileEvents`; in `AppDelegate`'s delegate, canonicalize both the event path and the root (`URL.resolvingSymlinksInPath()`, since FSEvents reports `/private/var/...` for `/var/...`, review R2-6), drop events whose path is under `<root>/memory/` or `<root>/models/`, reload only if any event remains. Test: injected events → reload count; a real save + pool append leaves `LanguageModelManager.reloadGeneration` unchanged.

### D14. Storage path
`UserPhraseLocationHelper.learningDataDirectory = applicationSupportRoot/memory`, created lazily with 0700; independent of the custom user-phrase location; per-PID temp root under XCTest.

### D15. Test harness
Shared XCTest setUp sets learning off (and resets the store) so existing tests behave as before; personal-memory tests enable it explicitly, inject clock, secure-input check, confirmer and store directory.

## Risks / Trade-offs

- Pool grows without bound (~100 KB/day heavy typing) → acceptable; fine-tune change adds compaction.
- Plain-text personal data on disk → user-only permissions, FileVault; Time Machine copies it (notice says "not sent over the network").
- Context keys use the current walk values of the two preceding nodes, which differ between stock walk and R-walk; right after launch (R still loading) or after a breaker trip, remembered picks made under R-walk can miss (review F10). Accepted; covered by a test that documents the behavior. Observed (task 3.5, `ContextKeyAcrossWalksTests`): a pick made under R-walk is not suggested with the stock walk when the preceding node differs (妳中 vs 你中), and vice versa.
- Secure input is system-wide → learning may pause while some app holds it (logged).
- Debounced save loses ≤ 3 s of picks on a crash or kill.

## Migration Plan

No existing persisted data; previous builds ignore `memory/`. Rollback = previous build.

## Open Questions

None blocking.
