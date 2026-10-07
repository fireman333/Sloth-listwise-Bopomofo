# Verification Report: personal-memory

Verified: 2026-10-07, branch `personal-memory`, uncommitted diff + untracked files (27 modified, 5 new source/test files). Method: `openspec-verify-change` (completeness / correctness / coherence) against proposal.md, design.md (D1–D15), tasks.md and the three spec deltas, by reading the code and tests. `openspec validate personal-memory --strict` → **valid**. Test suites were not re-run (last run per tasks.md: engine ctest 177/177, Rescorer 52/52, Debug 261 XCTest / 4 skips + 96 Swift Testing). One extra measurement was taken to settle a latency doubt (see S1).

## Summary

| Dimension    | Status |
|--------------|--------|
| Completeness | 18/20 tasks done (5.1 real-install test, 6.1 wrap-up open); 15/15 requirements have implementation |
| Correctness  | 15/15 requirements implemented; 33/36 scenarios covered by automated tests; 3 only by the pending task 5.1 (focus/keystrokes with notice, no network, real secure-input holder) |
| Coherence    | D1–D15 followed; 2 undocumented deviations (D13 canonicalization, D5 `u` vs `q` precedence) |

Known state (not new findings): task 5.1 not done (user remote, computer use denied; only install + codesign + stock McBopomofo unchanged checked in 5.0). Task 6.1 open (`/simplify` done, this verify was its last item). D6 was changed during apply to "only when the client process holds secure input" by user decision; design and spec are updated accordingly.

## Requirement → implementation → test map

| Requirement (spec) | Implementation | Tests |
|---|---|---|
| Long-term selection memory (personal-memory) | `Source/Engine/UserOverrideModel.cpp` (Policy ctor, `Observation::update` latestPickWins, `best`, `refresh`, `suggestWithKey`); `Source/LanguageModelManager.mm` `configureLearningEnabled:` (5000 / 7 d / {true,true}); `Source/KeyHandler.mm:289` observe, `:431` `_shouldObservePick` (D9 Plain skip) | gtests `LongTermSuggestedAfterThreeDays`, `LongTermFadesAfter140Days` (139/141 d), `LongTermLatestPickWins`, `RefreshKeepsMemoryFresh`, `CapacityEvictionWithRefresh`; XCTests `RefreshAndPlainModeTests` (6) |
| Memory survives restarts | `Source/AFMLearningStore.mm:355-431` (debounce, flush, atomic write), `:435-469` load + set-aside; `InputMethodController.swift:206` flush on deactivate | `LearningStoreTests.testPickFlushRestartRestoresSuggestion`, `testDebouncedSaveWritesAfterDelay`, `testMalformedFileIsSetAsideAndTypingWorks`, timing test; gtests Serialize* / DeserializeMalformed* |
| Sentence pool | `KeyHandler.mm:402` `clear` snapshot, `:462` `recordCommittedText:`, `:2645` stale drop, `:2775` `_lastComposingText`, `:841` Shift+Space reorder; `InputMethodController.swift:495`; `AFMLearningStore.mm:189` encoding, `:473` append | `SentencePoolCaptureTests` (14) incl. 7 commit paths, Esc / Ctrl+Enter / punctuation / Plain, Simplified both styles, replacement + macro, m/q/u, pref off |
| Nothing is learned in secure input | `AFMLearningStore.mm:233-250` (client-held check), `:322`; `KeyHandler.mm:431-441`, `:470` | `SecureInputLearningTests.testNothingLearnedUnderSecureInput` (injected check) — see W1 |
| Learning switch | `Preferences.swift:594` `setLearningEnabled`; `AFMLearningStore.mm:335-348`; `LanguageModelManager.mm` `configureLearningEnabled:` | `LearningStoreTests.testTurningOffSavesOnceThenNothing`, `testStockSessionModelNeverWritesMemoryFile`, `testLearningBackOnRestoresSavedPick`; `SentencePoolCaptureTests.testLearningTurnedOffStopsRecordingWithoutReactivation`; `RefreshAndPlainModeTests.testPlainBopomofoPickObservedWithLearningOff` |
| Clearing learning data | `AFMLearningStore.mm:493-514` (generation bump, delete); `LanguageModelManager.mm` `clearLearningData`; `PreferencesModel.swift` `clearLearningData()` | `LearningStoreTests.testClearRemovesDataAndEmptiesMemory`, `testQueuedWritesBeforeClearDoNotRecreateFiles`; `LearningPreferencesViewModelTests.testConfirmClears`, `testCancelDeletesNothing` |
| First-run notice | `Source/LearningNoticePanel.swift` (panel, controller); `InputMethodController.swift` activateServer | `LearningNoticeTests` (8). Focus/keystroke scenario → task 5.1 only |
| Local storage | `Preferences.swift` `learningDataDirectory`; `AFMLearningStore.mm:72-152` (0700 / 0600); `AppDelegate.swift:341-386` filter; FSEventStreamHelper public `Event` + FileEvents flag | `PersonalMemoryTests` (8), `LearningStoreTests.testFileAndFolderModes`, `testSaveAndPoolAppendDoNotReloadUserPhrases`, `LearningPreferencesTests.testLearningDataDirectoryIgnoresCustomUserPhraseLocation`. No-network → task 5.1 only |
| Selection-model preferences tab (MODIFIED) | `PreferencesView.swift` row 「學習：」, `PreferencesModel.swift`; strings in zh-Hant / en / Base | `LearningPreferencesViewModelTests` (4), render test (`PreciseModeTests`) |
| When preference changes take effect (MODIFIED) | as Learning switch + Clearing | same as above |
| User choices take precedence over model scores (MODIFIED) | `KeyHandler.mm:770-781` (`suggestWithKey` + `findOverrideTarget` + `_overrideTargetHasUserPhrase` at `:444`); `reading_grid.cpp` `findOverrideTarget` shared with `overrideCandidate` | `UserPhraseGuardTests` (3, R-walk on/off, precondition asserts); gtest `FindOverrideTargetMatchesOverrideCandidate`; existing `testUserOverrideSuggestionAppliedOverModel` |

## CRITICAL (must fix before archive)

- **C1. Task 5.1 incomplete (known).** Real-install interactive test not run: notice does not steal focus, files appear with 0600 after > 3 s, implicit commit on app switch, restart restores, clear from UI, learning off writes nothing, 0 sockets. Covers the three scenarios with no automated test (First activation focus, no network, real secure-input holder). Fix: run 5.1 at the Mac and record results in `../PLAN.md`, then tick `tasks.md:46`.
- **C2. Task 6.1 incomplete (known).** `/simplify` and test runs are done; only `/opsx:verify` was pending. Fix: after addressing or accepting W1–W2 below, tick `tasks.md:50` and reference this report.

Neither blocks a commit of work in progress; both block archive.

## WARNING (should fix)

- **W1. The "Secure input held by another process" scenario has no test of the real check.** All tests inject `secureInputCheck`, which bypasses `AFMSystemSecureInputCheck` (`Source/AFMLearningStore.mm:233-250`) and the pid lookup (`Source/InputMethodController.swift:194-200`). The pid comparison, the unknown-pid → not-secure rule and the `kCGSSessionSecureInputPID` parsing are untested, and this is the logic that was changed during apply. Fix: extract a pure function, e.g. `AFMSecureInputHeldByClient(BOOL systemEnabled, int holderPID, pid_t clientPID)`, call it from `AFMSystemSecureInputCheck`, and add an XCTest table: (enabled, holder == client) → secure; (enabled, holder ≠ client, e.g. loginwindow) → not secure; (enabled, client 0) → not secure; (disabled) → not secure. Optionally have 5.1 check Terminal Secure Keyboard Entry.
- **W2. An existing but unreadable memory file is treated as missing and later overwritten, not set aside.** `Source/AFMLearningStore.mm:443-452`: `dataWithContentsOfFile:` returns nil for both ENOENT and e.g. EACCES / EIO, so `exists` stays NO; the next save renames `.tmp` over it. Spec "Memory survives restarts" says an *unreadable or malformed* file is kept renamed aside. Fix: read with `dataWithContentsOfFile:options:error:` and, when the error is not "no such file", rename it to `override-model.bad-<epoch>.txt` as in the malformed branch; add a test with a 000-mode file (or accept and narrow the spec wording to "malformed").

## SUGGESTION (nice to fix)

- **S1. Latency of the real secure-input check is not in the recorded numbers.** Task 3.6 injected `secureInputCheck = { false }` (`McBopomofoTests/NeuralSelectorTests.swift`, `runInProcessLatency`). On this Mac `IsSecureEventInputEnabled()` is always true, so every pick and every `recordCommittedText:` calls `CGSessionCopyCurrentDictionary()`. Measured here (2,000 calls, -O, same Mac): p50 0.05 ms, p95 0.11 ms, max 23.8 ms (single outlier, likely first call). Within the 1 ms / 10 ms p95 budgets, but it is ~10x the recorded `recordCommittedText` p95 (0.015 ms). Fix: add these numbers to `../PLAN.md`, or run the latency test once with the real check; optionally cache the result per key event.
- **S2. D13 deviation not in design.md.** Design D13 says `URL.resolvingSymlinksInPath()`; implementation uses `realpath` on the longest existing prefix (`Source/AppDelegate.swift:367-386`). The reason is only in tasks.md 2.1. Fix: update D13 text to the implemented method and its reason.
- **S3. `u` vs `q` precedence differs from D5 wording.** `Source/KeyHandler.mm:509-513` labels a user-phrase node `u` even inside a precise walk; D5 defines `q` as "every node of the applied fused walk that is not a fixed position (manual pick, associated phrase, memory suggestion)", which would make it `q`. The implementation is reasonable (keeps the provenance a fine-tune needs). Fix: add "user-phrase nodes are `u`, also in a precise walk" to D5.
- **S4. Picks made between turning learning off and the next activation can be persisted later.** After `setLearningEnabled(false)` the long-term model stays active until the next activation (allowed by spec), and picks still enter it (`KeyHandler.mm:289`, `markDirty` no-ops). If learning is turned back on before reactivation, the next save writes those off-period picks. Fix (optional): in `_shouldObservePick` (`KeyHandler.mm:431`) also return NO when `!AFMLearningStore.sharedStore.recordingEnabled` and the long-term model is active, or document it as accepted in design Risks.
- **S5. Client pid lookup takes the first process with the bundle id.** `InputMethodController.swift:194-200` uses `runningApplications(withBundleIdentifier:).first`; with several instances of the same app the pid can be wrong, so secure input of the real client could be missed (learning not paused) or vice versa. Fix: compare `kCGSSessionSecureInputPID` against all pids of that bundle id, or document the limitation in D6.
- **S6. Custom user-phrase location scenario only checks the path.** `LearningPreferencesTests.testLearningDataDirectoryIgnoresCustomUserPhraseLocation` (`McBopomofoTests/PersonalMemoryTests.swift:500-519`) asserts directory strings, not that "nothing is written to the custom location". Fix: with a custom location set, do a pick + commit + flush and assert the custom folder has no new files and `memory/` under the root has both files.
- **S7. tasks.md 1.1 note lists an accessor that no longer exists.** It mentions `decayConstant()`; `Source/Engine/UserOverrideModel.h` has only `capacity()`, `policy()`, `size()`, `clear()` (probably removed by `/simplify`). Fix: edit the 1.1 note.

## Design adherence (D1–D15)

D1 followed (policy flags, refresh, `suggestWithKey`, to_chars/from_chars, all-or-nothing, capacity on load; extra validation documented in task 1.2). D2 followed (actual-configuration comparison via `learningConfigurationActive`; final synchronous save in `setRecordingEnabled:NO`). D3 followed (debounce 3 s, long-term-only saves, atomic generation, set-aside). D4 followed (single choke point, pre-OpenCC text match, Shift+Space reorder, drop rule). D5 followed except S3. D6 followed as revised. D7 followed (shared `findOverrideTarget`, overlap check on `_latestWalk`). D8 followed (refresh only on accepted snapshot, value check). D9 followed (policy used as the "learning on" proxy). D10–D12 followed (localized strings present; panel properties as specified; flag set only when visible; R3-1 wording already fixed). D13 followed with the S2 deviation. D14, D15 followed (principal-class observer).

## Final assessment

2 critical issues, both the known open tasks (5.1, 6.1): fix before archiving. 2 warnings, 7 suggestions. No requirement is unimplemented and no code contradicts the spec on the tested paths.

**Safe to commit** the current diff (no correctness defect found that would make the branch worse than its parent; W1/W2 are a test gap and an edge case). Archive stays blocked by 5.1.
