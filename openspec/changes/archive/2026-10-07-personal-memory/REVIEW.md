# Adversarial review: personal-memory

Reviewed: 2026-10-07T17:28:11+08:00. Scope: proposal.md, design.md, tasks.md, specs/{personal-memory,selection-pipeline,preferences}, checked against the source tree (not only the docs). Settled product decisions (persist stock UOM, 5,000 / 7 d, pool with manual flag, secure-input skip, on by default + notice, user phrase > memory > R, no training) are not re-litigated; where a finding touches one, it only points out a consequence the docs do not state.

Severity: P1 = must fix on paper before `/opsx:apply`; P2 = fix before apply unless explicitly accepted; P3 = fix during apply / spec wording; P4 = minor; P5 = nit.

Counts: P1 x3, P2 x5, P3 x7, P4 x7, P5 x1.

Design claims verified as correct: UOM structure / score / 2^-20 threshold (`Source/Engine/UserOverrideModel.cpp:39,190-242`); global instance and pointer (`Source/LanguageModelManager.mm:33-38,521-524`, `Source/KeyHandler.mm:83,171`); observe at `KeyHandler.mm:248`, suggest at `KeyHandler.mm:595-602` (Bopomofo only); `resetUserOverrideModelForTesting` move-assigns in place (`LanguageModelManager.mm:91-94`), so pointer stability under swap holds; `Node::currentUnigram()` returns the unigram including `isUserPhrase` (`Source/Engine/gramambular2/reading_grid.cpp:899-901`, flag set in `McBopomofoLM.cpp:166-169,233-239,400`); per-PID test root and isolated test defaults suite (`Source/Preferences.swift:99-101,234-240,584-594`); all UOM access is on the main thread (only KeyHandler touches it), so the main-thread swap + serialize-on-main + write-on-serial-queue plan is thread-safe as designed.

---

## P1 — must fix before apply

### F1 (P1). Memory writes DO trigger user-phrase reloads: the watcher is recursive over the application-support root
Evidence:
- Default user-phrase folder is the application-support root itself: `Source/Preferences.swift:615-618` (`defaultUserPhraseLocation` returns `applicationSupportRoot`); `LanguageModelManager.mm:477-484`.
- The FSEvents stream watches that path with `kFSEventStreamCreateFlagNone` (`Packages/FSEventStreamHelper/.../FSEventStreamHelper.swift:55-74`). FSEvents streams are recursive over the directory tree, so `<root>/memory/` changes are delivered.
- The delegate reloads user phrases and phrase replacement on *any* event, without looking at the path (`Source/AppDelegate.swift:326-333`). Each load bumps `gReloadGeneration` (`LanguageModelManager.mm:100,108`), which is part of the precise-mode request key (`KeyHandler.mm:2673`).
- Consequence: every debounced UOM save (every 3 s while picking) and every pool append (every commit) re-parses the user phrase file on the main thread about 1 s later (stream latency 1.0) and invalidates any in-flight precise request. This contradicts proposal "Impact > Data" ("Separate from the user-phrase folder so the phrase file watcher is not triggered") and the spec requirement "Local storage: writing it SHALL NOT trigger a reload of user phrases".
Fix: in design D11, state the actual mechanism. Recommended: filter events by path in the delegate (ignore events whose path is under `<root>/memory/` or `<root>/models/`) and reload only if at least one event remains. `FSEventStreamHelper.Event.path` is currently internal (`FSEventStreamHelper.swift:31-35`, no `public`), so the package must expose it; add `kFSEventStreamCreateFlagFileEvents` if per-file paths are wanted. Add a task + XCTest: after a save and a pool append, `LanguageModelManager.reloadGeneration` is unchanged and `loadUserPhrases` was not called (inject the delegate or count reloads). Alternative (moving memory outside the root) would contradict the "Local storage" requirement, so prefer filtering.

### F2 (P1). Commit capture is wrong for the common commit paths; several sites clear the walk before committing, and implicit commits are not covered
Evidence:
- `_handleEnterWithState` calls `[self clear]` (which empties `_latestWalk`, `KeyHandler.mm:350-356`) *before* creating the Committing state (`KeyHandler.mm:1300-1306`). Same order at Ctrl+` Big5 (`KeyHandler.mm:784-789`), Ctrl+\ feature menu (`KeyHandler.mm:798-803`), force commit (`KeyHandler.mm:366-371`: walk captured only as text via `buildInputtingState`). Design "Context" says "the composing text comes from `_latestWalk` / `_grid->readings()` before `[self clear]`" — the committed text actually comes from `state.composingBuffer`, and the walk is already gone at most sites.
- Implicit commits are not `InputStateCommitting` at all: `InputMethodController.handle(state: .Empty, previous: NotEmpty)` commits the previous buffer (`Source/InputMethodController.swift:523-533`), as do `.Deactivated` (`:494-508`) and the simple-candidate-window handlers (`:789-790,807-808`). These are hit by Caps Lock (`KeyHandler.mm:437-441`), Shift+letter with letterBehavior 0 (`KeyHandler.mm:854-859`), mode switch (`InputMethodController.swift:200-202`), Command-key input-source switch (`:231-234`), and every app switch / click elsewhere (`deactivateServer`, `:185-189`).
- `deactivateServer` calls `keyHandler.clear()` *before* `handle(state: .Deactivated())` commits the buffer (`InputMethodController.swift:185-189`), so no KeyHandler-side "right before commit" hook can see the walk on the most frequent real-world commit path (switching apps / clicking away). The same order applies to `_manualPickInComposition`, which D5 resets in `clear`.
- Task 3.2's site list ("Enter, candidate-window single-candidate commit, force commit, any other site emitting InputStateCommitting") misses all implicit commits; D4 "right before each Bopomofo-mode commit" spreads capture over ~10 call sites that will drift.
Fix (single choke point): in `-[KeyHandler clear]`, move (not copy; O(1)) `_latestWalk`, the manual flag, the input mode and a composition id into `_pendingPoolSnapshot` before resetting. Make `InputMethodController.commit(text:client:)` (`InputMethodController.swift:474-492`, the only place text reaches the client) call `keyHandler.recordCommittedText(text)` *before* OpenCC conversion; the KeyHandler accepts the snapshot only if it is Bopomofo mode and the snapshot's joined node values equal `text`, then drops it. Any insert/next composition drops a stale snapshot (so Esc-discard, Ctrl+Enter reading/ruby/braille output, numpad/caps letters, Big5 output never produce a record). Update D4/D5/tasks 3.2 accordingly, and add XCTests for: commit via `deactivateServer` with a non-empty buffer, via `commitComposition` (mouse click), via Caps Lock, via mode switch, Esc discard (no record), Ctrl+Enter reading output (no record).

### F3 (P1). The chosen notice component steals focus from the app the user is typing in
Evidence: `NonModalAlertWindowController.show` raises the window to `CGShieldingWindowLevel() + 1`, calls `makeKeyAndOrderFront` and `NSApp.activate(ignoringOtherApps: true)` (`Source/NonModalAlertWindowController.swift:86-90`). Shown from `activateServer` (D10) it makes the IME process the active app, which deactivates the client (triggering the implicit commit in F2), and the user's next keystrokes go to the alert; Return hits the default button 「打開設定」. This violates the spec ("SHALL NOT block or delay key handling", scenario "typing continues without delay"). It is fine for the existing user-initiated "Check for Updates" flow, not for an unsolicited notice.
Fix: specify a presentation that does not take key focus: e.g. a non-activating panel (`NSPanel` with `.nonactivatingPanel`, `becomesKeyOnlyIfNeeded`, `orderFrontRegardless`, normal floating level, no `NSApp.activate`) or the existing `NotifierUI` package, shown after a short delay; buttons only by mouse. Alternatively show the notice the first time the user opens the menu / preferences. Add a manual check to task 5.1: type immediately after first activation; every keystroke lands in the client and the client stays frontmost.

---

## P2 — fix before apply unless explicitly accepted

### F4 (P2). "Composed characters, before Simplified conversion" is false in the model conversion style (and with phrase replacement / macros)
Evidence: with `ChineseConversionStyle == model`, conversion happens inside the LM (`KeyHandler.mm:122,206`; `McBopomofoLM.cpp:392-397`), so `node->value()` is already Simplified; `InputMethodController.commit` skips OpenCC in that style (`InputMethodController.swift:476-484`). Phrase replacement and macro expansion also rewrite `value` (`McBopomofoLM.cpp:370-384`). D4 records "the node's composed value before OpenCC", so the spec scenario "Simplified Chinese output → record holds 這裡" fails in model style, and the pool mixes scripts and expanded macros (e.g. dates). Test 3.2 only names one style.
Fix: record `currentUnigram().rawValue()` (pre-replacement, pre-macro, pre-converter value) with fallback to `value()` when `rawValue` is empty (mono-syllable user phrases are rebuilt with an empty rawValue, `McBopomofoLM.cpp:236-239`); optionally also store the displayed value. Spec: say "the dictionary characters before phrase replacement, macros and Simplified conversion". Tests: both conversion styles, phrase replacement on, a date macro.

### F5 (P2). The user-phrase guard checks the wrong node set, and its test can pass without the guard
Evidence:
- `overrideCandidate(loc, value, type)` picks the first node among *all* nodes overlapping `loc` that has the value (`reading_grid.cpp:821-848`) and then resets every node overlapping that node's span (`:850-863`). A force-high-score multi-character suggestion (UOM case 2, `UserOverrideModel.cpp:118-126,136-137`) can therefore replace a user-phrase node *next to* the cursor node. D7 only checks "the node of `_latestWalk` covering the cursor".
- The suggestion key head is `(reading, unigrams()[0].value())` (`UserOverrideModel.cpp:263-264`). Adding a user phrase usually changes `unigrams()[0]` for that reading (user unigrams are inserted first, `McBopomofoLM.cpp:217-243`), so the key no longer matches and `suggest()` returns empty regardless of the guard. The guard only matters when the user phrase has the same value as the previous top unigram, or covers a different span than the head. Task 3.4's scenario ("a user phrase for the same readings that the walk selects") will typically pass vacuously.
Fix: D7 should resolve the override target first (mirror `overrideCandidate`'s search; add a small const grid helper such as `ReadingGrid::findOverrideTarget(loc, value) -> NodeInSpan` with gtest) and skip the suggestion if any walk node intersecting the target's span is a user phrase. Task 3.4: construct the case where the key still matches (user phrase equal to the stock top value, e.g. user re-asserting 中 after a remembered 鍾; and a 2-char forced suggestion adjacent to a 1-char user phrase), and assert as a precondition that `suggest()` is non-empty for that walk.

### F6 (P2). Toggling learning races with the debounced save
Evidence/design gap: D2 swaps the global model on activation when the setting changed; D3 has a pending 3 s timer and a background queue. If learning is turned off with a dirty model, (a) the last picks are lost unless flushed before the swap, and (b) a timer that fires after the swap serializes the stock session model over `override-model.txt`, violating "Learning off: no memory or pool file ... is created or modified". Symmetrically, pool records built before the switch may be appended after it. Also, per spec, pool recording continues until the next activation after the user turns learning off, which is surprising for a privacy switch.
Fix: in D2/D3: on every config change, flush (if enabled and dirty) *then* swap; give the store a configuration generation and have timers/queued writes no-op when their generation is stale; check `Preferences.learningEnabled` at record time so pool recording stops immediately when turned off (cheap; keep the UOM swap at activation). Tests: dirty model + turn off + wait > 3 s → file mtime/bytes unchanged; turn off → commit → no record even before reactivation.

### F7 (P2). With a 7-day half-life, corrections become sticky; design claim "overridable by picking again" is false
Evidence: score = `count/totalCount * decay` (`UserOverrideModel.cpp:232-242`), and `suggest` takes the max (`:204-218`). Example: candidate A picked 3 times, then the user corrects to B one day later: A = 3/4 x 0.5^(1/7) = 0.68, B = 1/4 = 0.25, so A is still suggested; the user needs 3 more corrections (about N for N earlier picks). With stock 1.5 h half-life the old picks have decayed to ~2^-16 after a day, so one correction wins. Design "Risks" says a wrong pick is "still overridable by picking again"; no spec scenario covers correction.
Fix: decide and spec it. Least invasive option within "extend the stock UOM": in `Observation::update`, when the observed candidate differs from the current best, scale down the other candidates' counts (e.g. reset them to 0 or halve) so the latest manual pick wins immediately; gtest "pick A x3, pick B once a day later → B suggested". If the stickiness is accepted instead, fix the Risks text and add a scenario stating how many corrections are needed.

### F8 (P2). Pool schema is collected permanently but cannot support the planned fine-tune without per-node provenance
Gap: the pool's only purpose is a later on-device fine-tune of R. With one composition-level `manual` flag, a record with `manual:false` is mostly R's own output, and a record with `manual:true` does not say which node was picked. Training on it risks self-reinforcement, and the information cannot be reconstructed later for already-collected records. Adding fields does not change the settled decision ("record every commit with a manual-pick flag"); it only makes the data usable.
Fix: keep `manual`, and add per node a source tag, e.g. `["ㄓㄨㄥ-ㄏㄨㄚˊ","中華","p"]` with `p` = manual pick, `a` = associated phrase, `m` = memory suggestion, `q` = precise result, `u` = user phrase, `r`/`s` = R-walk / stock walk; plus record-level `walk` (stock / r / precise), `app` version and R model id. Node provenance is available at capture time (override type on the node, `_latestWalkUsedNeural`, `_preciseOverride`). Update D4, the spec field list and test 3.1/3.2.

---

## P3 — fix during apply / wording

### F9 (P3). "Pick remembered after a day" and gtest 1.2 do not distinguish long-term from stock
Stock 1.5 h half-life still suggests after 24 h (24/1.5 = 16 half-lives; 2^-16 > threshold 2^-20, `UserOverrideModel.cpp:39,234-236`); stock stops only after 30 h. So the 24 h scenario and gtest 1.2 pass with the stock configuration. Fix: use 3 days (stock = 0, long-term ~0.74) in the spec scenario and gtest; add the boundary pair 139 d suggested / 141 d not, and assert the same input gives no suggestion with the stock config.

### F10 (P3). Memory hits depend on R-walk context; misses at startup / breaker trip
The key uses the *current* values of the previous two nodes (`UserOverrideModel.cpp:276-292`), which come from R-walk when it is active (`KeyHandler.mm:2611-2616`) and from the precise walk if one was applied when the pick happened (`KeyHandler.mm:2600-2605`, `prevWalk` at `:229`). After a restart R loads in the background, so early keystrokes use the stock walk and may produce different context values, i.e. persisted picks silently miss until R is ready (and when the breaker trips). Fix: document in design Risks; add an XCTest that a pick made with R on is (or is not) suggested with R off, so the behavior is known; optionally observe with the R/stock walk rather than a precise walk.

### F11 (P3). Applied suggestions never refresh the memory
`suggest` does not touch LRU order or timestamps (`UserOverrideModel.cpp:190-220`); only manual picks do. A remembered pick that works (so the user never re-picks) expires after 140 days and is LRU-evicted before rarely used contexts that were picked more recently. With a session-only stock UOM this did not matter. Decide: accept (document) or, on commit, refresh the timestamp/LRU position of contexts whose suggestion was committed unchanged (cheap with the F2 snapshot). Needs a user decision; record it in design.

### F12 (P3). `IsSecureEventInputEnabled` is system-wide
It reports secure input enabled by *any* process (e.g. Terminal "Secure Keyboard Entry", password managers, apps that leak it), while macOS already switches to a Roman layout in real password fields. So the check mostly produces false positives: learning silently stops, possibly for hours. Fix: spec wording "while secure event input is enabled anywhere on the system"; show a status line in the 「選字模型」 tab ("學習已暫停：其他程式啟用了安全輸入") or log once per transition; keep the conservative skip.

### F13 (P3). Test harness interactions with the new default
- `resetUserOverrideModelForTesting` always restores the stock 500 / 5400 model (`LanguageModelManager.mm:91-94`) while learning defaults to on; D2's "cheap when unchanged" check would then believe the long-term model is active. Make the check compare the actual model configuration, and make the reset use the current configuration (or set learning off by default in test setUp).
- With learning on by default in the test host, every existing Bopomofo XCTest writes pool records and arms save timers in the per-PID root, which can fire during unrelated tests. Recommend: learning off in the shared test setUp, enabled explicitly in personal-memory tests; inject the store and the clock.

### F14 (P3). "Stock when learning off" and the project's upstream-identity rule
- The guard (D7) applies with learning off, so "With learning off the input method behaves like stock McBopomofo" (proposal) is not exact; personal-memory spec scopes it to "for remembering picks" (fine), the proposal does not.
- `openspec/config.yaml` design rule: "Engine changes must keep the default (alpha=0 / model absent) behavior identical to upstream McBopomofo, except for entries removed by the reading-policy capability." Learning on by default (5,000 / 7 d / persisted) and the guard both change default behavior. The selection-pipeline delta's final clause ("behave exactly as in stock McBopomofo, apart from the user-phrase rule") also ignores the long-term memory.
Fix: add an explicit exception in design ("Deviation from upstream defaults") and reword the selection-pipeline clause to "apart from the user-phrase rule and the personal-memory capability".

### F15 (P3). Plain Bopomofo picks still feed the persistent memory
`fixNodeWithReading` (with `observe`) is called for Plain Bopomofo candidate picks too (`InputMethodController+CandidateControllerDelegate.swift:56-71`, `KeyHandler.mm:246-249`), so with learning on, plain-mode picks are persisted for 140 days and share the key space used by Bopomofo-mode suggestions. Stock observes in-session as well, so this may be intended, but the spec only excludes plain mode from the pool. Fix: state the rule in "Long-term selection memory" (recommend: do not observe in plain mode when learning is on, or state that it does) and test it.

---

## P4 — minor

### F16 (P4). Serialization details (D1)
`observationCount` is redundant (always the sum of candidate counts, `UserOverrideModel.cpp:222-230`) and repeated per line; either drop it and recompute, or treat a mismatch as malformed. Timestamps must round-trip exactly: write with `std::to_chars` / 17 significant digits, read with `std::from_chars` (locale-independent). gtest: round trip compares raw timestamps bit-exactly.

### F17 (P4). Termination flush has no hook today and IMEs are usually killed
There is no `applicationWillTerminate` in `Source/AppDelegate.swift`; the installer and logout frequently kill the process. The real guarantee is the 3 s debounce + `deactivateServer` flush. Spec "Restart" scenario should say "after the save interval"; task 5.1 should wait > 3 s after the pick before killing.

### F18 (P4). Load cost at activation is on the main thread
D3 allows up to 50 ms on activation; keystrokes typed right after app switch queue behind it. Load once at launch (`applicationDidFinishLaunching`) and only reload on an actual config change; measure activation-to-first-keystroke in task 2.2.

### F19 (P4). Spec testability / project-rule gaps
- "Nothing is learned in secure input", "Learning switch" and "Memory survives restarts" touch key handling or activation but state no numeric budget or fallback (config.yaml spec rule). Add "within the 10 ms p95 budget" / load budget and "on failure, typing continues with an empty memory".
- "Same context" is undefined: define it as the same two preceding nodes (readings and characters) as the remembered pick.
- "Clear and type again → THEN the stock choice is shown": with R-walk on it is the R-walk choice; say "the choice made without memory".
- "Learning off → SHALL NOT read or write the memory folder" conflicts with clearing while off; add "except when the user clears learning data".

### F20 (P4). Record-drop rule needs a precise definition
"Compositions with no Bopomofo syllable" — define in D4 as: no node whose reading is a Bopomofo syllable (readings starting with `_` such as `_punctuation_*` / `_letter_*`, and `" "`, do not count). Same predicate for unit tests.

### F21 (P4). Missing verifications in tasks
- Clearing leaves `data.txt`, excluded phrases, replacements, models and preferences untouched (spec "Clearing learning data").
- Writing memory / pool does not change `reloadGeneration` (F1).
- A commit whose buffer shows an applied precise result records the precise text with correct provenance (F8).
- First-run notice: decision function plus "flag set even if the window is never shown" behaviour stated.

### F22 (P4). Ordering
Task 3.2 depends on the snapshot / store API (F2) and on 2.2's serial queue; state "3 after 2" explicitly. The F1 watcher fix must land before or with 2.2, otherwise the first real save already causes reloads.

---

## P5 — nit

### F23 (P5). Notice wording
"This data stays on this Mac and is never sent anywhere": Time Machine and other local backups will copy `~/Library/Application Support`. Consider "is never sent over the network by this input method".

---

## Round 2

Reviewed: 2026-10-07 (revised proposal.md, design.md D1–D15, tasks.md, three spec deltas), re-checked against the source. User decisions F3 / F7 / F11 taken as settled.

Counts (new findings only): P1 x0, P2 x3, P3 x5, P4 x4, P5 x1.

### Status of round-1 findings
- F1 resolved (D13 path filter + public `Event.path`, task 2.1/2.3, spec "No phrase reload on save"); see R2-6 for a path-canonicalization gap.
- F2 partially resolved: the choke point (D4) is right and covers Enter, force commit, deactivate, Caps Lock, mode/input-source switch (`SwitchingInputSource` is a subclass of `Empty`, `InputState.swift:79`, so it commits via the Empty handler). Still open: R2-2 (annotation text mismatch) and R2-3 (Shift+Space site).
- F3 resolved in principle (D12 non-activating panel); details in R2-7.
- F4 resolved (`rawValue()` with fallback; both conversion styles, replacement and macro in task 3.1).
- F5 resolved (D7 `findOverrideTarget`, task 1.3, task 3.3 precondition assert, new selection-pipeline scenario).
- F6 not resolved: the flush/disable ordering is now internally contradictory (R2-1).
- F7 resolved (latestPickWins, gtest 1.1, spec "Correction wins immediately").
- F8 resolved in schema; provenance tracking has a keying flaw (R2-4).
- F9 resolved (3 d, 139/141 d, stock contrast in gtest 1.1).
- F10 resolved (Risks + task 3.5).
- F11 resolved (D1 `refresh`, D8, spec scenario); trigger condition ambiguous (R2-5).
- F12 resolved (spec wording, os_log on transition).
- F13 resolved (D2 compares actual configuration, D15 learning off in shared setUp).
- F14 resolved ("Deviation from upstream defaults"; selection-pipeline clause amended).
- F15 resolved (D9, task 3.4).
- F16 resolved (sum recomputed, `to_chars`/`from_chars`, bit-exact gtest).
- F17 resolved (D3 note, spec Restart waits > 3 s).
- F18 resolved (load only at launch / setting change).
- F19 resolved (budgets, context definition, "choice made without the memory", clear exception).
- F20 resolved (D4 drop rule).
- F21 resolved (tasks 2.3, 3.1).
- F22 resolved (order 1 → 6, watcher first).
- F23 resolved (spec notice wording).

### R2-1 (P2). Turning learning off: "pending changes are saved first" cannot happen as designed, and spec / scenario / task contradict each other
Evidence:
- D2 "disabled": flush to the serial queue, *then* bump the store generation. D3: "Every queued block captures the generation and no-ops if it changed (clear / disable)". The flush block captured the old generation, so the bump right after it cancels the flush itself.
- D2 also says the pref setter calls `setRecordingEnabled(false)` immediately; D3's save and `flush` ("does the same immediately") run only "when ... recording is enabled". So the dirty picks at toggle time are never written. The later `configureLearning(disabled)` flush is also gated off.
- personal-memory spec "Learning switch" requirement: "pending changes made while it was on are saved first", but its scenario "Learning off" says no file is modified "including by a save that was pending when learning was turned off". Task 2.3 tests "dirty model + learning off + wait > 3 s → file bytes/mtime unchanged". These three cannot all hold.
Fix: decide one behavior, recommended "flush once at the moment of switching off, then nothing". Implement `setRecordingEnabled(false)` as: if dirty, serialize and enqueue the write with the *current* generation, then set recording off and bump the generation for anything still pending (timer). The disable path in `configureLearning` then has nothing to flush. Reword the scenario: "after learning is turned off (and any pending save completed at that moment), no file in the memory folder is created or modified". Task 2.3: "dirty + off → file written exactly once at the switch; further picks/commits + wait > 3 s → unchanged".

### R2-2 (P2). The text match silently drops every record when Bopomofo font annotation is enabled
Evidence: with `Preferences.bopomofoFontAnnotationSupportEnabled`, `buildInputtingState` writes `variantAnnotator->annotate(...)` output (variant selectors / PUA code points) into the composing buffer instead of `node->value()` (`KeyHandler.mm:2486-2507`). The committed text is that buffer, so D4's "joined `value()`s equal the text" never matches. All pool records are lost, and D8 refresh too if it depends on the match. Nothing in tasks covers this preference.
Fix: compare against the same string the buffer is built from. Factor the composed-string part of `buildInputtingState` into a helper (`-composedStringForWalk:`) and store its result in the snapshot at `clear` time (O(n), only at clear), or compare against both the plain join and the annotated string. Add a task 3.1 case with annotation on.

### R2-3 (P2). Shift+Space (and Space when it is not the candidate key) commits before `clear`, so the snapshot is missing
Evidence: `KeyHandler.mm:658-666` commits `composingBuffer` via `InputStateCommitting` and only then calls `[self clear]`. With D4, `recordCommittedText` runs with no snapshot for this composition (the previous one was dropped on the first insert), and the following `" "` commit does not match. The composition is lost. Not listed in task 3.1.
Fix: move `[self clear]` above the first `stateCallback(committing)` at that site (the buffer string is already captured). Add "Shift+Space at end of buffer → one record" to task 3.1. Also add a debug assertion in `recordCommittedText`: a Bopomofo-mode commit that equals neither the snapshot nor a known non-composition output is logged, to catch future sites.

### R2-4 (P3). Provenance map keyed by span location drifts when the buffer is edited mid-way
D5 keys provenance "by span location". `insertReading` / `deleteReadingBeforeCursor` at a mid-buffer cursor shift every later location, so `p`/`m`/`q` tags (and the stored UOM key for D8) are attributed to the wrong nodes. Fix: key by node identity (`NodePtr`, held weakly or compared by pointer at record time). The grid rebuilds nodes around an edit, and that also drops their overrides, so a missing node correctly means "no longer this source". For `q`, define it as "nodes in the precise walk that are not in the R-walk / stock walk of the same grid", and clear `q` when `_preciseOverride` is reset (`KeyHandler.mm:2600-2606`). "overridden positions or the whole buffer" leaves the choice to the implementer.

### R2-5 (P3). D8 trigger is ambiguous
"At `recordCommittedText` (or at commit when learning on, even if the record is dropped for not matching)" would also refresh on non-composition outputs (Ctrl+Enter reading/ruby/braille, `KeyHandler.mm:741-770`), where the user did not commit the suggested characters. Fix: refresh only when the snapshot matched the committed text (after R2-2). Test: Ctrl+Enter output does not refresh.

### R2-6 (P3). Watcher path filter needs canonical paths
FSEvents reports resolved paths: under XCTest the root is `NSTemporaryDirectory()` (`/var/folders/...`, `Preferences.swift:588-592`), but events arrive as `/private/var/folders/...`. A plain prefix check fails there, so the "real save leaves reloadGeneration unchanged" test in 2.3 fails (or, with only injected events, the bug goes unnoticed). Fix: compare `URL(fileURLWithPath:).resolvingSymlinksInPath().standardized` of both sides, with a trailing `/` on the directory prefix (so `memory-old/` does not match). Treat `kFSEventStreamEventFlagMustScanSubDirs` / root-dir events as "reload" (current behavior).

### R2-7 (P3). Notice panel can be missed or linger
- D12 sets `AFMLearningNoticeShown` before showing. Without `collectionBehavior` `.canJoinAllSpaces`/`.fullScreenAuxiliary`, a panel ordered front while the client is in a full-screen Space may not be visible, and the one-time notice is spent unseen.
- Mouse-only buttons and no timeout: if the user ignores it, a floating panel stays above every app indefinitely.
- `canBecomeKey` is read-only on `NSWindow`, so it needs an `NSPanel` subclass override. `hidesOnDeactivate` defaults to true for panels; clicking 「打開設定」 activates the app, so returning to the client hides the panel. Set it explicitly to the behavior you want. Existing precedent: `Packages/TooltipUI/.../TooltipController.swift:41-44` and the candidate panels use `[.borderless, .nonactivatingPanel]`.
Fix: set collectionBehavior, a subclass with `canBecomeKey/canBecomeMain = false`, auto-close after e.g. 30 s or at the next `deactivateServer`, and set the flag when the panel was actually ordered on screen (`isVisible` / `occlusionState`). Task 4.2: assert collectionBehavior and the subclass.

### R2-8 (P3). Saves should be gated on the model's configuration, not only on `recordingEnabled`
D2 turns recording on immediately in the pref setter, while the model swap waits for the next `activateServer`. In that window a `markDirty` on the stock session model would let the timer serialize the 500-context stock model over the saved `override-model.txt`. This is unlikely in practice (returning to a client normally activates first), but it would destroy data. Fix: `markDirty`/save are no-ops unless the current UOM is the long-term configuration and that configuration was loaded from (or created for) the current generation.

### R2-9 (P4). `generation` is read on the serial queue and written on main
Make it `std::atomic<uint64_t>` (or capture and compare on main before enqueuing, and only bump on main between blocks).

### R2-10 (P4). Snapshot edge cases
- `clear` is also called on activation and in other places without a commit. It should not overwrite an unconsumed non-empty snapshot with an empty walk (cheap guard), so an ordering change elsewhere cannot erase a pending record.
- An implicit commit with an unfinished syllable (the buffer contains raw ㄅㄆㄇ from `_bpmfReadingBuffer`, `KeyHandler.mm:2581-2583`) never matches, so the whole sentence is dropped. Either compare after stripping the reading-buffer part, or state it in the spec ("compositions committed with an unfinished syllable are not recorded").

### R2-11 (P4). Preferences delta vs personal-memory on timing
preferences "Turning learning off" scenario: "from the next activation no learning data is written". personal-memory: recording and saving stop "immediately". These don't strictly contradict, but they test different things. Align the preferences scenario to "immediately (pool) / next activation (memory configuration)".

### R2-12 (P4). Clear during an open composition
A composition typed before the clear and committed after it is recorded after the clear (D11 states this). The spec scenario "no pool data from before the clear" can be read as forbidding it. Add one sentence to the requirement: "compositions committed after the clear are recorded as new data".

### R2-13 (P5). Wording
- personal-memory "Sentence pool": "by a candidate or associated-phrase pick that commits" only happens in Plain Bopomofo mode, which is not recorded. Drop the phrase.
- D1: "when the observed candidate is observed" → "when a candidate is observed".

---

## Round 3

Reviewed: 2026-10-07 (design D2/D3/D4/D5/D8/D12/D13, specs/personal-memory, specs/preferences, tasks.md).

### Status of round-2 findings
- R2-1 resolved: `setRecordingEnabled(false)` cancels the timer, then does one synchronous final save, then turns recording off (D2). Clear is now the only thing that bumps the generation, so the final save is no longer cancelled. Spec requirement, both "Learning off" scenarios and task 2.3 now agree.
- R2-2 resolved: the match uses `_lastComposingText`, the exact composing-buffer string including annotation (D4); task 3.1 has an annotation-on case.
- R2-3 resolved: the Shift+Space site is reordered to clear-then-commit (D4); Shift+Space is in the spec text and task 3.1.
- R2-4 resolved: provenance is keyed by `NodePtr`, and `q` is defined as every non-fixed node of the applied fused walk (D5); task 3.1 tests an insertion before a picked node.
- R2-5 resolved: refresh happens only on an accepted snapshot (D8).
- R2-6 resolved: both sides are canonicalized with `resolvingSymlinksInPath` (D13). Applying the same function to both sides keeps the comparison consistent, even though Foundation normalizes `/private/var` to `/var`.
- R2-7 resolved: subclass with `canBecomeKey`/`canBecomeMain` false, collectionBehavior set, and the flag is set when the panel is ordered front (D12). Staying until clicked is accepted as a decision. See R3-1.
- R2-8 resolved: saving is gated on the active UOM being the long-term configuration (D3); task 2.3 says the stock session model never writes the file.
- R2-9 resolved (`std::atomic<uint64_t>`, captured at enqueue).
- R2-10 resolved: `clear` with an empty walk keeps the pending snapshot, and the unfinished-syllable rule is in the spec and task 3.1.
- R2-11 resolved (preferences scenario now says "from that moment" for the pool and saves, and "next activation" for the configuration).
- R2-12 resolved (spec Clearing: "a composition committed after the clear is recorded as new data"; task 2.3).
- R2-13 resolved (pool wording; D1 wording).

### New findings
No new P1 or P2.

### R3-1 (P4). Task 4.2 still says "flag set before showing"
Task 4.2 says both "flag set before showing" and "flag set only when the panel is ordered front". D12 specifies the latter. Delete "flag set before showing" from 4.2.
