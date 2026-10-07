## ADDED Requirements

### Requirement: Gated background re-ranking request
In Bopomofo (sentence) input mode, when precise mode is enabled, the heavy re-ranking model is loaded and ready, and a key event left the composing buffer re-evaluated with R-walk, the system SHALL, once per key event after the final re-evaluation of that event, enumerate R-walk's best candidate results that keep every fixed position (see "Fixed positions take precedence over re-ranking"), estimate R-walk's confidence in its shown result with the calibrated gate, and request a background re-ranking only when the gate judges that confidence below the calibrated threshold. No request SHALL be made when the gate judges R-walk confident, when fewer than two such candidate results remain, when R-walk was not used for that re-evaluation (model unavailable, disabled, tripped, or buffer over the neural limit), when the shown result is itself an applied re-ranking result, or when the request would be identical (same readings, candidate characters, fixed positions, composition and loaded dictionaries) to the most recent request made or result applied. Applying a re-ranking result SHALL NOT itself trigger a request. The foreground work added by precise mode SHALL keep every keystroke within the 10 ms p95 per-keystroke budget; while the heavy model is not ready, no candidate enumeration or gate SHALL run and the keystroke SHALL be handled exactly as with precise mode off.

#### Scenario: Uncertain R-walk result
- **WHEN** precise mode is enabled and ready and the user types a syllable for which the gate judges R-walk below the threshold
- **THEN** the composing buffer immediately shows the R-walk result and exactly one background re-ranking request is made for that key event

#### Scenario: Confident R-walk result
- **WHEN** precise mode is enabled and ready and the gate judges R-walk confident for the current readings
- **THEN** no background request is made and the composing buffer keeps the R-walk result

#### Scenario: Heavy model not ready
- **WHEN** precise mode is enabled but the heavy model is still loading, missing, or failed to load
- **THEN** keystrokes are handled exactly as with precise mode off, with no added work

#### Scenario: User-override suggestion in the same key event
- **WHEN** a key event applies a user-override-model suggestion and re-evaluates the buffer
- **THEN** at most one request is made, after the final re-evaluation, and it reflects the suggestion

#### Scenario: No request loop after applying
- **WHEN** a re-ranking result has been applied and nothing else changes
- **THEN** no further request is made for that buffer

### Requirement: Re-ranking never blocks typing
Heavy-model inference SHALL run off the key-handling thread, and no keystroke SHALL wait for it; every keystroke SHALL stay within the 10 ms p95 per-keystroke budget while inference runs in the background. At most one heavy inference SHALL run at a time across all client applications; when a newer request is made while one is running or queued, older queued requests SHALL be dropped so that only the latest request is processed next. A heavy inference that takes longer than the configured timeout, fails, or cannot score the candidates (for example input too long for the model) SHALL have its result discarded, leaving the R-walk result. After three consecutive timed-out inferences, the system SHALL stop making background requests for the rest of the input-method process lifetime, even if precise mode is turned off and on again, SHALL log the event once, and SHALL continue with R-walk.

#### Scenario: Fast typing
- **WHEN** the user types several syllables faster than the heavy model can score them
- **THEN** every keystroke is handled within the per-keystroke budget, and only the request for the latest readings is scored after the running one finishes

#### Scenario: Repeated timeouts
- **WHEN** three consecutive heavy inferences exceed the configured timeout
- **THEN** no further background requests are made until the input method process restarts, and typing continues with R-walk

#### Scenario: Heavy model cannot score
- **WHEN** the heavy model returns no usable scores for a request
- **THEN** the composing buffer keeps the R-walk result and the next request is attempted normally

### Requirement: Applying a re-ranking result to the composing buffer
When a background re-ranking result arrives, the system SHALL deliver it only to the client session that made the request, and SHALL replace that session's composing-buffer text with the fused result only if, since the request was made, the readings, the candidate characters at every position, the fixed positions, the loaded dictionaries and the input mode are all unchanged, the session is still showing the plain inputting state with the same composing text (not a candidate window, marking, associated-phrase, menu or any other state, and not committed or cleared), and the fused result differs from what is shown. Otherwise the result SHALL be discarded without any visible change. Applying a result SHALL keep the cursor position, SHALL NOT commit text, SHALL NOT be recorded as a user choice, SHALL NOT recompute R-walk inference, and SHALL take no more than 10 ms (p95) on the key-handling thread. Any later re-evaluation of the buffer (a new syllable, a deletion, a candidate pick, a user-override suggestion) SHALL start again from R-walk for the whole buffer, followed by the gate as usual.

#### Scenario: Result arrives while the user pauses
- **WHEN** a background result arrives, nothing has changed since the request, and the fused result differs from the shown R-walk result
- **THEN** the composing buffer shows the fused result with the cursor at the same position

#### Scenario: Result arrives after more typing
- **WHEN** the user typed another syllable after the request was made and the result for the earlier readings arrives
- **THEN** the result is discarded and the composing buffer is unchanged by it

#### Scenario: Result arrives after commit
- **WHEN** the user committed the composing buffer before the result arrived
- **THEN** the result is discarded and the committed text is unchanged

#### Scenario: Result arrives in another state
- **WHEN** the candidate window, marking, an associated-phrase list, or a menu is shown when the result arrives
- **THEN** the result is discarded and that state is unaffected

#### Scenario: Result for another client
- **WHEN** the user switched to another application after the request was made
- **THEN** the result is not applied to the other application's composing buffer

#### Scenario: Re-evaluation after an applied result
- **WHEN** a fused result was applied and the user then picks a candidate at one position
- **THEN** the buffer is re-evaluated with R-walk and the pick, and positions changed only by the fused result may return to the R-walk choice until a new result is applied

### Requirement: Fixed positions take precedence over re-ranking
Positions fixed by a manual candidate pick or by an applied user-override-model suggestion SHALL keep their characters after any re-ranking result is applied: candidate results that change a fixed position SHALL be excluded before the gate, and a fused result that would change a fixed position SHALL be discarded. A result applied by precise mode SHALL NOT be fed to the user override model as a user choice. Enforcing this SHALL add no work to keystroke handling beyond the 10 ms p95 per-keystroke budget; when precise mode is unavailable, fixed positions behave as defined by the selection-pipeline capability.

#### Scenario: Manual pick before the result arrives
- **WHEN** the user picks a candidate after a request was made and before its result arrives
- **THEN** the result is discarded and the picked character stays

#### Scenario: Manual pick earlier in the buffer
- **WHEN** a position was picked manually before the request and the fused result is applied
- **THEN** the picked character stays at that position

### Requirement: Heavy model and calibration files
The system SHALL load the heavy re-ranking model and its calibration parameters from the `models/qwen/` subdirectory of the application-support directory defined by the app-identity capability, and SHALL NOT require them to be bundled in the application. The model and calibration SHALL be treated as one unit: if either file is missing, unreadable, or invalid, or if the calibration's R-walk weight differs from the R-walk weight in use, precise mode SHALL be unavailable, the failure SHALL be logged once, and typing SHALL continue with R-walk within the 10 ms p95 per-keystroke budget and without any blocking error.

#### Scenario: Calibration file missing or invalid
- **WHEN** the model file exists but the calibration file is missing or contains an unknown key
- **THEN** precise mode reports unavailable and typing uses R-walk

#### Scenario: Calibration weight mismatch
- **WHEN** the calibration file's R-walk weight differs from the configured R-walk weight
- **THEN** precise mode reports failed to load and typing uses R-walk

#### Scenario: Model directory missing
- **WHEN** precise mode is enabled and `models/qwen/` does not exist
- **THEN** the input method keeps working with R-walk and the preferences show that the model was not found

### Requirement: Heavy model lifecycle
Precise mode SHALL be off by default. The heavy model SHALL be loaded off the main thread whenever precise mode is enabled — at input-method start or activation if the preference is already on, and immediately when the user turns it on — and SHALL be released when precise mode is disabled; pending and in-flight results SHALL then be discarded, including results from an earlier enable/disable cycle. Loading, failure, and release SHALL NOT block key handling, activation, or candidate display (every keystroke within the 10 ms p95 budget), and SHALL NOT crash the input method. While the heavy model is not ready, R-walk alone SHALL be used.

#### Scenario: Default install
- **WHEN** the input method runs with default preferences and the heavy model files are installed
- **THEN** the heavy model is not loaded and only R-walk is used

#### Scenario: Preference already on at start
- **WHEN** the input method process starts with precise mode enabled in preferences
- **THEN** the heavy model starts loading in the background without the user opening preferences

#### Scenario: Turning precise mode off
- **WHEN** the user disables precise mode while the heavy model is loaded
- **THEN** the heavy model is released, no further background requests are made, and any result still in flight is discarded

#### Scenario: Rapid off and on
- **WHEN** the user turns precise mode off and on again before an earlier load or inference finished
- **THEN** no result or loaded model from the earlier cycle is used

### Requirement: On-device only
Heavy-model loading and inference SHALL run entirely on the local device and SHALL NOT open any network connection; typed readings and candidates SHALL NOT leave the device and SHALL NOT be written to logs.

#### Scenario: Typing with precise mode on
- **WHEN** precise mode is enabled and loaded and the user types in Bopomofo mode
- **THEN** the input method process opens no outbound network connection and the system log contains no typed readings or candidate text

### Requirement: Plain Bopomofo mode excluded
In Plain Bopomofo (per-character) input mode, precise mode SHALL make no background requests and SHALL add no latency.

#### Scenario: Plain mode with precise mode on
- **WHEN** precise mode is enabled and loaded and the input mode is Plain Bopomofo
- **THEN** no background request is made and behavior is identical to precise mode being off
