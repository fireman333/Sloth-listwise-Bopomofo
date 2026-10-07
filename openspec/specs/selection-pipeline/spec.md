# selection-pipeline Specification

## Purpose

Defines how the input method chooses the characters shown in the composing buffer on every keystroke: neural in-walk scoring (R-walk) when available, the stock McBopomofo walk otherwise, with user choices always taking precedence and typing never blocked by the model.

## Requirements

### Requirement: Neural in-walk selection in Bopomofo mode
In Bopomofo (sentence) input mode, whenever the composing buffer is re-evaluated and it holds at most 40 syllables, the system SHALL select the composing-buffer characters with the neural in-walk scorer (R-walk) if the R model is loaded and enabled; a precise-mode result MAY later replace that selection as defined by the precise-mode capability. The latency added by R-walk (inference plus scored walk, plus candidate enumeration and gate when precise mode is ready) SHALL stay within a p95 budget of 10 ms per keystroke on Apple Silicon for buffers of up to 40 syllables. When the R model is not available, or the buffer holds more than 40 syllables, the system SHALL use the stock McBopomofo walk for that keystroke without delaying key handling.

#### Scenario: Model ready
- **WHEN** the R model is loaded and enabled and the user types a syllable in Bopomofo mode
- **THEN** the composing buffer shows the R-walk result for the current readings

#### Scenario: Model not yet loaded
- **WHEN** the user types before the R model has finished loading
- **THEN** the composing buffer shows the stock walk result and key handling is not delayed by the loading

#### Scenario: Buffer longer than the neural limit
- **WHEN** the composing buffer holds more than 40 syllables
- **THEN** the composing buffer shows the stock walk result for that keystroke

#### Scenario: One forward pass per reading sequence
- **WHEN** the composing buffer is re-evaluated more than once for the same readings and the same candidate characters (for example after applying a user-override suggestion, or after a cursor move)
- **THEN** the neural scores computed for those readings are reused rather than recomputed

#### Scenario: Same length, different readings
- **WHEN** the user deletes a syllable and types a different one so that the buffer length is unchanged but the readings differ
- **THEN** neural scores are recomputed for the new readings and previous scores are not applied

### Requirement: Fallback to the stock walk
The fallback order for selection SHALL be precise mode, then R-walk, then the stock McBopomofo walk. Precise mode SHALL only refine an R-walk result in the background; whenever precise mode is disabled, unavailable, or fails, the composing buffer SHALL keep the R-walk result. Any neural failure (inference library missing, model missing, load failure, inference error, empty or mismatched score table) SHALL result in the stock walk for that keystroke within the same 10 ms p95 per-keystroke budget, and SHALL NOT crash the input method, produce an error visible to the user, drop the keystroke, or leave the composing buffer empty. When R-walk is not used for a keystroke, precise mode SHALL NOT be used for it either.

#### Scenario: Inference error
- **WHEN** the R model returns no scores for the current readings
- **THEN** the composing buffer shows the stock walk result, the failure is not remembered as valid scores, and the next keystroke attempts R-walk again

#### Scenario: Inference library missing
- **WHEN** the bundled inference library cannot be loaded at launch
- **THEN** the input method starts and works with the stock walk

#### Scenario: Precise mode unavailable
- **WHEN** precise mode is enabled but its model cannot be loaded and the R model is ready
- **THEN** the composing buffer shows the R-walk result

#### Scenario: R model unavailable with precise mode on
- **WHEN** precise mode is enabled and loaded but the R model is not ready
- **THEN** the composing buffer shows the stock walk result and no background request is made

### Requirement: Slow-inference circuit breaker
The system SHALL measure, once per keystroke that performs neural inference, the time spent in inference plus the scored walk, plus candidate enumeration and gate when precise mode is ready. If that time exceeds 30 ms on 3 consecutive measured keystrokes, the system SHALL stop using R-walk and use the stock walk for the remainder of the input-method process lifetime, which also stops precise mode, and SHALL record the event once in the system log. Keystrokes that perform no inference (reused scores, cursor moves, buffers over the neural limit) SHALL neither extend nor reset the streak.

#### Scenario: Repeatedly slow inference
- **WHEN** three consecutive measured keystrokes each take more than 30 ms
- **THEN** subsequent keystrokes use the stock walk until the input method process restarts

#### Scenario: Isolated slow keystroke
- **WHEN** a single measured keystroke exceeds 30 ms and the next measured keystroke does not
- **THEN** R-walk remains in use

#### Scenario: Tripped with precise mode on
- **WHEN** the breaker trips while precise mode is ready
- **THEN** subsequent keystrokes use the stock walk and no background re-ranking request is made

### Requirement: User choices take precedence over model scores
Candidates the user picked manually in the candidate window SHALL appear in the composing buffer regardless of neural scores. Suggestions from the user override model SHALL be applied with the same override type as stock McBopomofo. Overridden nodes SHALL receive no neural score. Phrases from the user's own phrase file SHALL NOT be disadvantaged by neural scoring: at every position a user phrase covers, it SHALL receive the best neural score available at that position, so that against any competing candidate covering exactly the same positions, R-walk SHALL choose the user phrase whenever stock McBopomofo would; competitors that segment the readings differently are still scored as usual. Applying a user choice SHALL stay within the 10 ms p95 per-keystroke budget; when the model is unavailable, user choices and user phrases behave exactly as in stock McBopomofo.

#### Scenario: Manual pick survives re-evaluation
- **WHEN** the user picks a candidate for a position and then types another syllable
- **THEN** the picked character stays at that position in the composing buffer

#### Scenario: Override model suggestion
- **WHEN** the user override model suggests a candidate for the cursor position after an R-walk
- **THEN** the suggestion is applied with the stock override type and the buffer is re-evaluated

#### Scenario: User phrase made of rare characters
- **WHEN** the user's phrase file contains 錏錒 with reading ㄧㄚˋ-ㄚ and the user types ㄧㄚˋㄚ with R-walk enabled
- **THEN** the composing buffer shows 錏錒, as it does with R-walk disabled

### Requirement: Committed text is never modified
The selection pipeline SHALL only change text inside the composing buffer and SHALL NOT alter text already committed to the client application, including when a background re-ranking result arrives after the composing buffer was committed. All neural inference SHALL run on the local device; typed readings SHALL NOT be sent off the device.

#### Scenario: Commit then continue typing
- **WHEN** the user commits the composing buffer and keeps typing
- **THEN** the previously committed text is unchanged and only the new composing buffer is subject to selection

#### Scenario: Background result after commit
- **WHEN** a background re-ranking result for a composing buffer arrives after that buffer was committed
- **THEN** the committed text and the current composing buffer are unchanged by the result

### Requirement: Plain Bopomofo mode keeps stock behavior
In Plain Bopomofo (per-character) input mode, the system SHALL NOT apply neural scoring or precise mode and SHALL behave as stock McBopomofo apart from entries removed by the reading-policy capability, adding no neural latency to any keystroke, whether or not the models are loaded.

#### Scenario: Plain mode typing
- **WHEN** the input mode is Plain Bopomofo and the R model is loaded
- **THEN** candidate selection and ordering are identical to stock McBopomofo apart from entries removed by the reading policy
