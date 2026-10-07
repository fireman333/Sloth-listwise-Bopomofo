## Purpose

Defines how the input method fine-tunes its selection model R on the user's Mac from the local sentence pool: which sentences are used, when training runs, how it is isolated from typing, when a trained version is adopted, how versions are rolled back, cleared or replaced after a factory model update, and the on-device guarantees.

## ADDED Requirements

### Requirement: Training data eligibility
Personalized training SHALL use only records of the local sentence pool defined by the personal-memory capability. A record SHALL be eligible when it is not marked as retracted (see "Retraction after commit") and either the user picked a candidate manually during the composition, or the default choice was kept. Before training, each node's reading SHALL be flattened into its syllables (a multi-syllable node such as a dictionary word or user phrase with reading ㄘㄜˋ-ㄕˋ contributes two syllables and its characters in order), a node whose character count differs from its syllable count SHALL make the record ineligible, and the record SHALL then be split at nodes that are not Bopomofo syllables (punctuation, symbols, spaces); only segments of 2 to 40 syllables whose characters and readings are all known to the R model SHALL be used, each with its committed dictionary characters as the target. Training SHALL use the selection dictionaries shipped with the input method and the reading-policy setting in effect, and SHALL NOT use the user's phrase files or the selection memory. The number of eligible segments SHALL be shown to the user (see "Status reporting").

#### Scenario: Record with a manual pick
- **WHEN** a pool record has the manual-pick flag set and is not retracted
- **THEN** its segments are eligible for training with the committed characters as targets

#### Scenario: Retracted record
- **WHEN** a pool record is marked as retracted
- **THEN** none of its segments is used for training or for the personal held-out test

#### Scenario: Multi-syllable node
- **WHEN** a record has the nodes ㄘㄜˋ-ㄕˋ → 測試 and ㄓㄨㄥ → 中
- **THEN** one segment of three syllables ㄘㄜˋ ㄕˋ ㄓㄨㄥ with target 測試中 is used

#### Scenario: Punctuation inside a sentence
- **WHEN** a record contains ㄓㄨㄥ ㄨㄣˊ, a comma, then ㄕㄨ ㄖㄨˋ
- **THEN** two segments of two syllables each are used and the comma is not

### Requirement: Retraction after commit
When learning is on and a composition was added to the sentence pool, and the next key the input method receives for the same client application within 10 seconds of that commit is Backspace or Delete with an empty composing buffer (so the key goes to the application), the system SHALL mark that pool record as retracted. Any other key, a new composition, or the 10-second limit SHALL end the window. A training run SHALL be discarded without adoption if a retraction recorded after the run read the pool marks any record whose segments the run used for training or testing. Marking SHALL NOT change the key's behavior, SHALL NOT modify committed text, and SHALL add no more than 1 ms (p95) on the key-handling thread within the 10 ms p95 per-keystroke budget; file writing SHALL happen off that thread and a write failure SHALL NOT affect typing. Nothing SHALL be marked while learning is off or in secure input.

#### Scenario: Delete right after commit
- **WHEN** learning is on, the user commits 中文 and presses Backspace 2 seconds later with an empty composing buffer
- **THEN** the character is deleted in the application as before and the record for 中文 is marked as retracted

#### Scenario: Backspace after typing something else
- **WHEN** the user commits a sentence, starts a new composition, and then presses Backspace
- **THEN** the Backspace edits the composing buffer and the earlier record is not marked

#### Scenario: Retraction during a run
- **WHEN** a training run has read the pool and the user then retracts a record that is in the run's personal test set
- **THEN** the run's result is not adopted and the model in use is unchanged

#### Scenario: Backspace after the window
- **WHEN** the first key after a commit is Backspace, pressed 15 seconds after the commit
- **THEN** the record is not marked

### Requirement: Training triggers
Training SHALL run only while learning is on (personal-memory "Learning switch") and only when at least 1,000 eligible segments exist (initial value). Automatic training SHALL start at most once per 24 hours, only while the Mac is on AC power, not in Low Power Mode, not under serious or critical thermal pressure, and the user has made no keyboard or mouse input for at least 10 minutes, and only when at least 200 eligible segments were added since the last attempt, or no attempt has been made yet on the installed factory model. An automatic run SHALL stop and discard its work when the Mac leaves AC power or enters Low Power Mode. The user SHALL be able to start training at any time with a 「立即學習」 button, which ignores the power, idle, 24-hour and new-segment conditions but not the learning switch or the 1,000-segment minimum, and to cancel a running training. All numeric thresholds in this requirement are initial values that MAY be changed by the personalization pack shipped with the factory model.

#### Scenario: Automatic run overnight
- **WHEN** learning is on, 1,500 eligible segments exist, no attempt has been made on the installed factory model, the Mac is on AC power and nobody uses it for 10 minutes
- **THEN** a training run starts without user action

#### Scenario: On battery
- **WHEN** the Mac is on battery
- **THEN** no automatic training starts, and an automatic run in progress stops and its work is discarded

#### Scenario: Too little data
- **WHEN** the user presses 「立即學習」 with 600 eligible segments
- **THEN** no training runs and the tab states that 1,000 segments are needed and 600 are available

#### Scenario: Cancel
- **WHEN** the user presses the cancel button during a run
- **THEN** the run stops, nothing is adopted, and the model in use is unchanged

### Requirement: Training never disturbs typing
Training SHALL run in a separate process from the input method, at background priority, and SHALL NOT block or slow key handling: while a training run is in progress, every keystroke SHALL stay within the 10 ms p95 per-keystroke budget and the selection model in use SHALL keep working, or fall back as defined by the selection-pipeline capability if it is unavailable. A crash, hang or resource failure of training SHALL NOT crash the input method, SHALL NOT change the model in use, and SHALL be reported only as a status (see "Status reporting"). A run SHALL be stopped if it exceeds 10 minutes of wall-clock time or 2 GB of resident memory (initial values). Training SHALL NOT show any window, alert or notification, and SHALL NOT take keyboard focus.

#### Scenario: Typing during training
- **WHEN** a training run is in progress and the user types in Bopomofo mode
- **THEN** every keystroke is handled within the 10 ms p95 budget with the current model

#### Scenario: Training process crashes
- **WHEN** the training process terminates abnormally
- **THEN** the input method keeps working with the current model and the tab shows the run as failed

### Requirement: Acceptance gate
Each run SHALL hold out the newest 30% of the eligible segments, but at least 300 and at most 400 (initial values), as a personal test set that is not used for training. A trained version SHALL be adopted only if, evaluated by the input method's own onnxruntime inference and R-walk with the exact weights that would be loaded (same quantized weights, same in-walk weight, the reading-policy setting in effect, the shipped dictionaries without user data), (1) its sentence accuracy on the personal test set exceeds the factory model's by at least 2 percentage points with p < 0.05 in a paired exact one-sided McNemar test (the trained version better than the factory model), (2) its sentence accuracy on the general test set shipped with the factory model is no more than 1.0 percentage point below the factory model's, and (3) its personal-test accuracy is not below that of the personalized version in use, if any. Otherwise the version SHALL be discarded and the model in use SHALL stay unchanged. The thresholds are initial values that MAY be changed by the personalization pack. The result and the measured accuracies SHALL be kept with the version and shown to the user.

#### Scenario: Clear improvement
- **WHEN** a trained version improves personal-test accuracy by 5 points with p = 0.01 and general accuracy drops by 0.4 points
- **THEN** it is adopted

#### Scenario: General regression
- **WHEN** a trained version improves personal-test accuracy by 6 points but general accuracy drops by 1.5 points
- **THEN** it is not adopted and the tab reports that general accuracy dropped too much

#### Scenario: Not significant
- **WHEN** a trained version improves personal-test accuracy by 2.5 points with p = 0.2
- **THEN** it is not adopted and the model in use is unchanged

### Requirement: Using an adopted version
After a version is adopted, the input method SHALL use R with that version's weights from the next input-method activation, loading it in the background as defined by the model-loading capability; until it is loaded, the model in use before SHALL continue, and every keystroke SHALL stay within the 10 ms p95 per-keystroke budget. Text already committed SHALL NOT change. The priority of user phrases and of the selection memory over R SHALL be unchanged. Sentence-pool records SHALL identify which personalized version, if any, produced them.

#### Scenario: Adoption while typing
- **WHEN** a version is adopted while the user is composing a sentence
- **THEN** the composition continues with the previous model, and the new version is used after the next activation once loaded

#### Scenario: User phrase still wins
- **WHEN** a personalized version prefers another candidate for readings covered by a user phrase
- **THEN** the composing buffer shows the user phrase as with the factory model

### Requirement: Binding to the evaluation configuration
A personalized version SHALL be used only with the factory R model, the R-walk weight (alpha), the reading-policy setting and the personalization pack it was evaluated with. When any of these differs from the current configuration, the system SHALL NOT use the version, SHALL use the factory weights (loading off the key-handling thread, every keystroke within the 10 ms p95 budget), SHALL report the version as not matching the current settings, and SHALL allow the next training run without waiting for new segments. The version SHALL be kept on disk unless the factory model changed (see "Factory model update"), so that restoring the setting it was evaluated with makes it usable again.

#### Scenario: Reading policy changed
- **WHEN** a personalized version was evaluated with acceptance of ㄧㄢˊ off and the user turns it on
- **THEN** from the next activation the factory weights are used, the tab reports that the version does not match the current settings, and a training run may start without new segments

#### Scenario: Setting restored
- **WHEN** the user turns acceptance of ㄧㄢˊ off again
- **THEN** from the next activation the personalized version is used again

### Requirement: Versions and rollback
The system SHALL keep at most two personalized versions: the one in use and the previous one. The user SHALL be able, with one action each, to return to the factory model and to switch to the previous version; the change SHALL take effect from the next activation without blocking typing. Returning to the factory model SHALL keep both versions on disk and SHALL turn automatic training off until the user presses 「立即學習」.

#### Scenario: Back to factory
- **WHEN** the user presses 「改回出廠模型」
- **THEN** from the next activation the factory weights are used, and no automatic training starts until the user presses 「立即學習」

#### Scenario: Previous version
- **WHEN** a previous version exists and the user presses 「改用上一版」
- **THEN** from the next activation the previous version is used and becomes the version in use

### Requirement: Factory model update
Personalized weights SHALL be bound to the exact factory R model they were trained from. When the installed factory model differs, the system SHALL NOT load personalized weights built for another factory model, SHALL delete them, SHALL use the factory model, SHALL keep the sentence pool, and SHALL allow the next training run (automatic or manual) to train on the new factory model from the retained pool without waiting for new segments.

#### Scenario: New factory model installed
- **WHEN** a new factory R model is installed while a personalized version exists for the old one
- **THEN** the input method uses the new factory model, the old personalized versions are deleted, and the next eligible automatic run trains on the new model from the existing pool

### Requirement: Clearing removes personalization
When the user clears learning data (personal-memory "Clearing learning data"), the system SHALL first stop using personalized weights — the factory weights are used for every keystroke from the next activation, and no personalized version selected or loaded before the clear SHALL be used afterwards — and SHALL then stop any running training, delete all personalized versions and their status, and discard any result, adoption or load that was started before the clear and completes after it. No file written by a training run that started before the clear SHALL remain afterwards. Clearing SHALL NOT block key handling (10 ms p95 per keystroke).

#### Scenario: Clear during training
- **WHEN** the user clears learning data while a training run is in progress
- **THEN** the run stops, no personalized version exists afterwards, and from the next activation the factory model is used

#### Scenario: Training finishes during the clear
- **WHEN** a training run finishes with a version that passed the gate while learning data is being cleared
- **THEN** the version is not adopted, no file of it remains, and the factory model is used

#### Scenario: Personalized load in flight during the clear
- **WHEN** the input method is loading a personalized version in the background and the user clears learning data
- **THEN** the loaded personalized weights are discarded and the factory model is used

### Requirement: Status reporting
The 「選字模型」 tab SHALL show: whether the factory model or a personalized version is in use (with its creation date), the number of eligible segments, the time and outcome of the last run (adopted, not adopted with the reason, failed, cancelled, or not enough data) with the measured personal-test and general accuracies, and, while a run is in progress, its phase and progress. The status SHALL update while the tab is open without blocking key handling (10 ms p95 per keystroke). Logs SHALL NOT contain typed text, readings or candidates.

#### Scenario: After a rejected run
- **WHEN** the last run was not adopted because the improvement was not significant
- **THEN** the tab shows the run time, "not adopted" with that reason, and both accuracies, and the model in use is unchanged

### Requirement: On-device training only
Training SHALL run entirely on this Mac. The training process SHALL NOT open any network connection, SHALL read only the learning data folder, the R model folder and the input method's own bundle, and SHALL write only inside the learning data folder with permissions readable and writable only by the user. Training data, personalized weights and evaluation results SHALL NOT be sent off the device.

#### Scenario: Network during training
- **WHEN** a training run is in progress
- **THEN** the training process has no open network socket

#### Scenario: Files written
- **WHEN** a training run finishes
- **THEN** every file it created is inside `memory/` under the application-support root and has user-only permissions
</content>
</invoke>
