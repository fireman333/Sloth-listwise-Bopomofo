## Purpose

Defines how the input method learns from the user on the device: a long-term, persistent memory of manual candidate picks, a local pool of committed sentences kept for later on-device personalization, the learning switch and data clearing, and the privacy guarantees for this data.

## ADDED Requirements

### Requirement: Long-term selection memory
When learning is on, the user override model SHALL remember up to 5,000 contexts, evicting the least recently used context when full, and SHALL weight each remembered pick by a decay with a half-life of 7 days measured in wall-clock time, so that time while the input method is not running also counts. A context is the same as in stock McBopomofo: the readings and characters of the two nodes before the picked node, and the reading and top dictionary value of the picked node. Suggestions SHALL be made and applied as in stock McBopomofo (same override types, Bopomofo mode only), with two differences while learning is on: (1) when the user manually picks a candidate in a context, that candidate SHALL be the one suggested next for that context, regardless of how often other candidates were picked there before; (2) when a composition is committed with a remembered suggestion still unchanged at its position, that context SHALL count as recently used and its decay SHALL restart from the commit time. Manual picks in Plain Bopomofo mode SHALL NOT be added to the memory while learning is on. Making and applying a suggestion SHALL stay within the 10 ms p95 per-keystroke budget and SHALL NOT depend on any neural model being available; if the memory is unavailable, typing continues with no suggestions.

#### Scenario: Pick remembered after three days
- **WHEN** learning is on, the user picks a candidate in a context, and types the same readings in the same context 3 days later
- **THEN** the remembered candidate is suggested and shown in the composing buffer

#### Scenario: Old pick fades
- **WHEN** a context was last used 141 days ago
- **THEN** no suggestion is made for it, whereas a context last used 139 days ago is still suggested

#### Scenario: Correction wins immediately
- **WHEN** the user picked 鍾 three times in a context and then picks 中 once in the same context
- **THEN** the next time the same readings are typed in that context, 中 is suggested

#### Scenario: Suggestion used keeps the memory fresh
- **WHEN** a remembered pick is suggested and the user commits it unchanged 100 days after the original pick, and types the same context again 100 days after that commit
- **THEN** the pick is still suggested

#### Scenario: Capacity reached
- **WHEN** 5,000 contexts are remembered and the user picks a candidate in a new context
- **THEN** the least recently used context is forgotten and the new one is remembered

### Requirement: Memory survives restarts
When learning is on, the selection memory SHALL be saved to the memory folder (see "Local storage") no more than 3 seconds after a change and when the input method is deactivated, and SHALL be restored when the input method starts. Saving SHALL NOT run on the key-handling thread beyond serializing a snapshot outside any keystroke, and loading SHALL happen at launch or when learning is turned on, not during a keystroke (every keystroke stays within the 10 ms p95 budget). If the memory file is missing, the input method SHALL start with an empty memory; if it is unreadable or malformed, the input method SHALL start with an empty memory, SHALL keep the unreadable file renamed aside instead of deleting it, and SHALL NOT crash or show an error while typing.

#### Scenario: Restart
- **WHEN** learning is on, the user picks a candidate, waits more than 3 seconds, and the input-method process is terminated and started again
- **THEN** typing the same readings in the same context shows the remembered candidate

#### Scenario: Corrupted file
- **WHEN** the memory file contains malformed data at start-up
- **THEN** typing works, no suggestion comes from the malformed file, and the file is kept under a different name in the memory folder

### Requirement: Sentence pool
When learning is on, each time a composition from Bopomofo input mode is committed to the client application — whether by Enter, Shift+Space, or implicitly (switching applications, clicking elsewhere, changing input mode or input source, Caps Lock) — the system SHALL append one record to a local sentence pool. A record SHALL contain the commit time, whether the user picked a candidate manually during the composition, the selection method that produced the composition (stock walk, R-walk, or precise mode), and for each node of the composition its reading, its dictionary characters before phrase replacement, macro expansion and Simplified Chinese conversion, and its source (manual pick, associated phrase, remembered suggestion, precise-mode result, user phrase, or model/dictionary). A composition that is discarded (for example with Esc), output as readings or other non-composed text, or contains no Bopomofo syllable SHALL NOT be recorded, and a syllable still being typed (not yet a complete reading) at commit time SHALL NOT be part of the record; nothing from Plain Bopomofo mode SHALL be recorded. Records SHALL be kept until the user clears learning data; the pool SHALL NOT be read by the selection pipeline in this version. Recording SHALL NOT add more than 1 ms (p95) on the key-handling thread; writing to disk SHALL happen off that thread, and a write failure SHALL NOT affect typing.

#### Scenario: Commit with a manual pick
- **WHEN** learning is on and the user types ㄓㄨㄥㄏㄨㄚˊ, picks a different candidate for one position, and presses Enter
- **THEN** the pool gains one record with the two readings, the committed characters per node, the manual-pick flag set, and that node marked as a manual pick

#### Scenario: Commit without a manual pick
- **WHEN** learning is on and the user types a sentence and commits it without opening the candidate window
- **THEN** the pool gains one record with the manual-pick flag not set

#### Scenario: Implicit commit on application switch
- **WHEN** learning is on and the user types a sentence and switches to another application without pressing Enter
- **THEN** the pool gains one record for the committed text

#### Scenario: Discarded composition
- **WHEN** the user types a sentence and presses Esc to discard it
- **THEN** no record is added

#### Scenario: Plain Bopomofo mode
- **WHEN** the input mode is Plain Bopomofo and the user commits a character
- **THEN** no record is added

#### Scenario: Simplified Chinese output
- **WHEN** Simplified Chinese output is on (in either conversion style) and the user commits 這裡
- **THEN** the record holds the dictionary characters 這裡, not the converted text

### Requirement: Nothing is learned in secure input
When learning is on, while secure event input is enabled by the client application the user is typing in (for example a password field, or a terminal's secure keyboard entry), the system SHALL NOT add a record to the sentence pool and SHALL NOT add or update anything in the selection memory. The check SHALL stay within the 10 ms p95 per-keystroke budget; composition and suggestions are otherwise unchanged. Secure input held by another process (for example left enabled by the login window) SHALL NOT pause learning.

#### Scenario: Commit in secure input
- **WHEN** learning is on, the client application holds secure event input, and the user picks a candidate and commits the composition
- **THEN** the sentence pool and the selection memory are unchanged

#### Scenario: Secure input held by another process
- **WHEN** learning is on and secure event input is enabled by a process other than the client application
- **THEN** picks and commits are learned as usual

### Requirement: Learning switch
Learning SHALL be on by default. When learning is off, the system SHALL remember picks as stock McBopomofo does (a session-only memory of at most 500 contexts with a 1.5-hour half-life and stock scoring), SHALL NOT read or write the memory folder except when the user clears learning data, and SHALL NOT add records to the sentence pool. Data already saved SHALL be kept, and SHALL be used again when learning is turned back on. Turning learning off SHALL save, once and immediately, the picks made while it was on that were not saved yet, and SHALL then stop pool recording and memory saving immediately; switching the memory configuration SHALL take effect no later than the next input-method activation. Neither SHALL block key handling (10 ms p95 per keystroke).

#### Scenario: Learning off
- **WHEN** the user turns learning off and then types, picks candidates and commits
- **THEN** apart from that single save of earlier picks at the moment learning is turned off, no file in the memory folder is created or modified

#### Scenario: Learning back on
- **WHEN** learning was turned off with a saved memory and is turned on again
- **THEN** from the next activation the saved picks are suggested again

### Requirement: Clearing learning data
The user SHALL be able to clear all learning data with one action from the preferences, after confirming. Clearing SHALL delete the selection memory file, the sentence pool, and any files set aside as unreadable, and SHALL empty the in-memory selection memory immediately, so that no previously learned pick is suggested afterwards. A save or record that was pending before the clear SHALL NOT recreate cleared data; a composition committed after the clear is recorded as new data. Clearing SHALL NOT change user phrases, excluded phrases, phrase replacements, models or other preferences, and SHALL NOT change the learning switch.

#### Scenario: Clear and type again
- **WHEN** the user clears learning data and types readings in a context that had a remembered pick
- **THEN** the choice made without the memory is shown, and the memory folder holds no memory or pool data from before the clear

#### Scenario: Cancel the confirmation
- **WHEN** the user presses the clear button and then cancels the confirmation
- **THEN** no data is deleted

#### Scenario: Other user data untouched
- **WHEN** the user clears learning data
- **THEN** the user phrase, excluded phrase and phrase replacement files, the model files and all preferences are unchanged

### Requirement: First-run notice
The first time the input method is activated with learning on and the notice has not been shown before, the system SHALL show a notice in Traditional Chinese stating that the input method learns from the user's typing and manual picks, that this data is kept on this Mac and is never sent over the network by the input method, and that learning can be turned off and the data cleared in the 「選字模型」 preferences tab. The notice SHALL offer a button 「好」 that closes it and a button 「打開設定」 that opens that tab, operable with the mouse. The notice SHALL NOT take keyboard focus or activate the input method's process: keystrokes typed while it is visible SHALL go to the client application within the 10 ms p95 budget. It SHALL be shown at most once per user and SHALL NOT change any setting by itself.

#### Scenario: First activation
- **WHEN** the input method is activated for the first time on a fresh install and the user keeps typing
- **THEN** the notice is shown once, the client application stays frontmost, and every keystroke reaches the composing buffer

#### Scenario: Later activations
- **WHEN** the notice was shown before and the input method is activated again
- **THEN** no notice is shown

### Requirement: Local storage
Learning data SHALL be stored only in `memory/` under the application-support root defined by the app-identity capability, regardless of any custom user-phrase location, with file permissions readable and writable only by the user. Learning data SHALL NOT be sent off the device. Writing learning data SHALL NOT cause user phrases or phrase replacements to be reloaded.

#### Scenario: Custom user-phrase location
- **WHEN** the user has configured a custom user-phrase location and learning is on
- **THEN** learning data is written to `~/Library/Application Support/Sloth-listwise-Bopomofo/memory/` and nothing is written to the custom location

#### Scenario: File permissions
- **WHEN** the memory folder files are created
- **THEN** they are readable and writable by the user only

#### Scenario: No phrase reload on save
- **WHEN** the memory is saved and a pool record is written with the default user-phrase location
- **THEN** user phrases are not reloaded
