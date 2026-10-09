# reading-policy Specification

## Purpose

Defines reading policies applied to the built-in dictionary and associated phrases — currently that 癌 is read ㄞˊ by default and the ㄧㄢˊ reading is accepted only when the user opts in — independent of any neural model.

## Requirements

### Requirement: 癌 is read ㄞˊ by default
Unless the user has enabled acceptance of the ㄧㄢˊ reading for 癌, the system SHALL NOT offer, in either Bopomofo or Plain Bopomofo input mode, any built-in dictionary entry in which the character 癌 corresponds to the syllable ㄧㄢˊ — neither the single character nor any multi-character phrase containing it — in the composing buffer or the candidate window. Entries in which 癌 corresponds to ㄞˊ, and all other entries for the same readings, SHALL be unaffected, with their original order and scores. The filtering SHALL keep every keystroke within the 10 ms p95 per-keystroke budget and SHALL NOT depend on any neural model being available.

#### Scenario: Typing the single syllable
- **WHEN** acceptance of ㄧㄢˊ is off and the user types ㄧㄢˊ and opens the candidate window
- **THEN** 癌 is not among the candidates, and the other characters for ㄧㄢˊ are offered as in stock McBopomofo

#### Scenario: Typing a phrase with the ㄧㄢˊ reading
- **WHEN** acceptance of ㄧㄢˊ is off and the user types ㄎㄤˋㄧㄢˊ
- **THEN** the composing buffer does not show 抗癌 and the candidate window does not offer 抗癌

#### Scenario: Typing the ㄞˊ reading
- **WHEN** the user types ㄈㄟˋㄞˊ
- **THEN** 肺癌 is offered as in stock McBopomofo, whether acceptance of ㄧㄢˊ is on or off

#### Scenario: Plain Bopomofo mode
- **WHEN** acceptance of ㄧㄢˊ is off, the input mode is Plain Bopomofo, and the user types ㄧㄢˊ
- **THEN** 癌 is not among the candidates

#### Scenario: Previously learned choice
- **WHEN** the user override model remembers 癌 for ㄧㄢˊ from a time when acceptance was on, and acceptance is now off
- **THEN** 癌 is not shown for ㄧㄢˊ

### Requirement: Associated phrases follow the reading policy
Unless acceptance of the ㄧㄢˊ reading for 癌 is enabled, the system SHALL NOT offer associated-phrase suggestions whose reading aligns 癌 with ㄧㄢˊ; associated phrases that align 癌 with ㄞˊ and all other associated phrases SHALL be unaffected. Selecting an associated-phrase suggestion that contains 癌 SHALL insert exactly the characters shown in the suggestion. Filtering SHALL stay within the 10 ms p95 per-keystroke budget.

#### Scenario: Associated phrase after 乳
- **WHEN** acceptance of ㄧㄢˊ is off and associated phrases are shown after the user enters 乳
- **THEN** no suggestion reads 癌 as ㄧㄢˊ, and any 乳癌 suggestion offered uses ㄞˊ and inserts 癌

### Requirement: Opt-in acceptance of ㄧㄢˊ
When the user enables acceptance of the ㄧㄢˊ reading for 癌, entries and associated phrases in which 癌 corresponds to ㄧㄢˊ SHALL be offered exactly as in stock McBopomofo, with their original scores.

#### Scenario: Acceptance enabled
- **WHEN** acceptance of ㄧㄢˊ is on and the user types ㄎㄤˋㄧㄢˊ
- **THEN** 抗癌 is offered as in stock McBopomofo

### Requirement: User phrases are not filtered
Phrases the user added to their own user-phrase file SHALL NOT be removed by the reading policy, including phrases in which 癌 is read ㄧㄢˊ.

#### Scenario: User adds the ㄧㄢˊ reading explicitly
- **WHEN** acceptance of ㄧㄢˊ is off and the user's phrase file contains 癌 with reading ㄧㄢˊ
- **THEN** typing ㄧㄢˊ offers 癌 from the user phrase

### Requirement: Policy changes affect only new composition
A change to the reading policy SHALL take effect no later than the next input-method activation, SHALL be applied without blocking key handling (within the 10 ms p95 per-keystroke budget), and SHALL NOT modify text already committed to the client application.

#### Scenario: Toggling acceptance
- **WHEN** the user turns acceptance of ㄧㄢˊ on or off in preferences and returns to typing
- **THEN** from the next activation on, candidates follow the new setting, and previously committed text is unchanged

### Requirement: Citation and sandhi tones of 一 and 不 find the same phrases
For a reading of two or more syllables, the system SHALL offer, in addition to the built-in entries listed under that exact reading, every built-in multi-character entry listed under a reading that differs from it only in syllables where the typed syllable and the listed syllable are both tone forms of 一 (ㄧ, ㄧˊ, ㄧˋ) and the entry's character at that position is 一, or both tone forms of 不 (ㄅㄨˋ, ㄅㄨˊ) and the entry's character at that position is 不. An added entry SHALL keep the score it is listed with, and the candidates for the reading SHALL remain ordered by score; an entry found under several readings SHALL be offered once, with the highest score it is listed with. At most the first three such syllables of a reading SHALL be considered. Entries whose character at such a position is not 一 or 不 SHALL NOT be added. Readings of a single syllable SHALL be unaffected. The added entries SHALL be subject to the reading policy for 癌 and to the user's excluded phrases like any built-in entry; user phrases, associated phrases and remembered user choices SHALL be unchanged. This lookup SHALL keep every keystroke within the 10 ms p95 per-keystroke budget and SHALL NOT depend on any neural model being available.

#### Scenario: Citation tone finds a sandhi-listed phrase
- **WHEN** the user types ㄅㄨˋㄧㄠˋ and the built-in dictionary lists 不要 only under ㄅㄨˊ-ㄧㄠˋ
- **THEN** 不要 is offered for ㄅㄨˋㄧㄠˋ with the score it is listed with

#### Scenario: Sandhi tone finds a citation-listed phrase
- **WHEN** the user types a sandhi form of 一 (for example ㄧˊ before a fourth-tone syllable) and a phrase with 一 at that position is listed only under ㄧ
- **THEN** that phrase is offered

#### Scenario: Other characters with the same syllable are not added
- **WHEN** the user types ㄧˊ-ㄉㄧㄥˋ and the dictionary lists 宜定 under ㄧ-ㄉㄧㄥˋ (宜 is not 一)
- **THEN** 宜定 is not offered for ㄧˊ-ㄉㄧㄥˋ

#### Scenario: A phrase listed under both readings appears once
- **WHEN** 一定 is listed under both ㄧ-ㄉㄧㄥˋ and ㄧˊ-ㄉㄧㄥˋ and the user types either reading
- **THEN** 一定 is offered once

#### Scenario: Single syllable unchanged
- **WHEN** the user types ㄅㄨˋ alone
- **THEN** the candidates are exactly those listed under ㄅㄨˋ, as in stock McBopomofo

#### Scenario: Works without the neural model
- **WHEN** the R model is not loaded and the user types ㄅㄨˋㄧㄠˋ
- **THEN** 不要 is still offered and composed by the stock walk
