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
