# Spec Delta

## ADDED Requirements

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
