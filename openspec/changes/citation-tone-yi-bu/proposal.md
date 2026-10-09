# Proposal: citation-tone-yi-bu

## Why

`openspec/project.md` states that for 「一」 and 「不」 both the citation tone (本調) and the tone-sandhi form (變調) are accepted, but the built-in dictionary does not honor this for phrases. Many common phrases are listed under one reading only: 不要 (ㄅㄨˊ-ㄧㄠˋ), 不是 (ㄅㄨˊ-ㄕˋ) and 不便 (ㄅㄨˊ-ㄅㄧㄢˋ) exist only with the sandhi reading of 不, so a user who types the citation tone (ㄅㄨˋ-ㄧㄠˋ) never gets 不要 as a phrase — the composing buffer has to assemble it from single characters and the candidate window does not offer the phrase. In the stock data, 976 phrase entries are reachable only through a sandhi reading. macOS built-in Zhuyin accepts both forms. In the Python benchmark (R-walk α 0.5, general test, `../PLAN.md` 2026-10-09) adding the citation-tone entries changes sentence accuracy only from 76.4 to 76.5 %, because single characters often rescue the sentence; the user-visible gain is mainly in phrase candidates and in sentences where a phrase is needed.

## What Changes

- When a multi-syllable reading contains a syllable of 一 (ㄧ, ㄧˊ, ㄧˋ) or 不 (ㄅㄨˋ, ㄅㄨˊ), the language model also returns built-in phrases listed under the other tone forms of that syllable, but only phrases whose character at that position is 一 (for the ㄧ forms) or 不 (for the ㄅㄨ forms). The added entries keep the score they are listed with; the merged list stays ordered by score.
- Both directions: typing the citation tone finds sandhi-listed phrases (ㄅㄨˋ-ㄧㄠˋ → 不要) and typing a sandhi form finds citation-listed phrases.
- Single-syllable readings are unchanged (a single 一 or 不 already exists under every form).
- The reading policy for 癌, the user exclusion list and the per-keystroke budget apply to the added entries as to any built-in entry. User phrases, associated phrases and the user override model are unchanged.

## Capabilities

### New Capabilities
<!-- none -->

### Modified Capabilities
- `reading-policy`: adds a requirement that the citation and sandhi tones of 一 and 不 find the same built-in phrases.

## Impact

- `Source/Engine/McBopomofoLM.cpp/.h`: `getUnigrams` / `hasUnigrams` look up tone variants of 一 / 不 syllables for multi-syllable keys (bounded number of extra dictionary lookups, only for keys containing those syllables).
- Engine gtest for the new behavior; latency re-measured with the existing in-process keystroke test.
- No dictionary file, model, preference or UI change.
