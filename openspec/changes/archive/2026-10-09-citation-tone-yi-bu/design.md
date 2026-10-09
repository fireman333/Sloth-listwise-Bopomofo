# Design

## Context

- `McBopomofoLM::getUnigrams(key)` collects excluded phrases, user phrases and built-in entries (`languageModel_`, a `ParselessLM` over the memory-mapped `data.txt`) for one reading key such as `ㄅㄨˋ-ㄧㄠˋ`; the 癌 reading policy is applied there on raw built-in entries (`violatesReadingPolicy(syllables, value)`), before `filterAndTransformUnigrams` (exclusions, phrase replacement, macros, dedup).
- `hasUnigrams(key)` decides whether the grid creates a node; today it answers from the dictionaries directly.
- In stock `data.txt`, 976 multi-character entries containing 一 or 不 are listed only under a sandhi reading (measured with `../bench/src/decoder.py add_citation_tone_variants`); most 一 phrases are already listed under ㄧ and a sandhi form, while 不 phrases are often listed only under ㄅㄨˊ (不要, 不是, 不便).
- The grid's node score is the score of the first unigram, so the list returned by `getUnigrams` must stay sorted by score.

## Goals / Non-Goals

**Goals:**
- A multi-syllable key containing a 一 / 不 syllable returns built-in entries listed under the other tone forms of that syllable when the entry has 一 / 不 at that position, with the listed score, sorted, deduplicated (user decision 2026-10-09: runtime lookup, both directions).
- Same behavior in Bopomofo and Plain Bopomofo modes for multi-syllable keys (Plain Bopomofo composes single syllables only, so it is effectively unaffected).

**Non-Goals:**
- Changing `data.txt` or the data compilers.
- User phrases (kept exactly as the user wrote them), associated phrases, the user override model.
- Other sandhi (e.g. third-tone sandhi) or other characters.

## Decisions

### D1 Runtime lookup in `getUnigrams`, not a data-file change
For a multi-syllable key, find positions whose syllable is in {ㄧ, ㄧˊ, ㄧˋ} or {ㄅㄨˋ, ㄅㄨˊ}. Enumerate the alternative keys obtained by replacing those syllables with the other members of their set (excluding the original key). For each alternative key that `languageModel_` has, take its raw entries and keep an entry only if its character count equals the syllable count and, at every replaced position, its character is 一 (ㄧ set) or 不 (ㄅㄨ set); apply the 癌 reading policy with the alternative key's syllables. Append the kept entries to the raw built-in list of the original key, keep the highest score per value, then stable-sort by score (descending) before `filterAndTransformUnigrams`. Alternatives considered: adding ~1,000 lines to `data.txt` at build time (simpler at runtime, but edits the upstream data pipeline and is harder to keep in sync on upstream merges) — rejected by the user.

### D2 Bounded work
At most the first 3 such positions in a key are expanded (≤ 3³ − 1 = 26 extra lookups; real keys rarely have more than one). Keys without such syllables pay only the syllable scan. Each lookup is the same binary search `ParselessLM` already does per key.

### D3 `hasUnigrams`
If the dictionaries have nothing for the key and the key is multi-syllable with a 一 / 不 syllable, answer `!getUnigrams(key).empty()`; otherwise unchanged.

### D4 Upstream-identical default behavior (config rule)
The project rule requires engine changes to keep the model-absent behavior identical to upstream McBopomofo except for entries removed by the reading policy. This change adds entries under the `reading-policy` capability, so the stock walk (R absent or α = 0) changes for keys with 一 / 不 syllables. This is the intended behavior of the requirement (project.md already states that both tones are accepted); the change records it as a deliberate exception to that rule, limited to keys containing 一 / 不 syllables.

## Risks / Trade-offs

- A sandhi-listed phrase now competes under the citation reading; where two such phrases tie (不便 / 不變, scores −4.70 / −4.68), the walk or R decides as for any homophones.
- Extra lookups per keystroke for keys containing 一 / 不 syllables; re-measured with the in-process latency test (budget p95 < 10 ms).
- Benchmark effect on sentence accuracy is small (76.4 → 76.5 in the Python R-walk benchmark); the value is phrase availability for citation-tone typists.
