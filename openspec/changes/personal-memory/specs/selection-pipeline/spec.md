## MODIFIED Requirements

### Requirement: User choices take precedence over model scores
Candidates the user picked manually in the candidate window SHALL appear in the composing buffer regardless of neural scores. Suggestions from the user override model SHALL be applied with the same override type as stock McBopomofo, except that a suggestion SHALL NOT be applied when any node it would replace comes from the user's own phrase file (user phrases take precedence over remembered picks, and remembered picks over neural scores). Overridden nodes SHALL receive no neural score. Phrases from the user's own phrase file SHALL NOT be disadvantaged by neural scoring: at every position a user phrase covers, it SHALL receive the best neural score available at that position, so that against any competing candidate covering exactly the same positions, R-walk SHALL choose the user phrase whenever stock McBopomofo would; competitors that segment the readings differently are still scored as usual. Applying a user choice SHALL stay within the 10 ms p95 per-keystroke budget; when the model is unavailable, user choices and user phrases behave exactly as in stock McBopomofo, apart from the user-phrase rule for suggestions above and the long-term memory defined by the personal-memory capability.

#### Scenario: Manual pick survives re-evaluation
- **WHEN** the user picks a candidate for a position and then types another syllable
- **THEN** the picked character stays at that position in the composing buffer

#### Scenario: Override model suggestion
- **WHEN** the user override model suggests a candidate for the cursor position after an R-walk
- **THEN** the suggestion is applied with the stock override type and the buffer is re-evaluated

#### Scenario: User phrase made of rare characters
- **WHEN** the user's phrase file contains 錏錒 with reading ㄧㄚˋ-ㄚ and the user types ㄧㄚˋㄚ with R-walk enabled
- **THEN** the composing buffer shows 錏錒, as it does with R-walk disabled

#### Scenario: Remembered pick versus a user phrase
- **WHEN** the selection memory holds a pick for a context, the user then adds a user phrase for the same readings, and the walk now selects that user phrase at the cursor
- **THEN** the remembered pick is not applied and the composing buffer shows the user phrase, whether R-walk is on or off

#### Scenario: Multi-character suggestion next to a user phrase
- **WHEN** a remembered multi-character pick would replace a node next to the cursor that comes from the user's phrase file
- **THEN** the remembered pick is not applied
