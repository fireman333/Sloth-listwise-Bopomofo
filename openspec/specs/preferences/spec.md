# preferences Specification

## Purpose

Defines the 「選字模型」 (Selection Model) preferences tab: the user-visible switches for neural selection, precise mode and reading policy, live model status, when changes take effect, and the domain-pack placeholders.

## Requirements

### Requirement: Selection-model preferences tab
The preferences window SHALL include a tab titled 「選字模型」 (Traditional Chinese) / "Selection Model" (English) containing:
- a switch for neural selection (R-walk), default on;
- a switch for precise mode, default off;
- a switch to accept the ㄧㄢˊ reading for 癌, default off;
- a domain-pack section listing the 14 categories 工業、中醫、生物學、西醫、佛教、法律與政治、物理學與化學、建築、軍事、商業、教育、通訊、運動、電腦, each shown disabled and labelled as not yet available.
The domain-pack entries SHALL NOT be selectable and SHALL NOT change typing behavior. The R-walk weight SHALL NOT be shown in the tab. All values SHALL be stored in the preferences domain defined by the app-identity capability.

#### Scenario: Opening the tab with default preferences
- **WHEN** the user opens the preferences window on a fresh install and selects 「選字模型」
- **THEN** neural selection is on, precise mode is off, acceptance of ㄧㄢˊ is off, and the 14 domain packs are shown disabled as not yet available

#### Scenario: Clicking a domain pack
- **WHEN** the user clicks a domain-pack entry
- **THEN** nothing is toggled and typing behavior is unchanged

### Requirement: Model status display
The tab SHALL show, next to each of the neural-selection and precise-mode switches, the current status of the corresponding model as one of: off, loading, ready, model not found, failed to load, or stopped because too slow. When a switch is off, its status SHALL be shown as off regardless of whether a model is loaded. The status SHALL update while the tab is open, without reopening the window. Reading or updating the status SHALL NOT block key handling (every keystroke within the 10 ms p95 budget).

#### Scenario: Precise mode enabled without the model installed
- **WHEN** the user turns precise mode on and `models/qwen/` is missing
- **THEN** the precise-mode status shows model not found and typing continues with R-walk

#### Scenario: Neural selection switched off with the model loaded
- **WHEN** the user turns neural selection off while the R model is loaded
- **THEN** the neural-selection status shows off

#### Scenario: Precise mode loading
- **WHEN** the user turns precise mode on with the model installed
- **THEN** the status shows loading, then ready once the heavy model has loaded

### Requirement: When preference changes take effect
Changes to the neural-selection switch, the precise-mode switch, and acceptance of ㄧㄢˊ SHALL take effect for typing no later than the next input-method activation. Turning precise mode on SHALL start loading the heavy model immediately, and turning it off SHALL release it immediately, both without blocking the preferences window or key handling (every keystroke within the 10 ms p95 budget); if a model is unavailable, typing falls back as defined by the selection-pipeline capability. No preference change SHALL modify text already committed to the client application.

#### Scenario: Turning precise mode on and returning to an app
- **WHEN** the user turns precise mode on, closes the preferences window, and types in another application
- **THEN** precise mode is used once the heavy model has loaded

#### Scenario: Turning neural selection off
- **WHEN** the user turns neural selection off and returns to typing
- **THEN** from the next activation the composing buffer shows the stock walk result, and precise mode makes no requests
