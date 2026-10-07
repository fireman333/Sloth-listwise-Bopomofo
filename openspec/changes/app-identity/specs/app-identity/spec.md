# Spec Delta

## Purpose

Defines the input method's own identity — bundle identifier and every identifier derived from it, user-visible names, install location, preferences domain, user-data and model root directory, and minimum macOS version — so that it can be installed and used side by side with a stock McBopomofo without sharing any identity or system resource.

## ADDED Requirements

### Requirement: Bundle identifier and derived identifiers
The input method application SHALL use the bundle identifier `io.github.fireman333.inputmethod.Sloth-listwise-Bopomofo`. Its input source ID (`TISInputSourceID`) SHALL equal the bundle identifier. Its two input modes SHALL use the identifiers `io.github.fireman333.inputmethod.Sloth-listwise-Bopomofo.Bopomofo` (sentence mode) and `io.github.fireman333.inputmethod.Sloth-listwise-Bopomofo.PlainBopomofo` (plain mode), and the identifiers the running input method uses to recognise its active mode SHALL be exactly these strings. Its input-method connection name, as declared in the bundle and as registered by the running server, SHALL be `io.github.fireman333.inputmethod.Sloth-listwise-Bopomofo_Connection`. No identifier listed in this requirement SHALL begin with `org.openvanilla.` or contain `McBopomofo`.

#### Scenario: Built bundle declares the new identifiers
- **WHEN** the Release app bundle is built and its Info.plist is inspected
- **THEN** `CFBundleIdentifier` and `TISInputSourceID` equal `io.github.fireman333.inputmethod.Sloth-listwise-Bopomofo`, the input-mode list contains exactly the `.Bopomofo` and `.PlainBopomofo` identifiers derived from it, and `InputMethodConnectionName` equals `io.github.fireman333.inputmethod.Sloth-listwise-Bopomofo_Connection`

#### Scenario: Mode selection uses the new mode identifiers
- **WHEN** the user selects the plain-mode input source of this input method
- **THEN** the input method switches to plain Bopomofo behaviour, and selecting the sentence-mode input source switches it back to sentence behaviour

### Requirement: User-visible names
In every bundled localization (Base, en, zh-Hant), the input method's bundle name and display name SHALL be `Sloth-listwise Bopomofo`, the sentence-mode input source SHALL be shown as `Sloth-listwise Bopomofo`, and the plain-mode input source SHALL be shown as `Sloth-listwise Plain Bopomofo`. The input method's own menu items and window titles that name the product (About, Preferences, the user-phrases service item) SHALL use `Sloth-listwise Bopomofo` instead of `McBopomofo` or `小麥注音`. The installer application SHALL present itself as installing `Sloth-listwise Bopomofo`.

#### Scenario: Input source menu in Traditional Chinese
- **WHEN** the system language is Traditional Chinese and the user opens the input source menu with both modes enabled
- **THEN** the entries read `Sloth-listwise Bopomofo` and `Sloth-listwise Plain Bopomofo`, and neither reads `小麥注音` nor `傳統注音`

#### Scenario: Input method menu
- **WHEN** this input method is active and the user opens its menu
- **THEN** the About and Preferences items name `Sloth-listwise Bopomofo`

### Requirement: Install location
The application bundle SHALL be named `Sloth-listwise-Bopomofo.app` and SHALL be installed per user at `~/Library/Input Methods/Sloth-listwise-Bopomofo.app`. Installing, upgrading or reinstalling this input method SHALL NOT remove, replace, move to Trash, or terminate any process of a bundle at `~/Library/Input Methods/McBopomofo.app` or any other bundle whose identifier differs from this input method's.

#### Scenario: Install next to an existing McBopomofo
- **WHEN** a stock McBopomofo is installed at `~/Library/Input Methods/McBopomofo.app` and this input method is installed or upgraded
- **THEN** `~/Library/Input Methods/Sloth-listwise-Bopomofo.app` exists, `~/Library/Input Methods/McBopomofo.app` is unchanged, and a running McBopomofo process is not terminated

### Requirement: Separate preferences domain
All preferences of this input method SHALL be stored in the user-defaults domain `io.github.fireman333.inputmethod.Sloth-listwise-Bopomofo`. The input method SHALL NOT read or write the `org.openvanilla.inputmethod.McBopomofo` defaults domain.

#### Scenario: Changing a preference does not affect McBopomofo
- **WHEN** the user changes a setting in this input method's preferences window
- **THEN** the change appears in `defaults read io.github.fireman333.inputmethod.Sloth-listwise-Bopomofo` and `defaults read org.openvanilla.inputmethod.McBopomofo` is unchanged

### Requirement: Application-support root and model directory
The input method's application-support root SHALL be `~/Library/Application Support/Sloth-listwise-Bopomofo/`. When no custom user-phrase location is configured, user phrase, excluded-phrase and phrase-replacement files SHALL be read from and created in this root. Model files for neural selection SHALL be located under `models/` of this root, with the R-walk model in `models/r-walk/`; the model directory SHALL be derived from the root regardless of any custom user-phrase location. The input method SHALL NOT read or write `~/Library/Application Support/McBopomofo/` unless the user explicitly configures that folder as the custom user-phrase location.

#### Scenario: First write of user data with default settings
- **WHEN** the input method first writes user data with no custom user-phrase location
- **THEN** `~/Library/Application Support/Sloth-listwise-Bopomofo/` is created with the template user-data files only, and nothing is created or modified in `~/Library/Application Support/McBopomofo/`

#### Scenario: Model directory with a custom user-phrase location
- **WHEN** the user has configured a custom user-phrase location
- **THEN** user phrase files are read from the custom location, and the model directory is still `~/Library/Application Support/Sloth-listwise-Bopomofo/models/r-walk/`

### Requirement: Fresh start without automatic migration
On first use the input method SHALL start with empty user data and default preferences, and SHALL NOT automatically copy, import, or delete user phrases, excluded phrases, phrase replacements, user-override history, or preferences from McBopomofo.

#### Scenario: Existing McBopomofo user data
- **WHEN** `~/Library/Application Support/McBopomofo/data.txt` contains user phrases and this input method is activated for the first time
- **THEN** the new root's `data.txt` contains only the template content and the McBopomofo file is unchanged

### Requirement: Coexistence with stock McBopomofo
With a stock McBopomofo installed and enabled, this input method SHALL be installable, enableable, and usable at the same time, both SHALL appear as separate entries in the system input-source list, and switching between them SHALL leave each one's typing behaviour, user data and preferences unaffected by the other. Changing the identity SHALL NOT add work to keystroke handling: per-keystroke latency SHALL remain within the project budget of 10 ms p95, and when no model is present typing SHALL fall back to the stock walk exactly as it would under the previous identity.

#### Scenario: Both input methods in Input Sources
- **WHEN** both input methods are installed and the user opens System Settings > Keyboard > Input Sources > Add
- **THEN** both `McBopomofo` (小麥注音) and `Sloth-listwise Bopomofo` are listed and can be enabled together

#### Scenario: Switching between the two while typing
- **WHEN** the user types a phrase with this input method, switches to McBopomofo, types, and switches back
- **THEN** each input method composes and commits text normally, and a user phrase added in one does not appear in the other

### Requirement: Minimum macOS version
The input method application, its installer, and its test bundle SHALL declare a minimum supported macOS version of 26.0, and the application's `LSMinimumSystemVersion` SHALL be `26.0`.

#### Scenario: Built bundle minimum version
- **WHEN** the Release app bundle is built and its Info.plist is inspected
- **THEN** `LSMinimumSystemVersion` is `26.0`

### Requirement: No upstream update channel
The input method SHALL NOT query McBopomofo's update endpoint or direct the user to McBopomofo's website to download a new version.

#### Scenario: Automatic update check
- **WHEN** the input method's periodic or manual update check runs
- **THEN** no network request is made to `mcbopomofo.openvanilla.org` and no "new version available" prompt referring to McBopomofo is shown
