# model-loading Specification

## Purpose

Defines where the R model lives on disk, when and how the input method loads it, how it can be switched off, and what happens when the files are missing or broken, so that model problems never stop the user from typing.

## Requirements

### Requirement: Model location
The system SHALL load the R model from the `models/r-walk/` subdirectory of the application-support directory defined by the app-identity capability, and SHALL NOT require the model to be bundled inside the application. When a personalized version adopted by the personal-finetune capability is selected for use and matches the configuration it was evaluated with (exactly the installed factory model, the R-walk weight and the reading-policy setting in use, as defined by the personal-finetune capability), the system SHALL load the factory model from `models/r-walk/` with that version's weights from `memory/r-personal/` replacing the corresponding factory weights in memory; the factory model files SHALL NOT be modified. The model files and personalized weights SHALL be read locally only; the system SHALL NOT download models in this capability.

#### Scenario: Model present
- **WHEN** the model directory contains a valid R model
- **THEN** the system loads it and R-walk becomes available

#### Scenario: Model directory missing
- **WHEN** the model directory does not exist
- **THEN** the system uses the stock walk, logs the missing path once, and typing works normally

#### Scenario: Personalized version in use
- **WHEN** a personalized version matching the installed factory model and current settings is selected for use
- **THEN** R-walk runs with the personalized weights and the bytes of `models/r-walk/` are unchanged

### Requirement: No network access
The system SHALL NOT open any network connection while loading or running the R model, including connections initiated by bundled inference libraries (for example usage telemetry). Model failures SHALL still fall back to the stock walk; disabling network access SHALL NOT add latency to key handling.

#### Scenario: Typing with the model loaded
- **WHEN** the input method loads the R model and the user types in Bopomofo mode
- **THEN** the input method process opens no outbound network connection

### Requirement: Background loading
The system SHALL load the R model off the main thread after the input method starts, and SHALL NOT block key handling, input-source activation, or candidate display while loading: activation and every keystroke during loading SHALL stay within the 10 ms p95 per-keystroke budget. Until loading completes, the stock walk SHALL be used.

#### Scenario: Typing during load
- **WHEN** the user activates the input method and types immediately
- **THEN** keystrokes are handled with the stock walk with no added delay, and R-walk is used once loading completes

### Requirement: Load failure handling
If the model files are corrupt, incompatible, or fail to load for any reason, the system SHALL use the stock walk for the rest of the process lifetime, SHALL log the failure once, and SHALL NOT crash or show a blocking error to the user. If only the personalized weights are missing, corrupt, built for a different factory model or configuration, or fail to load, the system SHALL load the factory model without them instead, SHALL log the failure once, SHALL report the version as unusable in the personalization status, and SHALL NOT crash or show a blocking error; loading SHALL stay off the key-handling thread so that every keystroke stays within the 10 ms p95 budget.

#### Scenario: Corrupt model file
- **WHEN** the model file exists but cannot be loaded
- **THEN** the input method keeps working with the stock walk and a single failure entry is logged

#### Scenario: Corrupt personalized weights
- **WHEN** the selected personalized weights file is truncated
- **THEN** R-walk runs with the factory weights, a single failure entry is logged, and the preferences show the version as unusable
</content>
</invoke>

### Requirement: Enable switch and weight via user defaults
R-walk SHALL be controlled by a user-defaults boolean that defaults to enabled and is also exposed as the neural-selection switch in the preferences window, and its in-walk weight (alpha) by a user-defaults number that defaults to the tuned value 0.5 and is not shown in the preferences window. Setting the switch to disabled or alpha to 0 SHALL make selection identical to stock McBopomofo apart from entries removed by the reading-policy capability. Changes made either with `defaults write` or in the preferences window SHALL take effect no later than the next input-method activation, including turning the switch on after starting with it off.

#### Scenario: Disabled by user defaults
- **WHEN** the enable switch is set to false
- **THEN** the composing buffer shows the stock walk result even though the model is loaded

#### Scenario: Alpha zero
- **WHEN** alpha is set to 0
- **THEN** selection is identical to stock McBopomofo apart from entries removed by the reading policy

#### Scenario: Enabled after starting disabled
- **WHEN** the input method started with the switch off and the user turns it on
- **THEN** R-walk is used from the next activation on, once the model has loaded

#### Scenario: Disabled in the preferences window
- **WHEN** the user turns neural selection off in the preferences window
- **THEN** `defaults read` shows the switch as false and, from the next activation, the composing buffer shows the stock walk result
