# Spec Delta

## Purpose

Defines where the R model lives on disk, when and how the input method loads it, how it can be switched off, and what happens when the files are missing or broken, so that model problems never stop the user from typing.

## ADDED Requirements

### Requirement: Model location
The system SHALL load the R model from the `models/r-walk/` subdirectory of the application-support directory defined by the app-identity capability, and SHALL NOT require the model to be bundled inside the application. The model files SHALL be read locally only; the system SHALL NOT download models in this capability.

#### Scenario: Model present
- **WHEN** the model directory contains a valid R model
- **THEN** the system loads it and R-walk becomes available

#### Scenario: Model directory missing
- **WHEN** the model directory does not exist
- **THEN** the system uses the stock walk, logs the missing path once, and typing works normally

### Requirement: Background loading
The system SHALL load the R model off the main thread after the input method starts, and SHALL NOT block key handling, input-source activation, or candidate display while loading: activation and every keystroke during loading SHALL stay within the 10 ms p95 per-keystroke budget. Until loading completes, the stock walk SHALL be used.

#### Scenario: Typing during load
- **WHEN** the user activates the input method and types immediately
- **THEN** keystrokes are handled with the stock walk with no added delay, and R-walk is used once loading completes

### Requirement: Load failure handling
If the model files are corrupt, incompatible, or fail to load for any reason, the system SHALL use the stock walk for the rest of the process lifetime, SHALL log the failure once, and SHALL NOT crash or show a blocking error to the user.

#### Scenario: Corrupt model file
- **WHEN** the model file exists but cannot be loaded
- **THEN** the input method keeps working with the stock walk and a single failure entry is logged

### Requirement: Enable switch and weight via user defaults
R-walk SHALL be controlled by a user-defaults boolean that defaults to enabled, and its in-walk weight (alpha) by a user-defaults number that defaults to the tuned value 0.5. Setting the switch to disabled or alpha to 0 SHALL make selection identical to stock McBopomofo. Changes SHALL take effect no later than the next input-method activation, including turning the switch on after starting with it off.

#### Scenario: Disabled by user defaults
- **WHEN** the enable switch is set to false
- **THEN** the composing buffer shows the stock walk result even though the model is loaded

#### Scenario: Alpha zero
- **WHEN** alpha is set to 0
- **THEN** selection is identical to stock McBopomofo

#### Scenario: Enabled after starting disabled
- **WHEN** the input method started with the switch off and the user turns it on
- **THEN** R-walk is used from the next activation on, once the model has loaded
