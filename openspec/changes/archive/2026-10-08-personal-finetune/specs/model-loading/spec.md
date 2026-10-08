## MODIFIED Requirements

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
