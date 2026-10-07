## MODIFIED Requirements

### Requirement: Heavy model and calibration files
The system SHALL load the heavy re-ranking model and its calibration parameters from the `models/qwen/` subdirectory of the application-support directory defined by the app-identity capability, and SHALL NOT require them to be bundled in the application. The model and calibration SHALL be treated as one unit: if either file is missing, unreadable, or invalid, or if the calibration's R-walk weight differs from the R-walk weight in use, precise mode SHALL be unavailable, the failure SHALL be logged once, and typing SHALL continue with R-walk within the 10 ms p95 per-keystroke budget and without any blocking error. When R runs with a personalized version (personal-finetune capability), the R-dependent part of the calibration (the confidence temperatures) SHALL be taken from the calibration refitted for that version and stored with it, and the remaining parameters from `models/qwen/`; if that refitted calibration is missing, invalid, or belongs to a different version, precise mode SHALL be unavailable in the same way.

#### Scenario: Calibration file missing or invalid
- **WHEN** the model file exists but the calibration file is missing or contains an unknown key
- **THEN** precise mode reports unavailable and typing uses R-walk

#### Scenario: Calibration weight mismatch
- **WHEN** the calibration file's R-walk weight differs from the configured R-walk weight
- **THEN** precise mode reports failed to load and typing uses R-walk

#### Scenario: Model directory missing
- **WHEN** precise mode is enabled and `models/qwen/` does not exist
- **THEN** the input method keeps working with R-walk and the preferences show that the model was not found

#### Scenario: Personalized R with its calibration
- **WHEN** precise mode is enabled and R runs with a personalized version whose refitted calibration is present
- **THEN** the gate uses the refitted temperatures and the other parameters from `models/qwen/`

#### Scenario: Personalized R without matching calibration
- **WHEN** precise mode is enabled and R runs with a personalized version whose refitted calibration is missing
- **THEN** precise mode reports failed to load and typing uses R-walk with the personalized weights
</content>
</invoke>
