## MODIFIED Requirements

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
