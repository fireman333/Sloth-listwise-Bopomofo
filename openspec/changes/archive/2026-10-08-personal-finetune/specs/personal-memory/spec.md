## MODIFIED Requirements

### Requirement: Clearing learning data
The user SHALL be able to clear all learning data with one action from the preferences, after confirming. Clearing SHALL delete the selection memory file, the sentence pool, any files set aside as unreadable, and all personalized model versions with their training status (personal-finetune "Clearing removes personalization"), SHALL empty the in-memory selection memory immediately, so that no previously learned pick is suggested afterwards, and SHALL return the selection model to the factory weights from the next activation. A save, record, training result or personalized load that was pending before the clear SHALL NOT recreate cleared data or be used afterwards; a composition committed after the clear is recorded as new data. Clearing SHALL NOT change user phrases, excluded phrases, phrase replacements, the factory model files or other preferences, and SHALL NOT change the learning switch.

#### Scenario: Clear and type again
- **WHEN** the user clears learning data and types readings in a context that had a remembered pick
- **THEN** the choice made without the memory is shown, and the memory folder holds no memory or pool data from before the clear

#### Scenario: Cancel the confirmation
- **WHEN** the user presses the clear button and then cancels the confirmation
- **THEN** no data is deleted

#### Scenario: Other user data untouched
- **WHEN** the user clears learning data
- **THEN** the user phrase, excluded phrase and phrase replacement files, the model files and all preferences are unchanged

#### Scenario: Clear with a personalized version in use
- **WHEN** the user clears learning data while a personalized version of the selection model is in use
- **THEN** every personalized version and its status are deleted, the factory model files are unchanged, and from the next activation the factory weights are used
