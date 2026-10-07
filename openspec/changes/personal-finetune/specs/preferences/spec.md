## MODIFIED Requirements

### Requirement: Selection-model preferences tab
The preferences window SHALL include a tab titled 「選字模型」 (Traditional Chinese) / "Selection Model" (English) containing:
- a switch for neural selection (R-walk), default on;
- a switch for precise mode, default off;
- a switch to accept the ㄧㄢˊ reading for 癌, default off;
- a switch 「學習我的選字」 / "Learn from my typing", default on, as defined by the personal-memory capability;
- a button 「清除學習資料…」 / "Clear Learning Data…" that asks for confirmation and then clears learning data as defined by the personal-memory capability (which also removes personalization, as defined by the personal-finetune capability);
- a personalization section 「個人化模型」 / "Personalized Model" showing the personalization status defined by the personal-finetune capability, with a button 「立即學習」 / "Learn Now" (replaced by 「取消」 / "Cancel" while a run is in progress), a button 「改回出廠模型」 / "Use Factory Model" enabled while a personalized version is in use, and a button 「改用上一版」 / "Use Previous Version" enabled only when a previous version exists; 「立即學習」 SHALL be disabled while learning is off;
- a domain-pack section listing the 14 categories 工業、中醫、生物學、西醫、佛教、法律與政治、物理學與化學、建築、軍事、商業、教育、通訊、運動、電腦, each shown disabled and labelled as not yet available.
The domain-pack entries SHALL NOT be selectable and SHALL NOT change typing behavior. The R-walk weight SHALL NOT be shown in the tab. All values SHALL be stored in the preferences domain defined by the app-identity capability. Pressing a personalization button SHALL NOT block the preferences window or key handling (every keystroke within the 10 ms p95 budget).

#### Scenario: Opening the tab with default preferences
- **WHEN** the user opens the preferences window on a fresh install and selects 「選字模型」
- **THEN** neural selection is on, precise mode is off, acceptance of ㄧㄢˊ is off, learning is on, the clear-data button is shown, the personalization section shows the factory model in use with 「改回出廠模型」 and 「改用上一版」 disabled, and the 14 domain packs are shown disabled as not yet available

#### Scenario: Clicking a domain pack
- **WHEN** the user clicks a domain-pack entry
- **THEN** nothing is toggled and typing behavior is unchanged

#### Scenario: Clearing from the tab
- **WHEN** the user clicks 「清除學習資料…」 and confirms
- **THEN** learning data is cleared and the learning switch keeps its value

#### Scenario: Learning now from the tab
- **WHEN** learning is on, enough eligible segments exist, and the user clicks 「立即學習」
- **THEN** a training run starts, its progress is shown with a 「取消」 button, and the window stays responsive

#### Scenario: Learning off
- **WHEN** learning is off
- **THEN** 「立即學習」 is disabled
</content>
</invoke>
