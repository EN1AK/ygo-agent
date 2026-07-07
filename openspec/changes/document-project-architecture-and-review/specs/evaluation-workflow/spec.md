## ADDED Requirements

### Requirement: Evaluation workflows declare asset preconditions
Evaluation commands SHALL document the required locale database, strings file, YGOPro scripts, decks, checkpoints, embeddings, and code-list inputs before execution.

#### Scenario: Evaluation starts with missing assets
- **WHEN** a user starts an evaluation workflow without required runtime assets
- **THEN** the workflow fails early or documents the missing asset path instead of producing an unrelated import or native runtime error

### Requirement: Random evaluation provides a smoke test
The project SHALL provide a random-agent evaluation path that can validate the native environment, Python wrappers, and locale assets without requiring a trained checkpoint.

#### Scenario: Running the random smoke test
- **WHEN** a user runs the documented random evaluation command after building the environment and provisioning assets
- **THEN** the command exercises the environment loop and reports duel outcomes or runtime errors

### Requirement: Checkpoint evaluation is reproducible
Checkpoint-based evaluation MUST require explicit model checkpoint, deck, language, seed, and environment configuration inputs or documented defaults.

#### Scenario: Running checkpoint evaluation
- **WHEN** a user evaluates a trained model checkpoint
- **THEN** the selected checkpoint, deck configuration, language assets, and evaluation options are visible in command arguments or logged configuration

### Requirement: Battle evaluation isolates player configurations
Battle workflows SHALL distinguish the two competing player configurations, including deck, checkpoint, model, and random or bot policy choices.

#### Scenario: Running a battle
- **WHEN** two agents are evaluated against each other
- **THEN** each side's configuration is independently specified or reported so results can be attributed to the correct player setup
