## ADDED Requirements

### Requirement: WindBot can be configured as an opponent
The project SHALL provide configuration for using IceYGO WindBot as a distinct opponent mode without replacing existing random, greedy, or self-play modes.

#### Scenario: Selecting WindBot opponent mode
- **WHEN** a user selects the WindBot opponent mode in a supported training or evaluation workflow
- **THEN** the workflow uses WindBot-specific configuration instead of the built-in greedy or random bot

#### Scenario: Existing bot modes remain available
- **WHEN** a user selects random, greedy, or self-play modes
- **THEN** those modes continue to run without requiring WindBot binaries or .NET/Mono runtime dependencies

### Requirement: WindBot executable and runtime assets are externally provisioned
The WindBot integration MUST treat WindBot source, build outputs, executable files, bot decks, and copied WindBot runtime assets as external local artifacts unless explicitly vendored by a future change.

#### Scenario: WindBot binary is missing
- **WHEN** a user selects WindBot mode without configuring a valid WindBot executable
- **THEN** the workflow fails before training starts with an actionable setup error

#### Scenario: WindBot cards database is missing
- **WHEN** the configured WindBot runtime directory lacks the required `cards.cdb`
- **THEN** the workflow reports the missing asset and does not start a training run

### Requirement: WindBot process lifecycle is controlled
The project SHALL launch, monitor, timeout, log, and clean up WindBot processes used by automated workflows.

#### Scenario: WindBot exits unexpectedly
- **WHEN** WindBot exits before a duel or episode finishes
- **THEN** the workflow captures its exit code/logs, marks the episode or smoke test as failed, and cleans up related resources

#### Scenario: Training is interrupted
- **WHEN** the user interrupts a WindBot-backed workflow
- **THEN** the workflow terminates any child WindBot processes it started

### Requirement: WindBot connection settings are explicit
The integration MUST expose connection settings required by WindBot, including host, port, host info or password, bot name, deck, dialog setting, and connection timeout.

#### Scenario: Connection succeeds
- **WHEN** a configured local YGOPro-compatible host is listening and WindBot is launched with matching connection settings
- **THEN** WindBot connects and can participate as the opponent side

#### Scenario: Connection times out
- **WHEN** WindBot does not connect within the configured timeout
- **THEN** the workflow fails early and reports host, port, and timeout values

### Requirement: WindBot compatibility is validated before training
The project SHALL provide a WindBot smoke test that validates executable, assets, deck configuration, connection, and at least one duel or short episode before long training jobs use WindBot.

#### Scenario: Smoke test passes
- **WHEN** WindBot setup is valid and the local host/adapter can complete a short duel or episode
- **THEN** the smoke test reports success and records the WindBot deck and executable path used

#### Scenario: Deck is incompatible
- **WHEN** the configured WindBot deck references cards unsupported by local card database or scripts
- **THEN** the smoke test fails with card/deck compatibility information before training begins

### Requirement: Training and evaluation expose WindBot opponent selection
Supported training and evaluation entrypoints SHALL expose WindBot as an opponent option where the local environment/adapter supports it.

#### Scenario: Evaluation against WindBot
- **WHEN** a user runs evaluation with WindBot selected
- **THEN** the result output identifies WindBot as the opponent and includes the configured WindBot deck

#### Scenario: Training against WindBot
- **WHEN** a user starts a supported training workflow with WindBot selected as the opponent
- **THEN** rollouts use WindBot for the opponent side or fail before training starts if the WindBot adapter is unavailable

### Requirement: WindBot integration is reproducible
The integration SHALL record WindBot source revision or executable version when available, configured deck, asset paths, connection settings, and random seeds used for WindBot-backed runs.

#### Scenario: Run metadata is emitted
- **WHEN** a WindBot-backed evaluation or training run starts
- **THEN** logs or run metadata include enough WindBot configuration to reproduce the opponent setup
