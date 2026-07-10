## ADDED Requirements

### Requirement: Code list supports metadata-compatible lines
The project SHALL define code-list entries as whitespace-separated lines where the first token is the card code and later tokens are optional metadata.

#### Scenario: Reading a generated two-column code list
- **WHEN** a reader loads a line formatted as `<card_code> <has_script>`
- **THEN** it uses `<card_code>` as the card id while preserving line order

#### Scenario: Reading a legacy single-column code list
- **WHEN** a reader loads a line formatted as `<card_code>`
- **THEN** it uses `<card_code>` as the card id without requiring metadata

### Requirement: Code list generation records script availability
The project SHALL generate code-list files from the active card database and script directory with deterministic card-code ordering and script availability metadata.

#### Scenario: Generating from active assets
- **WHEN** the asset update workflow runs code-list generation
- **THEN** the output records each card code and whether a corresponding Lua script exists

#### Scenario: Script exists outside the card database
- **WHEN** a script file references a card code absent from the selected `cards.cdb`
- **THEN** generation fails with the missing code information instead of writing an inconsistent code list

### Requirement: Python tools share code-list parsing
Python training, evaluation, and embedding utilities SHALL use shared code-list parsing behavior instead of assuming the whole line is a single integer.

#### Scenario: Loading embeddings with a two-column code list
- **WHEN** embedding loading reads the generated code list
- **THEN** it uses the first column to determine embedding order and reports missing embeddings by card code

#### Scenario: Updating embeddings with a two-column code list
- **WHEN** the embedding generation script reads the generated code list
- **THEN** it preserves existing card order and appends newly discovered deck cards as card-code entries

### Requirement: Native environment readers tolerate optional metadata
Native environment backends SHALL parse at least the first card-code column from code-list lines and ignore metadata they do not require.

#### Scenario: YGOPro-v0 reads current code list
- **WHEN** YGOPro-v0 initializes with a two-column code list
- **THEN** it uses the first column as the card code and does not reject the metadata column

#### Scenario: EDOPro reads current code list
- **WHEN** EDOPro initializes with a two-column code list
- **THEN** it uses the first column as the card code and does not reject the metadata column
