## ADDED Requirements

### Requirement: OpenSpec changes accompany behavioral changes
Changes that alter build behavior, runtime asset requirements, evaluation workflows, inference APIs, or security posture MUST update the corresponding OpenSpec capability.

#### Scenario: Reviewing a behavior change
- **WHEN** a pull request changes project behavior covered by a capability spec
- **THEN** reviewers can find a matching OpenSpec delta or request one before accepting the change

### Requirement: Verification commands are recorded
The project SHALL maintain verification commands for native build, random evaluation smoke tests, checkpoint evaluation, and inference service startup where applicable.

#### Scenario: Preparing a commit
- **WHEN** a contributor prepares a change affecting runtime behavior
- **THEN** the contributor can run or explicitly skip the relevant documented verification commands with a reason

### Requirement: Security-sensitive code paths are tracked
The project MUST track risks involving unsafe deserialization, shell command construction, disabled assertions, and externally supplied model or asset files.

#### Scenario: Reviewing security-sensitive changes
- **WHEN** a change touches checkpoint loading, embedding loading, shell command execution, runtime validation, or external asset parsing
- **THEN** reviewers evaluate the change against the tracked risks and require mitigation or explicit acceptance

### Requirement: Platform dependency risks are visible
The project SHALL document known dependency and platform compatibility risks before they block users at install or runtime.

#### Scenario: Installing on a constrained platform
- **WHEN** a user installs the project on a platform with limited package availability
- **THEN** documentation identifies known incompatible dependencies and available workflow-specific alternatives
