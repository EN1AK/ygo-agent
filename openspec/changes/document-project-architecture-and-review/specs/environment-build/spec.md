## ADDED Requirements

### Requirement: Native YGO environment build is reproducible
The project SHALL define a reproducible native build path for the `ygopro_ygoenv` Python extension using xmake and the repository package definitions.

#### Scenario: Building the native extension
- **WHEN** a developer runs the documented xmake build command with required compiler and Python prerequisites installed
- **THEN** the build produces the platform-specific `ygopro_ygoenv` extension for the active Python runtime

### Requirement: Linux and Windows build prerequisites are documented separately
The build documentation MUST distinguish Linux prerequisites from native Windows prerequisites, including compiler toolchain, Python headers and libraries, xmake, pybind11, and SQLite/Lua dependencies.

#### Scenario: Selecting a platform setup
- **WHEN** a developer follows the build documentation on Linux or Windows
- **THEN** the documentation lists the platform-specific prerequisites and does not imply that one platform's commands are sufficient for the other

### Requirement: Windows Python discovery is explicit
The Windows build SHALL support explicit Python and pybind11 discovery through `PYBIND11_INCLUDE`, `PYTHON_INCLUDE`, `PYTHON_LIBDIR`, and `PYTHON_VERSION_NODOT`.

#### Scenario: Windows build uses explicit Python paths
- **WHEN** xmake cannot safely infer the target Python installation on Windows
- **THEN** the build can consume the explicit environment variables and link against the intended Python runtime

### Requirement: Generated build outputs stay ignored
Compiled extensions, xmake caches, object files, downloaded package caches, and other generated build outputs MUST remain outside version control.

#### Scenario: Reviewing build artifacts
- **WHEN** a native build succeeds locally
- **THEN** generated binaries and cache files are ignored or cleaned rather than staged as source changes
