## ADDED Requirements

### Requirement: Inference service exposes health and prediction APIs
The `ygoinf` service SHALL expose a basic health endpoint and prediction endpoints for duel initialization and action prediction.

#### Scenario: Service health check
- **WHEN** a client requests the service root endpoint
- **THEN** the service returns a successful response that can be used as a deployment health check

#### Scenario: Client requests prediction
- **WHEN** a client submits a valid duel state payload to the prediction API
- **THEN** the service converts the payload into model features and returns action scores or the selected action according to the configured model backend

### Requirement: Model and card metadata inputs are required
The inference service MUST require configured model artifacts and card metadata inputs such as checkpoints, TFLite models, code lists, embeddings, and locale databases before serving predictions.

#### Scenario: Starting without model artifacts
- **WHEN** the inference service starts without the configured model or card metadata files
- **THEN** startup or the first prediction request fails with a clear configuration error rather than silently serving invalid predictions

### Requirement: Duel state lifecycle is bounded
The inference service SHALL bound stored duel state by an expiry or cleanup policy so long-running deployments do not retain stale duel sessions indefinitely.

#### Scenario: Duel state expires
- **WHEN** a duel session has not been updated within the configured expiry window
- **THEN** the service removes or refuses stale state before it can affect later predictions

### Requirement: Platform-specific inference dependencies are isolated
JAX and TFLite inference dependencies SHALL be declared as platform-aware or optional dependencies when a platform cannot install all backends reliably.

#### Scenario: Installing inference on Windows
- **WHEN** a user installs the inference package on a platform where `tflite-runtime` is unavailable
- **THEN** installation guidance or package extras allow the user to install supported backends without blocking unrelated workflows
