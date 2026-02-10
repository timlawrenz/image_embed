# accelerator-info-logging Specification

## Purpose
Provide non-fatal, debug-level logging for failures when querying optional accelerator/device properties during startup (e.g., ROCm-only fields).
## Requirements
### Requirement: Accelerator info logging is non-fatal
The system SHALL attempt to query optional accelerator/device properties during startup for diagnostic purposes. If querying an optional property fails, the system SHALL log a debug-level message and SHALL continue without raising.

#### Scenario: Optional device property query fails
- **WHEN** `_log_accelerator_info()` attempts to read an optional device property (e.g., `gcnArchName`) and the underlying call raises an exception
- **THEN** the system logs a debug-level message including the exception
- **AND** the function returns normally without raising

