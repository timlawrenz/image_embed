## Why

On some systems (especially ROCm builds), optional accelerator properties like `gcnArchName` may not be available or may throw. Today we silently swallow that exception, which makes debugging device/driver issues harder.

## What Changes

- Log a debug-level message when querying optional accelerator properties fails, instead of swallowing the exception.
- Keep behavior non-fatal: model loading/device detection should continue even if optional properties can’t be read.

## Capabilities

### New Capabilities
- `accelerator-info-logging`: Provide non-fatal, debuggable logging around optional accelerator/device property queries during startup.

### Modified Capabilities

## Impact

- `app/core/model_loader.py`: Improve exception handling inside `_log_accelerator_info()` to emit useful logs without changing device selection behavior.
