## Context

`app/core/model_loader.py` logs basic accelerator information at startup. For optional/ROCm-specific properties (e.g., `gcnArchName`), failures are currently swallowed (`except Exception: pass`), which hides useful diagnostics when device queries misbehave.

## Goals / Non-Goals

**Goals:**
- Preserve non-fatal startup behavior even when optional accelerator property queries fail.
- Provide a debuggable signal (log message) when those optional queries fail.

**Non-Goals:**
- Changing device selection logic (CUDA vs CPU) or error handling for required device queries.
- Adding new external dependencies or changing the logging configuration.

## Decisions

### Decision 1: Log and continue for optional property failures
When reading optional properties under the inner `try:` block, catch exceptions and emit a `logger.debug(...)` message (optionally with `exc_info=True`) and continue.

**Rationale:**
- Keeps behavior non-fatal (matches current intent) while making failures observable.
- Debug level avoids noisy logs for normal users, but is available when troubleshooting.

## Risks / Trade-offs

- Debug logs may still be considered noisy in some deployments → mitigated by using debug level (and keeping the message scoped to optional properties only).
