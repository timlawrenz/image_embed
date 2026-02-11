## Why

`describe_image` currently uses BLIP, which is dated and tends to produce short captions. We want higher-quality, instruction-following descriptions using `google/gemma-3-27b-it`.

## What Changes

- Update the `describe_image` operation to produce its final description text using `google/gemma-3-27b-it`.
- Keep the existing API contract (`type=describe_image`, targets, response fields) unchanged.
- Document the new description model behavior in `README.md`.

## Capabilities

### New Capabilities
- `describe-image`: Generate high-quality image descriptions via an LLM-backed description pipeline.

### Modified Capabilities

## Impact

- `app/services/description_service.py`: generate a base caption from the image and use Gemma to produce the final description.
- `app/core/model_loader.py`: add cached loader for Gemma model + tokenizer.
- `README.md`: update describe_image documentation.
- Tests: update/extend unit tests for the new behavior.
