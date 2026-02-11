## Why

We want `describe_image` to produce dense, token-rich, single-paragraph captions suitable for DiT-style text-to-image training datasets.
The current BLIP-based captioning (and BLIP→Gemma text rewrite) is not sufficient and does not leverage Gemma 3’s multimodal image understanding.

## What Changes

- Switch `describe_image` to send the image and a training-caption prompt directly to the multimodal model `google/gemma-3-27b-it` (no BLIP base caption step).
- Increase the default `max_length` for `describe_image` to ~300 tokens and define it as the output token budget (`max_new_tokens`).
- **BREAKING**: Remove the non-fatal fallback behavior for `describe_image` (Gemma load/generation failures return an error result for that task).
- **BREAKING**: Require GPU for `describe_image` by default to avoid accidental CPU OOM/unusable latency.

## Capabilities

### New Capabilities

<!-- none -->

### Modified Capabilities

- `describe-image`: Replace BLIP-based caption generation with direct multimodal Gemma 3 captioning, update `max_length` defaults/semantics, and change failure behavior to error (no fallback).

## Impact

- **API behavior**: `describe_image` output becomes a single dense paragraph aimed at training captions; default length changes; failures become task `status="error"` (not fallback).
- **Runtime requirements**: `describe_image` will require CUDA-capable GPU for Gemma 27B.
- **Dependencies**: Update Transformers dependency to a version that supports Gemma 3 multimodal inference (HF docs indicate `transformers>=4.50.0`), and use `AutoProcessor` + `Gemma3ForConditionalGeneration`.
- **Code areas**:
  - `app/core/model_loader.py` (Gemma multimodal loader)
  - `app/services/description_service.py` (prompting + generation)
  - `main.py` (`describe_image` param defaults/semantics)
  - unit/integration tests + `README.md` documentation.
