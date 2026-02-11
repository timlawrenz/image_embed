## Context

`describe_image` currently uses BLIP to caption an image region. We want to upgrade output quality by using `google/gemma-3-27b-it` (instruction-tuned LLM) to produce the final description.

## Goals / Non-Goals

**Goals:**
- Preserve the existing `describe_image` API shape.
- Produce richer final text using Gemma.
- Keep model loading cached (load-on-first-use) and keep requests non-fatal if Gemma is unavailable.

**Non-Goals:**
- Introducing a new endpoint or changing task params.
- Guaranteeing Gemma is preloaded at startup (the model is large; keep it lazy-loaded).

## Decisions

### Decision 1: Two-stage pipeline
1) Use existing vision captioning (BLIP) to obtain a base caption from the (optionally cropped) image.
2) Prompt Gemma with the base caption (and minimal context) to produce a higher-quality final description.

Rationale: `google/gemma-3-27b-it` is a text LLM; the vision-to-text step remains BLIP.

### Decision 2: Safe fallback behavior
If Gemma cannot load or generate, return the base caption so `describe_image` remains usable in smaller environments.

## Risks / Trade-offs

- Gemma 27B is very large (VRAM/RAM) → mitigated by lazy loading + fallback.
- Prompting may produce verbose output → mitigated by bounding generation length and using a consistent instruction prompt.
