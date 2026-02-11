## Context

The service already supports `embed_dino_v2` via timm DINOv2 models. We want to add a parallel `embed_dino_v3` operation using a HuggingFace-hosted DINOv3 checkpoint.

## Goals / Non-Goals

**Goals:**
- Add `embed_dino_v3` as a first-class operation (same targets as `embed_dino_v2`).
- Default to `facebook/dinov3-vitl16-pretrain-lvd1689m`.
- Keep model loading cached and non-fatal per-request (load-on-first-use).

**Non-Goals:**
- Changing existing `embed_dino_v2` behavior.
- Re-training classifiers or changing the remote training pipeline.

## Decisions

### Decision 1: Load DINOv3 via Transformers Auto* APIs
Implement `model_loader.get_dino_v3_model_and_processor()` using `transformers.AutoModel` and `transformers.AutoImageProcessor` with caching similar to other models.

### Decision 2: Robust embedding extraction
Because output structures can vary between HF models, extract an embedding using the first available of:
- `outputs.image_embeds`
- `outputs.pooler_output`
- `outputs.last_hidden_state[:, 0]`

## Risks / Trade-offs

- DINOv3 checkpoints may require custom code or different output keys → mitigated by robust extraction and unit tests that mock the loader.
- Large checkpoints can have significant memory/latency costs → mitigated by caching and keeping this as an opt-in operation.
