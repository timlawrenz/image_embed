## Why

We want to offer a modern DINOv3 embedding option so clients can request the latest visual embeddings (including a large/strong checkpoint) without changing the API shape.

## What Changes

- Add a new analysis task type `embed_dino_v3`, parallel to `embed_dino_v2`.
- Implement `embed_dino_v3` using the HuggingFace checkpoint `facebook/dinov3-vitl16-pretrain-lvd1689m` by default.
- Advertise the new operation via the `/available_operations/` endpoint.
- Document the new operation in `README.md`.

## Capabilities

### New Capabilities
- `embed-dino-v3`: Generate DINOv3 image embeddings for whole images or detected regions (person/face), similar to DINOv2.

### Modified Capabilities

## Impact

- `main.py`: add `embed_dino_v3` to `AVAILABLE_OPERATIONS` and handle it in the request task dispatcher.
- `app/core/model_loader.py`: add a cached loader for DINOv3 (HF model + image processor).
- `app/services/embedding_service.py`: add `get_dino_v3_embedding()`.
- `README.md`: document `embed_dino_v3`.
- Tests: add unit coverage for the new embedding function.
