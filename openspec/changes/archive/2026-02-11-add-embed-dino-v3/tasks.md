## 1. Core implementation

- [x] 1.1 Add cached DINOv3 model+processor loader to `app/core/model_loader.py`.
- [x] 1.2 Add `get_dino_v3_embedding()` to `app/services/embedding_service.py` (crop + base64 behavior matches DINOv2).
- [x] 1.3 Add `embed_dino_v3` to `AVAILABLE_OPERATIONS` and implement request handling in `main.py`.

## 2. Tests

- [x] 2.1 Add unit tests for `get_dino_v3_embedding()` (whole image + crop).

## 3. Docs

- [x] 3.1 Update `README.md` to document `embed_dino_v3` and its default checkpoint.

## 4. Verify

- [x] 4.1 Run pytest.
