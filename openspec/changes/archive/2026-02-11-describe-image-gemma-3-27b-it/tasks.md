## 1. Core implementation

- [x] 1.1 Add cached Gemma model+tokenizer loader to `app/core/model_loader.py`.
- [x] 1.2 Update `app/services/description_service.py` to generate a base caption and then produce the final description via Gemma (with fallback).

## 2. Tests

- [ ] 2.1 Update/add unit tests for `get_image_description()` for Gemma path and fallback path.

## 3. Docs

- [ ] 3.1 Update `README.md` to note Gemma is used for `describe_image` and document fallback behavior.

## 4. Verify

- [ ] 4.1 Run pytest.
