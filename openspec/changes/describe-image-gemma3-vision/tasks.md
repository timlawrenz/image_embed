## 1. Dependencies and defaults

- [x] 1.1 Update `requirements.txt` to require Transformers with Gemma 3 multimodal support (`transformers>=4.50.0`)
- [x] 1.2 Update `main.py` to default `describe_image.max_length` to 300 and define it as output token budget
- [x] 1.3 Update `app/services/description_service.py` function default `max_length` to 300 for consistency

## 2. Gemma 3 multimodal model loading

- [x] 2.1 Replace `get_gemma_text_model_and_tokenizer()` with a multimodal loader (e.g., `get_gemma_vision_model_and_processor()`) using `Gemma3ForConditionalGeneration` + `AutoProcessor`
- [x] 2.2 Ensure Gemma model/processor are cached in `app/core/model_loader.py` and loaded on the configured device with appropriate dtype

## 3. Direct image prompting for describe_image

- [x] 3.1 Remove BLIP caption generation from `get_image_description()` and generate captions directly from Gemma using image+prompt chat template
- [x] 3.2 Preserve existing crop behavior and returned `cropped_image_base64` / `cropped_image_bbox`
- [x] 3.3 Enforce CUDA requirement for `describe_image` (fail with task `status="error"` if CUDA is unavailable)
- [x] 3.4 Ensure output is a single dense paragraph and strip any prompt/template echoes robustly

## 4. Tests and documentation

- [x] 4.1 Update unit tests in `tests/test_unit/test_services/test_description_service.py` to mock multimodal processor/model and assert `max_new_tokens == max_length`
- [x] 4.2 Add/adjust tests to verify GPU-required error behavior (unit or integration, whichever fits existing test style)
- [x] 4.3 Update `README.md` to document new `describe_image` behavior and the new default `max_length` (300 output tokens)

## 5. Verification

- [x] 5.1 Run `python -m pytest` and fix any failures related to the change
- [x] 5.2 Run `openspec validate describe-image-gemma3-vision`
- [x] 5.3 Ensure tasks are checked off and the change is ready to archive
