## Context

Today `describe_image` is implemented in `app/services/description_service.py` as a BLIP captioning step (vision) followed by a Gemma text-only rewrite step. Gemma is currently loaded via `AutoModelForCausalLM` + `AutoTokenizer`, which cannot accept image inputs.

The new goal is to use **Gemma 3 multimodal** (`google/gemma-3-27b-it`) directly with image+prompt to produce a dense, dry, single-paragraph training caption suitable for DiT-style dataset creation.

Constraints:
- The 27B checkpoint is large and should not run on CPU in production.
- The repo already centralizes model loading/caching and device selection in `app/core/model_loader.py`.
- `describe_image` already supports optional cropping (prominent person/face) and returns the cropped image base64; that behavior should be preserved.

## Goals / Non-Goals

**Goals:**
- Replace BLIP usage for `describe_image` with direct Gemma 3 multimodal inference on the image and a dataset-caption prompt.
- Enforce output formatting: **single dense paragraph**, dry tone, no lists.
- Enforce grounding: do not invent details; if unclear, state “unclear/indistinct”.
- Redefine `max_length` as the **output token budget** (`max_new_tokens`) for Gemma, and set the default to ~300.
- Remove fallback: if Gemma cannot load or generate, the task returns `status="error"`.
- Require CUDA device for `describe_image` (including ROCm via `torch.device("cuda")`).

**Non-Goals:**
- Fine-tuning Gemma, adding LoRA/QLoRA, or changing model weights.
- Building a separate batch captioning pipeline (beyond existing helper scripts).
- Adding photographer metadata guesses (focal length, ISO, aperture).

## Decisions

1) **Model API (multimodal): `Gemma3ForConditionalGeneration` + `AutoProcessor`**
- Use Transformers’ recommended multimodal path:
  - `Gemma3ForConditionalGeneration.from_pretrained(model_id, ...)`
  - `AutoProcessor.from_pretrained(model_id)`
  - `processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")`
- Rationale: This is the supported interface for image+text chat prompting for Gemma 3, and avoids ad-hoc image preprocessing.
- Alternative: `pipeline("image-text-to-text")`.
  - Not chosen because we already have explicit model lifecycle management and want finer control over device/dtype/caching.

2) **Device policy: require CUDA for `describe_image`**
- Enforce a runtime check: if `model_loader.get_device() != "cuda"`, fail the task (raise an exception that becomes `status="error"`).
- Rationale: avoids accidental CPU execution that is likely unusably slow or may OOM for 27B.
- Alternative: allow CPU fallback.
  - Rejected due to predictably poor UX and resource risk.

3) **Transformers version bump**
- Update dependency to `transformers>=4.50.0` (per HF Gemma 3 docs).
- Rationale: ensures multimodal Gemma 3 classes/processor behavior exist.
- Trade-off: may subtly affect other HF models; mitigate via unit tests and pinned lower bounds only (not exact pin).

4) **Prompting format**
- Use a short system message that encodes policy (dry tone, one paragraph, no inventions, required facets).
- User message contains the image plus the captioning instruction text.
- Rationale: aligns with instruction-tuned chat behavior, encourages consistent formatting.

5) **`max_length` semantics**
- Define `max_length` for `describe_image` as output `max_new_tokens`.
- Default should be 300 tokens.
- Rationale: aligns with “token-rich” request and removes confusion between BLIP caption length vs LLM generation.

6) **No fallback**
- Remove non-fatal fallback to BLIP caption.
- Rationale: BLIP is removed, and silently returning a weak caption is undesirable for dataset quality.

## Risks / Trade-offs

- [High VRAM requirement / OOM] → Mitigation: require CUDA; consider `device_map="auto"` + `torch_dtype=torch.bfloat16`/`float16` in implementation; document VRAM expectations.
- [Dependency bump affects other HF models] → Mitigation: run full test suite; keep the change limited to a minimum version bump.
- [Model may echo prompt or include formatting] → Mitigation: decode only generated continuation past input length; keep “one paragraph” constraint; strip leading/trailing whitespace.
- [Hallucinations] → Mitigation: strict prompt language (“do not invent; if unclear say unclear”); do_sample=False.

## Migration Plan

- Update `requirements*.txt` and install in the venv.
- Update `model_loader` to load/cached Gemma multimodal model+processor.
- Update `description_service` to use Gemma directly, preserving crop behavior.
- Update `main.py` to set `describe_image` default `max_length=300`.
- Update tests and README.
- Rollback strategy: revert to prior commit and/or restore BLIP-based implementation; note this is a breaking behavior change.

## Open Questions

- Exact dtype/device-map policy for production (single GPU `.to("cuda")` vs `device_map="auto"`).
- Whether to expose additional prompt controls via API (e.g., a `style` or `prompt` override) or keep it fixed for dataset consistency.
