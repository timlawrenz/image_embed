# Crop & Embedding Cache Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Eliminate redundant image crops and base64 encodes within a single request by extracting a shared crop cache in `shared_context` and having all embedding functions consume it.

**Architecture:** A new `get_cropped_image()` helper in `image_utils.py` checks `shared_context` for a cached `(cropped_PIL, base64_str)` pair keyed by bbox tuple. All embedding functions and `describe_image` call this helper instead of doing their own crop. `_perform_analysis` populates bbox detection into `shared_context` once, then all downstream consumers reuse the same crop cache. Embeddings are cached per `{emb_type}_{target}` key and shared between `_perform_analysis` and `classify_embedding_from_image`.

**Tech Stack:** Python 3.12+, PIL/Pillow, FastAPI

---

## Root Cause Analysis

**Log evidence** (from user's excerpt):
```
Classifying for collection 6 using embed_dino_v2 on prominent_person
→ Detection: bbox [227, 7, 926, 1709]
→ Cropping image for DINO embedding with bbox: [227, 7, 926, 1709]   ← crop #1

Classifying for collection 7 using embed_clip_vit_b_32 on prominent_person
→ Cropping image for CLIP embedding with bbox: [227, 7, 926, 1709]   ← crop #2 (identical!)
```

**Three problems found:**

1. **Each embedding function does its own crop+encode.** `get_clip_embedding`, `get_dino_embedding`, `get_dino_v3_embedding`, `get_dino_v3_patch_embedding` each contain ~15 lines of identical `pil_image.crop(bbox)` → `BytesIO` → `save(PNG)` → `base64`. Meanwhile `image_utils.py` already has a perfectly good `crop_image_and_get_base64()` — only used by `description_service.py`.

2. **No per-request crop cache.** When two different embedding types request the same bbox on the same source image, the crop is executed twice. The cropped PIL Image and its base64 encoding should be cached in `shared_context` and reused.

3. **Two separate embedding caches.** `_perform_analysis` uses keys like `embedding_{target}` (only for CLIP), while `classify_embedding_from_image` uses `embedding_{emb_type}_{target}`. Both live in `shared_context` but don't share — a CLIP embedding computed by the legacy path won't be visible to `classify_embedding_from_image`.

---

### Task 1: Add `get_cropped_image()` with shared_context cache to image_utils

**Objective:** Create a helper that returns `(cropped_PIL, base64_str)` from a cache in `shared_context`, cropping only on cache miss.

**Files:**
- Modify: `app/services/image_utils.py` (append new function)

**Step 1: Add the function**

Add after the existing `crop_image_and_get_base64` function:

```python
def get_cropped_image(
    pil_image: Image.Image,
    bbox: List[int],
    shared_context: Optional[dict] = None,
) -> Tuple[Image.Image, str]:
    """
    Returns (cropped_image, base64_str) for the given bbox.
    
    If shared_context is provided, caches the result under the key
    f"crop_{tuple(bbox)}" so that subsequent calls with the same bbox
    on the same request reuse the crop without re-executing it.
    """
    if shared_context is not None:
        cache_key = f"crop_{tuple(bbox)}"
        cached = shared_context.get(cache_key)
        if cached is not None:
            logger.debug("Reusing cached crop for bbox %s", bbox)
            return cached
        cropped, b64 = crop_image_and_get_base64(pil_image, bbox)
        shared_context[cache_key] = (cropped, b64)
        return cropped, b64
    return crop_image_and_get_base64(pil_image, bbox)
```

**Step 2: Verify the file parses**

```bash
python -c "from app.services.image_utils import get_cropped_image; print('OK')"
```

Expected: `OK`

**Step 3: Run existing tests**

```bash
python -m pytest tests/ -x -q
```

Expected: all existing tests pass (the new function isn't called yet).

---

### Task 2: Refactor embedding functions to use `get_cropped_image()`

**Objective:** Replace the duplicate crop+encode blocks in all four embedding functions with `get_cropped_image()`. Pass a new optional `shared_context` parameter so the cache is request-scoped.

**Files:**
- Modify: `app/services/embedding_service.py` (all 4 embedding functions)

**Step 1: Add `shared_context` parameter to all embedding function signatures**

Each function gets a new optional parameter:

```python
def get_clip_embedding(
    pil_image_rgb: Image.Image, 
    clip_model_name: str,
    crop_bbox: Optional[List[int]] = None,
    shared_context: Optional[dict] = None,  # NEW
) -> Tuple[List[float], Optional[str], Optional[List[int]]]:
```

Same for `get_dino_embedding`, `get_dino_v3_embedding`, `get_dino_v3_patch_embedding`.

**Step 2: Replace crop+encode block in each function**

Current pattern (in all 4 functions):
```python
if crop_bbox:
    logger.info(f"Cropping image for ... with bbox: {crop_bbox}")
    xmin, ymin, xmax, ymax = crop_bbox
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("Invalid bounding box for cropping.")
    image_to_embed = pil_image_rgb.crop(crop_bbox)       # ← direct crop
    if image_to_embed.width == 0 or image_to_embed.height == 0:
        raise ValueError("Cropped image for embedding is empty.")
    actual_crop_bbox = crop_bbox
    buffered = io.BytesIO()
    image_to_embed.save(buffered, format="PNG")          # ← direct encode
    base64_cropped_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
```

New pattern (in all 4 functions):
```python
if crop_bbox:
    logger.info(f"Cropping image for ... with bbox: {crop_bbox}")
    xmin, ymin, xmax, ymax = crop_bbox
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("Invalid bounding box for cropping.")
    from app.services.image_utils import get_cropped_image
    image_to_embed, base64_cropped_image = get_cropped_image(
        pil_image_rgb, crop_bbox, shared_context
    )
    if image_to_embed.width == 0 or image_to_embed.height == 0:
        raise ValueError("Cropped image for embedding is empty.")
    actual_crop_bbox = crop_bbox
```

Also remove the now-unused `import io` and `import base64` from `embedding_service.py` if they're no longer needed. (They are only used in the crop blocks, so they can be removed after this refactor.)

**Step 3: Update tests to pass `shared_context=None` (backward compatible)**

The existing tests call embedding functions without `shared_context` — that's fine since the parameter defaults to `None`. Run:

```bash
python -m pytest tests/test_unit/test_services/test_embedding_service.py -x -q
```

Expected: all 4 tests pass.

---

### Task 3: Update main.py `_perform_analysis` to pass `shared_context`

**Objective:** Wire `shared_context` through all embedding calls in `_perform_analysis` so the crop cache is populated and reused.

**Files:**
- Modify: `main.py` (lines 189-191, 273-282)

**Step 1: CLIP embedding via `get_embedding_for_target`**

At line 190, change:
```python
embedding_list, b64_img, bbox_used = get_clip_embedding(pil_image_rgb, MODEL_NAME_CLIP, crop_box)
```
to:
```python
embedding_list, b64_img, bbox_used = get_clip_embedding(
    pil_image_rgb, MODEL_NAME_CLIP, crop_box, shared_context=shared_context
)
```

**Step 2: DINO embedding branches (lines 273-282)**

For each DINO variant, add `shared_context=shared_context`. Example for DINOv2:
```python
embedding_list, b64_img, bbox_used = get_dino_embedding(
    pil_image_rgb, crop_bbox=crop_box_for_dino, shared_context=shared_context
)
```

Same for `get_dino_v3_patch_embedding` and `get_dino_v3_embedding`.

**Step 3: description_service already uses `crop_image_and_get_base64`**

Add `shared_context` support there too for consistency:
```python
from app.services.image_utils import get_cropped_image
# Replace: image_to_process, b64_image = crop_image_and_get_base64(pil_image_rgb, crop_box)
image_to_process, b64_image = get_cropped_image(pil_image_rgb, crop_box, shared_context)
```

The `get_image_description` signature would need a `shared_context` param. Wire it from `_perform_analysis` at line 351.

**Step 4: Run integration tests**

```bash
python -m pytest tests/test_integration/ -x -q
```

Expected: integration tests pass.

---

### Task 4: Unify embedding cache keys between `_perform_analysis` and `classify_embedding_from_image`

**Objective:** Both code paths should use the same `shared_context` keys for embedding results so they share work. Currently `_perform_analysis` caches CLIP as `embedding_{target}` (line 154) while `classify_embedding_from_image` caches as `embedding_{emb_type}_{target}` (line 146). Standardize on `embedding_{emb_type}_{target}`.

**Files:**
- Modify: `main.py` (`get_embedding_for_target` inner function, lines 149-193)

**Step 1: Change cache key in `get_embedding_for_target`**

Current (lines 154-156):
```python
embedding_cache_key = f"embedding_{target}"
if "face" in target:
    embedding_cache_key += f"_{face_context}"
```

Change to:
```python
# Use the same key scheme as classify_embedding_from_image
embedding_cache_key = f"embedding_embed_clip_vit_b_32_{target}"
if "face" in target:
    embedding_cache_key += f"_{face_context}"
```

(The function is only used for CLIP embeddings, so `emb_type` is always `embed_clip_vit_b_32`.)

**Step 2: Remove the duplicate `get_embedding_for_target` — make `classify_embedding_from_image` read from the shared cache**

In `classify_embedding_from_image` (classification_service.py, lines 148-168), the embedding lookup already uses the correct key scheme `embedding_{emb_type}_{target}`. After Task 4 Step 1, both paths use the same keys. No further changes needed to classification_service.py.

**Step 3: Verify**

```bash
python -m pytest tests/ -x -q
```

Expected: all tests pass, no regressions.

---

### Task 5: Add unit test for crop cache behavior

**Objective:** Prove that `get_cropped_image` returns the cached result on the second call.

**Files:**
- Create: `tests/test_unit/test_services/test_image_utils.py`

**Step 1: Write the test**

```python
from PIL import Image
from app.services.image_utils import get_cropped_image

def test_get_cropped_image_caches_in_shared_context():
    """Second call with same bbox returns the cached result without re-cropping."""
    img = Image.new('RGB', (200, 200), color='red')
    bbox = [50, 50, 150, 150]
    ctx = {}

    # First call — should crop
    cropped1, b64_1 = get_cropped_image(img, bbox, shared_context=ctx)
    assert cropped1.size == (100, 100)
    assert isinstance(b64_1, str)
    cache_key = f"crop_{tuple(bbox)}"
    assert cache_key in ctx

    # Second call — should return cached instance
    cropped2, b64_2 = get_cropped_image(img, bbox, shared_context=ctx)
    assert cropped2 is cropped1  # same object (cached)
    assert b64_2 == b64_1

def test_get_cropped_image_without_shared_context():
    """Without shared_context, still returns valid crop (no caching)."""
    img = Image.new('RGB', (200, 200), color='blue')
    bbox = [0, 0, 100, 100]
    cropped, b64 = get_cropped_image(img, bbox)
    assert cropped.size == (100, 100)
    assert isinstance(b64, str)
```

**Step 2: Run the new tests**

```bash
python -m pytest tests/test_unit/test_services/test_image_utils.py -v
```

Expected: 2 tests pass.

**Step 3: Run full test suite**

```bash
python -m pytest tests/ -x -q
```

Expected: all tests pass.

---

### Task 6: Integration test — verify single crop per bbox per request

**Objective:** End-to-end test proving that two `classify` tasks on the same image/target only crop once.

**Files:**
- Create/modify: `tests/test_integration/test_api_endpoints.py`

**Step 1: Add test case**

```python
def test_single_crop_per_bbox_within_request(test_client, mock_model_loader):
    """Two classify tasks with same target should crop only once."""
    # Patch the crop function to track calls
    with patch('app.services.image_utils.crop_image_and_get_base64', wraps=crop_image_and_get_base64) as mock_crop:
        # ... setup request with 2 classify tasks, both prominent_person ...
        response = test_client.post("/analyze_image/", json={...})
        assert response.status_code == 200
        # crop_image_and_get_base64 should be called at most once per unique bbox
        # (detection produces one bbox, then both embedding calls hit the cache)
        assert mock_crop.call_count <= 1  # or 1 per unique bbox if face + person
```

**Note:** This test requires a more complete mock setup. If the integration test harness doesn't easily support this level of observation, a simpler approach: add a log-based assertion that "Reusing cached crop" appears in the logs, or add a unit test at the `_perform_analysis` level.

**Step 2: Run**

```bash
python -m pytest tests/test_integration/test_api_endpoints.py -v -k test_single_crop
```

Expected: test passes.

---

## Verification Checklist

After all tasks:

- [ ] `python -m pytest tests/ -v` — all tests pass
- [ ] Manual smoke test with 2 classify tasks on same image — log shows "Reusing cached crop" for second task
- [ ] `shared_context` crop cache is request-scoped (cleared per request, no cross-request pollution)
- [ ] `shared_context` embedding cache unifies keys between `_perform_analysis` and `classify_embedding_from_image`
- [ ] `import io`, `import base64` removed from `embedding_service.py` if unused

---

## Implementation Notes

- **No API changes.** All changes are internal — no endpoint signatures, request/response schemas, or `OperationResult` format changes.
- **Backward compatible.** All new parameters default to `None`, so existing callers of embedding functions continue to work.
- **Cache scope.** The `shared_context` dict is created fresh per request in `_perform_analysis` (line 143), so cross-request contamination is impossible.
- **Memory.** Each cropped PIL Image is kept in `shared_context` for the request lifetime. For typical images (<10MB), this is negligible. The `shared_context` dict is garbage-collected after `_perform_analysis` returns.
