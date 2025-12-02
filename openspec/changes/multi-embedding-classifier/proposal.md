# Multi-Embedding Classifier Support

**Change ID:** `multi-embedding-classifier`  
**Status:** Proposed  
**Created:** 2024-12-02  
**Author:** AI Assistant with Tim Lawrenz

## Problem

Currently, the `classify` operation in image_embed always uses CLIP embeddings, even when classifiers were trained on DINO embeddings. This causes incorrect predictions for collections focused on DINO features.

### Current Behavior

1. **Training:** `train_classifiers.py` downloads training data from crawlr that is already filtered by collection_focus (CLIP or DINO)
2. **Classifier storage:** No metadata about embedding type is saved with the classifier
3. **Runtime classification:** The `classify` operation hardcodes `get_clip_embedding()` regardless of what the classifier was trained on
4. **Result:** DINO-trained classifiers receive CLIP embeddings → meaningless predictions

### Why This Matters

In crawlr, collections can focus on different embedding types:
- CLIP whole_image for semantic similarity
- DINO prominent_face for visual features like composition/color
- CLIP prominent_person for semantic person matching
- DINO whole_image for visual texture similarity

Each collection has exactly ONE collection_focus that specifies both:
- `derivative_type` (whole_image, prominent_person, prominent_face)
- `embedding_type` (embed_clip_vit_b_32, embed_dino_v2)

## Proposed Solution

Store embedding type metadata with classifiers and use it to generate the correct embedding type during classification.

### High-Level Approach

1. **During training:** Fetch collection_focus metadata from crawlr and save it alongside classifier
2. **During classification:** Load metadata and generate appropriate embedding type (CLIP or DINO)
3. **No API changes needed:** Classification behavior becomes automatically correct

### Why This Design

- ✅ **One source of truth:** Collection's focus defined in crawlr only
- ✅ **Zero runtime overhead:** No additional API calls during classification
- ✅ **Fail-safe:** Metadata stored with classifier ensures they stay in sync
- ✅ **Clean API:** No need to expose embedding_type as a parameter (which would allow invalid requests)

## Implementation Plan

### Part 1: Enhance Crawlr API

**Status:** The data already exists in crawlr models! We just need to expose it via the API.

**File:** `crawlr/app/controllers/collections_controller.rb`

The `/collections.json` endpoint currently returns:
```ruby
format.json { render json: @collections.as_json(only: %i[id name]) }
```

Update it to include the existing collection_focus associations:

```ruby
format.json { 
  render json: @collections.as_json(
    only: [:id, :name],
    include: {
      collection_focus: {
        include: {
          derivative_type: { only: [:name] },
          embedding_type: { only: [:name] }
        }
      }
    }
  )
}
```

**Expected JSON output:**
```json
[
  {
    "id": 42,
    "name": "Portraits",
    "collection_focus": {
      "id": 123,
      "derivative_type": {
        "name": "prominent_face"
      },
      "embedding_type": {
        "name": "embed_dino_v2"
      }
    }
  }
]
```

**Note:** No model changes needed - all associations already exist!

### Part 2: Store Metadata During Training

**File:** `image_embed/scripts/train_classifiers.py`

**Changes:**

1. Extract focus metadata when fetching collections:
```python
def fetch_collections():
    # ... existing code ...
    collections = json.loads(cleaned_text)
    
    # Extract and validate collection_focus
    for collection in collections:
        focus = collection.get('collection_focus')
        if focus:
            collection['derivative_type'] = focus.get('derivative_type_name')
            collection['embedding_type'] = focus.get('embedding_type_name')
        else:
            logger.warning(f"Collection {collection.get('id')} has no focus, will skip.")
    
    return collections
```

2. Pass metadata to training function:
```python
def train_and_save_model(collection_id, collection_name, training_data, derivative_type, embedding_type):
    # ... existing training logic ...
    
    # Return metadata along with best model file
    return {
        'model_file': best_model_filename,
        'derivative_type': derivative_type,
        'embedding_type': embedding_type,
        'dimensionality': len(X[0])  # Inferred from training data
    }
```

3. Enhance `best_models.json` structure:
```python
# Old format:
# {
#   "42": "collection_42_classifier_2024-12-02_135230.pkl"
# }

# New format:
# {
#   "42": {
#     "model_file": "collection_42_classifier_2024-12-02_135230.pkl",
#     "derivative_type": "prominent_face",
#     "embedding_type": "embed_dino_v2",
#     "dimensionality": 768
#   }
# }
```

4. Skip collections without focus:
```python
for collection in collections:
    # ... existing code ...
    
    derivative_type = collection.get('derivative_type')
    embedding_type = collection.get('embedding_type')
    
    if not derivative_type or not embedding_type:
        logger.warning(f"Skipping collection '{collection_name}' (ID: {collection_id}) - missing focus.")
        continue
    
    # ... proceed with training ...
```

### Part 3: Use Correct Embedding During Classification

**File:** `image_embed/app/core/model_loader.py`

Add method to load classifier metadata:

```python
def get_classifier_metadata(collection_id: int) -> dict:
    """
    Retrieves metadata for a classifier.
    
    Returns dict with keys: model_file, derivative_type, embedding_type, dimensionality
    Raises FileNotFoundError if no metadata exists for collection_id.
    """
    config_path = os.path.join(CLASSIFIER_DIR, 'best_models.json')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Classifier metadata file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        metadata = json.load(f)
    
    collection_key = str(collection_id)
    if collection_key not in metadata:
        raise FileNotFoundError(f"No classifier found for collection_id {collection_id}")
    
    return metadata[collection_key]
```

**File:** `image_embed/app/services/classification_service.py`

Update to fetch correct embedding:

```python
def classify_embedding_from_image(
    pil_image: Image.Image, 
    collection_id: int,
    shared_context: Dict[str, Any],
    timing_stats: Dict[str, float]
) -> Dict[str, Any]:
    """
    Classifies an image for a collection by generating the correct embedding type.
    
    Args:
        pil_image: PIL Image in RGB format
        collection_id: ID of the collection classifier
        shared_context: Shared detection/embedding cache
        timing_stats: Performance timing dictionary
    
    Returns:
        Dict with keys: is_in_collection (bool), probability (float)
    """
    from app.core import model_loader
    from app.services.detection_service import get_prominent_person_bbox, get_prominent_face_bbox_in_region
    from app.services.embedding_service import get_clip_embedding, get_dino_embedding
    
    # Get classifier metadata
    metadata = model_loader.get_classifier_metadata(collection_id)
    embedding_type = metadata['embedding_type']
    derivative_type = metadata['derivative_type']
    
    # Map derivative_type to target
    target_map = {
        'whole_image': 'whole_image',
        'prominent_person': 'prominent_person',
        'prominent_face': 'prominent_face'
    }
    target = target_map.get(derivative_type)
    
    if not target:
        raise ValueError(f"Invalid derivative_type '{derivative_type}' for collection {collection_id}")
    
    # Determine crop_box based on target
    crop_box = None
    if target == 'prominent_person':
        if 'prominent_person_bbox' not in shared_context:
            detection_start = time.time()
            shared_context['prominent_person_bbox'] = get_prominent_person_bbox(pil_image)
            timing_stats['detection'] += time.time() - detection_start
        crop_box = shared_context.get('prominent_person_bbox')
    
    elif target == 'prominent_face':
        if 'prominent_face_bbox' not in shared_context:
            detection_start = time.time()
            if 'prominent_person_bbox' not in shared_context:
                shared_context['prominent_person_bbox'] = get_prominent_person_bbox(pil_image)
            person_bbox = shared_context.get('prominent_person_bbox')
            shared_context['prominent_face_bbox'] = get_prominent_face_bbox_in_region(pil_image, person_bbox)
            timing_stats['detection'] += time.time() - detection_start
        crop_box = shared_context.get('prominent_face_bbox')
        
        if not crop_box:
            raise ValueError(f"No face detected for collection {collection_id} with face target")
    
    # Generate correct embedding type
    embedding_cache_key = f"embedding_{embedding_type}_{target}"
    
    if embedding_cache_key not in shared_context:
        embedding_start = time.time()
        
        if embedding_type == 'embed_clip_vit_b_32':
            from main import MODEL_NAME_CLIP
            embedding_list, _, _ = get_clip_embedding(pil_image, MODEL_NAME_CLIP, crop_box)
        elif embedding_type == 'embed_dino_v2':
            embedding_list, _, _ = get_dino_embedding(pil_image, crop_box)
        else:
            raise ValueError(f"Unsupported embedding_type '{embedding_type}' for collection {collection_id}")
        
        timing_stats['embedding'] += time.time() - embedding_start
        shared_context[embedding_cache_key] = embedding_list
    else:
        embedding_list = shared_context[embedding_cache_key]
    
    # Classify using the existing function
    return classify_embedding(embedding_list, collection_id)
```

**File:** `image_embed/main.py`

Update the `classify` operation to use new service function:

```python
elif op_type == "classify":
    collection_id = op_params.get("collection_id")
    if collection_id is None:
        raise ValueError("'collection_id' param is required for 'classify' operation.")
    
    classification_start = time.time()
    try:
        from app.services.classification_service import classify_embedding_from_image
        current_result_data = classify_embedding_from_image(
            pil_image_rgb, 
            int(collection_id),
            shared_context,
            timing_stats
        )
        # No cropped image data for classification
        current_cropped_image_base64 = None
        current_cropped_image_bbox = None
    except FileNotFoundError as e:
        raise ValueError(f"Classifier not found for collection_id {collection_id}.") from e
    timing_stats["classification"] += time.time() - classification_start
```

## Migration Path

### Step 1: Update Crawlr (crawlr repo)
1. Add helper methods to `CollectionFocus` model
2. Update `CollectionsController#index` to include focus metadata
3. Deploy to production
4. Verify `/collections.json` returns expected structure

### Step 2: Update Image_Embed Training (image_embed repo)
1. Update `train_classifiers.py` to fetch and store metadata
2. Run training script: `python scripts/train_classifiers.py`
3. Verify `trained_classifiers/best_models.json` has new structure
4. Commit and push updated `best_models.json`

### Step 3: Update Image_Embed Runtime (image_embed repo)
1. Add `get_classifier_metadata()` to `model_loader.py`
2. Add `classify_embedding_from_image()` to `classification_service.py`
3. Update `main.py` classify operation to use new function
4. Deploy to production

### Step 4: Validate
1. Test CLIP-focused collection classification
2. Test DINO-focused collection classification
3. Monitor logs for correct embedding type usage
4. Verify classification probabilities are meaningful

## Testing Strategy

### Unit Tests
- Test `get_classifier_metadata()` with valid/invalid collection IDs
- Test `classify_embedding_from_image()` with CLIP metadata
- Test `classify_embedding_from_image()` with DINO metadata
- Test error handling for missing face when face target required

### Integration Tests
- End-to-end classification with CLIP-focused collection
- End-to-end classification with DINO-focused collection
- Verify embedding caching works correctly across multiple classify operations
- Verify timing stats are accurate

### Manual Testing
- Run training script and inspect `best_models.json`
- Send classify requests to API for known collections
- Compare probabilities before/after fix for DINO collections

## Risks & Mitigations

### Risk: Breaking existing classifiers
**Mitigation:** Keep backward compatibility in model_loader. If `best_models.json` is a string (old format), default to CLIP.

### Risk: Collections without focus
**Mitigation:** Training script skips collections without focus metadata. Log warnings clearly.

### Risk: Migration timing
**Mitigation:** Can deploy code before retraining. Old `best_models.json` will fail gracefully with clear error messages.

### Risk: Dimensionality mismatch
**Mitigation:** Store dimensionality in metadata and validate against classifier expected dimensions.

## Success Criteria

1. ✅ Training script successfully stores embedding_type metadata for all collections
2. ✅ Classification uses correct embedding type (CLIP or DINO) based on metadata
3. ✅ DINO-focused collections receive meaningful probability scores
4. ✅ No performance regression (embedding caching works correctly)
5. ✅ Clear error messages when classifiers or metadata are missing

## Future Enhancements

- Support for additional embedding types (e.g., future models)
- Versioning of metadata format for easier migrations
- API endpoint to query classifier metadata
- Metrics/monitoring for classifier accuracy by embedding type
