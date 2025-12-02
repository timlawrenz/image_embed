# Implementation Tasks: Multi-Embedding Classifier Support

## Phase 1: Crawlr API Enhancement

### Task 1.1: Add CollectionFocus Helper Methods
**File:** `crawlr/app/models/collection_focus.rb`

- [ ] Add `derivative_type_name` method that returns `derivative_type&.name`
- [ ] Add `embedding_type_name` method that returns `embedding_type&.name`
- [ ] Add tests for these methods in `spec/models/collection_focus_spec.rb`

**Acceptance:**
- Methods return correct string values when associations exist
- Methods return nil when associations are missing
- Tests pass

### Task 1.2: Update Collections API Endpoint
**File:** `crawlr/app/controllers/collections_controller.rb`

- [ ] Modify `index` action JSON format to include collection_focus
- [ ] Include `derivative_type_name` and `embedding_type_name` methods
- [ ] Update/add API tests in `spec/requests/collections_request_spec.rb`

**Acceptance:**
- `/collections.json` returns focus metadata
- Response includes `collection_focus` with `derivative_type_name` and `embedding_type_name`
- Existing clients not broken (only adds data, doesn't remove)

### Task 1.3: Deploy Crawlr Changes
- [ ] Create PR for crawlr changes
- [ ] Get code review
- [ ] Deploy to production
- [ ] Verify endpoint manually: `curl https://crawlr.lawrenz.com/collections.json`

## Phase 2: Image_Embed Training Enhancement

### Task 2.1: Update Training Script to Fetch Metadata
**File:** `image_embed/scripts/train_classifiers.py`

- [ ] Modify `fetch_collections()` to extract collection_focus from response
- [ ] Add validation for derivative_type and embedding_type presence
- [ ] Log warnings for collections missing focus metadata
- [ ] Update `train_and_save_model()` signature to accept metadata parameters

**Acceptance:**
- Script fetches collections with focus metadata
- Skips collections without focus (with warning log)
- Passes metadata to training function

### Task 2.2: Store Metadata in best_models.json
**File:** `image_embed/scripts/train_classifiers.py`

- [ ] Change `best_models_map` structure from `{id: filename}` to `{id: {metadata}}`
- [ ] Include: `model_file`, `derivative_type`, `embedding_type`, `dimensionality`
- [ ] Infer dimensionality from training data shape
- [ ] Update JSON writing logic in `main()` function

**Acceptance:**
- `best_models.json` has new nested structure
- All required metadata fields present for each collection
- Dimensionality matches embedding vector length

### Task 2.3: Run Training and Verify Output
- [ ] Run: `python scripts/train_classifiers.py`
- [ ] Inspect `trained_classifiers/best_models.json` structure
- [ ] Verify metadata for at least one CLIP and one DINO collection
- [ ] Commit updated `best_models.json` to git

**Acceptance:**
- Training completes successfully
- Metadata format matches spec
- File committed to repository

## Phase 3: Image_Embed Runtime Enhancement

### Task 3.1: Add Metadata Loading in model_loader
**File:** `image_embed/app/core/model_loader.py`

- [ ] Add `get_classifier_metadata(collection_id)` function
- [ ] Load and parse `best_models.json`
- [ ] Return metadata dict for collection_id
- [ ] Raise FileNotFoundError if collection not found
- [ ] Add backward compatibility for old string format (optional)

**Acceptance:**
- Function loads metadata correctly
- Raises appropriate exceptions
- Tests pass (unit tests for this function)

### Task 3.2: Create Enhanced Classification Service
**File:** `image_embed/app/services/classification_service.py`

- [ ] Add `classify_embedding_from_image()` function
- [ ] Fetch metadata using model_loader
- [ ] Map derivative_type to target (whole_image/person/face)
- [ ] Handle detection based on target
- [ ] Generate correct embedding type (CLIP or DINO)
- [ ] Cache embeddings in shared_context
- [ ] Call existing `classify_embedding()` function

**Acceptance:**
- Function generates CLIP embeddings for CLIP metadata
- Function generates DINO embeddings for DINO metadata
- Detection logic works correctly for all targets
- Embeddings cached to avoid redundant computation
- Tests pass

### Task 3.3: Update Main API Endpoint
**File:** `image_embed/main.py`

- [ ] Update `classify` operation in `_perform_analysis()`
- [ ] Replace direct `get_embedding_for_target()` call
- [ ] Call new `classify_embedding_from_image()` instead
- [ ] Remove old target/face_context handling from classify block
- [ ] Update error handling

**Acceptance:**
- Classification uses new function
- Timing stats still captured correctly
- Error messages remain helpful
- No breaking changes to API response format

### Task 3.4: Add Tests
**File:** `image_embed/tests/test_unit/test_classification_service.py`

- [ ] Test `classify_embedding_from_image()` with CLIP metadata
- [ ] Test `classify_embedding_from_image()` with DINO metadata
- [ ] Test with whole_image target
- [ ] Test with prominent_person target
- [ ] Test with prominent_face target
- [ ] Test error when face not found for face target
- [ ] Test embedding caching behavior

**File:** `image_embed/tests/test_integration/test_classify_operation.py`

- [ ] Test end-to-end classify with CLIP-focused collection
- [ ] Test end-to-end classify with DINO-focused collection
- [ ] Test multiple classify operations share embeddings
- [ ] Verify timing stats are correct

**Acceptance:**
- All tests pass
- Coverage includes happy path and error cases

### Task 3.5: Update Documentation
**File:** `image_embed/README.md`

- [ ] Update `classify` operation description
- [ ] Note that embedding type is automatically determined
- [ ] Update example to show classification works with any embedding type
- [ ] Document metadata requirement

**File:** `image_embed/openspec/project.md`

- [ ] Update classifier description to mention multi-embedding support
- [ ] Document metadata storage in best_models.json

**Acceptance:**
- Documentation is clear and accurate
- Users understand automatic embedding type selection

## Phase 4: Deployment and Validation

### Task 4.1: Deploy Image_Embed
- [ ] Create PR for image_embed changes
- [ ] Get code review
- [ ] Run tests in CI
- [ ] Deploy to production
- [ ] Monitor logs for errors

**Acceptance:**
- Deployment successful
- No errors in logs
- Service responds to health checks

### Task 4.2: Manual Testing
- [ ] Identify a CLIP-focused collection in crawlr
- [ ] Send classify request, verify probability is reasonable
- [ ] Identify a DINO-focused collection in crawlr
- [ ] Send classify request, verify probability is reasonable (should differ from before)
- [ ] Check logs for correct embedding type usage

**Acceptance:**
- CLIP collections work as before
- DINO collections show different (hopefully better) probabilities
- Logs confirm correct embedding types used

### Task 4.3: Monitor and Iterate
- [ ] Monitor classification request logs for 24 hours
- [ ] Check for any errors or unexpected behavior
- [ ] Review classification probabilities in crawlr
- [ ] Gather feedback from collection moderation

**Acceptance:**
- No critical errors
- Classification probabilities are meaningful
- System performs as expected

## Phase 5: Archive Proposal
- [ ] Move proposal to `openspec/changes/archive/multi-embedding-classifier/`
- [ ] Document completion date and final notes
- [ ] Update project.md if needed

## Notes

### Dependencies
- Phase 2 depends on Phase 1 (needs API changes deployed)
- Phase 3 depends on Phase 2 (needs best_models.json with metadata)
- Phase 4 depends on Phase 3 (needs code changes)

### Rollback Plan
If issues arise:
1. Revert main.py classify operation to use old logic
2. Keep metadata in best_models.json for future use
3. Investigate and fix issues
4. Redeploy when ready

### Estimated Timeline
- Phase 1: 1-2 hours
- Phase 2: 2-3 hours
- Phase 3: 4-6 hours (includes testing)
- Phase 4: 1-2 hours
- Total: ~8-13 hours of development time
