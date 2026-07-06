from PIL import Image

def test_analyze_image_classification_on_face(client, mocker):
    """
    Tests a full pipeline: detect person -> detect face -> get embedding -> classify.
    Verifies that intermediate results are shared between tasks.
    """
    # 1. Arrange: Mock all external dependencies and service calls
    # We patch the functions in the namespace where they are *used* (main.py),
    # not where they are defined. This is a common requirement when using mock.
    mocker.patch("main.download_image", return_value=Image.new('RGB', (800, 600)))

    # Mock the internals used by the classify operation (which delegates into the service layer)
    mocker.patch(
        "app.core.model_loader.get_classifier_metadata",
        return_value={"embedding_type": "embed_clip_vit_b_32", "derivative_type": "prominent_face"},
    )
    mock_get_person = mocker.patch(
        "app.services.detection_service.get_prominent_person_bbox",
        return_value=[100, 50, 300, 250],
    )
    mock_get_face = mocker.patch(
        "app.services.detection_service.get_prominent_face_bbox_in_region",
        return_value=[110, 70, 160, 130],
    )
    mock_get_embedding = mocker.patch(
        "app.services.embedding_service.get_clip_embedding",
        return_value=([0.1] * 512, "base64_string", [110, 70, 160, 130]),
    )
    mock_classify = mocker.patch(
        "app.services.classification_service.classify_embedding",
        return_value={"is_in_collection": True, "probability": 0.98},
    )

    # The second task reads the cached result; ensure we don't re-run detection via main.
    mock_main_get_person = mocker.patch("main.get_prominent_person_bbox")

    # 2. Arrange: Define a complex request with multiple dependent tasks
    request_data = {
        "image_url": "http://example.com/image.jpg",
        "tasks": [
            {
                "operation_id": "classify_face",
                "type": "classify",
                "params": {
                    "target": "prominent_face",
                    "collection_id": 42
                }
            },
            {
                "operation_id": "find_person",
                "type": "detect_bounding_box",
                "params": {
                    "target": "prominent_person"
                }
            }
        ]
    }

    # 3. Act
    response = client.post("/analyze_image/", json=request_data)

    # 4. Assert
    assert response.status_code == 200
    results = response.json()["results"]

    # Check classification result
    assert results["classify_face"]["status"] == "success"
    assert results["classify_face"]["data"]["is_in_collection"] is True
    assert results["classify_face"]["cropped_image_bbox"] == [110, 70, 160, 130]

    # Check person detection result
    assert results["find_person"]["status"] == "success"
    assert results["find_person"]["data"] == [100, 50, 300, 250]

    # IMPORTANT: Assert that expensive operations were only called once
    # This proves the internal caching (`shared_context`) is working.
    mock_get_person.assert_called_once()
    mock_get_face.assert_called_once()
    mock_get_embedding.assert_called_once()
    mock_classify.assert_called_once()
    mock_main_get_person.assert_not_called()


def test_analyze_image_describe_image(client, mocker):
    """
    Tests the 'describe_image' operation through the API.
    """
    # 1. Arrange
    mocker.patch("main.download_image", return_value=Image.new('RGB', (800, 600)))
    mock_get_description = mocker.patch(
        "main.get_image_description", 
        return_value=("A detailed description of the image.", None, None)
    )

    request_data = {
        "image_url": "http://example.com/image.jpg",
        "tasks": [
            {
                "operation_id": "describe_whole_image",
                "type": "describe_image",
                "params": {
                    "target": "whole_image"
                }
            }
        ]
    }

    # 3. Act
    response = client.post("/analyze_image/", json=request_data)

    # 4. Assert
    assert response.status_code == 200
    results = response.json()["results"]

    assert "describe_whole_image" in results
    assert results["describe_whole_image"]["status"] == "success"
    assert results["describe_whole_image"]["data"] == "A detailed description of the image."
    assert results["describe_whole_image"]["cropped_image_base64"] is None
    assert results["describe_whole_image"]["cropped_image_bbox"] is None

    mock_get_description.assert_called_once()


def test_crop_cache_across_classify_tasks(client, mocker):
    """
    Two classify tasks targeting the same person bbox should only
    invoke crop_image_and_get_base64 once — the second hit reuses
    the cache in shared_context.
    """
    from app.services.image_utils import crop_image_and_get_base64

    # 1. Arrange
    mocker.patch("main.download_image", return_value=Image.new('RGB', (800, 600)))

    # Classifier metadata: two collections, different embedding types, same target
    metadata_map = {
        10: {"embedding_type": "embed_dino_v2", "derivative_type": "prominent_person"},
        20: {"embedding_type": "embed_clip_vit_b_32", "derivative_type": "prominent_person"},
    }

    def _get_meta(cid):
        return metadata_map[cid]

    mocker.patch(
        "app.core.model_loader.get_classifier_metadata",
        side_effect=_get_meta,
    )

    # Detection produces the same bbox for both
    mocker.patch(
        "app.services.detection_service.get_prominent_person_bbox",
        return_value=[100, 50, 300, 250],
    )

    # Mock embedding generation — return dummy embeddings
    mocker.patch(
        "app.services.embedding_service.get_dino_embedding",
        return_value=([0.1] * 768, "dino_b64", [100, 50, 300, 250]),
    )
    mocker.patch(
        "app.services.embedding_service.get_clip_embedding",
        return_value=([0.2] * 512, "clip_b64", [100, 50, 300, 250]),
    )

    # Mock classification
    mocker.patch(
        "app.services.classification_service.classify_embedding",
        return_value={"is_in_collection": False, "probability": 0.5},
    )

    # Spy on the source-of-truth crop function
    mock_crop_base = mocker.patch(
        "app.services.image_utils.crop_image_and_get_base64",
        wraps=crop_image_and_get_base64,
    )

    request_data = {
        "image_url": "http://example.com/img.jpg",
        "tasks": [
            {
                "operation_id": "classify_a",
                "type": "classify",
                "params": {"target": "prominent_person", "collection_id": 10},
            },
            {
                "operation_id": "classify_b",
                "type": "classify",
                "params": {"target": "prominent_person", "collection_id": 20},
            },
        ],
    }

    # 2. Act
    response = client.post("/analyze_image/", json=request_data)

    # 3. Assert
    assert response.status_code == 200

    # The crop should only happen once because:
    #  - classify_a (dino_v2) crops → cache hit set
    #  - classify_b (clip_vit_b_32) reuses cached crop
    assert mock_crop_base.call_count <= 1, (
        f"Expected crop_image_and_get_base64 to be called ≤ 1 time, "
        f"but was called {mock_crop_base.call_count} times"
    )


# ──────────────────────────────────────────────────────────────────────
# AuraFace identity embedding integration tests
# ──────────────────────────────────────────────────────────────────────


def test_analyze_image_auraface_on_prominent_face(client, mocker):
    """Full pipeline: face detection → auraface embedding on prominent_face."""
    mocker.patch("main.download_image", return_value=Image.new("RGB", (800, 600)))

    mocker.patch(
        "main.get_prominent_person_bbox",
        return_value=[100, 50, 300, 250],
    )
    mocker.patch(
        "main.get_prominent_face_bbox_in_region",
        return_value=[110, 70, 160, 130],
    )
    mock_get_auraface = mocker.patch(
        "app.services.embedding_service.get_auraface_embedding",
        return_value=([0.15] * 512, "face_b64", [110, 70, 160, 130]),
    )

    request_data = {
        "image_url": "http://example.com/image.jpg",
        "tasks": [
            {
                "operation_id": "face_id",
                "type": "embed_auraface",
                "params": {"target": "prominent_face"},
            }
        ],
    }

    response = client.post("/analyze_image/", json=request_data)

    assert response.status_code == 200
    results = response.json()["results"]
    assert results["face_id"]["status"] == "success"
    assert len(results["face_id"]["data"]) == 512
    assert results["face_id"]["cropped_image_base64"] == "face_b64"
    assert results["face_id"]["cropped_image_bbox"] == [110, 70, 160, 130]

    mock_get_auraface.assert_called_once()


def test_analyze_image_auraface_skipped_when_no_face(client, mocker):
    """Returns skipped when face detection finds no face."""
    mocker.patch("main.download_image", return_value=Image.new("RGB", (800, 600)))

    mocker.patch(
        "main.get_prominent_person_bbox",
        return_value=[100, 50, 300, 250],
    )
    mocker.patch(
        "main.get_prominent_face_bbox_in_region",
        return_value=None,  # No face found
    )

    request_data = {
        "image_url": "http://example.com/image.jpg",
        "tasks": [
            {
                "operation_id": "no_face",
                "type": "embed_auraface",
                "params": {"target": "prominent_face"},
            }
        ],
    }

    response = client.post("/analyze_image/", json=request_data)

    assert response.status_code == 200
    results = response.json()["results"]
    assert results["no_face"]["status"] == "skipped"
    assert "No prominent face found" in results["no_face"]["error_message"]


def test_analyze_image_auraface_rejects_whole_image_target(client, mocker):
    """Returns skipped when target is not prominent_face."""
    mocker.patch("main.download_image", return_value=Image.new("RGB", (800, 600)))

    request_data = {
        "image_url": "http://example.com/image.jpg",
        "tasks": [
            {
                "operation_id": "bad_target",
                "type": "embed_auraface",
                "params": {"target": "whole_image"},
            }
        ],
    }

    response = client.post("/analyze_image/", json=request_data)

    assert response.status_code == 200
    results = response.json()["results"]
    assert results["bad_target"]["status"] == "skipped"
    assert "Unsupported target" in results["bad_target"]["error_message"]
