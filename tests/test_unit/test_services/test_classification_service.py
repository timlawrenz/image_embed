import pytest
import numpy as np
from unittest.mock import MagicMock
from app.services.classification_service import classify_embedding

def test_classify_embedding_success(mocker):
    """
    Tests successful classification when the model predicts 'True'.
    """
    # 1. Arrange: Create a mock model that simulates a scikit-learn classifier.
    mock_model = MagicMock()
    mock_model.classes_ = [False, True]
    # predict() returns a numpy array of predictions
    mock_model.predict.return_value = np.array([True])
    # predict_proba() returns probabilities for each class: [[prob_false, prob_true]]
    mock_model.predict_proba.return_value = np.array([[0.05, 0.95]])

    # Mock the model loader to return our fake model
    mocker.patch("app.core.model_loader.get_classifier_model", return_value=mock_model)

    # 2. Act: Call the function under test
    result = classify_embedding(embedding=[0.1] * 512, collection_id=123)

    # 3. Assert: Check the output is as expected
    assert result["is_in_collection"] is True
    assert result["probability"] == 0.95
    # Verify the model loader was called with the correct collection_id
    mock_model.predict.assert_called_once()


def test_classify_embedding_model_not_found(mocker):
    """
    Tests that classify_embedding correctly re-raises FileNotFoundError.
    """
    # 1. Arrange: Mock the model loader to raise the expected exception
    mocker.patch(
        "app.core.model_loader.get_classifier_model",
        side_effect=FileNotFoundError("Model not found")
    )

    # 2. Act & Assert: Use pytest.raises to confirm the exception is propagated
    with pytest.raises(FileNotFoundError):
        classify_embedding(embedding=[0.1] * 512, collection_id=999)


def test_classify_embedding_from_image_supports_auraface(mocker):
    """
    A classifier trained on embed_auraface (prominent_face) must generate the
    AuraFace embedding and classify it — not raise "Unsupported embedding_type".
    """
    from PIL import Image
    from app.services.classification_service import classify_embedding_from_image

    mocker.patch(
        "app.core.model_loader.get_classifier_metadata",
        return_value={"embedding_type": "embed_auraface", "derivative_type": "prominent_face"},
    )
    mocker.patch(
        "app.services.detection_service.get_prominent_person_bbox",
        return_value=[0, 0, 100, 100],
    )
    mocker.patch(
        "app.services.detection_service.get_prominent_face_bbox_in_region",
        return_value=[10, 10, 90, 90],
    )
    mock_auraface = mocker.patch(
        "app.services.embedding_service.get_auraface_embedding",
        return_value=([0.1] * 512, None, [10, 10, 90, 90]),
    )
    mock_classify = mocker.patch(
        "app.services.classification_service.classify_embedding",
        return_value={"is_in_collection": True, "probability": 0.9},
    )

    pil_image = Image.new("RGB", (200, 200), "white")
    timing_stats = {"detection": 0.0, "embedding": 0.0, "classification": 0.0, "description": 0.0}

    result = classify_embedding_from_image(pil_image, collection_id=20, shared_context={}, timing_stats=timing_stats)

    mock_auraface.assert_called_once()
    assert result == {"is_in_collection": True, "probability": 0.9}
