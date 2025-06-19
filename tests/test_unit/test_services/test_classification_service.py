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
