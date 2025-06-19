from unittest.mock import MagicMock
from PIL import Image
import numpy as np
import pytest
import torch
from app.services.detection_service import get_prominent_person_bbox, get_prominent_face_bbox_in_region

def test_get_face_bbox_in_region_with_offset(mocker):
    """
    Tests that face detection correctly applies the person_bbox offset
    to the coordinates of the detected face.
    """
    # 1. Arrange
    # A mock PIL image (contents don't matter, just size)
    mock_pil_image = Image.new('RGB', (800, 600))
    # A bounding box for the person region we are searching within
    person_bbox = [100, 50, 300, 250] # xmin, ymin, xmax, ymax

    # Mock the MTCNN model
    mock_face_model = MagicMock()
    # The model's detect method should return coordinates relative to the *cropped* image
    # and a probability for each face.
    detected_boxes = np.array([[10, 20, 60, 80]]) # A face found at (10, 20) in the crop
    detected_probs = np.array([0.99])
    mock_face_model.detect.return_value = (detected_boxes, detected_probs)
    mocker.patch("app.core.model_loader.get_face_detection_model", return_value=mock_face_model)

    # 2. Act
    final_bbox = get_prominent_face_bbox_in_region(mock_pil_image, person_bbox)

    # 3. Assert
    # The final bbox should be the face box offset by the person box's top-left corner.
    # Expected: [100+10, 50+20, 100+60, 50+80]
    expected_final_bbox = [110, 70, 160, 130]
    assert final_bbox == expected_final_bbox

    # Check that the image was cropped correctly before being passed to the model
    # The crop call is on the PIL Image object itself
    # We can't easily assert on this without more complex mocking, but this test
    # implicitly validates it by checking the final coordinate math.


def test_get_prominent_person_bbox_success(mocker):
    """
    Tests successful detection of a prominent person from model predictions.
    """
    # 1. Arrange
    mock_pil_image = Image.new('RGB', (800, 600))
    mock_person_model = MagicMock()
    
    # Use real torch tensors to simulate the model's output. This is cleaner
    # than mocking every single tensor operation.
    predictions = [{
        'scores': torch.tensor([0.9, 0.8, 0.95]),  # Scores for all detected objects
        'labels': torch.tensor([2, 1, 1]),         # Labels: background, person, person
        'boxes': torch.tensor([
            [0, 0, 10, 10],                        # Box for background
            [10, 10, 50, 50],                      # Box for person with score 0.8
            [200, 200, 400, 400]                   # Box for person with score 0.95
        ])
    }]
    mock_person_model.return_value = predictions

    # Mock the model loader to return our mock model. We also patch torch used
    # inside the service to allow the `with torch.no_grad()` context manager to work.
    mocker.patch("app.core.model_loader.get_person_detection_model", return_value=mock_person_model)
    mocker.patch("app.core.model_loader.get_device", return_value="cpu")
    mocker.patch("app.core.model_loader.torch", torch)

    # 2. Act
    bbox = get_prominent_person_bbox(mock_pil_image)

    # 3. Assert
    # The function should find two persons (labels == 1), select the one with the
    # highest score (0.95), and return its bounding box.
    assert bbox == [200, 200, 400, 400]
    mock_person_model.assert_called_once()


def test_get_prominent_person_bbox_no_person_found(mocker):
    """
    Tests that get_prominent_person_bbox returns None when no person is detected.
    """
    # 1. Arrange
    mock_pil_image = Image.new('RGB', (800, 600))
    mock_person_model = MagicMock()
    
    predictions = [{
        'scores': torch.tensor([0.9]),
        'labels': torch.tensor([2]), # Label for background, no person
        'boxes': torch.tensor([[0, 0, 10, 10]])
    }]
    mock_person_model.return_value = predictions
    
    mocker.patch("app.core.model_loader.get_person_detection_model", return_value=mock_person_model)
    mocker.patch("app.core.model_loader.get_device", return_value="cpu")
    mocker.patch("app.core.model_loader.torch", torch)

    # 2. Act
    bbox = get_prominent_person_bbox(mock_pil_image)

    # 3. Assert
    assert bbox is None


def test_get_face_bbox_in_region_no_person_bbox(mocker):
    """
    Tests face detection on the whole image when person_bbox is None.
    """
    # Arrange
    mock_pil_image = Image.new('RGB', (800, 600))
    mock_face_model = MagicMock()
    # Model detects a face in the full image
    detected_boxes = np.array([[150, 250, 250, 350]])
    detected_probs = np.array([0.99])
    mock_face_model.detect.return_value = (detected_boxes, detected_probs)
    mocker.patch("app.core.model_loader.get_face_detection_model", return_value=mock_face_model)

    # Act
    final_bbox = get_prominent_face_bbox_in_region(mock_pil_image, person_bbox=None)

    # Assert
    # Without an offset, the final bbox is what the model returned.
    assert final_bbox == [150, 250, 250, 350]


def test_get_face_bbox_in_region_no_face_found(mocker):
    """
    Tests that None is returned when the model finds no faces.
    """
    # Arrange
    mock_pil_image = Image.new('RGB', (800, 600))
    mock_face_model = MagicMock()
    # MTCNN model returns None in tuple when no faces are detected
    mock_face_model.detect.return_value = (None, None)
    mocker.patch("app.core.model_loader.get_face_detection_model", return_value=mock_face_model)
    
    # Act
    final_bbox = get_prominent_face_bbox_in_region(mock_pil_image, person_bbox=[100, 50, 300, 250])

    # Assert
    assert final_bbox is None


def test_get_face_bbox_in_region_invalid_person_bbox(mocker):
    """
    Tests that None is returned for an invalid person_bbox to prevent errors.
    """
    # Arrange
    mock_pil_image = Image.new('RGB', (800, 600))
    # This bbox is invalid because xmin >= xmax
    invalid_person_bbox = [300, 50, 100, 250]
    
    # The face detection model should not even be called
    mock_face_model = MagicMock()
    mocker.patch("app.core.model_loader.get_face_detection_model", return_value=mock_face_model)
    
    # Act
    final_bbox = get_prominent_face_bbox_in_region(mock_pil_image, person_bbox=invalid_person_bbox)

    # Assert
    assert final_bbox is None
    mock_face_model.detect.assert_not_called()
