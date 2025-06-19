from unittest.mock import MagicMock
from PIL import Image
import numpy as np
import pytest
from app.services.detection_service import get_prominent_face_bbox_in_region

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
    #

Of course. I will now create the testing files and structure we discussed.

First, I'll create a new file for development-specific dependencies.

```
requirements-dev.txt
<<<<<<< SEARCH
import pytest
from unittest.mock import MagicMock
from PIL import Image
import numpy as np
from app.services.detection_service import get_prominent_face_bbox_in_region

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
