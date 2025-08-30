import pytest
from unittest.mock import MagicMock, patch
from PIL import Image
from app.services.description_service import get_image_description

@pytest.fixture
def mock_captioning_model():
    """Fixture to provide a mocked image captioning model and processor."""
    mock_model = MagicMock()
    mock_processor = MagicMock()

    # Configure the mock processor to return a mock object that has a .to() method
    mock_inputs = MagicMock()
    mock_inputs.to.return_value = mock_inputs  # The .to() method can return itself for chaining
    mock_processor.return_value = mock_inputs
    
    # Configure the mock model to generate a dummy tensor output
    # The processor's decode method will be mocked to handle this
    mock_model.generate.return_value = ["dummy_output_tensor"]
    
    # Configure the mock processor's decode method to return a fixed caption
    mock_processor.decode.return_value = "a test caption"

    return mock_model, mock_processor

def test_get_image_description_whole_image(mock_captioning_model):
    """
    Tests generating a description for a whole image.
    """
    # 1. Arrange
    mock_model, mock_processor = mock_captioning_model
    with patch('app.core.model_loader.get_image_captioning_model_and_processor', return_value=(mock_model, mock_processor)):
        # Create a dummy PIL image
        dummy_image = Image.new('RGB', (100, 100), color = 'red')

        # 2. Act
        caption, b64_image, bbox_used = get_image_description(dummy_image)

        # 3. Assert
        assert caption == "A test caption"
        assert b64_image is None
        assert bbox_used is None
        mock_model.generate.assert_called_once()
        # The processor's decode should be called with the model's output
        mock_processor.decode.assert_called_once_with("dummy_output_tensor", skip_special_tokens=True)

def test_get_image_description_with_crop(mock_captioning_model):
    """
    Tests generating a description for a cropped region of an image.
    """
    # 1. Arrange
    mock_model, mock_processor = mock_captioning_model
    with patch('app.core.model_loader.get_image_captioning_model_and_processor', return_value=(mock_model, mock_processor)):
        dummy_image = Image.new('RGB', (200, 200), color = 'blue')
        crop_box = [50, 50, 150, 150]

        # 2. Act
        caption, b64_image, bbox_used = get_image_description(dummy_image, crop_box=crop_box)

        # 3. Assert
        assert caption == "A test caption"
        assert b64_image is not None
        assert isinstance(b64_image, str)
        assert bbox_used == crop_box
        # Ensure the model was called (implying the crop happened before it)
        mock_model.generate.assert_called_once()
