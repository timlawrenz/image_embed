import pytest
from unittest.mock import MagicMock, patch
from PIL import Image
from app.services.description_service import get_image_description

@pytest.fixture
def mock_captioning_model():
    """Fixture to provide a mocked Florence-2 model and processor."""
    mock_model = MagicMock()
    mock_processor = MagicMock()

    # Configure the mock processor to return a mock object that has a .to() method
    mock_inputs = MagicMock()
    mock_inputs.to.return_value = mock_inputs
    mock_processor.return_value = mock_inputs
    
    # Configure the mock model to generate a dummy tensor output
    mock_model.generate.return_value = ["dummy_output_tensor"]
    
    # Configure the mock processor's batch_decode method to return a fixed caption
    # This simulates the model's raw output which includes the prompt.
    mock_processor.batch_decode.return_value = ["<DETAILED_CAPTION> a test caption"]

    return mock_model, mock_processor

def test_get_image_description_whole_image(mock_captioning_model):
    """
    Tests generating a description for a whole image using the new model.
    """
    # 1. Arrange
    mock_model, mock_processor = mock_captioning_model
    with patch('app.core.model_loader.get_image_captioning_model_and_processor', return_value=(mock_model, mock_processor)):
        dummy_image = Image.new('RGB', (100, 100), color='red')

        # 2. Act
        caption, b64_image, bbox_used = get_image_description(dummy_image)

        # 3. Assert
        assert caption == "a test caption"
        assert b64_image is None
        assert bbox_used is None
        
        # Check that the processor was called with the correct prompt format
        mock_processor.assert_called_with(text="<DETAILED_CAPTION>", images=dummy_image, return_tensors="pt")
        
        mock_model.generate.assert_called_once()
        
        # The processor's batch_decode should be called with the model's output
        mock_processor.batch_decode.assert_called_once_with(["dummy_output_tensor"], skip_special_tokens=True)

def test_get_image_description_with_crop(mock_captioning_model):
    """
    Tests generating a description for a cropped region of an image.
    """
    # 1. Arrange
    mock_model, mock_processor = mock_captioning_model
    with patch('app.core.model_loader.get_image_captioning_model_and_processor', return_value=(mock_model, mock_processor)):
        dummy_image = Image.new('RGB', (200, 200), color='blue')
        crop_box = [50, 50, 150, 150]

        # 2. Act
        caption, b64_image, bbox_used = get_image_description(dummy_image, crop_box=crop_box)

        # 3. Assert
        assert caption == "a test caption"
        assert b64_image is not None
        assert isinstance(b64_image, str)
        assert bbox_used == crop_box
        
        # Ensure the model was called (implying the crop happened before it)
        mock_model.generate.assert_called_once()
        
        # Check that the processor was called with the cropped image
        processed_image = mock_processor.call_args[1]['images']
        assert processed_image.size == (100, 100) # The cropped size
