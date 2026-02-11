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
    
    # Configure the mock processor's decode method to return a fixed caption
    mock_processor.decode.return_value = "a test caption"

    return mock_model, mock_processor

def test_get_image_description_whole_image(mock_captioning_model, mocker):
    """Tests generating a Gemma-backed description for a whole image."""
    mock_model, mock_processor = mock_captioning_model

    # Mock Gemma model + tokenizer
    mock_llm = MagicMock()
    mock_tokenizer = MagicMock()
    input_ids = MagicMock(); input_ids.to.return_value = input_ids
    attention_mask = MagicMock(); attention_mask.to.return_value = attention_mask
    mock_tokenizer.return_value = {"input_ids": input_ids, "attention_mask": attention_mask}
    mock_llm.generate.return_value = ["gemma_tokens"]
    mock_tokenizer.decode.return_value = "Description: gemma description"

    mocker.patch(
        "app.core.model_loader.get_gemma_text_model_and_tokenizer",
        return_value=(mock_llm, mock_tokenizer),
    )

    with patch("app.core.model_loader.get_image_captioning_model_and_processor", return_value=(mock_model, mock_processor)):
        dummy_image = Image.new("RGB", (100, 100), color="red")

        caption, b64_image, bbox_used = get_image_description(dummy_image)

        assert caption == "gemma description"
        assert b64_image is None
        assert bbox_used is None
        mock_model.generate.assert_called_once()
        mock_processor.decode.assert_called_once_with("dummy_output_tensor", skip_special_tokens=True)

def test_get_image_description_with_crop(mock_captioning_model, mocker):
    """Tests generating a Gemma-backed description for a cropped region."""
    mock_model, mock_processor = mock_captioning_model

    mock_llm = MagicMock()
    mock_tokenizer = MagicMock()
    input_ids = MagicMock(); input_ids.to.return_value = input_ids
    attention_mask = MagicMock(); attention_mask.to.return_value = attention_mask
    mock_tokenizer.return_value = {"input_ids": input_ids, "attention_mask": attention_mask}
    mock_llm.generate.return_value = ["gemma_tokens"]
    mock_tokenizer.decode.return_value = "Description: gemma description"

    mocker.patch(
        "app.core.model_loader.get_gemma_text_model_and_tokenizer",
        return_value=(mock_llm, mock_tokenizer),
    )

    with patch("app.core.model_loader.get_image_captioning_model_and_processor", return_value=(mock_model, mock_processor)):
        dummy_image = Image.new("RGB", (200, 200), color="blue")
        crop_box = [50, 50, 150, 150]

        caption, b64_image, bbox_used = get_image_description(dummy_image, crop_box=crop_box)

        assert caption == "gemma description"
        assert b64_image is not None
        assert isinstance(b64_image, str)
        assert bbox_used == crop_box
        mock_model.generate.assert_called_once()


def test_get_image_description_fallback_to_base_caption(mock_captioning_model, mocker):
    """If Gemma fails to load/generate, we fall back to BLIP base caption."""
    mock_model, mock_processor = mock_captioning_model

    mocker.patch(
        "app.core.model_loader.get_gemma_text_model_and_tokenizer",
        side_effect=RuntimeError("nope"),
    )

    with patch("app.core.model_loader.get_image_captioning_model_and_processor", return_value=(mock_model, mock_processor)):
        dummy_image = Image.new("RGB", (100, 100), color="red")

        caption, _, _ = get_image_description(dummy_image)

        assert caption == "a test caption"

