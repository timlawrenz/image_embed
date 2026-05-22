from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image

from app.services.description_service import get_image_description


def test_get_image_description_whole_image(mocker):
    mocker.patch("app.core.model_loader.get_device", return_value="cuda")

    mock_llm = MagicMock()
    mock_llm.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    mock_processor.decode.return_value = "gemma description"

    mocker.patch(
        "app.core.model_loader.get_gemma_vision_model_and_processor",
        return_value=(mock_llm, mock_processor),
    )

    dummy_image = Image.new("RGB", (100, 100), color="red")
    caption, b64_image, bbox_used = get_image_description(dummy_image, max_length=7)

    assert caption == "gemma description"
    assert b64_image is None
    assert bbox_used is None

    _, kwargs = mock_llm.generate.call_args
    assert kwargs["max_new_tokens"] == 7


def test_get_image_description_with_crop(mocker):
    mocker.patch("app.core.model_loader.get_device", return_value="cuda")

    mock_llm = MagicMock()
    mock_llm.generate.return_value = torch.tensor([[1, 2, 3, 4]])

    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value = {
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.tensor([[1, 1]]),
    }
    mock_processor.decode.return_value = "gemma description"

    mocker.patch(
        "app.core.model_loader.get_gemma_vision_model_and_processor",
        return_value=(mock_llm, mock_processor),
    )

    mocker.patch(
        "app.services.description_service.get_cropped_image",
        return_value=(Image.new("RGB", (10, 10), color="blue"), "b64"),
    )

    dummy_image = Image.new("RGB", (200, 200), color="blue")
    crop_box = [50, 50, 150, 150]

    caption, b64_image, bbox_used = get_image_description(dummy_image, crop_box=crop_box)

    assert caption == "gemma description"
    assert b64_image == "b64"
    assert bbox_used == crop_box


def test_get_image_description_requires_cuda(mocker):
    mocker.patch("app.core.model_loader.get_device", return_value="cpu")
    get_loader = mocker.patch("app.core.model_loader.get_gemma_vision_model_and_processor")

    dummy_image = Image.new("RGB", (100, 100), color="red")
    with pytest.raises(RuntimeError, match="requires CUDA"):
        get_image_description(dummy_image)

    get_loader.assert_not_called()

