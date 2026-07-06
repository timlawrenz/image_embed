import pytest
from unittest.mock import MagicMock, patch
from PIL import Image
from app.services.embedding_service import get_dino_embedding, get_dino_v3_embedding

@pytest.fixture
def mock_dino_model():
    """Fixture to provide a mocked DINOv2 model and processor."""
    mock_model = MagicMock()
    mock_processor = MagicMock()

    # Configure the mock processor to return a mock tensor
    mock_inputs = MagicMock()
    mock_inputs.unsqueeze.return_value.to.return_value = mock_inputs
    mock_processor.return_value = mock_inputs
    
    # Configure the mock model to return a dummy tensor output
    mock_tensor = MagicMock()
    mock_tensor.cpu.return_value.numpy.return_value.tolist.return_value = [0.1, 0.2, 0.3]
    mock_model.return_value = [mock_tensor]
    
    return mock_model, mock_processor

def test_get_dino_embedding_whole_image(mock_dino_model):
    """
    Tests generating a DINO embedding for a whole image.
    """
    # 1. Arrange
    mock_model, mock_processor = mock_dino_model
    with patch('app.core.model_loader.get_dino_model_and_processor', return_value=(mock_model, mock_processor)):
        dummy_image = Image.new('RGB', (100, 100), color='red')

        # 2. Act
        embedding, b64_image, bbox_used = get_dino_embedding(dummy_image)

        # 3. Assert
        assert isinstance(embedding, list)
        assert b64_image is None
        assert bbox_used is None
        
        mock_processor.assert_called_once_with(dummy_image)
        mock_model.assert_called_once()

def test_get_dino_embedding_with_crop(mock_dino_model):
    """
    Tests generating a DINO embedding for a cropped image.
    """
    # 1. Arrange
    mock_model, mock_processor = mock_dino_model
    with patch('app.core.model_loader.get_dino_model_and_processor', return_value=(mock_model, mock_processor)):
        dummy_image = Image.new('RGB', (200, 200), color='blue')
        crop_box = [50, 50, 150, 150]

        # 2. Act
        embedding, b64_image, bbox_used = get_dino_embedding(dummy_image, crop_bbox=crop_box)

        # 3. Assert
        assert isinstance(embedding, list)
        assert b64_image is not None
        assert isinstance(b64_image, str)
        assert bbox_used == crop_box

        # Ensure the model was called (implying the crop happened before it)
        mock_model.assert_called_once()

        # Check that the processor was called with the cropped image
        processed_image = mock_processor.call_args[0][0]
        assert processed_image.size == (100, 100)


@pytest.fixture
def mock_dino_v3_model():
    """Fixture to provide a mocked DINOv3 model and image processor."""
    mock_model = MagicMock()
    mock_processor = MagicMock()

    # Processor returns a dict of tensors
    mock_pixel_values = MagicMock()
    mock_pixel_values.to.return_value = mock_pixel_values
    mock_processor.return_value = {"pixel_values": mock_pixel_values}

    # Model returns an object with pooler_output
    mock_embedding_row = MagicMock()
    mock_embedding_row.detach.return_value.cpu.return_value.numpy.return_value.tolist.return_value = [0.1, 0.2, 0.3]
    mock_embedding_batch = MagicMock()
    mock_embedding_batch.__getitem__.return_value = mock_embedding_row

    mock_outputs = MagicMock()
    mock_outputs.image_embeds = None
    mock_outputs.pooler_output = mock_embedding_batch
    mock_model.return_value = mock_outputs

    return mock_model, mock_processor


def test_get_dino_v3_embedding_whole_image(mock_dino_v3_model):
    mock_model, mock_processor = mock_dino_v3_model
    with patch('app.core.model_loader.get_dino_v3_model_and_processor', return_value=(mock_model, mock_processor)):
        dummy_image = Image.new('RGB', (100, 100), color='red')

        embedding, b64_image, bbox_used = get_dino_v3_embedding(dummy_image)

        assert isinstance(embedding, list)
        assert b64_image is None
        assert bbox_used is None
        mock_model.assert_called_once()
        mock_processor.assert_called_once()


def test_get_dino_v3_embedding_with_crop(mock_dino_v3_model):
    mock_model, mock_processor = mock_dino_v3_model
    with patch('app.core.model_loader.get_dino_v3_model_and_processor', return_value=(mock_model, mock_processor)):
        dummy_image = Image.new('RGB', (200, 200), color='blue')
        crop_box = [50, 50, 150, 150]

        embedding, b64_image, bbox_used = get_dino_v3_embedding(dummy_image, crop_bbox=crop_box)

        assert isinstance(embedding, list)
        assert b64_image is not None
        assert isinstance(b64_image, str)
        assert bbox_used == crop_box

        processed_image = mock_processor.call_args.kwargs["images"]
        assert processed_image.size == (100, 100)


# ──────────────────────────────────────────────────────────────────────
# AuraFace identity embedding tests
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_auraface_app():
    """Fixture providing a mock InsightFace AuraFace app with a detectable face."""
    mock_app = MagicMock()
    mock_face = MagicMock()
    mock_face.normed_embedding = __import__("numpy").array([0.1] * 512)
    mock_app.get.return_value = [mock_face]  # One face detected
    return mock_app


def test_get_auraface_embedding_requires_crop_bbox():
    """ValueError when no crop_bbox provided — AuraFace needs a face region."""
    from app.services.embedding_service import get_auraface_embedding

    dummy_image = Image.new("RGB", (200, 200), color="red")

    with pytest.raises(ValueError, match="requires a face crop"):
        get_auraface_embedding(dummy_image, crop_bbox=None)


def test_get_auraface_embedding_face_found(mock_auraface_app, mocker):
    """Returns 512-d list + base64 crop + bbox when a face is detected."""
    from app.core import model_loader
    from app.services.embedding_service import get_auraface_embedding

    mocker.patch.object(model_loader, "get_auraface_model", return_value=mock_auraface_app)

    dummy_image = Image.new("RGB", (200, 200), color="red")
    crop_box = [50, 50, 150, 150]

    embedding, b64_img, bbox_used = get_auraface_embedding(dummy_image, crop_bbox=crop_box)

    assert isinstance(embedding, list)
    assert len(embedding) == 512
    assert b64_img is not None
    assert isinstance(b64_img, str)
    assert bbox_used == crop_box
    mock_auraface_app.get.assert_called_once()


def test_get_auraface_embedding_no_face_detected(mocker):
    """ValueError raised when SCRFD finds zero faces (even after padding)."""
    from app.core import model_loader
    from app.services.embedding_service import get_auraface_embedding

    mock_app = MagicMock()
    mock_app.get.return_value = []  # No faces detected
    mocker.patch.object(model_loader, "get_auraface_model", return_value=mock_app)
    # cv2 is imported lazily inside the padding fallback
    mocker.patch.dict("sys.modules", {"cv2": MagicMock()})

    dummy_image = Image.new("RGB", (200, 200), color="red")
    crop_box = [50, 50, 150, 150]

    with pytest.raises(ValueError, match="No face found"):
        get_auraface_embedding(dummy_image, crop_bbox=crop_box)


def test_get_auraface_embedding_padding_fallback(mocker):
    """20% padding trick recovers detection on tight crops (ported from eidolon)."""
    from app.core import model_loader
    from app.services.embedding_service import get_auraface_embedding

    mock_app = MagicMock()
    mock_face = MagicMock()
    mock_face.normed_embedding = __import__("numpy").array([0.5] * 512)
    # First call fails, second (after padding) succeeds
    mock_app.get.side_effect = [[], [mock_face]]
    mocker.patch.object(model_loader, "get_auraface_model", return_value=mock_app)

    # Mock cv2 for the padding fallback
    mock_cv2 = MagicMock()
    mock_cv2.copyMakeBorder.return_value = MagicMock()
    mock_cv2.BORDER_CONSTANT = 0
    mocker.patch.dict("sys.modules", {"cv2": mock_cv2})

    dummy_image = Image.new("RGB", (100, 100), color="red")
    crop_box = [10, 10, 90, 90]

    embedding, b64_img, bbox_used = get_auraface_embedding(dummy_image, crop_bbox=crop_box)

    assert len(embedding) == 512
    assert bbox_used == crop_box
    # get() should have been called twice: original + padded
    assert mock_app.get.call_count == 2
    mock_cv2.copyMakeBorder.assert_called_once()

