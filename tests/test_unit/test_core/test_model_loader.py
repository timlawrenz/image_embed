import pytest
from unittest.mock import MagicMock


def test_get_auraface_model_loads_and_caches(mocker):
    """Tests that auraface model is loaded once and cached on subsequent calls."""
    from app.core import model_loader

    # Build a fake insightface module with a mock FaceAnalysis class
    mock_face_analysis_cls = MagicMock()
    mock_app_instance = mock_face_analysis_cls.return_value
    mock_app_instance.prepare.return_value = None

    fake_insightface_app = MagicMock()
    fake_insightface_app.FaceAnalysis = mock_face_analysis_cls

    # Inject the fake module so the lazy import inside get_auraface_model()
    # finds our mock instead of requiring the real insightface on disk.
    mocker.patch.dict("sys.modules", {
        "insightface": MagicMock(),
        "insightface.app": fake_insightface_app,
    })

    # Ensure clean cache
    model_loader._loaded_models.pop("auraface_model", None)

    # First call — should create the model
    app1 = model_loader.get_auraface_model()

    mock_face_analysis_cls.assert_called_once()
    mock_app_instance.prepare.assert_called_once()
    assert app1 is mock_app_instance
    assert "auraface_model" in model_loader._loaded_models

    # Second call — should return the cached instance
    app2 = model_loader.get_auraface_model()

    # Constructor should still have been called only once
    mock_face_analysis_cls.assert_called_once()
    assert app2 is app1


def test_get_auraface_model_missing_dependency(mocker):
    """Tests clear RuntimeError when insightface is not installed."""
    from app.core import model_loader

    # Ensure the model is not already cached
    model_loader._loaded_models.pop("auraface_model", None)

    # Make insightface.app unimportable
    original_import = __builtins__["__import__"]

    def mock_import(name, *args, **kwargs):
        if name in ("insightface", "insightface.app"):
            raise ImportError("No module named 'insightface'")
        return original_import(name, *args, **kwargs)

    mocker.patch("builtins.__import__", side_effect=mock_import)

    with pytest.raises(RuntimeError, match="insightface is required"):
        model_loader.get_auraface_model()
