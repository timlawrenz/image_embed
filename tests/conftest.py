import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

# We need to make sure the app can be imported.
# One way is to install your app in editable mode (`pip install -e .`)
# or adjust the python path.
from main import app

@pytest.fixture(scope="session", autouse=True)
def mock_model_loading():
    """
    This fixture is automatically used for the entire test session.
    It patches the model_loader.preload_all_models function to prevent
    it from running, which avoids downloading large models during testing.
    """
    with patch("app.core.model_loader.preload_all_models") as mock_preload:
        mock_preload.return_value = None
        yield

@pytest.fixture(scope="module")
def client():
    """
    Provides a TestClient instance for the entire test module.
    This is more efficient than creating one for every function.
    """
    with TestClient(app) as c:
        yield c
