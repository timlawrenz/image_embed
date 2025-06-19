import pytest
from fastapi.testclient import TestClient

# We need to make sure the app can be imported.
# One way is to install your app in editable mode (`pip install -e .`)
# or adjust the python path.
from main import app

@pytest.fixture(scope="module")
def client():
    """
    Provides a TestClient instance for the entire test module.
    This is more efficient than creating one for every function.
    """
    with TestClient(app) as c:
        yield c
