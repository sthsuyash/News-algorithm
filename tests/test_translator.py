import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from main import app

client = TestClient(app)


@pytest.fixture
def mock_translate_text():
    with patch("app.algorithms.translator.translate_text") as mock:
        yield mock


def test_translate_success(mock_translate_text):
    """
    
    """
    mock_translate_text.return_value = "This is a test."
    response = client.post(
        "/api/v1/translate",
        json={
            "text": "यो एक परीक्षण हो।",
            "source_language": "ne",
            "target_language": "en",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status_code"] == 200
    assert data["message"] == "Translation successful."
    assert data["data"]["translated_text"] == mock_translate_text.return_value


def test_translate_error(mock_translate_text):
    mock_translate_text.side_effect = Exception("Mock error")
    response = client.post(
        "/api/v1/translate",
        json={
            "text": "यो एक परीक्षण हो।",
            "source_language": "nep",
            "target_language": "eng",
        },
    )
    
    assert response.status_code == 200
    data = response.json()
    print(f"Data: {data}")
    assert data["status_code"] != 200
    assert "An error occurred while translating the text." in data["message"]
