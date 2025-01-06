import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from main import app

client = TestClient(app)


@pytest.fixture
def mock_summarizer():
    with patch("app.algorithms.summarizer.Summarizer.show_summary") as mock:
        yield mock


def test_summarize_success(mock_summarizer):
    mock_summarizer.return_value = "दाङ टिपरको ठक्करबाट सोमबार साँझ घोराहीमा मोटरसाइकल चालकको मृत्यु। टिपर चालक घोराही११ डोक्रेनाका ४० वर्षीय जितबहादुर चौधरीलाई नियन्त्रणमा राखी थप अनुसन्धान गरिरहेको डीएसपी थापाले बताए।"

    response = client.post(
        "/api/v1/summarize",
        json={
            "text": "दाङ — टिपरको ठक्करबाट सोमबार साँझ घोराहीमा एक मोटरसाइकल चालकको मृत्यु भएको छ । घोराहीबाट तुलसीपुरतर्फ जाँदै गरेको रा१ख १२९१ नम्बरको टिपरले सोही दिशातर्फ जाँदै गरेको रा७प ७४६ नम्बरको मोटरसाइकललाई घोराही उपमहानगरपालिका–१० झारबैरा चोकमा ठक्कर दिंदा चालक तुलसीपुर उपमहानगरपालिका–१९ कुटीचौरका १९ वर्षीय सुदीप बुढाथोकीको मृत्यु भएको हो । दुर्घटनामा घाइते बुढाथोकीको उपचारका क्रममा राप्ती स्वास्थ्य विज्ञान प्रतिष्ठान घोराहीमा मृत्यु भएको जिल्ला प्रहरी प्रवक्ता डीएसपी ईश्वर थापाले बताए । मोटरसाइकलमा सवार कुटीचौरकै २२ वर्षीय दीपक बस्नेत घाइते भएका छन् । उनको राप्ती स्वास्थ्य विज्ञान प्रतिष्ठान घोराहीमा उपचार भइरहेको उनले जानकारी दिए । टिपर चालक घोराही–११, डोक्रेनाका ४० वर्षीय जितबहादुर चौधरीलाई नियन्त्रणमा राखी थप अनुसन्धान गरिरहेको डीएसपी थापाले बताए ।",
            # "summary_length_ratio": 0.05,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status_code"] == 200
    assert data["message"] == "Summary generated successfully."
    assert data["data"]["summary"] == mock_summarizer.return_value
