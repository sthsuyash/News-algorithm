import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# Define your prediction_map here
prediction_map = {
    3: "Khelkud",
    4: "Manoranjan",
    5: "Prabas",
    6: "Sahitya",
    7: "SuchanaPrabidhi",
    8: "Swasthya",
    9: "Viswa",
}

category_texts = {
    "Khelkud": "नेपालको राष्ट्रिय खेल संघले आगामी वर्षको लागि नयाँ खेल नीति जारी गरेको छ।",
    "Manoranjan": "नेपालका प्रमुख चलचित्र निर्माता र निर्देशकहरूले दर्शकहरूको रुचि अनुसार नयाँ फिल्म परियोजनाहरू सुरु गरेका छन्, जसले नयाँ ट्रेण्ड र शैलीको अभ्यास गरिरहेका छन्।",
    "Prabas": "प्रवासी नेपालीहरूले विभिन्न देशहरूमा काम गरेर नेपालका लागि महत्वपूर्ण योगदान पुर्याएका छन्।",
    "Sahitya": "नेपालका प्रमुख लेखकहरूले नयाँ किताबहरू प्रकाशित गरेका छन् जसले समाजमा विचार र चिन्तनको नयाँ दृषटिकोन प्रस्तुत गर्दछ।",
    "SuchanaPrabidhi": "नेपालको सूचना प्रविधि क्षेत्रले सूचना प्रविधिमा नयाँ प्रविधिहरू र सुधार ल्याउँदै छ।",
    "Swasthya": "नेपालको स्वास्थ्य सेवा क्षेत्रमा सुधारका लागि सरकारले नयाँ अस्पताल र उपचार सुविधाहरू शुरु गरेको छ।",
    "Viswa": "विश्वभरिका देशहरूले जलवायु परिवर्तनको मुद्दालाई प्राथमिकतामा राखेका छन् र सहकार्यका लागि नयाँ पहलहरू थालेका छन्।",
}


@pytest.mark.parametrize("category, expected_category", prediction_map.items())
def test_predict_all_categories(category, expected_category):
    # Use the sample text for each category
    text_to_classify = category_texts.get(expected_category, "")

    # Simulate request for each category
    response = client.post(
        "/api/v1/classify",  # Endpoint for classification
        json={"text": text_to_classify},
    )

    # Check if the response status code is 200
    assert response.status_code == 200
    data = response.json()

    # Check if the returned status and message are as expected
    assert data["status_code"] == 200
    assert data["message"] == "News classification successful."

    # Ensure category matches the expected category
    assert data["data"]["category"] == expected_category

    # Ensure the label matches the expected category index (label should be the same as the category index)
    assert data["data"]["label"] == category
