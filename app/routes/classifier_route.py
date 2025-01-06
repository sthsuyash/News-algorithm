from fastapi import APIRouter
from pydantic import BaseModel

from app.core.model import ResponseModel
from app.core.logging import setup_logging
from app.algorithm.news_classifier import classifier

router = APIRouter()
logger = setup_logging()


class NewsClassificationRequest(BaseModel):
    text: str


@router.get("/classify", response_model=ResponseModel)
async def classify_news(request: NewsClassificationRequest):
    """
    Endpoint to classify the given news text into categories.

    Args:
        request (NewsClassificationRequest): The request body containing the text to classify.

    Returns:
        ResponseModel: The response model containing status, message, predicted category, and label.
    """
    try:
        text = request.text
        predicted_category, predicted_label = classifier.predict_category_with_label(
            text
        )

        return ResponseModel(
            status_code=200,
            message="News classification successful.",
            data={"category": predicted_category, "label": predicted_label},
            error=None,
        )

    except Exception as e:
        logger.error(f"An error occurred while classifying the news: {e}")
        return ResponseModel(
            status_code=500,
            message="An error occurred while classifying the news.",
            data=None,
            error=str(e),
        )
