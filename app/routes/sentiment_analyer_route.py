from fastapi import APIRouter
from pydantic import BaseModel

from app.core.model import ResponseModel
from app.core.logging import setup_logging
from app.algorithms.sentiment_analyzer import SentimentAnalyzer

router = APIRouter()
logger = setup_logging()
sa = SentimentAnalyzer()


class SentimentAnalysisRequest(BaseModel):
    text: str


@router.post("/analyze", response_model=ResponseModel)
async def analyze_sentiment(request: SentimentAnalysisRequest):
    """
    Endpoint to analyze the sentiment of the given text.

    Args:
        request (SentimentAnalysisRequest): The request body containing the text.

    Returns:
        ResponseModel: The response model containing status, message, and the sentiment.
    """
    try:
        text = request.text

        sentiment = sa.predict_sentiment(text)

        return ResponseModel(
            status_code=200,
            message="Sentiment analysis successful.",
            data={"sentiment": sentiment},
            error=None,
        )

    except Exception as e:
        logger.error(f"An error occurred while analyzing the sentiment: {e}")

        return ResponseModel(
            status_code=500,
            message="An error occurred while analyzing the sentiment.",
            data=None,
            error=str(e),
        )
