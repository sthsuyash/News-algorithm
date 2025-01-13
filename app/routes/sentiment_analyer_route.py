from fastapi import APIRouter
from pydantic import BaseModel

from app.core.response_model import ResponseModel
from app.core.logging import setup_logging
from app.algorithms.sentiment_analyzer import SentimentAnalyzer

router = APIRouter()
logger = setup_logging()
sentiment_analyzer = SentimentAnalyzer()


class SentimentAnalysisRequest(BaseModel):
    text: str


@router.post("/analyze-sentiment", response_model=ResponseModel)
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
        probability, sentiment = sentiment_analyzer.predict_sentiment(text)

        return ResponseModel(
            status_code=200,
            message="Sentiment analysis successful.",
            data={
                "sentiment": sentiment, 
                "probability": probability
            },
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
