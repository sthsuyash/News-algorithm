import os
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException

from app.core.response_model import ResponseModel
from app.core.logging import setup_logging
from app.algorithms.news_recommender import NewsRecommendationSystem

router = APIRouter()
logger = setup_logging()

# Define the path to the recommendation model
MODEL_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "../ml_models/news_recommendation_model.pkl"
    )
)

# Check if the model file exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

# Load the recommendation model
recommender = None
try:
    recommender = NewsRecommendationSystem.load_model(MODEL_PATH)
    logger.info("Recommendation model loaded successfully.")
except Exception as e:
    logger.error(f"An error occurred while loading the recommendation model: {e}")
    raise HTTPException(status_code=500, detail="Failed to load recommendation model.")


class NewsRecommendationRequest(BaseModel):
    article_id: str


@router.post("/recommend", response_model=ResponseModel)
async def recommend_news(request: NewsRecommendationRequest):
    """
    Endpoint to recommend news articles similar to the given article.

    Args:
        request (NewsRecommendationRequest): The request body containing the article ID.

    Returns:
        ResponseModel: The response model containing status, message, and the recommended articles.
    """
    try:
        article_id = request.article_id

        # Get recommendations from the model
        recommendations = recommender.recommend(article_id, limit=5)

        # Format the recommendations
        recommended_articles = [
            {"article_id": rec_id, "score": float(score)}
            for rec_id, score in recommendations
        ]

        return ResponseModel(
            status_code=200,
            message="News recommendation successful.",
            data={"recommendations": recommended_articles},
            error=None,
        )

    except Exception as e:
        logger.error(f"An error occurred while recommending news articles: {e}")

        return ResponseModel(
            status_code=500,
            message="An error occurred while recommending news articles.",
            data=None,
            error=str(e),
        )
