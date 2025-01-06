from fastapi import APIRouter
from pydantic import BaseModel

from app.core.model import ResponseModel
from app.core.logging import setup_logging
from app.algorithms.news_recommender import NewsRecommendationSystem

router = APIRouter()
logger = setup_logging()
recommender = NewsRecommendationSystem()


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

        recommended_articles = recommender.recommend_news(article_id)

        return ResponseModel(
            status_code=200,
            message="News recommendation successful.",
            data={"recommended_articles": recommended_articles},
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
