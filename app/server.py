from app.model import ResponseModel

from fastapi import FastAPI
from fastapi import APIRouter
from starlette.middleware.cors import CORSMiddleware

from app.config import settings


from app.routes import (
    summary,
    news_classifier,
    news_recommendation,
    sentiment_analysis
)

api_router = APIRouter()

# home route


@api_router.get("/")
async def home():
    return ResponseModel(
        status_code=200,
        message="Welcome to Nepali News Algorithm API",
        data=None,
        error=None
    )

api_router.include_router(summary.router)
api_router.include_router(news_classifier.router)
# api_router.include_router(news_recommendation.router)
# api_router.include_router(sentiment_analysis.router)


app = FastAPI(
    title="Nepali News API",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
)

# Set all CORS enabled origins
if settings.all_cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.all_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(api_router, prefix=settings.API_V1_STR)
