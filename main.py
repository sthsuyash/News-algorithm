from fastapi import FastAPI
from fastapi import APIRouter
from starlette.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.model import ResponseModel
from app.routes import (
    summarizer,
    translator,
    classifier,
    recommendor,
    sentiment_analyer,
)

# Define the API router
api_router = APIRouter()


@api_router.get("/")
async def home():
    return ResponseModel(
        status_code=200,
        message="Welcome to Nepali News Algorithm API",
        data=None,
        error=None,
    )


api_router.include_router(summarizer.router)
api_router.include_router(translator.router)
api_router.include_router(classifier.router)
api_router.include_router(recommendor.router)
api_router.include_router(sentiment_analyer.router)


app = FastAPI(
    title="Nepali News Algorithms API",
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