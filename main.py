from fastapi import FastAPI
from fastapi import APIRouter
from starlette.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.response_model import ResponseModel
from app.routes import (
    summarizer,
    translator,
    classifier,
    news_recommender,
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


api_router.include_router(summarizer)
api_router.include_router(translator)
api_router.include_router(classifier)
api_router.include_router(news_recommender)
print("news_recommender", news_recommender)
api_router.include_router(sentiment_analyer)


app = FastAPI(
    title="Nepali News Algorithms API",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_V1_STR)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
