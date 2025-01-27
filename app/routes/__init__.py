from .summarizer_route import router as summarizer
from .translator_route import router as translator
from .classifier_route import router as classifier
from .recommendor_route import router as news_recommender
from .sentiment_analyer_route import router as sentiment_analyer

_all__ = [
    summarizer,
    translator,
    classifier,
    news_recommender,
    sentiment_analyer,
]