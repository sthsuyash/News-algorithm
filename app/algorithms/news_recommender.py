# import numpy as np
# import pandas as pd
# from typing import List
# from datetime import datetime

# from sklearn.feature_extraction.text import TfidfVectorizer


# class NewsRecommendationSystem:
#     def __init__(self, k=5, metric="euclidean", time_decay_factor=0.1):
#         self.k = k
#         self.metric = metric
#         self.time_decay_factor = time_decay_factor
#         self.articles = []
#         self.feature_matrix = None
#         self.vectorizer = TfidfVectorizer()
#         self.category_index = {}

#     def _calculate_time_decay(self, date: pd.Timestamp) -> float:
#         """
#         Apply time decay to publication date.

#         Args:
#             date: Publication date of the article.

#         Returns:
#             Time decayed value of the publication date.
#         """
#         # Ensure date is a datetime object
#         if isinstance(date, pd.Timestamp):
#             date = date.date()  # Convert to a date object

#         days_since_pub = (datetime.now().date() - date).days
#         return np.exp(-self.time_decay_factor * days_since_pub)

#     def vectorize_articles(self) -> None:
#         """
#         Convert articles into a feature matrix.

#         The feature matrix consists of the following columns:
#         - TF-IDF vectors of article headlines
#         - One-hot encoded category vectors
#         - One-hot encoded sentiment vectors
#         - Normalized visit counts
#         - Time decayed publication dates
#         """
#         # Extract headline keywords using TF-IDF
#         headlines = [article["heading"] for article in self.articles]
#         headline_vectors = self.vectorizer.fit_transform(headlines).toarray()

#         # Encode categories
#         categories = list(set(article["category_name"] for article in self.articles))
#         self.category_index = {category: idx for idx, category in enumerate(categories)}
#         category_vectors = np.array(
#             [
#                 [
#                     1 if article["category_name"] == category else 0
#                     for category in categories
#                 ]
#                 for article in self.articles
#             ]
#         )

#         # Encode sentiments
#         sentiments = list(set(article["sentiment_name"] for article in self.articles))
#         sentiment_index = {sentiment: idx for idx, sentiment in enumerate(sentiments)}
#         sentiment_vectors = np.array(
#             [
#                 [
#                     1 if article["sentiment_name"] == sentiment else 0
#                     for sentiment in sentiments
#                 ]
#                 for article in self.articles
#             ]
#         )

#         # Normalize visit counts
#         visit_counts = np.array(
#             [article["visitCount"] for article in self.articles]
#         ).reshape(-1, 1)

#         visit_counts = (visit_counts - visit_counts.min()) / (
#             visit_counts.max() - visit_counts.min()
#         )

#         # Apply time decay to publication dates
#         time_decay = np.array(
#             [self._calculate_time_decay(article["date"]) for article in self.articles]
#         ).reshape(-1, 1)

#         # Combine features
#         self.feature_matrix = np.hstack(
#             (
#                 headline_vectors,
#                 category_vectors,
#                 sentiment_vectors,
#                 visit_counts,
#                 time_decay,
#             )
#         )

#     def fit(self, articles: List[dict]) -> None:
#         """
#         Load and preprocess articles.

#         Args:
#             articles: List of article dictionaries.
#         """
#         self.articles = articles
#         self.vectorize_articles()

#     def compute_distances(self, x):
#         x = np.array(x)
#         if self.metric == "euclidean":
#             return np.sqrt(np.sum((self.feature_matrix - x) ** 2, axis=1))
#         elif self.metric == "cosine":
#             x_norm = np.linalg.norm(x)
#             matrix_norms = np.linalg.norm(self.feature_matrix, axis=1)
#             return 1 - np.dot(self.feature_matrix, x) / (matrix_norms * x_norm)
#         else:
#             raise ValueError(f"Unknown metric: {self.metric}")

#     def recommend(self, article_id: str, limit: int = 5):
#         """
#         Recommend similar articles along with distances.

#         Args:
#             article_id: ID of the article to recommend similar articles for.
#             limit: Maximum number of recommendations to return.

#         Returns:
#             List of tuples containing (article ID, distance).
#         """
#         article = next(
#             (article for article in self.articles if article["id"] == article_id), None
#         )
#         if not article:
#             return []

#         article_vector = self.feature_matrix[self.articles.index(article)]
#         distances = self.compute_distances(article_vector)
#         neighbor_indices = np.argsort(distances)[: self.k + 1]

#         recommendations_with_distances = [
#             (self.articles[i]["id"], distances[i])
#             for i in neighbor_indices
#             if self.articles[i]["id"] != article_id
#         ][:limit]

#         return recommendations_with_distances


import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer


class NewsRecommendationSystem:

    def __init__(
        self, k: int = 5, metric: str = "euclidean", time_decay_factor: float = 0.1
    ):
        self.k = k
        self.metric = metric
        self.time_decay_factor = time_decay_factor
        self.articles: List[dict] = []
        self.feature_matrix: np.ndarray = None
        self.vectorizer: TfidfVectorizer = TfidfVectorizer()
        self.category_index: Dict[str, int] = {}
        self.sentiment_index: Dict[str, int] = {}

    def _calculate_time_decay(self, date: datetime) -> float:
        """Calculate time decay factor with numerical stability"""
        if not isinstance(date, datetime):
            date = pd.to_datetime(date).to_pydatetime()

        days_since_pub = (datetime.now() - date).days
        return np.exp(-self.time_decay_factor * days_since_pub)

    def _validate_articles(self, articles: List[dict]) -> None:
        """Validate article schema"""
        required_fields = {
            "id",
            "heading",
            "date",
            "visitCount",
            "category_name",
            "sentiment_name",
        }
        for article in articles:
            if not required_fields.issubset(article.keys()):
                missing = required_fields - article.keys()
                raise ValueError(
                    f"Article {article.get('id', '')} missing fields: {missing}"
                )

    def vectorize_articles(self) -> None:
        """Generate feature matrix with improved numerical stability"""
        # Validate before processing
        self._validate_articles(self.articles)

        # TF-IDF for headlines
        headlines = [a["heading"] for a in self.articles]
        headline_vectors = self.vectorizer.fit_transform(headlines).toarray()

        # Category encoding
        categories = list({a["category_name"] for a in self.articles})
        self.category_index = {cat: idx for idx, cat in enumerate(categories)}
        category_vectors = np.array(
            [
                [1 if a["category_name"] == cat else 0 for cat in categories]
                for a in self.articles
            ]
        )

        # Sentiment encoding
        sentiments = list({a["sentiment_name"] for a in self.articles})
        self.sentiment_index = {sent: idx for idx, sent in enumerate(sentiments)}
        sentiment_vectors = np.array(
            [
                [1 if a["sentiment_name"] == sent else 0 for sent in sentiments]
                for a in self.articles
            ]
        )

        # Visit count normalization
        visit_counts = np.array([a["visitCount"] for a in self.articles]).reshape(-1, 1)
        visit_counts = (visit_counts - visit_counts.min()) / (
            visit_counts.max() - visit_counts.min() + 1e-8  # Prevent division by zero
        )

        # Time decay
        time_decay = np.array(
            [self._calculate_time_decay(a["date"]) for a in self.articles]
        ).reshape(-1, 1)

        # Combine features
        self.feature_matrix = np.hstack(
            (
                headline_vectors,
                category_vectors,
                sentiment_vectors,
                visit_counts,
                time_decay,
            )
        )

    def fit(self, articles: List[dict]) -> None:
        """Train model with validation"""
        self.articles = articles
        self.vectorize_articles()

    def compute_distances(self, x: np.ndarray) -> np.ndarray:
        """Calculate distances with numerical stability"""
        x = np.array(x)
        if self.metric == "euclidean":
            return np.sqrt(np.sum((self.feature_matrix - x) ** 2, axis=1))
        elif self.metric == "cosine":
            epsilon = 1e-8  # Prevent division by zero
            x_norm = np.linalg.norm(x) + epsilon
            matrix_norms = np.linalg.norm(self.feature_matrix, axis=1) + epsilon
            return 1 - (np.dot(self.feature_matrix, x) / (matrix_norms * x_norm))
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

    def recommend(self, article_id: str, limit: int = 5) -> List[Tuple[str, float]]:
        """Get recommendations with input validation"""
        if not self.articles:
            raise ValueError("Model not trained - call fit() first")

        if limit < 1:
            raise ValueError("Limit must be at least 1")

        article = next((a for a in self.articles if a["id"] == article_id), None)
        if not article:
            return []

        idx = self.articles.index(article)
        distances = self.compute_distances(self.feature_matrix[idx])

        # Exclude self and get top matches
        sorted_indices = np.argsort(distances)
        recommendations = [
            (self.articles[i]["id"], float(distances[i]))
            for i in sorted_indices
            if self.articles[i]["id"] != article_id
        ][:limit]

        return recommendations

    def save_model(self, path: str) -> None:
        """Persist model to disk"""
        joblib.dump(
            {
                "k": self.k,
                "metric": self.metric,
                "time_decay_factor": self.time_decay_factor,
                "articles": self.articles,
                "vectorizer": self.vectorizer,
                "category_index": self.category_index,
                "sentiment_index": self.sentiment_index,
                "feature_matrix": self.feature_matrix,
            },
            path,
        )

    @classmethod
    def load_model(cls, path: str) -> "NewsRecommendationSystem":
        """Load persisted model"""
        data = joblib.load(path)
        model = cls(
            k=data["k"],
            metric=data["metric"],
            time_decay_factor=data["time_decay_factor"],
        )
        model.articles = data["articles"]
        model.vectorizer = data["vectorizer"]
        model.category_index = data["category_index"]
        model.sentiment_index = data["sentiment_index"]
        model.feature_matrix = data["feature_matrix"]
        return model


def train():
    import os
    import pandas as pd
    import joblib

    from sqlalchemy import create_engine

    # Load database URL from environment variables
    DATABASE_URL = os.getenv("DATABASE_URL")

    # Establish a connection to the database
    engine = create_engine(DATABASE_URL)

    QUERY = """
    SELECT 
        "Post".id,
        "Post".title AS heading,
        "Post"."createdAt" AS date,
        "Post"."visitCount",
        "Category".name AS category_name,
        "Sentiment".name AS sentiment_name
    FROM 
        "Post"
    LEFT JOIN "Category" ON "Post"."categoryId" = "Category".id
    LEFT JOIN "Sentiment" ON "Post"."sentimentId" = "Sentiment".id
    WHERE
        "Sentiment".name != 'NEGATIVE'
    """ 
    df = pd.read_sql_query(QUERY, engine)
    articles = df.to_dict(orient="records")
    
    recommender = NewsRecommendationSystem(k=5)
    recommender.fit(articles)

    MODEL_PATH = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../ml_models/news_recommendation_model.pkl")
    )
    
    try:
        recommender.save_model(MODEL_PATH)
        print(f"Model saved to: {MODEL_PATH}")
    except Exception as e:
        print(f"Failed to save model: {e}")

def test():
    import os
    MODEL_PATH = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../ml_models/news_recommendation_model.pkl")
    )
       
    recommender = NewsRecommendationSystem.load_model(MODEL_PATH)
    print(recommender.recommend("cm6cglzcu0003gjp8cl64f1im"))

if __name__ == "__main__":
    # train()
    test()