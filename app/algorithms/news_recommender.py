import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer


class NewsRecommendationSystem:
    def __init__(self, k=5, metric="euclidean", time_decay_factor=0.1):
        self.k = k
        self.metric = metric
        self.time_decay_factor = time_decay_factor
        self.articles = []
        self.feature_matrix = None
        self.vectorizer = TfidfVectorizer()
        self.category_index = {}

    def _calculate_time_decay(self, date):
        """
        Apply time decay to publication date.
        
        Args:
            date (str): The publication date of the article.
            
        Returns:
            float: The time decay factor
        """
        days_since_pub = (datetime.now() - datetime.strptime(date, "%Y-%m-%d")).days
        return np.exp(-self.time_decay_factor * days_since_pub)


    def vectorize_articles(self):
        """Convert articles into a feature matrix."""
        # Extract headline keywords using TF-IDF
        headlines = [article["heading"] for article in self.articles]
        headline_vectors = self.vectorizer.fit_transform(headlines).toarray()

        # Encode categories
        categories = list(set(article["category"] for article in self.articles))
        self.category_index = {category: idx for idx, category in enumerate(categories)}
        category_vectors = np.array(
            [
                [1 if article["category"] == category else 0 for category in categories]
                for article in self.articles
            ]
        )

        # Encode sentiments
        sentiments = list(set(article["sentiment"] for article in self.articles))
        sentiment_index = {sentiment: idx for idx, sentiment in enumerate(sentiments)}
        sentiment_vectors = np.array(
            [
                [1 if article["sentiment"] == sentiment else 0 for sentiment in sentiments]
                for article in self.articles
            ]
        )

        # Normalize visit counts
        visit_counts = np.array(
            [article["visitCount"] for article in self.articles]
        ).reshape(-1, 1)
        visit_counts = (visit_counts - visit_counts.min()) / (
            visit_counts.max() - visit_counts.min()
        )

        # Apply time decay to publication dates
        time_decay = np.array(
            [self._calculate_time_decay(article["date"]) for article in self.articles]
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

    def fit(self, articles):
        """
        Load and preprocess articles.

        Args:
            articles (list): A list of dictionaries containing article information.

        Returns:
            None
        """
        self.articles = articles
        self.vectorize_articles()

    def compute_distances(self, x):
        x = np.array(x)
        if self.metric == "euclidean":
            return np.sqrt(np.sum((self.feature_matrix - x) ** 2, axis=1))
        elif self.metric == "cosine":
            x_norm = np.linalg.norm(x)
            matrix_norms = np.linalg.norm(self.feature_matrix, axis=1)
            return 1 - np.dot(self.feature_matrix, x) / (matrix_norms * x_norm)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def recommend(self, article_id, limit=5):
        """
        Recommend similar articles based on the provided news article ID.

        Args:
            article_id (int): The ID of the news article.
            limit (int): The maximum number of recommendations to return.

        Returns:
            list: A list of recommended article IDs.
        """
        article = next(
            article 
            for article 
            in self.articles 
            if article["id"] == article_id
        )

        article_vector = self.feature_matrix[self.articles.index(article)]
        distances = self.compute_distances(article_vector)
        neighbor_indices = np.argsort(distances)[: self.k + 1]
        recommendations = [
            self.articles[i]["id"]
            for i in neighbor_indices
            if self.articles[i]["id"] != article_id
        ][:limit]

        return recommendations
