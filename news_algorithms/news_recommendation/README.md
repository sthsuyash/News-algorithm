# Nepali News Recommendation System

## Overview

This `NewsRecommendationSystem` class implements a `content-based recommendation system` for news articles. It recommends similar articles based on various features such as:

- headlines
- categories
- sentiments
- visit counts
- publication dates.

The algorithm uses feature engineering techniques to transform article attributes into numerical representations and employs distance-based similarity measures to find the most relevant articles.

---

## **Key Components of the Algorithm**

### **1. Initialization (`__init__` method)**

- Parameters:

  - `k`: The number of nearest neighbors to consider when making recommendations.
  - `metric`: The distance metric used for similarity calculation (`"euclidean"` or `"cosine"`).
  - `time_decay_factor`: A parameter controlling the influence of article recency on recommendations.

- Internal attributes:
  - `self.articles`: Stores the list of news articles.
  - `self.feature_matrix`: Stores the numerical representation of all articles.
  - `self.vectorizer`: A `TfidfVectorizer` instance to convert headlines into numerical features.
  - `self.category_index`: A dictionary to map category names to indices for one-hot encoding.

### **2. Time Decay Calculation (`_calculate_time_decay` method)**

This function applies an exponential decay to the publication date of each article to give more weight to recent articles.

- **Steps:**
  1. Convert the input `date` to a Python `date` object (if it's a Pandas `Timestamp`).
  2. Compute the number of days since publication.
  3. Apply the exponential decay formula:

$$
\text{decay} = e^{-\lambda \cdot \text{days\_since\_pub}}
$$

where, $$ \lambda\text{ } \text{is the 'time decay factor'}$$

### **3. Article Feature Extraction (`vectorize_articles` method)**

This method transforms the articles' attributes into a numerical feature matrix used for similarity calculations.

- **Feature Engineering Process:**

  1. **Headline TF-IDF Vectors:**

     - Extracts text from `heading` fields and applies Term Frequency-Inverse Document Frequency (TF-IDF) to convert text into numerical vectors.

  2. **Category Encoding (One-Hot Encoding):**

     - Unique categories are identified, and each article is converted into a one-hot vector representation of categories.

  3. **Sentiment Encoding (One-Hot Encoding):**

     - Similar to categories, sentiment labels are also one-hot encoded.

  4. **Visit Count Normalization:**

     - Visit counts are normalized to a [0,1] scale to ensure numerical consistency across features.

  5. **Time Decay Calculation:**
     - Applies time decay to the publication date to prioritize recent articles.

- **Final Feature Matrix:**
  - The extracted features (headline vectors, one-hot encodings, normalized visit counts, and time decay) are combined into a single matrix using `np.hstack()`.
  - This matrix is stored in `self.feature_matrix`.

### **4. Fitting the Model (`fit` method)**

This method loads the list of articles and processes them by calling `vectorize_articles`.

- **Steps:**
  1. Assigns the list of articles to `self.articles`.
  2. Calls `vectorize_articles` to transform the articles into numerical representations.

### **5. Similarity Calculation (`compute_distances` method)**

This function calculates the distance (or similarity) between the given article and all other articles based on the specified metric.

- **Supported distance metrics:**
- **Euclidean distance:**

  $$
  d(x, y) = \sqrt{\sum{(x_i - y_i)^2}}
  $$

- **Cosine similarity (converted to distance):**

  $$
  d(x, y) = 1 - \frac{x \cdot y}{\|x\| \cdot \|y\|}
  $$

- **Steps:**
  1. Convert input vector `x` into a NumPy array.
  2. Compute the distance from `x` to each row in the feature matrix using either Euclidean or Cosine distance.
  3. Return the computed distances.

### **6. Recommendation (`recommend` method)**

This method recommends similar articles based on the computed feature representations.

- **Steps:**
  1. Find the target article by its `id` from the list of articles.
  2. Retrieve the feature vector for the target article.
  3. Compute distances between the target article and all others using `compute_distances`.
  4. Sort articles by distance and select the closest `k` articles (excluding the input article itself).
  5. Return the list of recommended article IDs.

---

## **How the Algorithm Works Step-by-Step**

1. **Input Articles:**

   - The `fit` method is called with a list of articles containing fields like `heading`, `category_name`, `sentiment_name`, `visitCount` and `date`.

2. **Feature Extraction:**

   - The articles are converted into a numerical matrix using TF-IDF, one-hot encoding, normalization and time decay.

3. **Recommendation Request:**

   - When `recommend(article_id)` is called, the system identifies the requested article and finds similar articles based on the chosen distance metric.

4. **Result Generation:**
   - A list of recommended article IDs is returned based on similarity.

---

## **Example Usage**

```python
# Sample articles dataset
articles = [
    {
        "id": "1",
        "heading": "Stock market hits record high",
        "category_name": "Finance",
        "sentiment_name": "Positive",
        "visitCount": 5000,
        "date": pd.Timestamp("2024-01-20"),
    },
    {
        "id": "2",
        "heading": "Technology stocks rally after strong earnings",
        "category_name": "Technology",
        "sentiment_name": "Positive",
        "visitCount": 3000,
        "date": pd.Timestamp("2024-01-18"),
    }
]

# Initialize and train the recommendation system
recommender = NewsRecommendationSystem(k=3, metric="cosine")
recommender.fit(articles)

# Recommend articles similar to article with ID "1"
recommendations = recommender.recommend(article_id="1", limit=3)
print(recommendations)
```

---

## **Strengths of the Algorithm**

- **Personalization:**
  - Customizes recommendations based on the content features rather than popularity alone.
- **Time Awareness:**

  - Time decay ensures recent articles are given higher weight.

- **Versatile Similarity Metrics:**

  - Supports both Euclidean and Cosine distance, providing flexibility based on the application (but used only Euclidean distance in this application).

- **Scalability:**
  - TF-IDF and one-hot encoding are efficient for moderate datasets.

---

## **Potential Improvements**

1. **Hybrid Model:**

   - Combine content-based filtering with collaborative filtering for better personalization.

2. **Keyword Expansion:**

   - Use NLP techniques like word embeddings (e.g., Word2Vec) to enhance the understanding of article topics.

3. **Real-Time Updates:**

   - Implement incremental updates to include new articles dynamically.

4. **Weighting Features:**
   - Fine-tune the importance of different features (e.g., more weight to headlines vs. visit counts).
