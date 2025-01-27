# Tf-idf Vectorization in News Recommendation System

Here, the `TfidfVectorizer` converts the article headlines into numerical vectors that represent their importance and relevance in the dataset. It works in two main steps:

1. **Term Frequency (TF):**

   - Counts how often each word appears in a headline.
   - Frequent words get higher values.

2. **Inverse Document Frequency (IDF):**
   - Reduces the weight of common words that appear in many headlines.
   - Rare words get higher importance.

## **How it works in the code:**

```python
headlines = [article["heading"] for article in self.articles]
headline_vectors = self.vectorizer.fit_transform(headlines).toarray()
```

- `TfidfVectorizer` processes all article headlines.
- It assigns higher values to important words and lower values to common ones.
- The result is a numerical matrix where each row represents an article and each column represents a word's importance.

## In short

TF-IDF helps identify the most meaningful words in headlines by balancing frequency and uniqueness, making it easier to compare articles based on their content.
