# How the Time Decay Factor Works in News Recommendation Systems

The **time decay factor** in this news recommendation system works by **exponentially reducing the importance** of older articles based on how long ago they were published. This helps prioritize newer content when recommending articles, making the recommendations more relevant and fresh.

## Formula Used

The time decay is calculated using the formula:

$$
\text{decay} = e^{-\lambda \cdot \text{days\_since\_pub}}
$$

where,

$$ \lambda\text{ } \text{is the 'time decay factor that controls how quickly older articles lose importance.'}$$

$$ \text{days\_since\_pub} \text{ is the number of days since the article was published.}$$

$$ e \text{ (Euler's number, ≈2.718) ensures exponential decay.}$$

---

### **How It Works in Your Code**

1. **In the `_calculate_time_decay` method:**

   - The publication date is compared with today's date to calculate `days_since_pub`.
   - The exponential decay formula is applied to compute a decayed value (a number between 0 and 1).

   ```python
   def _calculate_time_decay(self, date: pd.Timestamp) -> float:
       # Convert the date to a standard format and calculate days since publication
       if isinstance(date, pd.Timestamp):
           date = date.date()

       days_since_pub = (datetime.now().date() - date).days
       return np.exp(-self.time_decay_factor * days_since_pub)
   ```

2. **When vectorizing articles:**

   - This decay value is computed for each article and added as a feature in the `feature_matrix`.
   - More recent articles (fewer days ago) will have a decay value closer to **1**, while older articles will have values closer to **0**.

   ```python
   time_decay = np.array([
       self._calculate_time_decay(article["date"])
       for article in self.articles
   ]).reshape(-1, 1)
   ```

3. **Effect on Recommendation:**
   - During distance calculations, older articles with lower time decay values contribute less to similarity computations.
   - This helps ensure that newer content is favored, even if older articles have very similar features.

---

### **Impact of `time_decay_factor` (λ)**

The value of `time_decay_factor` controls **how fast articles lose relevance**:

- **Small λ (e.g., 0.01):**
  - Slow decay → older articles stay relevant for a long time.
- **Large λ (e.g., 0.1 or 0.2):**
  - Fast decay → older articles quickly lose importance.

#### **Example Decay Values Over Time (λ = 0.1)**

| Days Since Published | Decay Value \(e^{-0.1 \cdot \text{days}}\) |
| -------------------- | ------------------------------------------ |
| 0 days               | 1.00                                       |
| 10 days              | 0.37                                       |
| 30 days              | 0.05                                       |
| 60 days              | 0.0025                                     |

---

### **How to Tune `time_decay_factor`**

To find the optimal value of `time_decay_factor`, you can:

1. **Experiment with different values:**

   - Try values like `0.01, 0.05, 0.1, 0.2` and evaluate recommendations.

2. **Analyze user engagement:**

   - If users prefer fresher content, increase the decay rate.

3. **Domain-specific tuning:**
   - News cycles (e.g., breaking news requires higher decay; evergreen content requires lower decay).

---

### **Key Takeaways**

- The time decay factor helps the model prioritize fresh articles over older ones.
- Exponential decay ensures older articles gradually lose relevance.
- Proper tuning of the factor can significantly impact recommendation relevance.
