# Flow of a News Classifier Notebook

### 1. **Loading the Dataset**

The first part of the code handles loading multiple CSV files containing text and category labels into a single pandas DataFrame.

```python
csv_dir = '../dataset/processed'
csv_files = glob.glob(os.path.join(csv_dir, '*.csv'))
combined_df = pd.DataFrame()

# Load and combine datasets
for file in csv_files:
    print(f"Loading: {file}")
    df = pd.read_csv(file)
    if not set(['category', 'text']).issubset(df.columns):
        raise ValueError(f"File {file} does not have the required columns: ['category', 'text']")
    combined_df = pd.concat([combined_df, df], ignore_index=True)
```

- **CSV Files**: This section looks for CSV files in the `processed` directory. Each CSV should contain `category` (label) and `text` (text data) columns.
- **Error Checking**: If any file is missing these columns, an error is raised.
- **Combining Data**: It concatenates the content of all CSV files into a single `combined_df` DataFrame.

### 2. **Inspecting the Data**

Here, we check for any missing or duplicate values in the dataset.

```python
print(f"Shape of combined DataFrame: {combined_df.shape}\n-----------")
print(f"Missing values:\n{combined_df.isnull().sum()}\n-----------")
print(f"Duplicates: {combined_df.duplicated().sum()}")
```

- **Shape**: Displays the number of rows and columns in the dataset.
- **Missing Values**: Checks for any missing values in the dataset.
- **Duplicates**: Identifies any duplicate rows.

### 3. **Preprocessing the Data**

Next, we clean the text data by applying a custom `preprocess` function. This might involve removing stop words, punctuation, converting to lowercase, and other typical text preprocessing steps.

```python
from utils.preprocess import preprocess

# Apply preprocessing
tqdm.pandas()
combined_df["text"] = combined_df["text"].progress_apply(preprocess)

# Assign numeric values to labels
classes = {c: i for i, c in enumerate(combined_df["category"].unique())}
combined_df["target"] = combined_df["category"].progress_apply(lambda x: classes[x])
```

- **Text Preprocessing**: The `preprocess` function is applied to each text entry using `progress_apply` for progress tracking.
- **Label Encoding**: The categorical labels (e.g., topic names) are converted into numeric labels (using a dictionary `classes`), which is required for machine learning models.

### 4. **Preparing Features and Target Variables**

We separate the text data (features) and the target labels.

```python
X = combined_df["text"]
y = combined_df["target"]
```

- `X` contains the preprocessed text data.
- `y` contains the numeric target labels.

### 5. **Hyperparameter Grids for Models**

This section defines the hyperparameter search spaces for the three models we are going to evaluate: Naive Bayes, Logistic Regression, and Random Forest.

```python
param_grid_nb = {
    'vectorizer__ngram_range': [(1, 1), (1, 2)],
    'classifier__alpha': [0.1, 0.5, 1.0]
}

param_grid_lr = {
    'vectorizer__ngram_range': [(1, 1), (1, 2)],
    'classifier__C': [0.1, 1.0, 10.0],
    'classifier__solver': ['liblinear', 'saga']
}

param_grid_rf = {
    'vectorizer__ngram_range': [(1, 1), (1, 2)],
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}
```

- **Naive Bayes (`param_grid_nb`)**: It tests different n-gram ranges and smoothing parameters (`alpha`).
- **Logistic Regression (`param_grid_lr`)**: It explores different regularization strengths (`C`) and solvers.
- **Random Forest (`param_grid_rf`)**: It tests different numbers of trees (`n_estimators`), tree depths (`max_depth`), and split thresholds.

### 6. **Defining Models**

Here, we define the three models that we will evaluate: Naive Bayes, Logistic Regression, and Random Forest.

```python
models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier()
}
```

Each model is associated with its respective sklearn class.

### 7. **Grid Search with Cross-Validation**

Now, we set up a `GridSearchCV` to search through the hyperparameter grids for each model and perform cross-validation to evaluate each configuration.

```python
kf = KFold(n_splits=5, shuffle=True, random_state=1)

for model_name, model in models.items():
    print(f"Evaluating {model_name}...")

    # Create pipeline
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(ngram_range=(1, 2))),
        ('tfidf', TfidfTransformer()),
        ('classifier', model)
    ])

    grid_search = GridSearchCV(
        pipeline,
        param_grids[model_name],
        cv=kf,
        scoring='accuracy',
        verbose=1
    )
    grid_search.fit(X, y)

    results[model_name] = {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_
    }

    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy for {model_name}: {grid_search.best_score_}\n")
```

- **KFold Cross-Validation**: 5-fold cross-validation is used to split the dataset into 5 subsets, training the model on 4 subsets and testing on the remaining 1 subset. This process is repeated for each subset.
- **GridSearchCV**: It automatically tunes hyperparameters and evaluates model performance using cross-validation. The best configuration and corresponding score are stored in the `results` dictionary.

### 8. **Model Evaluation and Metrics**

After hyperparameter tuning, the models are evaluated on the entire dataset (`X`, `y`).

```python
for model_name, model in models.items():
    print(f"Evaluating {model_name} on cross-validation set...")

    # Create pipeline
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(ngram_range=(1, 2))),
        ('tfidf', TfidfTransformer()),
        ('classifier', model)
    ])

    # Train the model with the best parameters
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X)

    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')
    f1 = f1_score(y, y_pred, average='weighted')

    # Store results
    results[model_name].update({
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy
    })

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}\n")
```

- The **best model** (from GridSearchCV) is evaluated on the full dataset (`X`, `y`).
- Evaluation metrics (accuracy, precision, recall, and F1-score) are calculated for each model and stored in the `results` dictionary.

### 9. **Visualizing the Results**

We create confusion matrices and a performance comparison bar chart.

#### **Confusion Matrix**:

```python
def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes.keys(), yticklabels=classes.keys())
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

for model_name, model in models.items():
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    plot_confusion_matrix(cm, model_name)
```

- **Confusion Matrix**: A confusion matrix shows how well the model classifies each category. It’s visualized with a heatmap, where the rows represent true labels and the columns represent predicted labels.

#### **Performance Comparison**:

```python
def plot_performance_comparison(results):
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    scores = {metric: [] for metric in metrics}

    for model_name in results:
        scores['accuracy'].append(results[model_name]['accuracy'])
        scores['precision'].append(results[model_name]['precision'])
        scores['recall'].append(results[model_name]['recall'])
        scores['f1_score'].append(results[model_name]['f1_score'])

    df_scores = pd.DataFrame(scores, index=models.keys())
    df_scores.plot(kind='bar', figsize

=(10, 6))
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.xlabel('Model')
    plt.xticks(rotation=45)
    plt.show()

plot_performance_comparison(results)
```

- **Bar Chart**: A bar chart compares the performance (accuracy, precision, recall, F1-score) of the models side by side.

### 10. **Saving the Best Models**

Finally, the best-performing models are saved as `.pkl` files for later use.

```python
for model_name, model in models.items():
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, f"final_{model_name}_model.pkl")
    print(f"Saved best {model_name} model.")
```

- **Saving**: The best model (after tuning) is saved using `joblib.dump`, which allows you to load the model later without retraining.

---

### Summary of Flow:

1. Load and combine dataset.
2. Preprocess text data.
3. Define models and their hyperparameter grids.
4. Use GridSearchCV with cross-validation for hyperparameter tuning.
5. Evaluate models on metrics like accuracy, precision, recall, and F1-score.
6. Visualize results using confusion matrices and bar charts.
7. Save the best models for future use.
