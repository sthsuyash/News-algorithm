import pandas as pd
import matplotlib.pyplot as plt


def plot_sentiment_distribution(
    df: pd.DataFrame,
    title: str,
    labels_dict: dict
) -> None:
    """
    Plots the distribution of sentiment labels in the given DataFrame with customizable labels.

    Args:
    - df (pd.DataFrame): The DataFrame containing the sentiment data.
    - title (str): The title to display on the plot.
    - labels_dict (dict): A dictionary mapping sentiment labels (e.g., 0, 1, 2) to sentiment names.

    Example:
    >>> data = {'Sentences': ['Text 1', 'Text 2', 'Text 3'], 'Sentiment': [0, 1, 2]}
    >>> df = pd.DataFrame(data)
    >>> labels_dict = {0: 'Neutral', 1: 'Positive', 2: 'Negative'}
    >>> plot_sentiment_distribution(df, 'Sentiment Distribution', labels_dict)
    """
    # Count the occurrences of each sentiment label
    sentiment_counts = df['Sentiment'].value_counts().sort_index()

    # Map sentiment labels to their names using the provided dictionary
    labels = [
        labels_dict.get(label, f"Label {label}")
        for label in sentiment_counts.index
    ]

    # Plot the sentiment distribution
    plt.figure(figsize=(6, 4))
    plt.bar(
        labels,
        sentiment_counts,
        color=['blue', 'green', 'red', 'orange'][:len(labels)]
    )
    plt.title(title)
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.show()
