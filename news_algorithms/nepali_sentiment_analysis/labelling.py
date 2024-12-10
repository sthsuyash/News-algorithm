import pandas as pd


def standardize_labels(
        df: pd.DataFrame,
        text_column: str,
        label_column: str,
        label_mapping: dict
) -> pd.DataFrame:
    """
    Standardizes the labels in the DataFrame by renaming columns and mapping the labels
    according to a provided label mapping.

    Args:
    - df (pd.DataFrame): The DataFrame containing the text and label columns.
    - text_column (str): The name of the column containing text data.
    - label_column (str): The name of the column containing label data.
    - label_mapping (dict): A dictionary to map old labels to new ones.

    Returns:
    - pd.DataFrame: The DataFrame with standardized column names and labels.

    Example:
    >>> data = {'Sentences': ["मुझे अच्छा लगा", "यह बहुत दुखद है"], 'Sentiment': [1, 2]}
    >>> df = pd.DataFrame(data)
    >>> label_mapping = {1: 'Positive', 2: 'Negative'}
    >>> standardize_labels(df, 'Sentences', 'Sentiment', label_mapping)
        Sentences     Sentiment
    0  मुझे अच्छा लगा    Positive
    1  यह बहुत दुखद है  Negative
    """
    # Rename columns for consistency
    df = df.rename(columns={
        text_column: 'Sentences',
        label_column: 'Sentiment'
    })

    # Map the labels according to the provided mapping
    df['Sentiment'] = df['Sentiment'].map(label_mapping)

    return df
