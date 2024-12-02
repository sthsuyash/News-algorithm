import numpy as np
import gensim

from nepalikit.tokenization import Tokenizer
from nepalikit.preprocessing import TextProcessor

tokenizer = Tokenizer()
processor = TextProcessor()


def text_to_embeddings(
        text: str,
        word2vec_model: gensim.models.KeyedVectors,
        max_length: int = 32
) -> np.ndarray:
    """
    Converts a given text into a fixed-size embedding matrix using a Word2Vec model.

    Args:
    - text (str): The input text to convert.
    - word2vec_model (gensim.models.KeyedVectors): The pre-trained Word2Vec model.
    - max_length (int): The fixed length for the output embedding matrix.

    Returns:
    - np.ndarray: A matrix of shape (max_length, vector_size), where each row is a word vector.

    Example:
    >>> text = "नेपाल कोड सुन्दर छ"
    >>> embedding = text_to_embeddings(text, word2vec_model, max_length=5)
    >>> embedding.shape
    (5, 300)  # Assuming word2vec_model.vector_size = 300
    """
    if not text:
        raise ValueError("Input text cannot be empty or None.")

    # Tokenize the text
    tokens = tokenizer.tokenize(text, level='word')

    # Precompute zero vector for efficiency
    zero_vector = np.zeros(word2vec_model.vector_size)

    # Convert tokens to word vectors
    review_vector = [
        word2vec_model[token]
        if token in word2vec_model else zero_vector
        for token in tokens
    ]

    # Pad or truncate the sequence to max_length
    if len(review_vector) > max_length:
        review_vector = review_vector[:max_length]
    else:
        review_vector.extend([zero_vector] * (max_length - len(review_vector)))

    return np.array(review_vector)
