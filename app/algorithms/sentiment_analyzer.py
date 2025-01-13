import os
import torch
import numpy as np

from app.helpers.preprocess import preprocess_text as preprocess
from app.helpers.embedding import Embeddings
from app.helpers.sentiment_analysis_transformer_model import SentimentTransformer


EMBEDDING_DIM = 300
MAX_LENGTH = 120
D_MODEL = 300
NUM_HEADS = 6
NUM_LAYERS = 6
NUM_CLASSES = 3
D_FF = 1200
DROPOUT = 0.1


class SentimentAnalyzer:
    def __init__(self):
        self.labels_dict = {
            0: "Neutral",
            1: "Positive",
            2: "Negative",
        }
        self.max_length = MAX_LENGTH
        self.word_vectors = Embeddings().load_vector()

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the trained model
        self.model = self._load_model()

    def _load_model(self) -> torch.nn.Module:
        """
        Loads the trained LSTM model from the given path.

        Returns:
            torch.nn.Module: Loaded LSTM model.
        """
        model = SentimentTransformer(
            D_MODEL, NUM_HEADS, NUM_LAYERS, NUM_CLASSES, D_FF, MAX_LENGTH, DROPOUT
        ).to(self.device)

        sentiment_model_file = os.path.join(
            os.path.dirname(__file__),
            "..",
            "ml_models",
            "sentiment_transformer_model.pth",
        )

        model.load_state_dict(
            torch.load(
                sentiment_model_file, map_location=self.device, weights_only=True
            )
        )
        model.eval()

        return model

    def _preprocess_review(self, text: str) -> torch.Tensor:
        """
        Preprocesses and converts the input text to a tensor for the model.

        Args:
            text (str): Input text for sentiment analysis.

        Returns:
            torch.Tensor: Preprocessed text as a tensor.
        """
        preprocessed_text = preprocess(text)
        tokens = preprocessed_text.split()

        embeddings = [
            (
                self.word_vectors[token]
                if token in self.word_vectors
                else np.zeros(EMBEDDING_DIM)
            )
            for token in tokens
        ]

        if len(embeddings) > MAX_LENGTH:
            embeddings = embeddings[:MAX_LENGTH]
        else:
            embeddings += [np.zeros(EMBEDDING_DIM)] * (MAX_LENGTH - len(embeddings))

        return (
            torch.tensor(np.array(embeddings), dtype=torch.float32)
            .unsqueeze(0)
            .to(self.device)
        )

    def predict_sentiment(self, text: str) -> dict:
        """
        Predicts the sentiment of the given text.

        Args:
            text (str): Input text for sentiment analysis.

        Returns:
            dict: Predicted sentiment label and its probability.
        """
        review_tensor = self._preprocess_review(text)
        output = self.model(review_tensor)

        probabilities = torch.softmax(output, dim=1)
        predicted_label = torch.argmax(probabilities, dim=1).item()
        probability = probabilities[0, predicted_label].item()

        return {
            self.labels_dict.get(predicted_label, "Unknown"),
            probability,
        }


if __name__ == "__main__":
    # Instantiate the analyzer
    analyzer = SentimentAnalyzer()

    # Predict sentiment for a sample sentence
    sample_sentence = "यो फोन खराब छ"
    result = analyzer.predict_sentiment(sample_sentence)

    print(f"Sentence: {result['text']}")
    print(f"Predicted Sentiment: {result['predicted_sentiment']}")
    print(f"Probability: {result['probability']}")
