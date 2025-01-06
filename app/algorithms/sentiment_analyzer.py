import torch
import joblib
import numpy as np
from app.ml_models import sentiment_analysis
from app.helpers.preprocess import preprocess


class SentimentAnalyzer:
    """
    A class for performing sentiment analysis using an LSTM model.
    """

    def __init__(
        self, 
        model_file: str, 
        word2vec_model, 
        tokenizer, 
        max_length=32, 
        device=None
    ):
        """
        Initializes the SentimentAnalyzer with model parameters and utilities.

        Args:
            model_file (str): Path to the trained LSTM model file.
            word2vec_model: Pre-trained word2vec model for embeddings.
            tokenizer: Tokenizer for text preprocessing.
            max_length (int): Maximum length of the review vector.
            device (str): Device to run the model ('cuda' or 'cpu').
        """
        self.labels_dict = {
            0: "Neutral",
            1: "Positive",
            2: "Negative",
        }
        self.max_length = max_length
        self.word2vec_model = word2vec_model
        self.tokenizer = tokenizer

        # Device configuration
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load the trained model
        self.model = self._load_model(model_file)

    def _load_model(self, model_file):
        """
        Loads the trained LSTM model from the given path.

        Args:
            model_file (str): Path to the trained LSTM model file.

        Returns:
            torch.nn.Module: Loaded LSTM model.
        """
        input_dim = 300  # Embedding dimension
        hidden_dim = 128  # Number of LSTM units
        output_dim = 1  # Number of classes (0 or 1)
        num_layers = 2  # Number of LSTM layers

        model = sentiment_analysis.LSTMBinaryModel(
            input_dim, hidden_dim, output_dim, num_layers
        ).to(self.device)

        model.load_state_dict(
            torch.load(model_file, map_location=self.device, weights_only=True)
        )
        model.eval()
        return model

    def preprocess_review(self, text: str) -> torch.Tensor:
        """
        Preprocesses and converts the input text to a tensor for the model.

        Args:
            text (str): Input text for sentiment analysis.

        Returns:
            torch.Tensor: Preprocessed text as a tensor.
        """
        preprocessed_text = preprocess(text)
        tokens = self.tokenizer.tokenize(preprocessed_text)

        review_vector = []
        for token in tokens:
            if token in self.word2vec_model:
                review_vector.append(self.word2vec_model[token])
            else:
                review_vector.append(np.zeros(self.word2vec_model.vector_size))

        # Pad or truncate to max_length
        if len(review_vector) > self.max_length:
            review_vector = review_vector[: self.max_length]
        else:
            review_vector.extend(
                [np.zeros(self.word2vec_model.vector_size)]
                * (self.max_length - len(review_vector))
            )

        return (
            torch.tensor(review_vector, dtype=torch.float32)
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
        review_tensor = self.preprocess_review(text)
        output = self.model(review_tensor)

        predicted_label = int((output > 0.5).item())
        probability = torch.sigmoid(output).item()

        return {
            "text": text,
            "predicted_sentiment": self.labels_dict.get(predicted_label, "Unknown"),
            "probability": probability,
        }


# Example usage
if __name__ == "__main__":
    # Load necessary components
    model_file = "src/models/outputs/lstm_binary_model.pth"
    word2vec_model = joblib.load("path/to/word2vec_model.pkl")
    tokenizer = joblib.load("path/to/tokenizer.pkl")

    # Instantiate the analyzer
    analyzer = SentimentAnalyzer(
        model_file=model_file,
        word2vec_model=word2vec_model,
        tokenizer=tokenizer,
        max_length=32,
    )

    # Predict sentiment for a sample sentence
    sample_sentence = "यो फोन खराब छ"
    result = analyzer.predict_sentiment(sample_sentence)

    print(f"Sentence: {result['text']}")
    print(f"Predicted Sentiment: {result['predicted_sentiment']}")
    print(f"Probability: {result['probability']}")
