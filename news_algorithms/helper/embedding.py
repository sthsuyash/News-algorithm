import os
import logging
from gensim.models import KeyedVectors

MODEL_PATH = "models/nepali_embeddings_word2vec.kv"


class Embeddings:
    """
    This class helps to load word embeddings in KeyedVectors format.
    """

    def __init__(self, model_path: str = MODEL_PATH):
        """
        Initializes the Embeddings class.

        Args:
            model_path (str): Path to the pre-trained Word2Vec model.
        """
        self.model_path = model_path
        self.word_vector = None
        self._validate_model_path()

    def _validate_model_path(self):
        """Validate if the model file exists at the specified path."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}")

    def load_vector(self) -> KeyedVectors:
        """
        Loads the word embedding model.

        Returns:
            KeyedVectors: Loaded word vector model.

        Raises:
            FileNotFoundError: If the model file does not exist.
            ValueError: If the model fails to load.
        """
        try:
            logging.info(f"Loading model from {self.model_path}...")
            self.word_vector = KeyedVectors.load(self.model_path)
            logging.info("Model loaded successfully.")
            return self.word_vector
        except FileNotFoundError as e:
            logging.error(f"File not found: {e}")
            raise
        except ValueError as e:
            logging.error(f"Failed to load model: {e}")
            raise

    def save_vector(self, save_path: str):
        """
        Saves the word vector model to a specified location.

        Args:
            save_path (str): Path where the model will be saved.
        """
        if self.word_vector is not None:
            try:
                logging.info(f"Saving model to {save_path}...")
                self.word_vector.save(save_path)
                logging.info("Model saved successfully.")
            except Exception as e:
                logging.error(f"Error saving model: {e}")
                raise
        else:
            logging.warning("No word vector model to save.")

    def __str__(self):
        """
        Returns a string representation of the object.
        """
        return f"Embeddings class with model path: {self.model_path}"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    try:
        embeddings = Embeddings()
        word_vector = embeddings.load_vector()
        if word_vector is not None:
            print(word_vector.most_similar("नेपाल", topn=5))

        else:
            logging.warning("No word vector model loaded.")
    except Exception as e:
        logging.error(f"Error occurred: {e}")
