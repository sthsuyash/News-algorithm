import os
import joblib
from app.helpers.preprocess import preprocess


class CategoryPredictor:
    """
    A class for predicting text categories using a trained model.
    """

    def __init__(self):
        """
        Initializes the CategoryPredictor with the model and mapping.
        """
        self.model_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "ml_models",
            "Logistic_Regression_classifier_model.pkl",
        )

        self.prediction_map = {
            0: "ArthaBanijya",
            1: "Bichar",
            2: "Desh",
            3: "Khelkud",
            4: "Manoranjan",
            5: "Prabas",
            6: "Sahitya",
            7: "SuchanaPrabidhi",
            8: "Swasthya",
            9: "Viswa",
        }

        self.model = self._load_model()

    def _load_model(self):
        """
        Loads the trained model from the specified path.

        Returns:
            object: The loaded model.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        return joblib.load(self.model_path)

    def predict(self, text: str):
        """
        Predicts the category for a given text.

        Args:
            text (str): The input text to classify.

        Returns:
            tuple: The predicted category name and its numeric label.
        """
        # Preprocess the input text (same as during training)
        preprocessed_text = preprocess(text)

        # Ensure the text is in the format expected by the model (e.g., a single string)
        if isinstance(preprocessed_text, list):
            preprocessed_text = " ".join(preprocessed_text)

        # Predict the category using the model
        prediction = self.model.predict([preprocessed_text])

        # Map the numeric prediction to the corresponding category name
        predicted_label = int(prediction[0])  # Numeric label
        predicted_category = self.prediction_map.get(predicted_label, "Unknown")

        return predicted_category, predicted_label
