import os
import joblib
from app.algorithms.helpers.preprocess import preprocess


CLASSIFIER_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "models", "Logistic_Regression_best_model.pkl"
)

prediction_map = {
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


def predict_category_with_label(text: str):
    # Load the trained model
    model = joblib.load(CLASSIFIER_MODEL_PATH)

    # Preprocess the input text (same as during training)
    preprocessed_text = preprocess(text)

    # Ensure the text is in the format expected by the model (e.g., a single string)
    if isinstance(preprocessed_text, list):
        preprocessed_text = " ".join(preprocessed_text)

    # Predict the category using the model
    prediction = model.predict([preprocessed_text])

    # Reverse mapping of numeric labels to category names
    classes = {i: c for i, c in enumerate(model.classes_)}

    # Map the numeric prediction to the corresponding category name
    predicted_category = prediction_map[classes[prediction[0]]]

    # Return the predicted category name and the numeric label
    predicted_label = int(prediction[0])  # Numeric label

    return predicted_category, predicted_label
