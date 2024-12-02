import pickle
from preprocess import preprocess_text

classes = {
    'Agriculture': 0,
    'automobiles': 1,
    'bank': 2,
    'business': 3,
    'economy': 4,
    'education': 5,
    'entertainment': 6,
    'health': 7,
    'politic': 8,
    'sports': 9,
    'technology': 10,
    'tourism': 11,
    'world': 12
}


def load_model_and_vectorizer(model_file, vectorizer_file):
    model = pickle.load(open(model_file, 'rb'))
    vectorizer = pickle.load(open(vectorizer_file, 'rb'))
    return model, vectorizer


def predict_category(text, model_file='news_pred_model.pickle', vectorizer_file='news_pred_vectorizer.pickle', stop_file="nepali_stopwords.txt", punctuation_file="nepali_punctuation.txt"):
    # Load stop words and punctuation words
    # stop words is in pre/stop_file
    # punctuation words is in pre/punctuation

    stop_words_file = 'pre/' + stop_file
    punctuation_file = 'pre/' + punctuation_file

    stop_words = []
    with open(stop_words_file, encoding='utf-8') as fp:
        lines = fp.readlines()
        stop_words = list(map(lambda x: x.strip(), lines))

    punctuation_words = []
    with open(punctuation_file, encoding='utf-8') as fp:
        lines = fp.readlines()
        punctuation_words = list(map(lambda x: x.strip(), lines))

    # Load the model and vectorizer
    model_file = 'models' + '/' + model_file
    vectorizer_file = 'models' + '/' + vectorizer_file
    model, vectorizer = load_model_and_vectorizer(model_file, vectorizer_file)

    # Preprocess the text
    preprocessed_text = preprocess_text([text], stop_words, punctuation_words)

    # Transform the preprocessed text using the loaded vectorizer
    transformed_text = vectorizer.transform(preprocessed_text)

    # Predict the category
    predicted_category = model.predict(transformed_text)

    nprd = []

    # Map the predicted category to the class name
    for k, v in classes.items():
        for p in predicted_category:
            if v == p:
                nprd.append(k)

    return nprd[0]
