from .utils import (
    remove_html_tags,
    remove_extra_whitespace,
    remove_special_characters
)

stopwords = None
with open('text_summarizer/nepali_stopwords.txt', 'r', encoding='utf-8') as f:
    stopwords = f.read().splitlines()


def preprocess(text: str) -> list:
    """
    Preprocess the input text by removing HTML tags, extra whitespaces, normalizing characters,
    and splitting into sentences.

    Args:
        text (str): The input raw text to be preprocessed.

    Returns:
        list: A list of preprocessed sentences with normalized characters and tokenized words.
    """
    # Step 1: Remove HTML tags and extra whitespaces
    text = remove_html_tags(text)
    text = remove_extra_whitespace(text)

    # Step 2: Split the text into sentences based on the Nepali punctuation '।' and filter out empty sentences
    sentences = [
        sentence.strip()
        for sentence in text.split(u"।")
        if sentence.strip()
    ]

    # Step 3: Basic letter normalization mapping
    letters_to_normalize = {
        "ी": "ि",
        "ू": "ु",
        "श": "स",
        "ष": "स",
        "व": "ब",
        "ङ": "न",
        "ञ": "न",
        "ण": "न",
        "ृ": "र",
        "ँ": "",
        "ं": "",
        "ः": "",
        "ं": ""  # Duplicate key removed
    }

    # Step 4: Process each sentence
    processed_sentences = []
    for sentence in sentences:
        # Remove special characters and tokenize the sentence
        sentence = remove_special_characters(sentence)
        tokens = sentence.split()

        # Remove stopwords from tokens
        tokens = [word for word in tokens if word not in stopwords]

        # Normalize letters in tokens
        tokens = [letters_to_normalize.get(word, word) for word in tokens]

        # Join tokens back into a processed sentence
        processed_sentence = " ".join(tokens)
        processed_sentences.append(processed_sentence)

    return processed_sentences
