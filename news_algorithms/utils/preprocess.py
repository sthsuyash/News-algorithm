import re
import os

stopwords_file = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),  # This points to the 'utils' directory
        # Access 'preprocessing_files/nepali_stopwords.txt'
        "preprocessing_files", "nepali_stopwords.txt"
    )
)


# Pre-compiled regex patterns for performance
HTML_TAGS_PATTERN = re.compile(r'<.*?>')
WHITESPACE_PATTERN = re.compile(r'\s+')
SPECIAL_CHARACTERS_PATTERN = re.compile(r'(?<!\d)\.(?!\d)|[^\u0900-\u097F\u0966-\u096F\s.]')
ENGLISH_DIGITS_PATTERN = re.compile(r'[0-9]')


def remove_html_tags(text: str) -> str:
    """
    Remove HTML tags from a string.

    Args:
        text (str): The input text that may contain HTML tags.

    Returns:
        str: The text with HTML tags removed.
    """
    return HTML_TAGS_PATTERN.sub('', text)


def remove_extra_whitespace(text: str) -> str:
    """
    Remove extra whitespaces from a string, replacing multiple spaces with a single space.

    Args:
        text (str): The input text with possible extra whitespaces.

    Returns:
        str: The text with normalized whitespace.
    """
    return WHITESPACE_PATTERN.sub(' ', text).strip()


def remove_special_characters(text: str) -> str:
    """
    Remove special characters, keeping only Nepali digits, Devanagari letters, spaces, and decimal numbers.

    Args:
        text (str): The input text to be processed.

    Returns:
        str: The cleaned text with special characters removed.
    """
    # Remove special characters except Nepali digits, Devanagari letters, whitespace, and decimal points
    text = SPECIAL_CHARACTERS_PATTERN.sub('', text)

    # Remove English digits (0-9)
    text = ENGLISH_DIGITS_PATTERN.sub('', text)

    return text


# Load stopwords with error handling
try:
    with open(stopwords_file, 'r', encoding='utf-8') as f:
        stopwords = f.read().splitlines()
except FileNotFoundError:
    raise RuntimeError(f"Stopwords file not found at: {stopwords_file}")

# Normalization mapping
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
    "ः": ""
}


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

    # Step 3: Process each sentence
    processed_sentences = []
    for sentence in sentences:
        # Remove special characters and tokenize the sentence
        sentence = remove_special_characters(sentence)
        tokens = sentence.split()

        # Normalize letters and remove stopwords in one loop
        tokens = [
            letters_to_normalize.get(word, word)
            for word in tokens
            if word not in stopwords
        ]

        # Join tokens back into a processed sentence
        processed_sentence = " ".join(tokens)
        processed_sentences.append(processed_sentence)

    return processed_sentences
