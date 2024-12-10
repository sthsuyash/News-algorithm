import os
from nepalikit.preprocessing import TextProcessor
from nepalikit.tokenization import Tokenizer
import re
# import nltk
# nltk.download("stopwords")
# from nltk.corpus import stopwords

# negative_stop_words = {
#     'न', 'नजिकै', 'नत्र', 'नयाँ', 'नै', 'निम्न', 'निम्नानुसार', 'बिरुद्ध', 'बाहेक', 'गैर', 'नै', 'भए', 'नै', 'दुई', 'नै', 'न'
# }

# stop_words = set(stopwords.words("nepali")) - negative_stop_words

stop_words = None

with open("dataset/nepali_stopwords.txt", "r", encoding="utf-8") as f:
    stop_words = f.read().splitlines()

tokenizer = Tokenizer()
processor = TextProcessor()


def preprocess_text(text: str, remove_stopwords=True) -> str:
    """
    Preprocesses the input text by performing the following operations:
    1. Removes HTML tags.
    2. Removes extra whitespace, including spaces between words.
    3. Removes special characters.
    4. Tokenizes the text.
    5. Removes Nepali numbers.
    6. Normalizes certain Nepali characters.

    Args:
    - text (str): The input text to preprocess.
    - remove_stopwords (bool): Whether to remove stopwords from the text.

    Returns:
    - str: The preprocessed 'Nepali' text.

    Example:
    >>> preprocess_text("यहाँ    केही  संख्या  छन्  १२३।")
    'यहाँ केही संख्या छन् ।'

    >>> preprocess_text("<p>प्रस्तावित  कोड</p>  मा   ३८  संख्या  छन्।")
    'प्रस्तावित कोड मा संख्या छन् ।'
    """
    # Remove HTML tags
    text = processor.remove_html_tags(text)

    # Remove extra whitespace, including extra spaces between words
    text = processor.remove_extra_whitespace(text)

    # Remove special characters
    text = processor.remove_special_characters(text)

    # Remove Nepali numbers (0-9)
    text = re.sub(r'[०-९]', '', text)

    # Tokenize text into words
    tokens = tokenizer.tokenize(text, level='word')

    # Remove stopwords (currently commented out, needs list of stopwords)
    if remove_stopwords:
        tokens = [word for word in tokens if word not in stop_words]

    # Basic letter normalization for specific Nepali characters
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
    }

    tokens = [letters_to_normalize.get(word, word) for word in tokens]

    # Join the tokens back into a string
    return " ".join(tokens)


# # test the function
# if __name__ == "__main__":

#     print(preprocess_text("यहाँ    केही  संख्या  छन्  १२३।"))
#     print(preprocess_text(
#         "<p>प्रस्तावित  कोड</p>  मा   ३८  संख्या  छन्।", remove_stopwords=False))
