import re

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
