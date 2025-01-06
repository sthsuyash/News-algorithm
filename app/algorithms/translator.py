from deep_translator import GoogleTranslator


def translate_text(text, source_lang="ne", target_lang="en"):
    """
    Translates text from Nepali to English.
    Args:
    - text (str): Text to translate.
    - source_lang (str): Source language code (Nepali).
    - target_lang (str): Target language code (English).

    Returns:
    - str: Translated text.
    """
    try:
        return GoogleTranslator(source=source_lang, target=target_lang).translate(text)
    except Exception as e:
        print(f"Error translating text: {e}")
        return text
