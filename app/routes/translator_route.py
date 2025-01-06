from fastapi import APIRouter
from pydantic import BaseModel

from app.core.model import ResponseModel
from app.core.logging import setup_logging
from app.algorithms.translator import translate_text

router = APIRouter()
logger = setup_logging()


class TranslationRequest(BaseModel):
    text: str
    source_language: str = "ne"
    target_language: str = "en"


@router.post("/translate", response_model=ResponseModel)
async def translate(request: TranslationRequest):
    """
    Endpoint to translate the given text to the target language.

    Args:
        request (TranslationRequest): The request body containing the text and target language.

    Returns:
        ResponseModel: The response model containing status, message, and the translated text.
    """
    try:
        text = request.text
        source_language = request.source_language
        target_language = request.target_language

        translated_text = translate_text(text, source_language, target_language)

        return ResponseModel(
            status_code=200,
            message="Translation successful.",
            data={"translated_text": translated_text},
            error=None,
        )

    except Exception as e:
        logger.error(
            f"An error occurred while translating the text from {source_language} to {target_language}: {e}"
        )

        return ResponseModel(
            status_code=500,
            message="An error occurred while translating the text.",
            data=None,
            error=str(e),
        )
