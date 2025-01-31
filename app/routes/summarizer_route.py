from fastapi import APIRouter
from pydantic import BaseModel

from app.core.response_model import ResponseModel
from app.core.logging import setup_logging
from app.algorithms.summarizer import Summarizer

router = APIRouter()
logger = setup_logging()
summarizer = Summarizer()


class SummarizationRequest(BaseModel):
    text: str
    summary_length_ratio: float = 0.01


@router.post("/summarize", response_model=ResponseModel)
async def summarize(request: SummarizationRequest):
    """
    Endpoint to summarize the given text.

    Args:
        request (SummarizationRequest): The request body containing the text and optional summary length ratio.

    Returns:
        ResponseModel: The response model containing status, message and the summary.
    """
    try:
        text = request.text
        summary_length_ratio = request.summary_length_ratio

        summary = summarizer.show_summary(
            text=text, 
            length_sentence_predict=summary_length_ratio
        )

        return ResponseModel(
            status_code=200,
            message="Summary generated successfully.",
            data={"summary": summary},
            error=None,
        )

    except Exception as e:
        logger.error(f"An error occurred while generating the summary: {e}")

        return ResponseModel(
            status_code=500,
            message="An error occurred while generating the summary.",
            data=None,
            error=str(e),
        )
