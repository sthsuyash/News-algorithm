from pydantic import BaseModel
from typing import Any, Optional


class ResponseModel(BaseModel):
    status_code: int
    message: str
    data: Optional[Any] = None
    error: Optional[Any] = None

    class Config:
        # Ensuring that all fields are converted to JSON-compatible format
        json_encoders = {
            # If you have custom objects, you can specify how to encode them.
        }
