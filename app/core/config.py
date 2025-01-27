import os
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    ENV_FILE_LOCATION: str = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    )
    model_config = SettingsConfigDict(
        env_file=ENV_FILE_LOCATION,
        env_ignore_empty=True,
        extra="ignore",
    )

    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: Literal["local", "staging", "production"] = "local"


settings = Settings()
