import os
import secrets
from typing import Annotated, Any, Literal

from pydantic import (
    AnyUrl,
    BeforeValidator,
    computed_field,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


def parse_cors(v: Any) -> list[str] | str:
    if isinstance(v, str) and not v.startswith("["):
        return [i.strip() for i in v.split(",")]
    elif isinstance(v, list | str):
        return v
    raise ValueError(v)


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
    SECRET_KEY: str = secrets.token_urlsafe(32)
    # 60 minutes * 24 hours * 8 days = 8 days
    FRONTEND_HOST: str = "http://localhost:5005"
    ENVIRONMENT: Literal["local", "staging", "production"] = "local"

    BACKEND_CORS_ORIGINS: Annotated[
        list[AnyUrl] | str, 
        BeforeValidator(parse_cors)
    ] = []

    @computed_field 
    @property
    def all_cors_origins(self) -> list[str]:
        return [
            str(origin).rstrip("/") 
            for origin in self.BACKEND_CORS_ORIGINS
        ] + [self.FRONTEND_HOST]


settings = Settings() 
