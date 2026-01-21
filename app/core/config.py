import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings


def _load_env_file() -> None:
    env_file = os.getenv("BACKEND_ENV_FILE")
    if env_file and os.path.exists(env_file):
        load_dotenv(env_file, override=False)
        return

    default_env = os.path.join(os.getcwd(), ".env")
    if os.path.exists(default_env):
        load_dotenv(default_env, override=False)


class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    BACKEND_HOST: str = "127.0.0.1"
    BACKEND_PORT: int = 8000
    # Add other settings here

    class Config:
        case_sensitive = True


_load_env_file()
settings = Settings()
