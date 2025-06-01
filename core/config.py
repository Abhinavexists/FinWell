from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    google_api_key: Optional[str] = Field(default=None, env="GOOGLE_API_KEY")
    default_model: str = Field(default="gemini-2.0-flash", env="DEFAULT_MODEL")

    finnhub_api_key: Optional[str] = Field(default=None, env="FINNHUB_API_KEY")
    timeout_seconds: int = 30
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
