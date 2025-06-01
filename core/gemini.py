import logging
from crewai import LLM
from core.config import settings

logger = logging.getLogger(__name__)

class Gemini:
    def __init__(self):
        self.gemini_instance = None
        self.initialize_gemini()

    def initialize_gemini(self):
        if not settings.google_api_key:
            logger.warning("Gemini API key not configured")
            return

        try:
            self.gemini_instance = LLM(
                model=settings.default_model,
                api_key=settings.google_api_key
            )
            logger.info("Gemini initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")

    def get_gemini(self) -> LLM:
        if not settings.google_api_key:
            raise ValueError(
                "Gemini API key not configured. Please set GOOGLE_API_KEY in your .env file.\n"
                "Get your API key from: https://makersuite.google.com/app/apikey"
            )
        
        if not self.gemini_instance:
            self.initialize_gemini()

        if self.gemini_instance:
            return self.gemini_instance

        raise ValueError("Gemini not available. Check your API key configuration.")

gemini = Gemini()

def get_gemini() -> LLM:
    return gemini.get_gemini()
