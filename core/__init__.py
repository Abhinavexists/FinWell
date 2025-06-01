"""Core package for financial AI agents."""

from .config import settings
from .gemini import get_gemini

__all__ = ["settings", "get_gemini"]
