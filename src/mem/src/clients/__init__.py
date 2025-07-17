"""
External service client module

Contains:
- llm_client: LLM client
- embedding_client: Embedding client
"""

from .llm_client import LLMClient
from .embedding_client import EmbeddingClient

__all__ = [
    "LLMClient",
    "EmbeddingClient"
]
