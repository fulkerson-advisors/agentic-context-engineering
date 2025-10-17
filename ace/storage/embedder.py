"""
Text embedding implementations using the OpenAI API.
"""

from __future__ import annotations

from typing import List, Optional
import logging
import os

from openai import OpenAI

from ace.core.interfaces import Embedder
from ace.config import get_openai_model

logger = logging.getLogger(__name__)


class OpenAIEmbedder(Embedder):
    """
    OpenAI embeddings-based implementation.
    
    Defaults to the model provided via ``OPENAI_EMBEDDING_MODEL`` or
    ``OPENAI_MODEL`` environment variables (falls back to
    ``text-embedding-3-small`` if unset).
    """

    def __init__(
        self,
        client: Optional[OpenAI] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize embedder.

        Args:
            client: Optional pre-configured OpenAI client.
            model: Embedding model name. If ``None`` the value is resolved from
                environment variables.
        """
        self.client = client or OpenAI()
        resolved_model = model or get_openai_model(
            default="text-embedding-3-small",
            env_var="OPENAI_EMBEDDING_MODEL",
        )

        # If we inherited the general OPENAI_MODEL and it doesn't look like an
        # embedding model, fall back to a sensible default.
        if (
            model is None
            and os.getenv("OPENAI_EMBEDDING_MODEL") is None
            and "embedding" not in resolved_model
        ):
            logger.warning(
                "Model '%s' is not an embedding model; falling back to 'text-embedding-3-small'. "
                "Set OPENAI_EMBEDDING_MODEL to override.",
                resolved_model,
            )
            resolved_model = "text-embedding-3-small"
        self.model = resolved_model
        model_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }
        self._dimension: Optional[int] = model_dimensions.get(resolved_model)

    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        response = self.client.embeddings.create(
            model=self.model,
            input=[text],
        )
        vector = response.data[0].embedding
        self._set_dimension_if_needed(vector)
        return vector

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        vectors = [item.embedding for item in response.data]
        if vectors:
            self._set_dimension_if_needed(vectors[0])
        return vectors

    def dimension(self) -> int:
        """Return embedding dimensionality."""
        if self._dimension is None:
            self._fetch_dimension()
        return self._dimension or 0
    
    def _set_dimension_if_needed(self, vector: List[float]) -> None:
        if self._dimension is None:
            self._dimension = len(vector)

    def _fetch_dimension(self) -> None:
        """Fetch embedding dimensionality by making a minimal request."""
        response = self.client.embeddings.create(
            model=self.model,
            input=["dimension probe"],
        )
        if response.data:
            self._dimension = len(response.data[0].embedding)
