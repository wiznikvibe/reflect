"""EmbeddingGemma wrapper. The real model loads on first embed, not at import."""

from __future__ import annotations

import hashlib
import math
import os
from typing import Protocol

DEFAULT_MODEL_ID = "google/embeddinggemma-300m"
# Retrieval prompts from the EmbeddingGemma model card. Applied inside the
# wrapper so catalog text stays `name | category | styles | material | description`.
QUERY_PROMPT = "task: search result | query: "
DOCUMENT_PROMPT = "title: none | text: "


class Embedder(Protocol):
    """Object that can embed catalog documents and search queries."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per document text."""

    def embed_query(self, text: str) -> list[float]:
        """Return one vector for a search query."""


class FakeEmbedder:
    """Deterministic unit-test embedder. Does not touch the network."""

    def __init__(self, dim: int = 32) -> None:
        if dim < 1:
            raise ValueError("dim must be positive")
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values: list[float] = []
        while len(values) < self.dim:
            for byte in digest:
                values.append((byte / 127.5) - 1.0)
                if len(values) == self.dim:
                    break
            if len(values) == self.dim:
                break
            digest = hashlib.sha256(digest).digest()
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return [value / norm for value in values]


class EmbeddingGemmaEmbedder:
    """Lazy local wrapper around ``google/embeddinggemma-300m``.

    ``transformers`` must be >= 4.57 so Gemma 3 uses bidirectional attention.
    Older releases silently emit the wrong vectors.
    """

    def __init__(self, model_id: str | None = None, model: object | None = None) -> None:
        if model_id is None:
            model_id = os.environ.get("EMBEDDING_MODEL", DEFAULT_MODEL_ID)
        self.model_id = model_id
        self._model = model

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _load(self) -> object:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_id)
        return self._model

    def _encode(self, texts: list[str], prompt: str) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        vectors = model.encode(
            list(texts),
            prompt=prompt,
            normalize_embeddings=True,
        )
        return [[float(value) for value in row] for row in vectors]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(list(texts), DOCUMENT_PROMPT)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text], QUERY_PROMPT)[0]


_CACHED_EMBEDDER: EmbeddingGemmaEmbedder | None = None


def get_embedder() -> EmbeddingGemmaEmbedder:
    """Return a shared embedder. The weights stay unloaded until the first embed."""

    global _CACHED_EMBEDDER
    if _CACHED_EMBEDDER is None:
        _CACHED_EMBEDDER = EmbeddingGemmaEmbedder()
    return _CACHED_EMBEDDER
