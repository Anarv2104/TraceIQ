"""Embedding backend with caching."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class SentenceTransformerEmbedder:
    """Embedder using sentence-transformers with LRU caching."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        max_content_length: int = 512,
        cache_size: int = 10000,
    ) -> None:
        self.model_name = model_name
        self.max_content_length = max_content_length
        self._model = None

        # Create cached embed function with specified size
        self._cached_embed = lru_cache(maxsize=cache_size)(self._embed_uncached)

    def _load_model(self) -> None:
        """Lazy-load the sentence-transformers model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is required for embedding. "
                    "Install it with: pip install sentence-transformers"
                ) from e
            self._model = SentenceTransformer(self.model_name)

    def _content_hash(self, content: str) -> str:
        """Generate a hash key for content."""
        return hashlib.sha256(content.encode()).hexdigest()

    def _embed_uncached(self, content_hash: str, content: str) -> tuple[float, ...]:
        """Embed content without caching (returns tuple for hashability)."""
        self._load_model()
        truncated = content[: self.max_content_length]
        embedding = self._model.encode(truncated, convert_to_numpy=True)
        return tuple(embedding.tolist())

    def embed(self, content: str) -> NDArray[np.float32]:
        """Embed content with caching."""
        content_hash = self._content_hash(content)
        embedding_tuple = self._cached_embed(content_hash, content)
        return np.array(embedding_tuple, dtype=np.float32)

    def embed_batch(self, contents: list[str]) -> list[NDArray[np.float32]]:
        """Embed multiple contents."""
        return [self.embed(content) for content in contents]

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cached_embed.cache_clear()

    @property
    def cache_info(self) -> dict:
        """Get cache statistics."""
        info = self._cached_embed.cache_info()
        return {
            "hits": info.hits,
            "misses": info.misses,
            "size": info.currsize,
            "maxsize": info.maxsize,
        }


class MockEmbedder:
    """Mock embedder for testing without sentence-transformers."""

    def __init__(
        self,
        embedding_dim: int = 384,
        seed: int | None = None,
    ) -> None:
        self.embedding_dim = embedding_dim
        self._rng = np.random.default_rng(seed)
        self._cache: dict[str, NDArray[np.float32]] = {}

    def _content_hash(self, content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()

    def embed(self, content: str) -> NDArray[np.float32]:
        """Generate deterministic pseudo-random embedding based on content hash."""
        content_hash = self._content_hash(content)
        if content_hash not in self._cache:
            # Use hash to seed a local RNG for deterministic output
            hash_int = int(content_hash[:16], 16)
            local_rng = np.random.default_rng(hash_int)
            embedding = local_rng.random(self.embedding_dim).astype(np.float32)
            # Normalize to unit vector
            embedding = embedding / np.linalg.norm(embedding)
            self._cache[content_hash] = embedding
        return self._cache[content_hash]

    def embed_batch(self, contents: list[str]) -> list[NDArray[np.float32]]:
        return [self.embed(content) for content in contents]

    def clear_cache(self) -> None:
        self._cache.clear()

    @property
    def cache_info(self) -> dict:
        return {
            "hits": 0,
            "misses": 0,
            "size": len(self._cache),
            "maxsize": None,
        }
