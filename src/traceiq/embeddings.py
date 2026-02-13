"""Embedding backend with caching."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class EmbeddingResult:
    """Result of an embedding operation, including truncation info."""

    def __init__(
        self, embedding: NDArray[np.float32], was_truncated: bool = False
    ) -> None:
        self.embedding = embedding
        self.was_truncated = was_truncated


class SentenceTransformerEmbedder:
    """Embedder using sentence-transformers with caching."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        max_content_length: int = 5000,
        cache_size: int = 10000,
    ) -> None:
        self.model_name = model_name
        self.max_content_length = max_content_length
        self._cache_size = cache_size
        self._model = None

        # Cache: hash -> (embedding, was_truncated)
        self._cache: dict[str, tuple[NDArray[np.float32], bool]] = {}
        self._cache_hits = 0
        self._cache_misses = 0

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

    def _content_key(self, content: str) -> tuple[str, str, bool]:
        """Generate cache key from truncated content.

        Returns (hash_key, truncated_content, was_truncated).
        """
        was_truncated = len(content) > self.max_content_length
        truncated = content[: self.max_content_length]
        hash_key = hashlib.sha256(truncated.encode("utf-8")).hexdigest()
        return hash_key, truncated, was_truncated

    def embed(self, content: str) -> NDArray[np.float32]:
        """Embed content with caching."""
        result = self.embed_with_info(content)
        return result.embedding

    def embed_with_info(self, content: str) -> EmbeddingResult:
        """Embed content with caching, returning truncation info."""
        hash_key, truncated, was_truncated = self._content_key(content)

        if hash_key in self._cache:
            self._cache_hits += 1
            cached_emb, _ = self._cache[hash_key]
            return EmbeddingResult(cached_emb.copy(), was_truncated)

        self._cache_misses += 1
        self._load_model()
        embedding = self._model.encode(truncated, convert_to_numpy=True)
        embedding = embedding.astype(np.float32)

        # Manage cache size (simple eviction: remove oldest entries)
        if len(self._cache) >= self._cache_size:
            # Remove first entry (oldest)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[hash_key] = (embedding, was_truncated)
        return EmbeddingResult(embedding.copy(), was_truncated)

    def embed_batch(self, contents: list[str]) -> list[NDArray[np.float32]]:
        """Embed multiple contents using real batch encoding for uncached items."""
        results = self.embed_batch_with_info(contents)
        return [r.embedding for r in results]

    def embed_batch_with_info(self, contents: list[str]) -> list[EmbeddingResult]:
        """Embed multiple contents, returning truncation info for each."""
        results: list[EmbeddingResult | None] = [None] * len(contents)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []
        uncached_truncated: list[bool] = []

        # Check cache first
        for i, content in enumerate(contents):
            hash_key, truncated, was_truncated = self._content_key(content)
            if hash_key in self._cache:
                self._cache_hits += 1
                cached_emb, _ = self._cache[hash_key]
                results[i] = EmbeddingResult(cached_emb.copy(), was_truncated)
            else:
                uncached_indices.append(i)
                uncached_texts.append(truncated)
                uncached_truncated.append(was_truncated)

        # Batch encode uncached items
        if uncached_texts:
            self._cache_misses += len(uncached_texts)
            self._load_model()
            embeddings = self._model.encode(uncached_texts, convert_to_numpy=True)
            if len(uncached_texts) == 1:
                embeddings = [embeddings]

            for idx, emb, truncated_text, was_truncated in zip(
                uncached_indices,
                embeddings,
                uncached_texts,
                uncached_truncated,
                strict=True,
            ):
                emb = emb.astype(np.float32)
                hash_key = hashlib.sha256(truncated_text.encode("utf-8")).hexdigest()

                # Manage cache size
                if len(self._cache) >= self._cache_size:
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]

                self._cache[hash_key] = (emb, was_truncated)
                results[idx] = EmbeddingResult(emb.copy(), was_truncated)

        return results  # type: ignore[return-value]

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    @property
    def cache_info(self) -> dict:
        """Get cache statistics."""
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "size": len(self._cache),
            "maxsize": self._cache_size,
        }


class MockEmbedder:
    """Mock embedder for testing without sentence-transformers."""

    def __init__(
        self,
        embedding_dim: int = 384,
        seed: int | None = None,
        max_content_length: int = 5000,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.max_content_length = max_content_length
        self._rng = np.random.default_rng(seed)
        self._cache: dict[str, NDArray[np.float32]] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def _content_key(self, content: str) -> tuple[str, bool]:
        """Generate cache key from truncated content."""
        was_truncated = len(content) > self.max_content_length
        truncated = content[: self.max_content_length]
        hash_key = hashlib.sha256(truncated.encode("utf-8")).hexdigest()
        return hash_key, was_truncated

    def embed(self, content: str) -> NDArray[np.float32]:
        """Generate deterministic pseudo-random embedding based on content hash."""
        result = self.embed_with_info(content)
        return result.embedding

    def embed_with_info(self, content: str) -> EmbeddingResult:
        """Embed content, returning truncation info."""
        hash_key, was_truncated = self._content_key(content)
        if hash_key in self._cache:
            self._cache_hits += 1
            return EmbeddingResult(self._cache[hash_key].copy(), was_truncated)

        self._cache_misses += 1
        # Use hash to seed a local RNG for deterministic output
        hash_int = int(hash_key[:16], 16)
        local_rng = np.random.default_rng(hash_int)
        embedding = local_rng.random(self.embedding_dim).astype(np.float32)
        # Normalize to unit vector
        embedding = embedding / np.linalg.norm(embedding)
        self._cache[hash_key] = embedding
        return EmbeddingResult(embedding.copy(), was_truncated)

    def embed_batch(self, contents: list[str]) -> list[NDArray[np.float32]]:
        """Embed multiple contents."""
        results = self.embed_batch_with_info(contents)
        return [r.embedding for r in results]

    def embed_batch_with_info(self, contents: list[str]) -> list[EmbeddingResult]:
        """Embed multiple contents, returning truncation info for each."""
        return [self.embed_with_info(content) for content in contents]

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    @property
    def cache_info(self) -> dict:
        """Get cache statistics."""
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "size": len(self._cache),
            "maxsize": None,
        }
