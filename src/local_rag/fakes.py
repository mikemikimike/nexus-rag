"""Deterministic stand-ins for external services, for tests and local experiments."""

import math
import zlib
from collections.abc import Iterable, Mapping, Sequence

from local_rag.sparse import tokenize
from local_rag.tokens import count_tokens


class FakeModelGateway:
    """Bag-of-words embeddings hashed into a fixed number of dimensions.

    Words listed together in a synonym group share one dimension, so passages and
    queries with no words in common can still be close.
    """

    def __init__(self, dimension: int = 64, synonyms: Mapping[str, Iterable[str]] | None = None):
        self.dimension = dimension
        self._canonical = {
            word: concept for concept, words in (synonyms or {}).items() for word in words
        }

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def count_tokens(self, text: str) -> int:
        return count_tokens(text)

    def rerank(self, query: str, texts: Sequence[str]) -> list[float]:
        """Relevance is the share of the query's words that appear in the text."""
        wanted = set(tokenize(query))
        return [len(wanted & set(tokenize(text))) / (len(wanted) or 1) for text in texts]

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in tokenize(text):
            concept = self._canonical.get(token, token)
            vector[zlib.crc32(concept.encode("utf-8")) % self.dimension] += 1.0
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]
