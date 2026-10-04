from collections.abc import Sequence
from typing import Protocol


class ModelGateway(Protocol):
    """The network-local service through which embeddings are reached."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...

    def count_tokens(self, text: str) -> int: ...

    def rerank(self, query: str, texts: Sequence[str]) -> list[float]: ...
