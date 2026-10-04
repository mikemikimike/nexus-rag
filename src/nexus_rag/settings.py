from dataclasses import dataclass, field
from enum import StrEnum

from nexus_rag.sparse import SparseParams


class RetrievalMode(StrEnum):
    HYBRID = "hybrid"  # dense vectors from an embedding model, and keywords
    KEYWORD = "keyword"  # keywords alone: needs no embedding model


@dataclass(frozen=True)
class Settings:
    """Everything the Knowledge index needs. Adapters build this from their environment."""

    qdrant_location: str = ":memory:"  # ":memory:" (in-process) or a Qdrant server URL
    qdrant_api_key: str | None = field(default=None, repr=False)
    collection: str = "knowledge"
    sparse: SparseParams = field(default_factory=SparseParams)
    retrieval: RetrievalMode = RetrievalMode.HYBRID  # what a new index is built to search by
    top_k: int = 6
    hybrid_candidates: int = 40
    gateway_base_url: str | None = (
        None  # LiteLLM, OpenAI-compatible root; None until an operator names it
    )
    gateway_api_key: str | None = field(default=None, repr=False)  # bearer key for the gateway
    embed_model: str = "embed"  # model name registered in the gateway for embeddings
    query_instruction: str = ""  # goes before a query embedding ("Instruct: ...\nQuery: ..."); empty leaves queries as they are; not part of the Index format
    rerank_model: str | None = None  # model name for /rerank; None leaves rerank off
    embed_batch_size: int = 16  # texts per /embeddings request
    embed_attempts: int = (
        3  # tries at one Document's embeddings in a Sync or Rebuild; search never retries
    )
    embed_retry_backoff: float = 0.5  # seconds before the second try, doubling for each later one
    gateway_timeout: float = 60.0  # seconds per gateway request
    liteparse_bin: str = "lit"  # LiteParse command that reads PDF, Office and image files
    liteparse_timeout: float = 300.0  # seconds LiteParse may take per file
    ocr: bool = False  # let LiteParse run OCR (needs OCR language data installed)
    chunk_tokens: int = 400  # target Passage size
    chunk_overlap: int = 60  # tokens shared between neighbouring Passages
    code_chunk_tokens: int = 800  # target size of a code Passage, which is split by lines
    max_passages_per_document: int = 2  # most Passages one Document may add to a result
    rerank: bool = False  # reorder results through the Model gateway's rerank
    sync_lock_stale_after: float = (
        120.0  # seconds without a heartbeat before a crashed Sync's lock is taken over
    )
    rebuild_failure_threshold: float = (
        0.0  # share of Documents (0 to 1) that may fail for a Rebuild to go live
    )

    @property
    def needs_model_gateway(self) -> bool:
        """Whether this installation reaches the Model gateway at all: Hybrid retrieval does, and
        so does a reranker; Keyword-only retrieval without one does not."""
        return self.retrieval is RetrievalMode.HYBRID or self.rerank

    def __post_init__(self):
        object.__setattr__(self, "retrieval", RetrievalMode(self.retrieval))
        if self.chunk_tokens < 1:
            raise ValueError(f"chunk_tokens must be at least 1, got {self.chunk_tokens!r}")
        if not 0 <= self.chunk_overlap < self.chunk_tokens:
            raise ValueError(
                f"chunk_overlap must be at least 0 and smaller than chunk_tokens "
                f"({self.chunk_tokens}), got {self.chunk_overlap!r}"
            )
        if self.code_chunk_tokens < 1:
            raise ValueError(
                f"code_chunk_tokens must be at least 1, got {self.code_chunk_tokens!r}"
            )
        if self.embed_attempts < 1:
            raise ValueError(f"embed_attempts must be at least 1, got {self.embed_attempts!r}")
        if self.embed_retry_backoff < 0:
            raise ValueError(
                f"embed_retry_backoff must not be negative, got {self.embed_retry_backoff!r}"
            )
        if not 0.0 <= self.rebuild_failure_threshold <= 1.0:  # also false for NaN
            raise ValueError(
                "rebuild_failure_threshold must be between 0 and 1, "
                f"got {self.rebuild_failure_threshold!r}"
            )
