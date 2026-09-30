from dataclasses import dataclass, field

from local_rag.sparse import SparseParams


@dataclass(frozen=True)
class Settings:
    """Everything the Knowledge index needs. Adapters build this from their environment."""

    qdrant_location: str = ":memory:"  # ":memory:" (in-process) or a Qdrant server URL
    qdrant_api_key: str | None = field(default=None, repr=False)
    collection: str = "knowledge"
    sparse: SparseParams = field(default_factory=SparseParams)
    top_k: int = 6
    hybrid_candidates: int = 40
    gateway_base_url: str = "http://localhost:4000/v1"  # LiteLLM, OpenAI-compatible root
    gateway_api_key: str | None = field(default=None, repr=False)  # bearer key for the gateway
    embed_model: str = "embed"  # model name registered in the gateway for embeddings
    rerank_model: str | None = None  # model name for /rerank; None leaves rerank off
    embed_batch_size: int = 16  # texts per /embeddings request
    gateway_timeout: float = 60.0  # seconds per gateway request
    liteparse_bin: str = "lit"  # LiteParse command that reads PDF, Office and image files
    liteparse_timeout: float = 300.0  # seconds LiteParse may take per file
    ocr: bool = False  # let LiteParse run OCR (needs OCR language data installed)
    chunk_tokens: int = 400  # target Passage size
    chunk_overlap: int = 60  # tokens shared between neighbouring Passages
    max_passages_per_document: int = 2  # most Passages one Document may add to a result
    rerank: bool = False  # reorder results through the Model gateway's rerank
    sync_lock_stale_after: float = 120.0  # seconds without a heartbeat before a crashed Sync's lock is taken over
