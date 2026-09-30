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
    chunk_tokens: int = 400  # target Passage size
    chunk_overlap: int = 60  # tokens shared between neighbouring Passages
