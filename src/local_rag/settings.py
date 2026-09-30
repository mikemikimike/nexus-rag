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
    max_passages_per_document: int = 2  # most Passages one Document may add to a result
    rerank: bool = False  # reorder results through the Model gateway's rerank
    sync_lock_stale_after: float = 120.0  # seconds without a heartbeat before a crashed Sync's lock is taken over
