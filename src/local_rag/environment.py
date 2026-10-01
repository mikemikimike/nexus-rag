"""Settings from environment variables: how the adapters (CLI, MCP server) configure the core.

Only variables that are set and non-empty override a default, so the defaults live in
Settings alone, except Qdrant's location: a command line that kept nothing because it
defaulted to in-process memory would be worse than useless, so the default is the local
Qdrant server. The variable names are the ones operators already know from earlier installations.
"""
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

from local_rag.settings import RetrievalMode, Settings
from local_rag.sparse import SparseParams


DEFAULT_QDRANT_URL = "http://127.0.0.1:6333"


class SettingsError(ValueError):
    """An environment variable holds something that cannot be used."""


def settings_from_environment(environ: Mapping[str, str], *,
                              embedding_required: bool = True) -> Settings:
    """The Settings the environment asks for; raises SettingsError for an unusable value.

    Hybrid retrieval, the default, needs an embedding model named. A caller that never
    embeds (the acceptance suite's checks of a host's units) passes ``embedding_required=False``.
    """
    def text(name: str) -> str | None:
        return environ.get(name) or None

    def number[T: (int, float)](name: str, kind: Callable[[str], T]) -> T | None:
        value = text(name)
        if value is None:
            return None
        try:
            return kind(value)
        except ValueError:
            raise SettingsError(f"{name} must be a number, got {value!r}") from None

    sparse = _given(k1=number("RAG_BM25_K1", float), b=number("RAG_BM25_B", float),
                    avg_doc_length=number("RAG_BM25_AVG_DL", float))
    rerank_model = text("RAG_RERANK_MODEL")
    retrieval = _retrieval_mode(text("RAG_RETRIEVAL"))
    if embedding_required and retrieval is RetrievalMode.HYBRID \
            and text("RAG_EMBED_MODEL") is None:
        # Settings would fill a default name in, and a default is no embedding model of ours.
        raise SettingsError(
            "RAG_RETRIEVAL is hybrid but RAG_EMBED_MODEL is not set: name the embedding model "
            "registered in the Model gateway, or set RAG_RETRIEVAL=keyword to run without one")
    given = _given(
        qdrant_location=text("QDRANT_URL") or DEFAULT_QDRANT_URL,
        qdrant_api_key=text("QDRANT_API_KEY"),
        collection=text("RAG_COLLECTION"),
        sparse=SparseParams(**sparse) if sparse else None,
        top_k=number("RAG_TOP_K", int),
        hybrid_candidates=number("RAG_HYBRID_CANDIDATES", int),
        max_passages_per_document=number("RAG_MAX_PER_DOC", int),
        gateway_base_url=text("LITELLM_BASE_URL"),
        gateway_api_key=text("LITELLM_API_KEY"),
        retrieval=retrieval,
        embed_model=text("RAG_EMBED_MODEL"),
        embed_batch_size=number("RAG_EMBED_BATCH", int),
        embed_attempts=number("RAG_EMBED_ATTEMPTS", int),
        embed_retry_backoff=number("RAG_EMBED_RETRY_BACKOFF", float),
        gateway_timeout=number("RAG_GATEWAY_TIMEOUT", float),
        rerank_model=rerank_model,
        rerank=True if rerank_model else None,  # naming a rerank model turns rerank on
        liteparse_bin=text("LIT_BIN") or _sibling_liteparse(),
        liteparse_timeout=number("RAG_LIT_TIMEOUT", float),
        ocr=(True if text("RAG_OCR") == "1" else None),
        chunk_tokens=number("RAG_CHUNK_TOKENS", int),
        chunk_overlap=number("RAG_CHUNK_OVERLAP", int),
        sync_lock_stale_after=number("RAG_SYNC_LOCK_STALE_AFTER", float),
        rebuild_failure_threshold=number("RAG_REBUILD_FAILURE_THRESHOLD", float),
    )
    try:
        return Settings(**given)
    except ValueError as error:
        raise SettingsError(str(error)) from None


def _retrieval_mode(value: str | None) -> RetrievalMode:
    if value is None:
        return RetrievalMode.HYBRID
    try:
        return RetrievalMode(value.strip().lower())
    except ValueError:
        raise SettingsError(
            f"RAG_RETRIEVAL must be hybrid or keyword, got {value!r}") from None


def _given(**values) -> dict:
    return {name: value for name, value in values.items() if value is not None}


def _sibling_liteparse() -> str | None:
    """The `lit` installed beside this Python, which a service's PATH does not include."""
    sibling = Path(sys.executable).parent / "lit"
    return str(sibling) if sibling.exists() else None
