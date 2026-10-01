from dataclasses import replace

import pytest

from local_rag import Settings
from local_rag.environment import SettingsError, settings_from_environment
from local_rag.settings import RetrievalMode
from local_rag.sparse import SparseParams

EMBEDDING = {"RAG_EMBED_MODEL": "embed",  # Hybrid retrieval, the default, needs these named
             "LITELLM_BASE_URL": "http://gateway.example.test:4000/v1"}
BGE = {**EMBEDDING, "RAG_EMBED_MODEL": "bge"}


def test_an_environment_naming_only_the_embedding_model_and_the_gateway_gives_the_defaults():
    settings = settings_from_environment(EMBEDDING)

    assert replace(settings, liteparse_bin="lit") == Settings(
        qdrant_location="http://127.0.0.1:6333", embed_model="embed",
        gateway_base_url="http://gateway.example.test:4000/v1")
    assert settings.retrieval is RetrievalMode.HYBRID


def test_hybrid_retrieval_is_the_default_and_needs_an_embedding_model_named():
    with pytest.raises(SettingsError, match="RAG_EMBED_MODEL") as error:
        settings_from_environment({})

    assert "RAG_RETRIEVAL=keyword" in str(error.value)  # the other way out
    with pytest.raises(SettingsError, match="RAG_EMBED_MODEL"):
        settings_from_environment({"RAG_RETRIEVAL": "hybrid"})


def test_keyword_retrieval_needs_no_embedding_model():
    settings = settings_from_environment({"RAG_RETRIEVAL": "keyword"})

    assert settings.retrieval is RetrievalMode.KEYWORD


def test_keyword_retrieval_still_takes_an_embedding_model_name_for_a_later_switch():
    settings = settings_from_environment({"RAG_RETRIEVAL": " Keyword ", "RAG_EMBED_MODEL": "bge"})

    assert (settings.retrieval, settings.embed_model) == (RetrievalMode.KEYWORD, "bge")


def test_a_retrieval_mode_that_is_neither_names_its_variable():
    with pytest.raises(SettingsError, match="RAG_RETRIEVAL") as error:
        settings_from_environment({"RAG_RETRIEVAL": "semantic", "RAG_EMBED_MODEL": "bge"})

    assert "hybrid" in str(error.value) and "keyword" in str(error.value)


def test_the_embedding_retry_settings_come_from_the_environment():
    settings = settings_from_environment({
        **BGE, "RAG_EMBED_ATTEMPTS": "5", "RAG_EMBED_RETRY_BACKOFF": "2"})

    assert (settings.embed_attempts, settings.embed_retry_backoff) == (5, 2.0)


def test_no_embedding_attempt_is_an_error():
    with pytest.raises(SettingsError, match="embed_attempts"):
        settings_from_environment({**BGE, "RAG_EMBED_ATTEMPTS": "0"})


def test_each_variable_reaches_its_setting():
    settings = settings_from_environment({
        "QDRANT_URL": "http://qdrant.test:6333", "QDRANT_API_KEY": "qk",
        "RAG_COLLECTION": "wiki", "RAG_TOP_K": "9", "RAG_HYBRID_CANDIDATES": "50",
        "RAG_MAX_PER_DOC": "3", "LITELLM_BASE_URL": "http://gw.test:4000/v1",
        "LITELLM_API_KEY": "gk", "RAG_EMBED_MODEL": "bge", "RAG_EMBED_BATCH": "8",
        "RAG_GATEWAY_TIMEOUT": "5", "RAG_RERANK_MODEL": "rr", "LIT_BIN": "/opt/lit",
        "RAG_LIT_TIMEOUT": "30", "RAG_OCR": "1", "RAG_CHUNK_TOKENS": "300",
        "RAG_CHUNK_OVERLAP": "30", "RAG_SYNC_LOCK_STALE_AFTER": "60",
        "RAG_REBUILD_FAILURE_THRESHOLD": "0.1", "RAG_BM25_K1": "1.5", "RAG_BM25_B": "0.5",
        "RAG_BM25_AVG_DL": "300", "RAG_RETRIEVAL": "hybrid", "RAG_EMBED_ATTEMPTS": "4",
        "RAG_EMBED_RETRY_BACKOFF": "0.25"})

    assert settings == Settings(
        qdrant_location="http://qdrant.test:6333", qdrant_api_key="qk", collection="wiki",
        sparse=SparseParams(k1=1.5, b=0.5, avg_doc_length=300.0), top_k=9,
        hybrid_candidates=50, gateway_base_url="http://gw.test:4000/v1",
        gateway_api_key="gk", embed_model="bge", rerank_model="rr", embed_batch_size=8,
        gateway_timeout=5.0, liteparse_bin="/opt/lit", liteparse_timeout=30.0, ocr=True,
        chunk_tokens=300, chunk_overlap=30, max_passages_per_document=3, rerank=True,
        sync_lock_stale_after=60.0, rebuild_failure_threshold=0.1, embed_attempts=4,
        embed_retry_backoff=0.25)


def test_empty_values_are_left_at_their_defaults():
    # The earlier .env.example listed variables with nothing after the "=".
    settings = settings_from_environment({**EMBEDDING, "RAG_RERANK_MODEL": "", "RAG_TOP_K": ""})

    assert (settings.rerank, settings.rerank_model, settings.top_k) == (False, None, 6)


@pytest.mark.parametrize("name", ["RAG_TOP_K", "RAG_BM25_K1", "RAG_REBUILD_FAILURE_THRESHOLD"])
def test_a_value_that_is_not_a_number_names_its_variable(name):
    with pytest.raises(SettingsError, match=name):
        settings_from_environment({**EMBEDDING, name: "lots"})


def test_a_value_the_settings_refuse_is_a_settings_error():
    with pytest.raises(SettingsError, match="rebuild_failure_threshold"):
        settings_from_environment({**EMBEDDING, "RAG_REBUILD_FAILURE_THRESHOLD": "1.5"})


@pytest.mark.parametrize("given, complaint", [
    ({"RAG_CHUNK_OVERLAP": "400"}, "chunk_overlap"),
    ({"RAG_CHUNK_TOKENS": "50"}, "chunk_overlap"),  # the default overlap no longer fits
    ({"RAG_CHUNK_TOKENS": "0", "RAG_CHUNK_OVERLAP": "0"}, "chunk_tokens"),
    ({"RAG_CHUNK_OVERLAP": "-1"}, "chunk_overlap"),
])
def test_chunk_settings_that_cannot_split_text_are_a_settings_error(given, complaint):
    with pytest.raises(SettingsError, match=complaint):
        settings_from_environment({**EMBEDDING, **given})
