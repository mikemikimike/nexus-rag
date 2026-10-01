from dataclasses import replace

import pytest

from local_rag import Settings
from local_rag.environment import SettingsError, settings_from_environment
from local_rag.sparse import SparseParams


def test_an_empty_environment_gives_the_defaults():
    settings = settings_from_environment({})

    assert replace(settings, liteparse_bin="lit") == Settings(
        qdrant_location="http://127.0.0.1:6333")


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
        "RAG_BM25_AVG_DL": "300"})

    assert settings == Settings(
        qdrant_location="http://qdrant.test:6333", qdrant_api_key="qk", collection="wiki",
        sparse=SparseParams(k1=1.5, b=0.5, avg_doc_length=300.0), top_k=9,
        hybrid_candidates=50, gateway_base_url="http://gw.test:4000/v1",
        gateway_api_key="gk", embed_model="bge", rerank_model="rr", embed_batch_size=8,
        gateway_timeout=5.0, liteparse_bin="/opt/lit", liteparse_timeout=30.0, ocr=True,
        chunk_tokens=300, chunk_overlap=30, max_passages_per_document=3, rerank=True,
        sync_lock_stale_after=60.0, rebuild_failure_threshold=0.1)


def test_empty_values_are_left_at_their_defaults():
    # The earlier .env.example listed variables with nothing after the "=".
    settings = settings_from_environment({"RAG_RERANK_MODEL": "", "RAG_TOP_K": ""})

    assert (settings.rerank, settings.rerank_model, settings.top_k) == (False, None, 6)


@pytest.mark.parametrize("name", ["RAG_TOP_K", "RAG_BM25_K1", "RAG_REBUILD_FAILURE_THRESHOLD"])
def test_a_value_that_is_not_a_number_names_its_variable(name):
    with pytest.raises(SettingsError, match=name):
        settings_from_environment({name: "lots"})


def test_a_value_the_settings_refuse_is_a_settings_error():
    with pytest.raises(SettingsError, match="rebuild_failure_threshold"):
        settings_from_environment({"RAG_REBUILD_FAILURE_THRESHOLD": "1.5"})


@pytest.mark.parametrize("given, complaint", [
    ({"RAG_CHUNK_OVERLAP": "400"}, "chunk_overlap"),
    ({"RAG_CHUNK_TOKENS": "50"}, "chunk_overlap"),  # the default overlap no longer fits
    ({"RAG_CHUNK_TOKENS": "0", "RAG_CHUNK_OVERLAP": "0"}, "chunk_tokens"),
    ({"RAG_CHUNK_OVERLAP": "-1"}, "chunk_overlap"),
])
def test_chunk_settings_that_cannot_split_text_are_a_settings_error(given, complaint):
    with pytest.raises(SettingsError, match=complaint):
        settings_from_environment(given)
