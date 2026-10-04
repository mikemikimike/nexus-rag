"""Whether an installation needs the Model gateway is decided once, and every caller agrees.

Hybrid retrieval needs it, and so does a named rerank model; Keyword-only retrieval without a
reranker needs none. An unnamed gateway is a settings error, never a guessed address.
"""

import itertools

import pytest
from qdrant_client import QdrantClient

from nexus_rag import KnowledgeIndex, Settings
from nexus_rag.environment import SettingsError, settings_from_environment
from nexus_rag.fakes import FakeModelGateway
from nexus_rag.gateway_litellm import LiteLLMGateway, gateway_from_settings
from nexus_rag.settings import RetrievalMode

ADDRESS = "http://gateway.example.test:4000/v1"


def environment(mode: str, rerank: bool, named: bool) -> dict[str, str]:
    return {
        "RAG_RETRIEVAL": mode,
        "RAG_EMBED_MODEL": "embed",
        "RAG_RERANK_MODEL": "rerank-model" if rerank else "",
        "LITELLM_BASE_URL": ADDRESS if named else "",
    }


ROWS = list(itertools.product(["hybrid", "keyword"], [False, True], [False, True]))


@pytest.mark.parametrize("mode, rerank, named", ROWS)
def test_every_caller_agrees_on_whether_the_model_gateway_is_needed(mode, rerank, named):
    needed = mode == "hybrid" or rerank

    if needed and not named:
        with pytest.raises(SettingsError, match="LITELLM_BASE_URL"):
            settings_from_environment(environment(mode, rerank, named))
        # Settings built by hand, past the reader: the other callers still agree.
        settings = Settings(retrieval=RetrievalMode(mode), rerank=rerank)
        assert settings.needs_model_gateway
        with pytest.raises(ValueError):
            gateway_from_settings(settings)
        with pytest.raises(ValueError, match="Model gateway"):
            KnowledgeIndex(settings, None, client=QdrantClient(":memory:"))
        return

    settings = settings_from_environment(environment(mode, rerank, named))

    assert settings.needs_model_gateway is needed
    gateway = gateway_from_settings(settings)
    if needed:
        assert isinstance(gateway, LiteLLMGateway)
    else:
        assert gateway is None
        KnowledgeIndex(settings, None, client=QdrantClient(":memory:"))  # needs none


def test_an_unnamed_gateway_is_refused_with_what_to_set_and_the_other_way_out():
    with pytest.raises(SettingsError) as error:
        settings_from_environment(environment("hybrid", False, False))

    message = str(error.value)
    assert "LITELLM_BASE_URL" in message and "RAG_RETRIEVAL=keyword" in message


def test_a_reranker_without_a_gateway_is_refused_naming_the_reranker():
    with pytest.raises(SettingsError) as error:
        settings_from_environment(environment("keyword", True, False))

    message = str(error.value)
    assert "LITELLM_BASE_URL" in message and "RAG_RERANK_MODEL" in message


def test_an_empty_address_counts_as_not_named():
    with pytest.raises(SettingsError, match="LITELLM_BASE_URL"):
        settings_from_environment({"RAG_EMBED_MODEL": "embed", "LITELLM_BASE_URL": ""})


def test_a_gateway_on_the_same_host_is_named_explicitly():
    settings = settings_from_environment(
        {"RAG_EMBED_MODEL": "embed", "LITELLM_BASE_URL": "http://localhost:4000/v1"}
    )

    assert settings.gateway_base_url == "http://localhost:4000/v1"


def test_a_caller_that_never_reaches_the_gateway_needs_none_named():
    settings = settings_from_environment({"RAG_EMBED_MODEL": "embed"}, gateway_required=False)

    assert settings.gateway_base_url is None


def test_settings_built_by_hand_with_a_gateway_in_hand_name_no_address():
    KnowledgeIndex(Settings(), FakeModelGateway(), client=QdrantClient(":memory:"))
