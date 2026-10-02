"""The Retrieval modes: a Keyword-only index needs no embedding model, a Hybrid one does.

Driven through the Knowledge index with the fake Model gateway, like the other index tests.
"""
import pytest

from local_rag import Access, IndexFormat, IndexReadOnly, KnowledgeIndex, Settings
from local_rag.fakes import FakeModelGateway
from local_rag.gateway_litellm import GatewayError
from local_rag.settings import RetrievalMode
from conftest import SYNONYMS
from index_states import record_format
from test_index_format import after_an_upgrade

KEYWORD = RetrievalMode.KEYWORD
HYBRID = RetrievalMode.HYBRID


class NoEmbeddingModel(FakeModelGateway):
    """A gateway whose embedding model does not exist: any attempt to use it is a failure."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attempts = 0

    def embed(self, texts):
        self.attempts += 1
        raise AssertionError("an embedding model was used")

    def embed_query(self, text):
        self.attempts += 1
        raise AssertionError("an embedding model was used")


def keyword_index(gateway=None, client=None, **settings):
    return KnowledgeIndex(Settings(retrieval=KEYWORD, **settings), gateway, client=client)


def test_a_keyword_only_index_is_synced_and_searched_with_no_gateway(corpus):
    index = keyword_index()

    report = index.sync(corpus)

    assert (report.added, report.failed) == (3, 0)
    found = index.search("ORA-00942")
    assert found and found[0].document_id == "default:dba/ora-00942"


def test_a_keyword_only_index_never_asks_the_gateway_to_embed(corpus):
    gateway = NoEmbeddingModel()
    index = keyword_index(gateway)

    index.sync(corpus)
    index.search("battery")
    index.rebuild([corpus])
    index.status()

    assert gateway.attempts == 0


# --- a mode that disagrees with the index ----------------------------------------------

def reopened(index, client, gateway=None, **changed):
    """The same Qdrant, opened by code whose settings differ from what wrote the index."""
    return after_an_upgrade(index, client, gateway, **changed)


def test_a_keyword_only_index_under_hybrid_settings_is_read_only_and_searched_by_keywords(
        corpus, client):
    keyword = keyword_index(client=client)
    keyword.sync(corpus)
    gateway = NoEmbeddingModel()

    hybrid = reopened(keyword, client, gateway, retrieval=HYBRID)

    status = hybrid.status()
    assert status.access == Access.READ_ONLY
    assert any("retrieval mode" in warning for warning in status.warnings)
    assert hybrid.search("ORA-00942")[0].document_id == "default:dba/ora-00942"
    with pytest.raises(IndexReadOnly):
        hybrid.sync(corpus)
    assert gateway.attempts == 0


def test_a_hybrid_index_under_keyword_settings_is_read_only_and_searched_by_keywords(
        index, corpus, client):
    index.sync(corpus)

    keyword = reopened(index, client, None, retrieval=KEYWORD)

    status = keyword.status()
    assert status.access == Access.READ_ONLY
    assert any("retrieval mode" in warning for warning in status.warnings)
    assert keyword.search("ORA-00942")[0].document_id == "default:dba/ora-00942"
    with pytest.raises(IndexReadOnly):
        keyword.sync(corpus)


def test_a_rebuild_brings_an_index_to_the_mode_the_code_runs_in(index, corpus, client):
    index.sync(corpus)
    keyword = reopened(index, client, None, retrieval=KEYWORD)

    assert keyword.rebuild([corpus]).promoted

    status = keyword.status()
    assert status.access == Access.READ_WRITE and status.warnings == ()
    assert status.format.retrieval == KEYWORD
    assert (status.format.embed_model, status.format.dimension) == (None, None)
    assert keyword.sync(corpus).unchanged == 3

    hybrid = reopened(index, client, FakeModelGateway(synonyms=SYNONYMS), retrieval=HYBRID)
    assert hybrid.status().access == Access.READ_ONLY
    assert hybrid.rebuild([corpus]).promoted
    status = hybrid.status()
    assert status.access == Access.READ_WRITE and status.warnings == ()
    assert (status.format.retrieval, status.format.dimension) == (HYBRID, 64)


def test_a_hybrid_index_records_its_mode_and_a_keyword_only_one_no_model(index, corpus):
    index.sync(corpus)
    keyword = keyword_index()
    keyword.sync(corpus)

    assert index.status().format.retrieval == HYBRID
    assert (index.status().format.embed_model, index.status().format.dimension) == ("embed", 64)
    assert keyword.status().format.retrieval == KEYWORD
    assert (keyword.status().format.embed_model, keyword.status().format.dimension) == (None, None)


# --- the Keyword fallback --------------------------------------------------------------

class FlakyGateway(FakeModelGateway):
    """Embeds until `down` is set, then fails the way an unreachable gateway does."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.down = False
        self.query_attempts = 0

    def embed_query(self, text):
        self.query_attempts += 1
        if self.down:
            raise GatewayError("gateway http://gateway.example.test/v1/embeddings unreachable")
        return super().embed_query(text)


@pytest.fixture
def flaky():
    return FlakyGateway(synonyms=SYNONYMS)


@pytest.fixture
def hybrid(flaky, corpus, client):
    index = KnowledgeIndex(Settings(), flaky, client=client)
    index.sync(corpus)
    return index


def test_a_search_with_the_embedding_model_up_carries_no_notice(hybrid):
    assert hybrid.search("ORA-00942").notice is None


def test_a_search_falls_back_to_keywords_and_says_so_when_the_embedding_call_fails(
        hybrid, flaky):
    hybrid.search("ORA-00942")  # the dimension check passes while the gateway is up
    flaky.down = True

    results = hybrid.search("ORA-00942")

    assert results[0].document_id == "default:dba/ora-00942"
    assert "keyword" in results.notice and "embedding" in results.notice


def test_a_fallback_does_not_stop_the_next_search_using_the_embedding_model(hybrid, flaky):
    hybrid.search("battery")
    flaky.down = True
    assert hybrid.search("battery").notice
    flaky.down = False

    # "vehicle" shares no word with the Document: only the dense side finds it.
    results = hybrid.search("vehicle ignition")

    assert results.notice is None
    assert results[0].document_id == "default:garage/battery"


def test_a_gateway_that_is_down_from_the_start_is_asked_once_per_search(hybrid, flaky, client):
    flaky.down = True
    restarted = KnowledgeIndex(Settings(), flaky, client=client)
    flaky.query_attempts = 0

    results = restarted.search("ORA-00942")

    assert results.notice and results[0].document_id == "default:dba/ora-00942"
    assert flaky.query_attempts == 1


def test_a_keyword_only_index_under_hybrid_settings_tells_the_caller(corpus, client):
    keyword = keyword_index(client=client)
    keyword.sync(corpus)

    results = reopened(keyword, client, NoEmbeddingModel(), retrieval=HYBRID).search("ORA-00942")

    assert "keyword" in results.notice and "Rebuild" in results.notice


def test_a_keyword_only_index_that_was_asked_for_carries_no_notice(corpus):
    index = keyword_index()
    index.sync(corpus)

    assert index.search("ORA-00942").notice is None


def test_an_unreachable_embedding_model_is_a_health_note_not_a_failure(hybrid, flaky):
    hybrid.search("battery")
    assert hybrid.health().notes == ()
    flaky.down = True
    hybrid.search("battery")

    health = hybrid.health()

    assert health.ok
    assert any("embedding" in note for note in health.notes)
    flaky.down = False
    hybrid.search("battery")
    assert hybrid.health().notes == ()


# --- records that disagree with the collection ------------------------------------------

def test_a_recorded_mode_that_disagrees_with_the_collection_is_reported_not_trusted(corpus, client):
    keyword = keyword_index(client=client)
    keyword.sync(corpus)
    generation = keyword.status().generation
    record_format(client, generation, IndexFormat.written_by(Settings(retrieval=HYBRID), 64))

    status = reopened(keyword, client).status()

    assert status.access == Access.REFUSED
    assert any("dense" in warning for warning in status.warnings)

