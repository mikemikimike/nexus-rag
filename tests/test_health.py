from local_rag import KnowledgeIndex, Settings, index_format
from local_rag.corpus import InMemoryCorpus
from local_rag.fakes import FakeModelGateway
from conftest import DOCUMENTS
from test_index_format import after_an_upgrade
from test_index_state import lose_the_live_collection


class UnreachableQdrant:
    """Stands in for a Qdrant client whose server cannot be reached."""

    def __getattr__(self, name):
        def fail(*args, **kwargs):
            raise ConnectionError("connection refused")
        return fail


def test_an_empty_index_is_healthy(index):
    health = index.health()

    assert (health.qdrant, health.knowledge_index, health.ok) == (True, True, True)
    assert health.notes == ()


def test_a_synced_index_is_healthy(index):
    index.sync(InMemoryCorpus(DOCUMENTS))

    assert index.health().ok


def test_an_unreachable_qdrant_is_reported_not_raised(gateway):
    index = KnowledgeIndex(Settings(), gateway, client=UnreachableQdrant())

    health = index.health()

    assert (health.qdrant, health.knowledge_index, health.ok) == (False, False, False)
    assert "Qdrant" in health.notes[0] and "connection refused" in health.notes[0]


def test_a_corrupt_index_is_reported_unreachable_while_qdrant_is_up(index):
    index.sync(InMemoryCorpus(DOCUMENTS))
    index.rebuild([InMemoryCorpus(DOCUMENTS)])
    lose_the_live_collection(index)

    health = index.health()

    assert (health.qdrant, health.knowledge_index, health.ok) == (True, False, False)
    assert "Rebuild" in health.notes[0]


def test_a_read_only_index_is_still_reachable_and_says_why(index):
    index.sync(InMemoryCorpus(DOCUMENTS))

    health = after_an_upgrade(index, embed_model="embed-v2").health()

    assert (health.qdrant, health.knowledge_index) == (True, True)
    assert any("embedding model" in note for note in health.notes)


def test_a_refused_index_is_not_reachable(index, monkeypatch):
    index.sync(InMemoryCorpus(DOCUMENTS))
    monkeypatch.setattr(index_format, "SCHEMA_VERSION", 2)

    health = index.health()

    assert (health.qdrant, health.knowledge_index) == (True, False)


def test_health_never_calls_the_model_gateway(index):
    class CountingQueries(FakeModelGateway):
        calls = 0

        def embed_query(self, text):
            CountingQueries.calls += 1
            return super().embed_query(text)

    index.sync(InMemoryCorpus(DOCUMENTS))
    upgraded = after_an_upgrade(index, CountingQueries(), embed_model="embed-v2")

    health = upgraded.health()

    assert health.ok and CountingQueries.calls == 0
