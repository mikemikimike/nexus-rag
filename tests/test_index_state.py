"""Corrupt or unreadable index state is reported, never treated as an empty index."""
import pytest
from qdrant_client import models

from local_rag import Access, CorruptIndex, KnowledgeIndex, Settings
from local_rag.corpus import InMemoryCorpus
from conftest import DOCUMENTS
from test_rebuild import collections
from test_sync import CountingGateway


def registry_payload(index) -> dict:
    return index._client.retrieve("knowledge.generations", ids=[1])[0].payload


def overwrite_registry(index, payload):
    index._client.upsert("knowledge.generations", points=[
        models.PointStruct(id=1, vector=[1.0], payload=payload)], wait=True)


def lose_the_live_collection(index):
    """What losing the Live index's data in Qdrant leaves behind."""
    index._client.delete_collection(index.status().generation)


def assert_reported_not_empty(index, gateway):
    embedded, before = len(gateway.embedded), collections(index)
    status = index.status()
    assert status.access == Access.REFUSED
    assert status.documents is None and status.passages is None
    assert status.warnings and "Rebuild" in status.warnings[0]
    for use in (lambda: index.search("ORA-00942"), index.list_spaces,
                lambda: index.get_document("dba/ora-00942"),
                lambda: index.sync(InMemoryCorpus(DOCUMENTS))):
        with pytest.raises(CorruptIndex, match="Rebuild"):
            use()
    assert len(gateway.embedded) == embedded  # no silent re-embed of everything
    assert collections(index) == before


@pytest.fixture
def gateway():
    return CountingGateway()


@pytest.fixture
def two_generations(gateway):
    index = KnowledgeIndex(Settings(), gateway)
    index.rebuild(InMemoryCorpus(DOCUMENTS))
    index.rebuild(InMemoryCorpus(DOCUMENTS))
    return index


def test_a_live_index_whose_data_is_gone_is_reported_not_treated_as_empty(
        two_generations, gateway):
    lose_the_live_collection(two_generations)

    assert_reported_not_empty(two_generations, gateway)


def test_a_live_index_whose_data_is_gone_is_reported_even_with_no_previous_index(gateway):
    index = KnowledgeIndex(Settings(), gateway)
    index.sync(InMemoryCorpus(DOCUMENTS))
    lose_the_live_collection(index)

    assert_reported_not_empty(index, gateway)


def test_a_lost_generation_registry_is_reported_not_read_as_an_earlier_version(
        two_generations, gateway):
    two_generations._client.delete_collection("knowledge.generations")

    assert_reported_not_empty(two_generations, gateway)


@pytest.mark.parametrize("damage", [
    lambda payload: {**payload, "formats": {n: {"garbage": 1} for n in payload["formats"]}},
    lambda payload: {"unexpected": True},
    lambda payload: {**payload, "generations": "not a list"},
    lambda payload: {**payload, "live": 42},
])
def test_an_unreadable_generation_registry_is_reported(two_generations, gateway, damage):
    overwrite_registry(two_generations, damage(registry_payload(two_generations)))

    assert_reported_not_empty(two_generations, gateway)


def test_a_collection_that_is_not_an_index_is_reported(two_generations, gateway):
    live = two_generations.status().generation
    two_generations._client.delete_collection(live)
    two_generations._client.create_collection(
        live, vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE))
    two_generations._client.update_collection_aliases(change_aliases_operations=[
        models.CreateAliasOperation(create_alias=models.CreateAlias(
            collection_name=live, alias_name="knowledge.live"))])

    reopened = KnowledgeIndex(Settings(), gateway, client=two_generations._client)
    assert_reported_not_empty(reopened, gateway)


@pytest.mark.parametrize("damage", [
    lose_the_live_collection,
    lambda index: overwrite_registry(index, {"unexpected": True}),
    lambda index: index._client.delete_collection("knowledge.generations"),
])
def test_a_rebuild_repairs_corrupt_state(two_generations, damage):
    damage(two_generations)

    report = two_generations.rebuild(InMemoryCorpus(DOCUMENTS))

    assert report.promoted
    assert two_generations.status().access == Access.READ_WRITE
    assert two_generations.search("ORA-00942")[0].document_id == "dba/ora-00942"
    assert two_generations.sync(InMemoryCorpus(DOCUMENTS)).unchanged == 3


def test_a_rebuild_over_an_unreadable_registry_says_so(two_generations):
    overwrite_registry(two_generations, {"unexpected": True})

    report = two_generations.rebuild(InMemoryCorpus(DOCUMENTS))

    assert report.promoted
    assert any("unreadable" in warning for warning in report.warnings)
    assert "unreadable" in two_generations.status().last_sync.detail


def test_a_rebuild_over_a_healthy_registry_has_nothing_to_warn_about(two_generations):
    assert two_generations.rebuild(InMemoryCorpus(DOCUMENTS)).warnings == ()
