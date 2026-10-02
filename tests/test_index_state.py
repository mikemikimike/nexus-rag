"""Corrupt or unreadable index state is reported, never treated as an empty index."""
import pytest

from local_rag import Access, CorruptIndex, KnowledgeIndex, Settings
from local_rag.corpus import InMemoryCorpus
from conftest import DOCUMENTS
from index_states import (
    collections, lose_the_collection, lose_the_registry, overwrite_registry, registry_payload,
    replace_with_a_non_index)
from test_sync import CountingGateway


def lose_the_live_collection(index, client):
    """What losing the Live index's data in Qdrant leaves behind."""
    lose_the_collection(client, index.status().generation)


def assert_reported_not_empty(index, client, gateway):
    embedded, before = len(gateway.embedded), collections(client)
    status = index.status()
    assert status.access == Access.REFUSED
    assert status.documents is None and status.passages is None
    assert status.warnings and "Rebuild" in status.warnings[0]
    for use in (lambda: index.search("ORA-00942"), index.list_spaces,
                lambda: index.get_document("default:dba/ora-00942"),
                lambda: index.sync(InMemoryCorpus(DOCUMENTS))):
        with pytest.raises(CorruptIndex, match="Rebuild"):
            use()
    assert len(gateway.embedded) == embedded  # no silent re-embed of everything
    assert collections(client) == before


@pytest.fixture
def gateway():
    return CountingGateway()


@pytest.fixture
def two_generations(gateway, client):
    index = KnowledgeIndex(Settings(), gateway, client=client)
    index.rebuild([InMemoryCorpus(DOCUMENTS)])
    index.rebuild([InMemoryCorpus(DOCUMENTS)])
    return index


def test_a_live_index_whose_data_is_gone_is_reported_not_treated_as_empty(
        two_generations, client, gateway):
    lose_the_live_collection(two_generations, client)

    assert_reported_not_empty(two_generations, client, gateway)


def test_a_live_index_whose_data_is_gone_is_reported_even_with_no_previous_index(
        index, client, gateway):
    index.sync(InMemoryCorpus(DOCUMENTS))
    lose_the_live_collection(index, client)

    assert_reported_not_empty(index, client, gateway)


def test_a_lost_generation_registry_is_reported_not_read_as_an_earlier_version(
        two_generations, client, gateway):
    lose_the_registry(client)

    assert_reported_not_empty(two_generations, client, gateway)


@pytest.mark.parametrize("damage", [
    lambda payload: {**payload, "formats": {n: {"garbage": 1} for n in payload["formats"]}},
    lambda payload: {"unexpected": True},
    lambda payload: {**payload, "generations": "not a list"},
    lambda payload: {**payload, "live": 42},
])
def test_an_unreadable_generation_registry_is_reported(two_generations, client, gateway, damage):
    overwrite_registry(client, damage(registry_payload(client)))

    assert_reported_not_empty(two_generations, client, gateway)


def test_a_collection_that_is_not_an_index_is_reported(two_generations, client, gateway):
    replace_with_a_non_index(client, two_generations.status().generation)

    reopened = KnowledgeIndex(Settings(), gateway, client=client)
    assert_reported_not_empty(reopened, client, gateway)


@pytest.mark.parametrize("damage", [
    lose_the_live_collection,
    lambda index, client: overwrite_registry(client, {"unexpected": True}),
    lambda index, client: lose_the_registry(client),
])
def test_a_rebuild_repairs_corrupt_state(two_generations, client, damage):
    damage(two_generations, client)

    report = two_generations.rebuild([InMemoryCorpus(DOCUMENTS)])

    assert report.promoted
    assert two_generations.status().access == Access.READ_WRITE
    assert two_generations.search("ORA-00942")[0].document_id == "default:dba/ora-00942"
    assert two_generations.sync(InMemoryCorpus(DOCUMENTS)).unchanged == 3


def test_a_rebuild_over_an_unreadable_registry_says_so(two_generations, client):
    overwrite_registry(client, {"unexpected": True})

    report = two_generations.rebuild([InMemoryCorpus(DOCUMENTS)])

    assert report.promoted
    assert any("unreadable" in warning for warning in report.warnings)
    assert "unreadable" in two_generations.status().last_sync.detail


def test_a_rebuild_over_a_healthy_registry_has_nothing_to_warn_about(two_generations):
    assert two_generations.rebuild([InMemoryCorpus(DOCUMENTS)]).warnings == ()
