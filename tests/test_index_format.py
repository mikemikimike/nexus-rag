import dataclasses

import pytest

from local_rag import Access, IndexFormat, IndexReadOnly, IndexRefused, KnowledgeIndex, Settings, index_format
from local_rag.corpus import InMemoryCorpus
from local_rag.fakes import FakeModelGateway
from local_rag.sparse import SparseParams
from conftest import DOCUMENTS
from test_rebuild import collections, legacy_index
from test_sync import CountingGateway, revised


def after_an_upgrade(index, gateway=None, **changed):
    """The same Qdrant, opened by code whose settings differ from what wrote the index."""
    settings = dataclasses.replace(index._settings, **changed)
    return KnowledgeIndex(settings, gateway or index._gateway, client=index._client)


def test_an_index_written_with_the_same_format_is_served_and_synced_as_usual(index):
    index.sync(InMemoryCorpus(DOCUMENTS))

    same = after_an_upgrade(index)

    assert same.status().access == Access.READ_WRITE
    assert same.sync(InMemoryCorpus(DOCUMENTS)).unchanged == 3


@pytest.mark.parametrize("changed, difference", [
    ({"embed_model": "embed-v2"}, "embedding model"),
    ({"sparse": SparseParams(k1=1.5)}, "sparse encoder"),
    ({"chunk_tokens": 200}, "chunk size"),
    ({"chunk_overlap": 20}, "chunk overlap"),
])
def test_a_differing_but_searchable_index_is_served_read_only_with_a_warning(
        index, changed, difference):
    index.sync(InMemoryCorpus(DOCUMENTS))
    upgraded = after_an_upgrade(index, **changed)

    status = upgraded.status()

    assert status.access == Access.READ_ONLY
    assert any(difference in warning for warning in status.warnings)
    assert any("Rebuild" in warning for warning in status.warnings)
    assert upgraded.search("ORA-00942")[0].document_id == "default:dba/ora-00942"
    assert upgraded.list_spaces() == ["DBA", "GARAGE"]
    assert upgraded.get_document("default:garage/battery") == DOCUMENTS[2].text


def test_a_read_only_index_is_logged_when_first_served(index, caplog):
    index.sync(InMemoryCorpus(DOCUMENTS))
    upgraded = after_an_upgrade(index, embed_model="embed-v2")

    upgraded.search("ORA-00942")
    upgraded.search("ORA-12154")

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1 and "embedding model" in warnings[0].getMessage()


def test_a_read_only_index_refuses_a_sync_and_never_rebuilds_on_its_own():
    gateway = CountingGateway()
    index = KnowledgeIndex(Settings(), gateway)
    index.sync(InMemoryCorpus(DOCUMENTS))
    embedded, before = len(gateway.embedded), collections(index)
    upgraded = after_an_upgrade(index, embed_model="embed-v2")

    with pytest.raises(IndexReadOnly, match="Rebuild"):
        upgraded.sync(InMemoryCorpus([*DOCUMENTS[:2], revised(DOCUMENTS[2], "2")]))
    upgraded.search("ORA-00942")
    upgraded.status()

    assert len(gateway.embedded) == embedded  # nothing re-embedded
    assert {n for n in collections(index) if ".gen-" in n} == \
        {n for n in before if ".gen-" in n}
    assert upgraded.get_document("default:garage/battery") == DOCUMENTS[2].text


def test_a_rebuild_makes_a_read_only_index_current_again(index):
    index.sync(InMemoryCorpus(DOCUMENTS))
    upgraded = after_an_upgrade(index, embed_model="embed-v2")

    report = upgraded.rebuild([InMemoryCorpus(DOCUMENTS)])

    assert report.promoted
    status = upgraded.status()
    assert status.access == Access.READ_WRITE and status.warnings == ()
    assert status.format.embed_model == "embed-v2"
    assert upgraded.sync(InMemoryCorpus(DOCUMENTS)).unchanged == 3
    upgraded.switch_back()  # the old index is still there to go back to
    assert upgraded.status().access == Access.READ_ONLY


def test_an_index_of_another_embedding_dimension_is_refused(index):
    index.sync(InMemoryCorpus(DOCUMENTS))
    upgraded = after_an_upgrade(
        index, FakeModelGateway(dimension=32), embed_model="embed-v2")

    status = upgraded.status()

    assert status.access == Access.REFUSED
    assert any("dimension" in warning and "64" in warning for warning in status.warnings)
    for use in (lambda: upgraded.search("ORA-00942"), upgraded.list_spaces,
                lambda: upgraded.get_document("default:dba/ora-00942"),
                lambda: upgraded.sync(InMemoryCorpus(DOCUMENTS))):
        with pytest.raises(IndexRefused, match="dimension"):
            use()


def test_an_index_of_another_schema_version_is_refused(index, monkeypatch):
    index.sync(InMemoryCorpus(DOCUMENTS))
    monkeypatch.setattr(index_format, "SCHEMA_VERSION", 2)

    assert index.status().access == Access.REFUSED
    with pytest.raises(IndexRefused, match="schema version 1"):
        index.search("ORA-00942")


def test_a_refused_index_is_replaced_by_a_rebuild_not_by_a_sync(index, monkeypatch):
    index.sync(InMemoryCorpus(DOCUMENTS))
    monkeypatch.setattr(index_format, "SCHEMA_VERSION", 2)

    assert index.rebuild([InMemoryCorpus(DOCUMENTS)]).promoted

    assert index.status().access == Access.READ_WRITE
    assert index.search("ORA-00942")[0].document_id == "default:dba/ora-00942"


def test_switching_back_to_a_refused_index_is_itself_refused(index, monkeypatch):
    index.sync(InMemoryCorpus(DOCUMENTS))
    monkeypatch.setattr(index_format, "SCHEMA_VERSION", 2)
    index.rebuild([InMemoryCorpus(DOCUMENTS)])
    live = index.status().generation

    with pytest.raises(IndexRefused, match="previous index"):
        index.switch_back()

    assert index.status().generation == live
    assert index.search("ORA-00942")[0].document_id == "default:dba/ora-00942"


def test_an_index_of_another_dimension_is_refused_even_under_the_same_model_name(index):
    index.sync(InMemoryCorpus(DOCUMENTS))
    upgraded = after_an_upgrade(index, FakeModelGateway(dimension=32))  # same settings

    status = upgraded.status()

    assert status.access == Access.REFUSED
    assert any("dimension" in warning for warning in status.warnings)
    with pytest.raises(IndexRefused, match="dimension"):
        upgraded.search("ORA-00942")


def test_an_index_without_a_recorded_format_is_refused_when_the_dimension_differs(index):
    legacy_index(index, DOCUMENTS)
    upgraded = after_an_upgrade(index, FakeModelGateway(dimension=32))

    with pytest.raises(IndexRefused, match="dimension"):
        upgraded.search("ORA-00942")


def test_an_unreachable_gateway_does_not_hide_a_differing_index_from_status(index):
    class Unreachable(FakeModelGateway):
        def embed_query(self, text):
            raise ConnectionError("gateway down")

    index.sync(InMemoryCorpus(DOCUMENTS))
    upgraded = after_an_upgrade(index, Unreachable(), embed_model="embed-v2")

    status = upgraded.status()

    assert status.access == Access.READ_ONLY
    assert any("could not be checked" in warning for warning in status.warnings)


def test_an_index_built_before_chunks_were_recorded_is_served_read_only_until_a_rebuild(
        index, monkeypatch):
    real = IndexFormat.to_payload

    def without_chunks(self):  # what an earlier version wrote
        return {k: v for k, v in real(self).items() if not k.startswith("chunk")}

    monkeypatch.setattr(IndexFormat, "to_payload", without_chunks)
    index.sync(InMemoryCorpus(DOCUMENTS))
    monkeypatch.setattr(IndexFormat, "to_payload", real)
    upgraded = after_an_upgrade(index)

    status = upgraded.status()

    assert status.access == Access.READ_ONLY
    assert any("chunk" in warning and "not recorded" in warning for warning in status.warnings)
    assert upgraded.search("ORA-00942")[0].document_id == "default:dba/ora-00942"
    with pytest.raises(IndexReadOnly, match="Rebuild"):
        upgraded.sync(InMemoryCorpus(DOCUMENTS))
    assert upgraded.rebuild([InMemoryCorpus(DOCUMENTS)]).promoted
    assert upgraded.status().access == Access.READ_WRITE


def test_changed_chunk_settings_never_trigger_a_rebuild_on_their_own():
    gateway = CountingGateway()
    index = KnowledgeIndex(Settings(), gateway)
    index.sync(InMemoryCorpus(DOCUMENTS))
    embedded, before = len(gateway.embedded), collections(index)
    upgraded = after_an_upgrade(index, chunk_tokens=200)

    with pytest.raises(IndexReadOnly, match="chunk size"):
        upgraded.sync(InMemoryCorpus(DOCUMENTS))
    upgraded.search("ORA-00942")
    upgraded.status()

    assert len(gateway.embedded) == embedded
    assert collections(index) == before


def test_a_rebuild_records_the_chunk_settings_it_used(index):
    index.sync(InMemoryCorpus(DOCUMENTS))
    upgraded = after_an_upgrade(index, chunk_tokens=200, chunk_overlap=20)

    assert upgraded.rebuild([InMemoryCorpus(DOCUMENTS)]).promoted

    status = upgraded.status()
    assert (status.format.chunk_tokens, status.format.chunk_overlap) == (200, 20)
    assert status.access == Access.READ_WRITE


def test_an_unreadable_recorded_chunk_size_is_reported_as_an_unreadable_format():
    with pytest.raises(ValueError, match="unreadable index format"):
        IndexFormat.from_payload({
            "schema_version": 1, "embed_model": "embed", "dimension": 64,
            "sparse": {"k1": 1.2, "b": 0.75, "avg_doc_length": 250.0},
            "chunk_tokens": "lots", "chunk_overlap": 60})


def test_an_index_built_before_corpus_ownership_was_recorded_is_served_read_only_until_a_rebuild(
        index, monkeypatch):
    real = IndexFormat.to_payload

    def without_ownership(self):  # what an earlier version wrote
        return {k: v for k, v in real(self).items() if k != "corpus_ownership"}

    monkeypatch.setattr(IndexFormat, "to_payload", without_ownership)
    index.sync(InMemoryCorpus(DOCUMENTS))
    monkeypatch.setattr(IndexFormat, "to_payload", real)
    upgraded = after_an_upgrade(index)

    status = upgraded.status()

    assert status.access == Access.READ_ONLY
    assert any("Corpus" in warning for warning in status.warnings)
    assert upgraded.search("ORA-00942")
    with pytest.raises(IndexReadOnly, match="Rebuild"):
        upgraded.sync(InMemoryCorpus(DOCUMENTS))
    assert upgraded.rebuild([InMemoryCorpus(DOCUMENTS)]).promoted
    assert upgraded.status().access == Access.READ_WRITE


def test_a_changed_query_instruction_leaves_the_index_current_and_needs_no_rebuild(index):
    upgraded = after_an_upgrade(index, query_instruction="Find the passage that answers the question")

    assert upgraded.status().access == Access.READ_WRITE
    upgraded.sync(InMemoryCorpus(DOCUMENTS))
