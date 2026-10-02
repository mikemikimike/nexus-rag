import dataclasses

from qdrant_client import QdrantClient, models

from local_rag import KnowledgeIndex, Settings
from local_rag.chunker import Chunker
from local_rag.corpus import Document, InMemoryCorpus
from local_rag.fakes import FakeModelGateway
from conftest import DOCUMENTS
from index_states import NAMES
from test_sync import CountingGateway, revised

SMALL = Settings(chunk_tokens=40, chunk_overlap=5)


def long_text(sentences: int, marker: dict[int, str] | None = None) -> str:
    """Sentences of ~8 tokens each; ``marker`` replaces the filler of chosen sentences."""
    return " ".join(
        (marker or {}).get(i, f"Filler sentence number {i} says nothing useful.")
        for i in range(sentences))


def manual(text: str, *, id: str = "wiki/manual", version: str = "1") -> Document:
    return Document(id=id, title="Operations manual", url=f"https://wiki.example.test/{id}",
                    space="OPS", version=version, text=text)


def passages_of(client: QdrantClient, document_id: str) -> list:
    """Every Passage of one Document in the Live index, read from Qdrant directly."""
    records, _ = client.scroll(
        NAMES.live, scroll_filter=models.Filter(must=[
            models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))],
            must_not=[models.FieldCondition(
                key="kind", match=models.MatchValue(value="document"))]),
        limit=1000, with_payload=True)
    return records


def test_a_document_longer_than_one_chunk_is_indexed_as_several_passages(client):
    index = KnowledgeIndex(SMALL, FakeModelGateway(), client=client)

    index.sync(InMemoryCorpus([manual(long_text(30))]))

    status = index.status()
    assert status.documents == 1
    assert status.passages > 1
    assert len(passages_of(client, "default:wiki/manual")) == status.passages


def test_a_short_document_still_yields_one_passage(index, corpus):
    index.sync(corpus)

    status = index.status()

    assert (status.documents, status.passages) == (3, 3)


def test_a_query_matching_only_the_tail_of_a_long_document_finds_it():
    index = KnowledgeIndex(SMALL, FakeModelGateway())
    text = long_text(40, {39: "Restart the flux capacitor with code ZX-9000 afterwards."})
    index.sync(InMemoryCorpus([manual(text), *DOCUMENTS]))

    found = index.search("ZX-9000")

    assert found[0].document_id == "default:wiki/manual"
    assert "ZX-9000" in found[0].text
    assert len(found[0].text) < len(text) / 2  # the Passage, not the whole Document
    assert (found[0].title, found[0].space, found[0].url) == (
        "Operations manual", "OPS", "https://wiki.example.test/wiki/manual")


def test_a_semantic_query_finds_the_tail_of_a_long_document_too():
    gateway = FakeModelGateway(
        dimension=1024, synonyms={"vehicle": {"vehicle", "car", "automobile"}})
    index = KnowledgeIndex(SMALL, gateway)
    text = long_text(40, {39: "Every automobile needs fuel."})
    index.sync(InMemoryCorpus([manual(text), *DOCUMENTS[:2]]))

    assert index.search("car")[0].document_id == "default:wiki/manual"


def test_every_passage_carries_the_documents_title_in_what_is_embedded():
    gateway = CountingGateway()
    index = KnowledgeIndex(SMALL, gateway)

    index.sync(InMemoryCorpus([manual(long_text(30))]))

    assert len(gateway.embedded) > 1
    assert all(text.startswith("Operations manual\n\n") for text in gateway.embedded)


def test_the_full_text_is_still_the_whole_document_however_many_passages():
    index = KnowledgeIndex(SMALL, FakeModelGateway())
    text = long_text(30)
    index.sync(InMemoryCorpus([manual(text)]))

    assert index.get_document("default:wiki/manual") == text


def indexed(corpus_documents, settings=SMALL, gateway=None, client=None):
    index = KnowledgeIndex(settings, gateway or FakeModelGateway(), client=client)
    index.sync(InMemoryCorpus(corpus_documents))
    return index


def test_replacing_a_document_replaces_all_its_old_passages():
    index = indexed([manual(long_text(30, {2: "Old marker QQ-1111 here."}))])
    before = index.status().passages

    report = index.sync(InMemoryCorpus([
        manual(long_text(30, {2: "New marker QQ-2222 here."}), version="2")]))

    assert (report.replaced, report.failed) == (1, 0)
    assert index.status().passages == before
    assert index.search("QQ-1111")[0].text.count("QQ-1111") == 0  # gone from every Passage
    assert "QQ-2222" in index.search("QQ-2222")[0].text


def test_a_document_that_shrinks_loses_its_surplus_passages():
    index = indexed([manual(long_text(40, {39: "Old tail marker QQ-1111."}))])
    assert index.status().passages > 2

    index.sync(InMemoryCorpus([manual("One short paragraph now.", version="2")]))

    status = index.status()
    assert (status.documents, status.passages) == (1, 1)
    assert not [p for p in index.search("QQ-1111", k=10) if "QQ-1111" in p.text]
    assert index.get_document("default:wiki/manual") == "One short paragraph now."


def test_a_document_that_grows_gains_passages():
    index = indexed([manual("Short.")])

    index.sync(InMemoryCorpus([manual(long_text(30), version="2")]))

    assert index.status().passages > 1


def test_removing_a_document_removes_all_its_passages(client):
    index = indexed([manual(long_text(30)), *DOCUMENTS], client=client)
    assert index.status().passages > 4

    report = index.sync(InMemoryCorpus(DOCUMENTS))

    assert report.removed == 1
    status = index.status()
    assert (status.documents, status.passages) == (3, 3)
    assert passages_of(client, "default:wiki/manual") == []
    assert index.get_document("default:wiki/manual") is None


def test_passage_ids_are_the_same_every_time_a_document_is_indexed():
    first, second = QdrantClient(location=":memory:"), QdrantClient(location=":memory:")
    indexed([manual(long_text(30))], client=first)
    indexed([manual(long_text(30))], client=second)

    ids = lambda client: sorted(p.id for p in passages_of(client, "default:wiki/manual"))
    assert ids(first) == ids(second) and len(ids(first)) > 1


def test_resyncing_the_same_text_under_a_new_version_does_not_duplicate_passages(client):
    index = indexed([manual(long_text(30))], client=client)
    before = sorted(p.id for p in passages_of(client, "default:wiki/manual"))

    index.sync(InMemoryCorpus([manual(long_text(30), version="2")]))

    assert sorted(p.id for p in passages_of(client, "default:wiki/manual")) == before


def test_a_failed_embed_of_a_changed_long_document_keeps_its_old_passages_searchable():
    class Flaky(FakeModelGateway):
        fail = False

        def embed(self, texts):
            if self.fail:
                raise ConnectionError("gateway down")
            return super().embed(texts)

    gateway = Flaky()
    index = indexed([manual(long_text(30, {20: "Old marker QQ-1111."}))], gateway=gateway)
    before = index.status().passages
    gateway.fail = True

    report = index.sync(InMemoryCorpus([manual(long_text(20), version="2")]))

    assert report.failed == 1
    assert index.status().passages == before
    assert "QQ-1111" in index.search("QQ-1111")[0].text


def test_a_crash_after_writing_the_new_passages_never_hides_the_document(client, monkeypatch):
    index = indexed([manual(long_text(40, {39: "Old tail marker QQ-1111."}))], client=client)

    def crash(*args, **kwargs):
        raise ConnectionError("Qdrant went away")

    with monkeypatch.context() as crashing:
        crashing.setattr(client, "delete", crash)
        report = index.sync(InMemoryCorpus([manual(long_text(10, {9: "New marker QQ-2222."}),
                                                   version="2")]))

    assert report.failed == 1
    assert index.search("QQ-2222")[0].document_id == "default:wiki/manual"  # new text already live
    assert index.search("QQ-1111")[0].document_id == "default:wiki/manual"  # nothing went missing
    again = index.sync(InMemoryCorpus([manual(long_text(10, {9: "New marker QQ-2222."}),
                                              version="2")]))
    assert again.replaced == 1  # the interrupted replace is finished by the next Sync
    assert not [p for p in index.search("QQ-1111", k=10) if "QQ-1111" in p.text]
    assert "QQ-2222" in index.search("QQ-2222")[0].text


def test_the_per_document_cap_still_limits_passages_of_one_document():
    text = long_text(40, {i: f"Alert ZX-9000 fires in case {i}." for i in range(40)})
    capped = indexed([manual(text)], dataclasses.replace(SMALL, max_passages_per_document=2))
    wider = indexed([manual(text)], dataclasses.replace(SMALL, max_passages_per_document=4))

    assert len(capped.search("ZX-9000", k=10)) == 2
    assert len(wider.search("ZX-9000", k=10)) == 4


def test_a_document_that_chunks_to_nothing_is_still_found_by_its_title(monkeypatch):
    # Extraction refuses empty text, so this is only a guard for the index's own contract.
    monkeypatch.setattr(Chunker, "pieces", lambda self, text, media_type="text/plain": [])
    index = indexed([manual("Nothing a splitter keeps.", id="wiki/empty")])

    status = index.status()
    assert (status.documents, status.passages) == (1, 1)
    assert index.search("Operations")[0].document_id == "default:wiki/empty"


def test_a_rebuild_splits_every_document_and_counts_its_passages():
    index = indexed([manual(long_text(30)), *DOCUMENTS])
    before = index.status().passages

    assert index.rebuild([InMemoryCorpus([manual(long_text(30)), *DOCUMENTS])]).promoted

    assert index.status().passages == before
