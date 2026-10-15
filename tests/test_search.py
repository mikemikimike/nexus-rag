import time
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from nexus_rag import KnowledgeIndex, Settings
from nexus_rag.fakes import FakeModelGateway
from nexus_rag.sparse import tokenize


def test_exact_identifier_search_returns_the_passage_containing_it(index, corpus):
    index.sync(corpus)

    passages = index.search("ORA-00942")

    assert passages[0].document_id == "default:dba/ora-00942"
    assert "ORA-00942" in passages[0].text
    assert passages[0].title == "Fixing ORA-00942"
    assert passages[0].url == "https://wiki.example.test/dba/ora-00942"
    assert passages[0].space == "DBA"


def test_semantic_query_without_shared_keywords_returns_closest_passage(index, corpus):
    index.sync(corpus)

    query = "vehicle refuses to ignite"

    passages = index.search(query)

    assert passages[0].document_id == "default:garage/battery"
    assert not set(tokenize(query)) & set(tokenize(passages[0].text))


def test_sync_reports_how_many_documents_were_added(index, corpus):
    report = index.sync(corpus)

    assert report.added == 3


def test_index_ignores_environment_and_working_directory(gateway, corpus, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("QDRANT_URL", "http://unreachable.invalid:6333")
    monkeypatch.setenv("RAG_BM25_K1", "not-a-number")
    index = KnowledgeIndex(Settings(), gateway)

    index.sync(corpus)

    assert index.search("ORA-12154")[0].document_id == "default:dba/ora-12154"
    assert list(tmp_path.iterdir()) == []


def test_settings_repr_hides_the_qdrant_api_key():
    assert "s3cret" not in repr(Settings(qdrant_api_key="s3cret"))


def test_index_rejects_an_unsupported_qdrant_location(gateway):
    with pytest.raises(ValueError, match="qdrant_location"):
        KnowledgeIndex(Settings(qdrant_location="./qdrant_data"), gateway)


class SlowUnavailableGateway(FakeModelGateway):
    def __init__(self):
        super().__init__()
        self.unavailable = True
        self.dimension_probes = 0

    def embed_query(self, text):
        if text == "dimension probe":
            self.dimension_probes += 1
            time.sleep(0.05)
            if self.unavailable:
                raise ConnectionError("gateway unavailable")
        return super().embed_query(text)


def test_unreachable_dimension_probe_is_cached_until_cooldown_expires(
    index, corpus, client, monkeypatch
):
    index.sync(corpus)
    gateway = SlowUnavailableGateway()
    index = KnowledgeIndex(Settings(), gateway, client=client)
    now = [0.0]
    monkeypatch.setattr(index, "_assessment_clock", lambda: now[0])

    first = index.search("vehicle refuses to ignite")
    second = index.search("car battery")

    assert first.notice and "keyword-only" in first.notice
    assert second.notice == first.notice
    assert gateway.dimension_probes == 1

    now[0] = 31.0
    gateway.unavailable = False
    recovered = index.search("car battery")

    assert gateway.dimension_probes == 2
    assert recovered.notice is None
    assert recovered[0].document_id == "default:garage/battery"


def test_concurrent_searches_share_one_unreachable_dimension_probe(index, corpus, client):
    index.sync(corpus)
    gateway = SlowUnavailableGateway()
    index = KnowledgeIndex(Settings(), gateway, client=client)
    workers = 6
    start = Barrier(workers)

    def search():
        start.wait()
        return index.search("gateway search fallback")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda _: search(), range(workers)))

    assert gateway.dimension_probes == 1
    assert all(result.notice and "keyword-only" in result.notice for result in results)
