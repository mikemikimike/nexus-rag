import pytest

from local_rag import KnowledgeIndex, Settings
from local_rag.sparse import tokenize


def test_exact_identifier_search_returns_the_passage_containing_it(index, corpus):
    index.sync(corpus)

    passages = index.search("ORA-00942")

    assert passages[0].document_id == "dba/ora-00942"
    assert "ORA-00942" in passages[0].text
    assert passages[0].title == "Fixing ORA-00942"
    assert passages[0].url == "https://wiki.example.test/dba/ora-00942"
    assert passages[0].space == "DBA"


def test_semantic_query_without_shared_keywords_returns_closest_passage(index, corpus):
    index.sync(corpus)

    query = "vehicle refuses to ignite"

    passages = index.search(query)

    assert passages[0].document_id == "garage/battery"
    assert not set(tokenize(query)) & set(tokenize(passages[0].text))


def test_sync_reports_how_many_documents_were_added(index, corpus):
    report = index.sync(corpus)

    assert report.added == 3


def test_index_ignores_environment_and_working_directory(
        gateway, corpus, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("QDRANT_URL", "http://unreachable.invalid:6333")
    monkeypatch.setenv("RAG_BM25_K1", "not-a-number")
    index = KnowledgeIndex(Settings(), gateway)

    index.sync(corpus)

    assert index.search("ORA-12154")[0].document_id == "dba/ora-12154"
    assert list(tmp_path.iterdir()) == []


def test_settings_repr_hides_the_qdrant_api_key():
    assert "s3cret" not in repr(Settings(qdrant_api_key="s3cret"))


def test_index_rejects_an_unsupported_qdrant_location(gateway):
    with pytest.raises(ValueError, match="qdrant_location"):
        KnowledgeIndex(Settings(qdrant_location="./qdrant_data"), gateway)
