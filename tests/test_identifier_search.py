"""Keyword search matches identifiers by their parts, in Hybrid and Keyword-only retrieval."""

import pytest

from nexus_rag import Access, IndexFormat, IndexReadOnly, KnowledgeIndex, Settings
from nexus_rag.corpus import Document, InMemoryCorpus
from nexus_rag.fakes import FakeModelGateway
from nexus_rag.settings import RetrievalMode
from test_index_format import after_an_upgrade


def doc(id, text):
    return Document(
        id=id, title=id, url=f"https://git.example.test/{id}", space="code", version="1", text=text
    )


CODE = InMemoryCorpus(
    [
        doc("camel", "def getUserById(identifier): return users[identifier]"),
        doc("dotted", "client = ledger.api.Client(timeout=3)"),
        doc("kebab", "deploy the payments-v2 service to production"),
        doc("pascal", "class OrderRepository: pass"),
        doc("snake", "call get_account_balance now"),
        doc("unrelated", "nothing of interest lives here"),
    ]
)

PARTS = [
    ("user by id", "camel"),
    ("ledger api client", "dotted"),
    ("payments v2", "kebab"),
    ("order repository", "pascal"),
    ("account balance", "snake"),
]
WHOLE = [
    ("getUserById", "camel"),
    ("ledger.api.Client", "dotted"),
    ("payments-v2", "kebab"),
    ("OrderRepository", "pascal"),
    ("get_account_balance", "snake"),
]


@pytest.fixture(params=[RetrievalMode.HYBRID, RetrievalMode.KEYWORD])
def code_index(request):
    index = KnowledgeIndex(Settings(retrieval=request.param), FakeModelGateway())
    index.sync(CODE)
    return index


@pytest.mark.parametrize("query, expected", PARTS)
def test_an_identifier_is_found_by_its_parts(code_index, query, expected):
    assert code_index.search(query)[0].document_id == f"default:{expected}"


@pytest.mark.parametrize("query, expected", WHOLE)
def test_an_identifier_is_still_found_whole(code_index, query, expected):
    assert code_index.search(query)[0].document_id == f"default:{expected}"


def test_an_index_without_identifier_splitting_is_read_only_until_a_rebuild(monkeypatch, client):
    real = IndexFormat.to_payload

    def without_splitting(self):  # what the previous version wrote
        return {k: v for k, v in real(self).items() if k != "identifier_splitting"}

    index = KnowledgeIndex(Settings(retrieval=RetrievalMode.KEYWORD), None, client=client)
    monkeypatch.setattr(IndexFormat, "to_payload", without_splitting)
    index.sync(CODE)
    monkeypatch.setattr(IndexFormat, "to_payload", real)
    upgraded = after_an_upgrade(index, client)

    status = upgraded.status()

    assert status.access == Access.READ_ONLY
    assert any("identifier" in warning for warning in status.warnings)
    with pytest.raises(IndexReadOnly, match="Rebuild"):
        upgraded.sync(CODE)
    assert upgraded.rebuild([CODE]).promoted
    assert upgraded.status().access == Access.READ_WRITE
    assert upgraded.search("user by id")[0].document_id == "default:camel"
