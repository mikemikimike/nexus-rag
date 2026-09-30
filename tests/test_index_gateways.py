"""The Knowledge index behaves the same over the fake gateway and the LiteLLM adapter."""
import pytest

from conftest import DOCUMENTS, SYNONYMS
from fake_litellm_server import FakeLiteLLM
from local_rag import KnowledgeIndex, Settings
from local_rag.corpus import InMemoryCorpus
from local_rag.fakes import FakeModelGateway
from local_rag.gateway_litellm import LiteLLMGateway
from local_rag.sparse import tokenize


@pytest.fixture(params=["fake", "litellm"])
def gateway(request):
    fake = FakeModelGateway(synonyms=SYNONYMS)
    if request.param == "fake":
        yield fake
        return
    server = FakeLiteLLM(fake).start()
    yield LiteLLMGateway.from_settings(Settings(
        gateway_base_url=server.base_url, gateway_api_key=server.api_key))
    server.stop()


@pytest.fixture
def index(gateway):
    return KnowledgeIndex(Settings(), gateway)


def test_sync_then_exact_identifier_search(index):
    report = index.sync(InMemoryCorpus(DOCUMENTS))

    assert (report.added, report.failed) == (3, 0)
    assert index.search("ORA-00942")[0].document_id == "dba/ora-00942"


def test_semantic_search_without_shared_keywords(index):
    index.sync(InMemoryCorpus(DOCUMENTS))
    query = "vehicle refuses to ignite"

    passages = index.search(query)

    assert passages[0].document_id == "garage/battery"
    assert not set(tokenize(query)) & set(tokenize(passages[0].text))


def test_an_unreachable_gateway_fails_documents_instead_of_crashing_sync():
    gateway = LiteLLMGateway("http://127.0.0.1:1/v1", timeout=1)
    index = KnowledgeIndex(Settings(), gateway)

    report = index.sync(InMemoryCorpus(DOCUMENTS))

    assert (report.added, report.failed) == (0, 3)
