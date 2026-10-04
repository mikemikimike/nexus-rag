"""Sync and Rebuild retry an embedding call that failed for a transient reason, then give up
on the Document, which counts as failed. Search never retries (see test_keyword_only).

The index is driven with the real gateway adapter against a local HTTP server that answers
as scripted, so which failures count as transient is observed in the requests it receives.
"""

import pytest

from conftest import DOCUMENTS
from fake_litellm_server import FakeLiteLLM
from nexus_rag import KnowledgeIndex, Settings
from nexus_rag.corpus import InMemoryCorpus
from nexus_rag.fakes import FakeModelGateway
from nexus_rag.gateway_litellm import GatewayError, LiteLLMGateway

ONE = InMemoryCorpus(DOCUMENTS[:1])


@pytest.fixture
def server():
    fake = FakeLiteLLM().start()
    yield fake
    fake.stop()


def index_over(server, **settings):
    settings = Settings(
        gateway_base_url=server.base_url,
        gateway_api_key=server.api_key,
        embed_retry_backoff=0.0,
        **settings,
    )
    return KnowledgeIndex(settings, LiteLLMGateway.from_settings(settings))


def embedding_requests(server):
    return [r for r in server.requests if r["path"].endswith("/embeddings")]


def test_a_transient_failure_that_clears_within_the_attempts_does_not_fail_the_document(server):
    index = index_over(server, embed_attempts=3)
    server.fail_with, server.fail_for = (503, "overloaded"), 2

    report = index.sync(ONE)

    assert (report.added, report.failed) == (1, 0)
    assert len(embedding_requests(server)) == 3


def test_a_document_whose_embedding_keeps_failing_is_failed_after_the_attempts(server):
    index = index_over(server, embed_attempts=3)
    server.fail_with = (503, "overloaded")

    report = index.sync(InMemoryCorpus(DOCUMENTS[:2]))

    assert (report.added, report.failed) == (0, 2)
    assert len(embedding_requests(server)) == 6  # three attempts for each Document
    assert "503" in next(iter(report.failures.values()))


@pytest.mark.parametrize("status", [401, 404, 400])
def test_a_misconfiguration_is_not_retried(server, status):
    index = index_over(server, embed_attempts=3)
    server.fail_with = (status, "no")

    report = index.sync(ONE)

    assert report.failed == 1
    assert len(embedding_requests(server)) == 1


@pytest.mark.parametrize("status", [429, 500, 502, 504])
def test_a_busy_or_failing_gateway_is_retried(server, status):
    index = index_over(server, embed_attempts=2)
    server.fail_with = (status, "later")

    index.sync(ONE)

    assert len(embedding_requests(server)) == 2


def test_one_attempt_means_no_retry(server):
    index = index_over(server, embed_attempts=1)
    server.fail_with = (503, "overloaded")

    index.sync(ONE)

    assert len(embedding_requests(server)) == 1


@pytest.mark.parametrize(
    "error, calls",
    [
        (ValueError("a bug, not a gateway"), 1),  # nothing says another try could help
        (GatewayError("unreachable"), 2),
        (GatewayError("wrong key", retryable=False), 1),
    ],
)
def test_only_an_error_that_says_it_may_pass_is_retried(error, calls):
    gateway = CountingFailure(error)
    index = KnowledgeIndex(Settings(embed_attempts=2, embed_retry_backoff=0.0), gateway)

    report = index.sync(ONE)

    assert report.failed == 1
    assert gateway.calls == calls


class CountingFailure(FakeModelGateway):
    def __init__(self, error):
        super().__init__()
        self.error, self.calls = error, 0

    def embed(self, texts):
        self.calls += 1
        raise self.error


def test_a_rebuild_whose_embedding_recovers_within_the_attempts_is_promoted(server):
    index = index_over(server, embed_attempts=2)
    index.sync(ONE)
    server.fail_with, server.fail_for = (503, "overloaded"), 1

    report = index.rebuild([ONE])

    assert report.promoted and report.failed == 0


def test_a_rebuild_whose_embedding_keeps_failing_is_refused_by_the_threshold(server):
    index = index_over(server, embed_attempts=2)
    index.sync(ONE)
    server.fail_with = (503, "overloaded")

    report = index.rebuild([ONE])

    assert not report.promoted and report.failed == 1
    assert "503" in report.failures["default:dba/ora-00942"]
    assert (
        index.search("ORA-00942")[0].document_id == "default:dba/ora-00942"
    )  # the Live index stays


def test_failed_attempts_are_a_health_note_and_a_success_clears_it(server):
    index = index_over(server, embed_attempts=1)
    server.fail_with = (503, "overloaded")
    index.sync(ONE)

    assert any("embedding" in note for note in index.health().notes)

    server.fail_with = None
    index.sync(ONE)
    assert index.health().notes == ()


@pytest.mark.parametrize("bad", [0, -1])
def test_at_least_one_attempt_is_needed(bad):
    with pytest.raises(ValueError, match="embed_attempts"):
        Settings(embed_attempts=bad)
