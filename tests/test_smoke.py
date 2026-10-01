"""The manual smoke check: what an operator runs on the installed host to see that the real
Model gateway, Qdrant, LiteParse, Confluence CLI and (on the systemd route) units work."""
import io
import os

import pytest
from qdrant_client import QdrantClient

from local_rag.fakes import FakeModelGateway
from local_rag.gateway_litellm import GatewayError
from local_rag.smoke import main
from conftest import DOCUMENTS
from local_rag import KnowledgeIndex, Settings
from local_rag.corpus import InMemoryCorpus
from test_health import UnreachableQdrant

ENVIRONMENT = {
    "LITELLM_BASE_URL": "http://gateway.example.test:4000/v1",
    "RAG_EMBED_MODEL": "embed-model",
    "MCP_TOKENS": "first-secret,second-secret",
}


class Host:
    """Stands in for the commands the check runs (lit, confluence, systemctl)."""

    def __init__(self, failing=()):
        self.failing = set(failing)
        self.calls = []

    def __call__(self, command):
        self.calls.append(list(command))
        if os.path.basename(command[0]) in self.failing:
            raise OSError(f"{command[0]} failed")
        return "9.9.9"


class DownGateway(FakeModelGateway):
    def embed_query(self, text):
        raise GatewayError("gateway http://gateway.example.test:4000/v1/embeddings unreachable")


@pytest.fixture
def smoke():
    def run(*argv, gateway=None, client=None, host=None, **environment):
        out, err = io.StringIO(), io.StringIO()
        code = main(list(argv), {**ENVIRONMENT, **environment},
                    gateway=gateway or FakeModelGateway(), client=client or QdrantClient(":memory:"),
                    run=host or Host(), stdout=out, stderr=err)
        return code, out.getvalue(), err.getvalue()
    return run


def line(output: str, name: str) -> str:
    [found] = [l for l in output.splitlines() if name in l]
    return found


def test_a_working_installation_passes_every_check(smoke):
    code, out, _ = smoke()

    assert code == 0, out
    assert "[FAIL]" not in out
    assert line(out, "Model gateway").startswith("[ OK ]")
    assert "64 dimensions" in line(out, "Model gateway")
    assert line(out, "Knowledge index").startswith("[ OK ]")


def test_the_tokens_are_counted_never_printed(smoke):
    code, out, err = smoke()

    assert "2 tokens" in line(out, "MCP_TOKENS")
    assert "first-secret" not in out + err and "second-secret" not in out + err


def test_without_a_token_the_check_fails_but_the_rest_still_runs(smoke):
    code, out, _ = smoke(MCP_TOKENS="")

    assert code == 1
    assert line(out, "MCP_TOKENS").startswith("[FAIL]")
    assert line(out, "Model gateway").startswith("[ OK ]")


def test_a_placeholder_token_fails(smoke):
    code, out, _ = smoke(MCP_TOKENS="CHANGE_ME_MCP_TOKEN")

    assert code == 1
    assert line(out, "MCP_TOKENS").startswith("[FAIL]")


def test_an_unset_model_gateway_fails_instead_of_using_a_default(smoke):
    code, out, _ = smoke(LITELLM_BASE_URL="")

    assert code == 1
    failed = line(out, "Model gateway")
    assert failed.startswith("[FAIL]") and "LITELLM_BASE_URL" in failed


def test_hybrid_retrieval_without_an_embedding_model_fails_naming_both_ways_out(smoke):
    code, out, _ = smoke(RAG_EMBED_MODEL="")

    assert code == 1
    failed = line(out, "settings")
    assert failed.startswith("[FAIL]") and "RAG_EMBED_MODEL" in failed
    assert "RAG_RETRIEVAL=keyword" in failed


def test_an_unreachable_gateway_fails_with_its_reason(smoke):
    code, out, _ = smoke(gateway=DownGateway())

    assert code == 1
    failed = line(out, "Model gateway")
    assert failed.startswith("[FAIL]") and "unreachable" in failed


def test_rerank_is_skipped_unless_a_model_is_named(smoke):
    _, out, _ = smoke()

    assert line(out, "Rerank").startswith("[SKIP]")


def test_a_named_rerank_model_is_asked_to_score(smoke):
    code, out, _ = smoke(RAG_RERANK_MODEL="rerank-model")

    assert code == 0, out
    assert line(out, "Rerank").startswith("[ OK ]")


def test_a_rerank_model_the_gateway_cannot_serve_fails(smoke):
    class NoRerank(FakeModelGateway):
        def rerank(self, query, texts):
            raise GatewayError("no rerank model configured")

    code, out, _ = smoke(RAG_RERANK_MODEL="rerank-model", gateway=NoRerank())

    assert code == 1
    assert line(out, "Rerank").startswith("[FAIL]")


def test_a_rerank_model_without_a_named_gateway_fails_with_a_reason():
    out = io.StringIO()
    environment = {**ENVIRONMENT, "LITELLM_BASE_URL": "", "RAG_RETRIEVAL": "keyword",
                   "RAG_RERANK_MODEL": "rerank-model"}

    code = main([], environment, client=QdrantClient(":memory:"), run=Host(), stdout=out)

    assert code == 1
    failed = line(out.getvalue(), "Rerank")
    assert failed.startswith("[FAIL]") and "no Model gateway" in failed


def test_an_unreachable_qdrant_fails_with_its_reason(smoke):
    code, out, _ = smoke(client=UnreachableQdrant())

    assert code == 1
    failed = line(out, "Knowledge index")
    assert failed.startswith("[FAIL]") and "connection refused" in failed


def test_the_index_check_reports_what_is_indexed(smoke):
    client = QdrantClient(":memory:")
    KnowledgeIndex(Settings(), FakeModelGateway(), client=client).sync(InMemoryCorpus(DOCUMENTS))

    code, out, _ = smoke(client=client)

    assert code == 0, out
    assert "3 documents" in line(out, "Knowledge index")


def test_a_missing_liteparse_fails(smoke):
    code, out, _ = smoke(host=Host(failing=["lit"]))

    assert code == 1
    assert line(out, "LiteParse").startswith("[FAIL]")


def test_liteparse_is_the_one_the_settings_name(smoke):
    host = Host()

    smoke(host=host, LIT_BIN="/opt/venv/bin/lit")

    assert ["/opt/venv/bin/lit", "--version"] in host.calls


def test_confluence_is_checked_only_when_it_is_configured(smoke):
    host = Host()
    _, out, _ = smoke(host=host)

    assert line(out, "Confluence").startswith("[SKIP]")
    assert not any(call[0] == "confluence" for call in host.calls)


def test_a_configured_confluence_that_cannot_be_read_fails(smoke):
    code, out, _ = smoke(host=Host(failing=["confluence"]), CONFLUENCE_DOMAIN="wiki.example.test")

    assert code == 1
    assert line(out, "Confluence").startswith("[FAIL]")


def test_a_configured_confluence_is_asked_through_the_named_command(smoke):
    host = Host()

    code, out, _ = smoke(host=host, CONFLUENCE_DOMAIN="wiki.example.test",
                         CONFLUENCE_BIN="/usr/local/bin/confluence")

    assert code == 0, out
    assert any(call[0] == "/usr/local/bin/confluence" for call in host.calls)


def test_the_systemd_units_are_checked_only_on_request(smoke):
    host = Host()
    _, out, _ = smoke(host=host)

    assert "systemd" not in out
    assert not any(call[0] == "systemctl" for call in host.calls)


def test_each_systemd_unit_must_be_active(smoke):
    host = Host()

    code, out, _ = smoke("--systemd", host=host)

    assert code == 0, out
    asked = {call[-1] for call in host.calls if call[0] == "systemctl"}
    assert asked == {"qdrant.service", "local-rag-mcp.service",
                     "local-rag-sync.timer", "local-rag-full.timer"}


def test_an_inactive_systemd_unit_fails(smoke):
    code, out, _ = smoke("--systemd", host=Host(failing=["systemctl"]))

    assert code == 1
    assert line(out, "local-rag-mcp.service").startswith("[FAIL]")


def test_an_unusable_setting_fails_the_check_instead_of_crashing(smoke):
    code, out, _ = smoke(RAG_CHUNK_TOKENS="many")

    assert code == 1
    assert "[FAIL]" in out and "RAG_CHUNK_TOKENS" in out


def test_an_unknown_argument_is_a_usage_error(smoke):
    code, _, _ = smoke("--nonsense")

    assert code == 2


# --- the Retrieval mode ----------------------------------------------------------------

KEYWORD_ONLY = {"RAG_RETRIEVAL": "keyword", "RAG_EMBED_MODEL": "", "LITELLM_BASE_URL": ""}


def smoke_without_a_gateway(gateway=None, **environment):
    out = io.StringIO()
    code = main([], {**ENVIRONMENT, **KEYWORD_ONLY, **environment}, gateway=gateway,
                client=QdrantClient(":memory:"), run=Host(), stdout=out)
    return code, out.getvalue()


def test_a_keyword_only_installation_passes_with_no_gateway_at_all():
    code, out = smoke_without_a_gateway()

    assert code == 0, out
    assert "[FAIL]" not in out
    skipped = line(out, "Model gateway")
    assert skipped.startswith("[SKIP]") and "keyword" in skipped
    assert line(out, "Rerank").startswith("[SKIP]")
    assert line(out, "Knowledge index").startswith("[ OK ]")


def test_a_keyword_only_installation_with_a_reranker_still_checks_the_gateway_for_it():
    code, out = smoke_without_a_gateway(
        FakeModelGateway(), LITELLM_BASE_URL="http://gateway.example.test:4000/v1",
        RAG_RERANK_MODEL="rerank-model")

    assert code == 0, out
    assert line(out, "Model gateway").startswith("[SKIP]")
    assert line(out, "Rerank").startswith("[ OK ]")


def test_hybrid_retrieval_with_a_dead_embedding_model_fails_not_skips(smoke):
    code, out, _ = smoke(gateway=DownGateway())

    assert code == 1
    assert line(out, "Model gateway").startswith("[FAIL]")
