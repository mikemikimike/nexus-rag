"""The MCP server, driven the way Claude Code drives it: MCP over HTTP with a bearer token.

The server runs for real (uvicorn on an ephemeral port) over a Knowledge index built from
the fakes, and is reached through the MCP client and plain HTTP.
"""
import io
import os
import socket
import sys
import threading
import time
from contextlib import contextmanager

import anyio
import httpx2
import pytest
import uvicorn
from mcp import Client, StdioServerParameters
from mcp.client.streamable_http import streamable_http_client

from local_rag import KnowledgeIndex, Settings
from local_rag.corpus import Document, InMemoryCorpus
from local_rag.fakes import FakeModelGateway
from local_rag.gateway_litellm import GatewayError
from local_rag.mcp_server import build_app, main
from local_rag.settings import RetrievalMode
from test_health import UnreachableQdrant
from test_index_state import lose_the_live_collection

TOKEN = "token-alice"


def serving(index, tokens=(TOKEN,)):
    """Serve the index's MCP server on localhost and yield its base URL."""
    return serving_app(build_app(index, tokens))


@contextmanager
def serving_app(app):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", lifespan="on"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.01)
    try:
        yield f"http://127.0.0.1:{sock.getsockname()[1]}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()


def call(base_url, tool, arguments=None, token=TOKEN):
    """Call one tool over MCP and return its text."""
    async def run():
        headers = {"Authorization": f"Bearer {token}"}
        async with httpx2.AsyncClient(headers=headers) as http:
            transport = streamable_http_client(f"{base_url}/mcp", http_client=http)
            async with Client(transport) as client:
                result = await client.call_tool(tool, arguments or {})
                assert not result.is_error, result
                return result.content[0].text
    return anyio.run(run)


@pytest.fixture
def server(index, corpus):
    index.sync(corpus)
    with serving(index) as base_url:
        yield base_url


def test_search_docs_finds_a_page_by_an_exact_identifier(server):
    text = call(server, "search_docs", {"query": "ORA-00942"})

    assert "Fixing ORA-00942" in text
    assert "https://wiki.example.test/dba/ora-00942" in text
    assert "space: DBA" in text
    assert "page_id: default:dba/ora-00942" in text


def test_search_docs_narrows_to_a_space(server):
    text = call(server, "search_docs", {"query": "battery", "space": "DBA"})

    assert "GARAGE" not in text


def test_search_docs_says_so_when_nothing_matches(index, corpus):
    with serving(index) as base_url:  # nothing synced yet
        assert call(base_url, "search_docs", {"query": "ORA-00942"}) == "No results."


def test_search_docs_lists_attachments_with_their_parent_page(index, corpus):
    index.sync(InMemoryCorpus([Document(
        id="10/attachments/runbook.pdf", title="Runbook / runbook.pdf",
        url="https://wiki.example.test/10/runbook.pdf", space="OPS", version="1",
        text="Restart the frobnicator nightly.", parent_id="10")]))
    with serving(index) as base_url:
        text = call(base_url, "search_docs", {"query": "frobnicator"})

    assert "page_id: default:10" in text
    assert "attachment: default:10/attachments/runbook.pdf" in text


def test_search_docs_returns_no_more_passages_than_asked_for(server):
    assert "[2]" in call(server, "search_docs", {"query": "ORA"})
    assert "[2]" not in call(server, "search_docs", {"query": "ORA", "top_k": 1})
    assert "[1]" in call(server, "search_docs", {"query": "ORA", "top_k": 0})  # at least one


def test_healthz_needs_no_token_and_reports_ok(server):
    response = httpx2.get(f"{server}/healthz")

    assert (response.status_code, response.text) == (200, "ok")


def test_healthz_reports_an_unreachable_qdrant_as_unavailable(gateway):
    index = KnowledgeIndex(Settings(), gateway, client=UnreachableQdrant())
    with serving(index) as base_url:
        response = httpx2.get(f"{base_url}/healthz")

    assert response.status_code == 503
    assert "Qdrant is unreachable" in response.text


def test_healthz_reports_a_corrupt_index_as_unavailable(index, corpus):
    index.sync(corpus)
    lose_the_live_collection(index)
    with serving(index) as base_url:
        response = httpx2.get(f"{base_url}/healthz")

    assert response.status_code == 503


def test_list_spaces_lists_the_spaces_in_the_index(server):
    assert call(server, "list_spaces") == "DBA, GARAGE"


def test_list_spaces_says_so_when_the_index_is_empty(index):
    with serving(index) as base_url:
        assert call(base_url, "list_spaces") == "none"


def test_get_page_returns_the_full_text(server):
    text = call(server, "get_page", {"page_id": "default:dba/ora-00942"})

    assert text == "ORA-00942: table or view does not exist. Check the schema owner and grants."


def test_get_page_truncates_to_max_chars(server):
    text = call(server, "get_page", {"page_id": "default:dba/ora-00942", "max_chars": 10})

    assert text == "ORA-00942:\n\n[truncated]"


def test_get_page_says_so_when_the_page_is_not_indexed(server):
    assert call(server, "get_page", {"page_id": "nope"}) == "Page nope is not in the index."


def post_initialize(base_url, **headers):
    return httpx2.post(
        f"{base_url}/mcp", headers={
            "Accept": "application/json, text/event-stream", **headers},
        json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-03-26", "capabilities": {},
            "clientInfo": {"name": "test", "version": "0"}}})


def test_a_request_without_a_token_is_rejected(server):
    assert post_initialize(server).status_code == 401


def test_a_request_with_a_wrong_token_is_rejected(server):
    assert post_initialize(server, Authorization="Bearer nope").status_code == 401


def test_a_request_with_a_valid_token_is_accepted(server):
    assert post_initialize(server, Authorization=f"Bearer {TOKEN}").status_code == 200


def test_every_configured_token_is_accepted(index, corpus):
    index.sync(corpus)
    with serving(index, tokens=("token-alice", "token-bob")) as base_url:
        assert "ORA-00942" in call(base_url, "search_docs", {"query": "ORA-00942"},
                                   token="token-bob")
        assert post_initialize(base_url, Authorization="Bearer token-alice").status_code == 200


def test_an_unauthenticated_client_cannot_use_a_tool(server):
    with pytest.raises(Exception):
        call(server, "search_docs", {"query": "ORA-00942"}, token="nope")


class ServeRecorder:
    """Stands in for uvicorn: records what `main` asks to serve instead of listening."""

    def __init__(self):
        self.calls = []

    def __call__(self, app, *, host, port, certfile, keyfile):
        self.calls.append({"app": app, "host": host, "port": port,
                           "certfile": certfile, "keyfile": keyfile})


def start(index, serve, **environ):
    err = io.StringIO()
    code = main([], environ, index=index, serve=serve, stderr=err)
    return code, err.getvalue()


def test_the_tokens_come_from_the_environment(index, corpus):
    index.sync(corpus)
    serve = ServeRecorder()

    code, _ = start(index, serve, MCP_TOKENS="token-alice, token-bob ,,")

    assert code == 0
    with serving_app(serve.calls[0]["app"]) as base_url:
        for token in ("token-alice", "token-bob"):
            assert "ORA-00942" in call(base_url, "search_docs", {"query": "ORA-00942"}, token)
        assert post_initialize(base_url, Authorization="Bearer").status_code == 401


def test_a_revoked_token_stops_working_after_a_restart(index, corpus):
    index.sync(corpus)
    before, after = ServeRecorder(), ServeRecorder()

    start(index, before, MCP_TOKENS="token-alice,token-bob")
    start(index, after, MCP_TOKENS="token-alice")  # the env file edited, the server restarted

    with serving_app(before.calls[0]["app"]) as base_url:
        assert post_initialize(base_url, Authorization="Bearer token-bob").status_code == 200
    with serving_app(after.calls[0]["app"]) as base_url:
        assert post_initialize(base_url, Authorization="Bearer token-bob").status_code == 401
        assert post_initialize(base_url, Authorization="Bearer token-alice").status_code == 200


def test_it_refuses_to_serve_without_tokens(index):
    serve = ServeRecorder()

    for environ in ({}, {"MCP_TOKENS": ""}, {"MCP_TOKENS": " , "}):
        code, err = start(index, serve, **environ)

        assert code == 2 and "MCP_TOKENS" in err
    assert serve.calls == []


def test_it_listens_where_the_environment_says(index):
    serve = ServeRecorder()

    start(index, serve, MCP_TOKENS="t", MCP_HOST="0.0.0.0", MCP_PORT="9000",
          MCP_TLS_CERT="/etc/tls/server.crt", MCP_TLS_KEY="/etc/tls/server.key")

    assert {k: v for k, v in serve.calls[0].items() if k != "app"} == {
        "host": "0.0.0.0", "port": 9000,
        "certfile": "/etc/tls/server.crt", "keyfile": "/etc/tls/server.key"}


def test_it_listens_on_localhost_without_tls_by_default(index):
    serve = ServeRecorder()

    start(index, serve, MCP_TOKENS="t")

    assert {k: v for k, v in serve.calls[0].items() if k != "app"} == {
        "host": "127.0.0.1", "port": 8765, "certfile": None, "keyfile": None}


def test_an_unusable_port_is_a_usage_error(index):
    code, err = start(index, ServeRecorder(), MCP_TOKENS="t", MCP_PORT="http")

    assert code == 2 and "MCP_PORT" in err


def test_stdio_mode_serves_the_tools_without_tokens():
    async def run():
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "local_rag.mcp_server", "--stdio"],
            env={**os.environ, "QDRANT_URL": ":memory:", "RAG_EMBED_MODEL": "embed",
                 "LITELLM_BASE_URL": "http://gateway.example.test:4000/v1"})
        async with Client(params) as client:
            tools = await client.list_tools()
            spaces = await client.call_tool("list_spaces", {})
            return sorted(tool.name for tool in tools.tools), spaces.content[0].text
    assert anyio.run(run) == (["get_page", "list_spaces", "search_docs"], "none")


# --- the Retrieval mode ----------------------------------------------------------------

def tool_descriptions(base_url):
    async def run():
        headers = {"Authorization": f"Bearer {TOKEN}"}
        async with httpx2.AsyncClient(headers=headers) as http:
            transport = streamable_http_client(f"{base_url}/mcp", http_client=http)
            async with Client(transport) as client:
                return {t.name: t.description for t in (await client.list_tools()).tools}
    return anyio.run(run)


class FlakyGateway(FakeModelGateway):
    def __init__(self):
        super().__init__()
        self.down = False

    def embed_query(self, text):
        if self.down:
            raise GatewayError("gateway unreachable")
        return super().embed_query(text)


def test_a_keyword_fallback_tells_claude_the_results_are_keyword_only(corpus):
    gateway = FlakyGateway()
    index = KnowledgeIndex(Settings(), gateway)
    index.sync(corpus)
    with serving(index) as base_url:
        healthy = call(base_url, "search_docs", {"query": "ORA-00942"})
        gateway.down = True
        degraded = call(base_url, "search_docs", {"query": "ORA-00942"})

    assert "keyword-only" not in healthy
    assert degraded.startswith("Note: Results are keyword-only")
    assert "embedding model could not be reached" in degraded
    assert "Fixing ORA-00942" in degraded  # the results are still there


def test_a_fallback_that_finds_nothing_still_says_why(corpus):
    gateway = FlakyGateway()
    index = KnowledgeIndex(Settings(), gateway)
    index.sync(corpus)
    gateway.down = True
    with serving(index) as base_url:
        text = call(base_url, "search_docs", {"query": "zzzzunknownterm"})

    assert "No results." in text and "keyword-only" in text


def test_a_keyword_only_server_describes_search_as_keyword_search(corpus):
    index = KnowledgeIndex(Settings(retrieval=RetrievalMode.KEYWORD), None)
    index.sync(corpus)
    with serving(index) as base_url:
        description = tool_descriptions(base_url)["search_docs"]
        text = call(base_url, "search_docs", {"query": "ORA-00942"})

    assert "keyword" in description.lower() and "semantic" not in description
    assert "Fixing ORA-00942" in text and "Note:" not in text


def test_a_hybrid_server_describes_search_as_semantic_and_keyword(server):
    assert "semantic" in tool_descriptions(server)["search_docs"]


def test_a_keyword_only_server_starts_with_no_embedding_model_and_no_gateway():
    async def run():
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "local_rag.mcp_server", "--stdio"],
            env={**os.environ, "QDRANT_URL": ":memory:", "RAG_RETRIEVAL": "keyword",
                 "RAG_EMBED_MODEL": ""})
        async with Client(params) as client:
            spaces = await client.call_tool("list_spaces", {})
            search = await client.call_tool("search_docs", {"query": "anything"})
            return spaces.content[0].text, search.content[0].text
    assert anyio.run(run) == ("none", "No results.")


def test_a_hybrid_server_without_an_embedding_model_refuses_to_start_and_says_why(capsys):
    code = main(["--stdio"], {"QDRANT_URL": ":memory:"})

    assert code != 0
    assert "RAG_EMBED_MODEL" in capsys.readouterr().err


def test_search_docs_narrows_to_a_repository_or_a_tracked_ref_and_shows_the_ref(index):
    from local_rag.corpus import RawDocument

    def code(repository, ref, primary, text):
        return RawDocument(
            id=f"{repository}@{ref}:a.py", title=f"{repository}/a.py", url="https://src.example.test/a",
            space="platform", version=text, content=text.encode(), media_type="text/x-python",
            repository=repository, ref=ref, primary=primary)
    index.sync(InMemoryCorpus([
        code("platform/pay", "main", True, "def settle(): pass  # zebrafish"),
        code("platform/pay", "feature/x", False, "def settle(): pass  # zebrafish"),
        code("platform/other", "main", True, "def settle(): pass  # zebrafish")], name="code"))
    with serving(index) as base_url:
        plain = call(base_url, "search_docs", {"query": "zebrafish"})
        feature = call(base_url, "search_docs", {"query": "zebrafish", "ref": "feature/x"})
        other = call(base_url, "search_docs",
                     {"query": "zebrafish", "repository": "platform/other"})

    assert "ref: feature/x" not in plain and plain.count("ref: main") == 2
    assert "ref: feature/x" in feature and "ref: main" not in feature
    assert "platform/other/a.py" in other and "platform/pay/a.py" not in other
