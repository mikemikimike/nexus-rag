"""The live acceptance suite, run offline against stand-ins for the infrastructure.

Every check passes against the good stand-ins, and each failure path has a deliberately
broken stand-in, so the suite's own logic is verified before it meets the real thing.
"""
import os
import socket
import subprocess
import sys

from acceptance_support import (
    ENVIRONMENT, MCP_TOKEN, SECRETS, TIMERS, Host, Qdrant, container_host, good_gateway, line,
    probe_synonyms, run_suite, systemd_host)
from fake_drop import REPO
from test_health import UnreachableQdrant
from test_mcp_server import serving, serving_app
from local_rag import KnowledgeIndex, Settings
from local_rag.corpus import Document, InMemoryCorpus
from local_rag.fakes import FakeModelGateway
from local_rag.gateway_litellm import GatewayError
from local_rag.mcp_server import build_server


def test_an_unknown_argument_is_a_usage_error():
    code, out, err = run_suite("--nonsense")

    assert code == 2
    assert out == ""


# --- the Model gateway ---------------------------------------------------------------

def gateway_only(**kwargs):
    return run_suite("--only", "gateway", **kwargs)


def test_the_gateway_answers_and_its_dimension_is_reported():
    code, out, _ = gateway_only()

    assert code == 0, out
    assert line(out, "gateway.embeddings").startswith("[ OK ]")
    assert "64 dimensions" in line(out, "gateway.embeddings")


class DownGateway(FakeModelGateway):
    def __init__(self, error):
        super().__init__()
        self.error = error

    def embed(self, texts):
        raise GatewayError(self.error)

    def embed_query(self, text):
        raise GatewayError(self.error)


def test_an_unreachable_gateway_fails_with_its_cause_and_the_guide_section():
    code, out, _ = gateway_only(gateway=DownGateway(
        "gateway http://gateway.example.test:4000/v1/embeddings unreachable: refused"))

    assert code == 1
    failed = line(out, "gateway.embeddings")
    assert failed.startswith("[FAIL]") and "unreachable" in failed
    assert "firewall or DNS" in failed
    assert "docs/agents/closed-network-setup.md section 10" in failed
    assert line(out, "gateway.dimension").startswith("[SKIP]")  # one failure, not a dozen
    assert "gateway.embeddings did not pass" in line(out, "gateway.dimension")


def test_a_gateway_that_refuses_the_key_names_the_key():
    _, out, _ = gateway_only(gateway=DownGateway(
        "gateway http://gateway.example.test:4000/v1/embeddings answered 401: bad key"))

    assert "LITELLM_API_KEY" in line(out, "gateway.embeddings")


def test_a_gateway_that_does_not_know_the_url_or_model_names_both():
    _, out, _ = gateway_only(gateway=DownGateway(
        "gateway http://gateway.example.test:4000/v1/embeddings answered 404: nope"))

    failed = line(out, "gateway.embeddings")
    assert "/v1" in failed and "RAG_EMBED_MODEL" in failed


def test_an_unset_gateway_is_refused_before_any_check():
    code, out, err = gateway_only(environ={"LITELLM_BASE_URL": ""})

    assert code == 2 and out == ""
    assert "LITELLM_BASE_URL" in err and "RAG_RETRIEVAL=keyword" in err


def test_a_gateway_address_that_is_not_http_is_reported_as_a_gateway_problem():
    code, out, _ = run_suite(
        "--only", "gateway", no_gateway=True,
        environ={"LITELLM_BASE_URL": "gateway.example.test:4000/v1"})

    assert code == 1
    assert "http(s)" in line(out, "gateway.embeddings")


def test_the_groups_that_do_not_reach_the_gateway_need_none_named():
    code, out, _ = run_suite("--only", "qdrant", environ={"LITELLM_BASE_URL": ""})

    assert code == 0, out


def test_hybrid_retrieval_without_an_embedding_model_is_refused_before_any_check():
    code, out, err = gateway_only(environ={"RAG_EMBED_MODEL": ""})

    assert code == 2 and out == ""
    assert "RAG_EMBED_MODEL" in err and "RAG_RETRIEVAL=keyword" in err


def test_the_groups_that_never_embed_need_no_embedding_model():
    code, out, _ = run_suite("--only", "qdrant", environ={"RAG_EMBED_MODEL": ""})

    assert code == 0, out


class DriftingGateway(FakeModelGateway):
    """Embeds documents and queries at different dimensions: a misconfigured model."""

    def embed_query(self, text):
        return super().embed_query(text)[:32]


def test_a_gateway_whose_dimension_differs_between_calls_fails():
    code, out, _ = gateway_only(gateway=DriftingGateway())

    assert code == 1
    failed = line(out, "gateway.dimension")
    assert failed.startswith("[FAIL]") and "64" in failed and "32" in failed


class EmptyVectors(FakeModelGateway):
    def embed(self, texts):
        return [[] for _ in texts]

    def embed_query(self, text):
        return []


def test_a_gateway_that_returns_empty_vectors_fails():
    code, out, _ = gateway_only(gateway=EmptyVectors())

    assert code == 1
    assert line(out, "gateway.embeddings").startswith("[FAIL]")
    assert "empty" in line(out, "gateway.embeddings")


class RaggedBatch(FakeModelGateway):
    """Answers a batch with vectors of different lengths."""

    def embed(self, texts):
        vectors = super().embed(texts)
        return [vector[: len(vector) - number] for number, vector in enumerate(vectors)]


def test_a_gateway_whose_vectors_differ_within_a_batch_fails():
    code, out, _ = gateway_only(gateway=RaggedBatch())

    assert code == 1
    assert line(out, "gateway.dimension").startswith("[FAIL]")


# --- Qdrant --------------------------------------------------------------------------

def qdrant_only(**kwargs):
    return run_suite("--only", "qdrant", **kwargs)


def test_qdrant_answers_with_its_key_and_refuses_a_wrong_one():
    code, out, _ = qdrant_only()

    assert code == 0, out
    assert "[ OK ]" in line(out, "qdrant.reachable")
    assert "refused" in line(out, "qdrant.wrong-key")
    assert "[ OK ]" in line(out, "qdrant.wrong-key")


def test_a_qdrant_that_refuses_the_configured_key_names_the_key_pair():
    code, out, _ = qdrant_only(qdrant=Qdrant(key="some-other-key"))

    assert code == 1
    failed = line(out, "qdrant.reachable")
    assert failed.startswith("[FAIL]") and "QDRANT_API_KEY" in failed
    assert "QDRANT__SERVICE__API_KEY" in failed and "closed-network-setup.md section 10" in failed
    assert line(out, "qdrant.wrong-key").startswith("[SKIP]")


def test_a_qdrant_that_accepts_any_key_fails_the_wrong_key_check():
    code, out, _ = qdrant_only(qdrant=Qdrant(accept_any_key=True))

    assert code == 1
    failed = line(out, "qdrant.wrong-key")
    assert failed.startswith("[FAIL]") and "accepted a wrong API key" in failed


def test_an_unreachable_qdrant_fails_with_its_cause():
    class Unreachable:
        def connect(self, location, api_key):
            raise ConnectionError("connection refused")

    code, out, _ = qdrant_only(qdrant=Unreachable())

    assert code == 1
    assert "unreachable" in line(out, "qdrant.reachable") and "NO_PROXY" in line(out, "qdrant.reachable")


def test_without_a_key_the_wrong_key_check_is_skipped_with_its_reason():
    code, out, _ = qdrant_only(qdrant=Qdrant(accept_any_key=True), environ={"QDRANT_API_KEY": ""})

    assert code == 0, out
    assert line(out, "qdrant.wrong-key").startswith("[SKIP]")
    assert "QDRANT_API_KEY is not set" in line(out, "qdrant.wrong-key")


def test_the_rerank_check_is_skipped_without_a_model_and_runs_with_one():
    _, without, _ = gateway_only()
    code, with_model, _ = gateway_only(environ={"RAG_RERANK_MODEL": "rerank-model"})

    assert line(without, "gateway.rerank").startswith("[SKIP]")
    assert "RAG_RERANK_MODEL is not set" in line(without, "gateway.rerank")
    assert code == 0 and line(with_model, "gateway.rerank").startswith("[ OK ]")


def test_a_reranker_that_cannot_tell_relevance_fails():
    class Flat(FakeModelGateway):
        def rerank(self, query, texts):
            return [0.5 for _ in texts]

    code, out, _ = gateway_only(gateway=Flat(), environ={"RAG_RERANK_MODEL": "rerank-model"})

    assert code == 1 and line(out, "gateway.rerank").startswith("[FAIL]")


def test_a_gateway_that_refuses_a_long_passage_blames_the_chunk_size():
    class Limited(FakeModelGateway):
        def embed(self, texts):
            if any(self.count_tokens(t) > 100 for t in texts):
                raise GatewayError("gateway answered 400: maximum context length is 100 tokens")
            return super().embed(texts)

    code, out, _ = gateway_only(gateway=Limited())

    assert code == 1
    failed = line(out, "gateway.input-limit")
    assert failed.startswith("[FAIL]") and "RAG_CHUNK_TOKENS" in failed


def test_a_long_text_is_split_so_every_passage_is_embedded():
    code, out, _ = gateway_only()

    assert code == 0
    assert "Passages" in line(out, "gateway.input-limit")


# --- the Knowledge index round trip --------------------------------------------------

INDEX_CHECKS = ["index.sync", "index.keyword", "index.dense", "index.tail-keyword",
                "index.tail-meaning", "index.status", "index.replace", "index.remove",
                "index.rebuild", "index.switch-back"]


def index_only(**kwargs):
    return run_suite("--only", "index", **kwargs)


def test_the_round_trip_passes_every_check_and_leaves_nothing_behind():
    qdrant = Qdrant()

    code, out, _ = index_only(qdrant=qdrant)

    assert code == 0, out
    for name in INDEX_CHECKS:
        assert line(out, name).startswith("[ OK ]"), out
    assert qdrant.names() == set()
    assert "[ OK ] cleanup.collections" in out and "[ OK ] cleanup.folder" in out


def test_the_live_index_is_untouched_by_a_run():
    qdrant = Qdrant()
    live = KnowledgeIndex(Settings(collection="knowledge"), good_gateway(), client=qdrant.client)
    live.sync(InMemoryCorpus([Document("a", "Alpha", "u", "ENG", "1", "alpha beta gamma")]))
    before = snapshot(qdrant)

    code, out, _ = index_only(qdrant=qdrant)

    assert code == 0, out
    assert snapshot(qdrant) == before
    assert [p.document_id for p in live.search("alpha")] == ["default:a"]


def snapshot(qdrant):
    names = sorted(qdrant.names())
    return names, {name: sorted(
        (str(r.id), str(r.payload)) for r in qdrant.client.scroll(name, limit=1000, with_payload=True)[0])
        for name in names}


def test_the_throwaway_collection_is_its_own_name_not_the_live_one():
    _, out, _ = index_only()

    assert "live collection:  knowledge (not touched)" in out
    assert "throwaway:        acceptance-" in out


class Inverted(FakeModelGateway):
    """An embedding model whose query vectors point away from the meaning of the query: what a
    model that never read the end of a Passage looks like to a search for that end."""

    def embed_query(self, text):
        return [-x for x in super().embed_query(text)]


def test_an_embedding_model_that_does_not_find_by_meaning_fails_the_dense_checks_only():
    code, out, _ = index_only(gateway=Inverted(synonyms=probe_synonyms()))

    assert code == 1
    assert line(out, "index.keyword").startswith("[ OK ]")  # the keyword side still works
    assert line(out, "index.tail-keyword").startswith("[ OK ]")  # and the tail was indexed
    assert line(out, "index.dense").startswith("[FAIL]")
    failed = line(out, "index.tail-meaning")
    assert failed.startswith("[FAIL]") and "RAG_CHUNK_TOKENS" in failed
    assert "closed-network-setup.md section 10" in failed


class Delegating:
    """A Qdrant client that does what the real one does, but for the calls it is told to break:
    each is a function from the real method to the method to use instead."""

    def __init__(self, client, **broken):
        self._client = client
        self._replaced = {name: make(getattr(client, name)) for name, make in broken.items()}

    def __getattr__(self, name):
        return self._replaced.get(name) or getattr(self._client, name)


class Wrapped(Qdrant):
    def __init__(self, **broken):
        super().__init__()
        self._broken = broken

    def connect(self, location, api_key):
        client = super().connect(location, api_key)
        return Delegating(client, **self._broken) if client is self.client else client


def ignore(real):
    return lambda *args, **kwargs: None


def forbid(message):
    def forbidden(real):
        def method(*args, **kwargs):
            raise RuntimeError(message)
        return method
    return forbidden


def test_a_qdrant_that_keeps_deleted_passages_fails_remove_and_is_still_cleaned():
    qdrant = Wrapped(delete=ignore)

    code, out, _ = index_only(qdrant=qdrant)

    assert code == 1
    assert line(out, "index.remove").startswith("[FAIL]")
    assert qdrant.names() == set()  # cleaned up despite the failures


def forbid_after(allowed, message):
    """A client method that works ``allowed`` times, then is forbidden."""
    def broken(real):
        calls = []

        def method(*args, **kwargs):
            calls.append(1)
            if len(calls) > allowed:
                raise RuntimeError(message)
            return real(*args, **kwargs)
        return method
    return broken


def test_a_rebuild_that_cannot_go_live_fails_and_is_still_cleaned():
    qdrant = Wrapped(update_collection_aliases=forbid_after(1, "alias operations are forbidden"))

    code, out, _ = index_only(qdrant=qdrant)

    assert code == 1
    assert line(out, "index.rebuild").startswith("[FAIL]")
    assert line(out, "index.switch-back").startswith("[SKIP]")
    assert qdrant.names() == set()


def test_a_failed_cleanup_is_a_failure_that_names_what_is_left():
    qdrant = Wrapped(delete_collection=forbid("delete forbidden"))

    code, out, _ = index_only(qdrant=qdrant)

    assert code == 1
    assert line(out, "cleanup.collections").startswith("[FAIL]")
    assert "delete forbidden" in line(out, "cleanup.collections")
    assert "[ OK ] cleanup.folder" in out


def test_without_a_working_gateway_the_index_group_is_skipped_and_creates_nothing():
    qdrant = Qdrant()

    code, out, _ = run_suite("--only", "gateway", "--only", "index", qdrant=qdrant,
                             gateway=DownGateway("gateway http://g unreachable"))

    assert code == 1
    assert line(out, "index.sync").startswith("[SKIP]")
    assert qdrant.names() == set()


def test_a_corpus_whose_documents_cannot_be_indexed_reports_the_reason():
    class Refusing(FakeModelGateway):
        def embed(self, texts):
            if any(t.startswith("long-note") for t in texts):
                raise GatewayError("gateway answered 400: input too long")
            return super().embed(texts)

    code, out, _ = index_only(gateway=Refusing(synonyms=probe_synonyms()))

    assert code == 1
    failed = line(out, "index.sync")
    assert "1 failed" in failed and "long-note" in failed and "RAG_CHUNK_TOKENS" in failed


# --- the MCP server -------------------------------------------------------------------

MCP_CHECKS = ["mcp.healthz", "mcp.no-token", "mcp.wrong-token", "mcp.search_docs",
              "mcp.list_spaces", "mcp.get_page"]


def served_index(*documents):
    index = KnowledgeIndex(Settings(), FakeModelGateway())
    if documents:
        index.sync(InMemoryCorpus(documents))
    return index


PAGE = Document("ops/guide", "Runbook", "https://wiki.example.test/guide", "OPS", "1",
                "Restart the scheduler after the failover.")


def mcp_only(url, *extra, **kwargs):
    return run_suite("--only", "mcp", "--mcp-url", url, *extra, **kwargs)


def test_a_working_mcp_server_passes_every_check():
    with serving(served_index(PAGE), tokens=(MCP_TOKEN,)) as url:
        code, out, err = mcp_only(url)

    assert code == 0, out
    for name in MCP_CHECKS:
        assert line(out, name).startswith("[ OK ]"), out
    assert "page default:ops/guide" in line(out, "mcp.get_page")
    assert "OPS" in line(out, "mcp.list_spaces")


def test_the_checks_also_pass_against_an_empty_index():
    with serving(served_index(), tokens=(MCP_TOKEN,)) as url:
        code, out, _ = mcp_only(url)

    assert code == 0, out
    assert "nothing indexed" in line(out, "mcp.get_page")


def test_a_token_passed_explicitly_is_used_instead_of_the_environments():
    with serving(served_index(PAGE), tokens=("explicit-token",)) as url:
        code, out, err = mcp_only(url, "--token", "explicit-token", environ={"MCP_TOKENS": ""})

    assert code == 0, out
    assert "explicit-token" not in out + err


def test_the_url_comes_from_mcp_host_and_port_when_not_given():
    with serving(served_index(PAGE), tokens=(MCP_TOKEN,)) as url:
        host, port = url.removeprefix("http://").split(":")
        code, out, _ = run_suite("--only", "mcp", environ={"MCP_HOST": host, "MCP_PORT": port})

    assert code == 0, out


def test_a_server_that_does_not_know_our_tokens_fails_the_calls_with_the_token_cause():
    with serving(served_index(PAGE), tokens=("another-token",)) as url:
        code, out, _ = mcp_only(url)

    assert code == 1
    assert line(out, "mcp.healthz").startswith("[ OK ]")
    assert line(out, "mcp.wrong-token").startswith("[ OK ]")
    failed = line(out, "mcp.search_docs")
    assert failed.startswith("[FAIL]") and "MCP_TOKENS" in failed and "restart" in failed
    assert '"401 from /mcp"' in failed


def test_a_server_that_lets_requests_in_without_a_token_fails():
    app = build_server(served_index(PAGE)).streamable_http_app(stateless_http=True)
    with serving_app(app) as url:
        code, out, _ = mcp_only(url)

    assert code == 1
    assert "not 401" in line(out, "mcp.no-token") and "[FAIL]" in line(out, "mcp.no-token")
    assert "[FAIL]" in line(out, "mcp.wrong-token")


def test_a_server_whose_qdrant_is_down_fails_healthz():
    index = KnowledgeIndex(Settings(), FakeModelGateway(), client=UnreachableQdrant())
    with serving(index, tokens=(MCP_TOKEN,)) as url:
        code, out, _ = mcp_only(url)

    assert code == 1
    failed = line(out, "mcp.healthz")
    assert failed.startswith("[FAIL]") and "503" in failed and "Qdrant" in failed
    assert line(out, "mcp.search_docs").startswith("[SKIP]")


def test_a_server_that_is_not_running_fails_with_its_cause():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    url = f"http://127.0.0.1:{sock.getsockname()[1]}"
    sock.close()  # nothing listens there now

    code, out, _ = mcp_only(url)

    assert code == 1
    failed = line(out, "mcp.healthz")
    assert failed.startswith("[FAIL]") and "MCP_PORT" in failed and "--mcp-url" in failed
    assert "closed-network-setup.md section 10" in failed


def test_without_any_token_the_tool_checks_fail_and_name_the_setting():
    with serving(served_index(PAGE), tokens=(MCP_TOKEN,)) as url:
        code, out, _ = mcp_only(url, environ={"MCP_TOKENS": ""})

    assert code == 1
    assert line(out, "mcp.no-token").startswith("[ OK ]")
    assert "MCP_TOKENS is empty" in line(out, "mcp.search_docs")


# --- the installation ------------------------------------------------------------------

def install_only(*route, host):
    return run_suite("--only", "install", *route, host=host)


def test_a_complete_systemd_installation_passes():
    host = systemd_host()

    code, out, _ = install_only("--systemd", host=host)

    assert code == 0, out
    for name in ("install.units", "install.active", "install.enabled", "install.env-file",
                 "install.timers"):
        assert line(out, name).startswith("[ OK ]"), out
    assert ["systemctl", "is-active", "--quiet", "local-rag-mcp.service"] in host.calls
    assert "not world-readable" in line(out, "install.env-file")


def test_a_missing_unit_fails_and_points_at_the_installer():
    code, out, _ = install_only("--systemd", host=systemd_host(failing=["cat --no-pager qdrant.service"]))

    failed = line(out, "install.units")
    assert code == 1 and failed.startswith("[FAIL]") and "qdrant.service" in failed
    assert "section 4" in failed
    assert line(out, "install.active").startswith("[SKIP]")


def test_an_inactive_unit_is_named():
    code, out, _ = install_only(
        "--systemd", host=systemd_host(failing=["is-active --quiet local-rag-mcp.service"]))

    failed = line(out, "install.active")
    assert code == 1 and "local-rag-mcp.service" in failed and "qdrant.service" not in failed


def test_a_disabled_unit_is_named():
    code, out, _ = install_only(
        "--systemd", host=systemd_host(failing=["is-enabled --quiet local-rag-sync.timer"]))

    assert code == 1 and "local-rag-sync.timer" in line(out, "install.enabled")


def test_a_world_readable_env_file_fails_without_printing_a_secret():
    host = systemd_host()
    host.answers["stat -c %a"] = "644\n"

    code, out, err = install_only("--systemd", host=host)

    failed = line(out, "install.env-file")
    assert code == 1 and failed.startswith("[FAIL]") and "644" in failed
    assert "/etc/local-rag/local-rag.env" in failed
    assert not any(secret in out + err for secret in SECRETS)


def test_a_timer_without_a_next_run_is_not_scheduled():
    host = systemd_host()
    host.answers["list-timers"] = TIMERS.format(
        sync="local-rag-sync", full="local-rag-full").replace("Thu 2026-10-01 13:00:00 UTC  40min left", "n/a  n/a")

    code, out, _ = install_only("--systemd", host=host)

    assert code == 1 and "local-rag-sync.timer" in line(out, "install.timers")


def test_a_complete_container_installation_passes():
    code, out, _ = install_only("--container", host=container_host())

    assert code == 0, out
    assert "qdrant and mcp are up" in line(out, "install.stack")
    assert "container.env" in line(out, "install.env-file")
    assert "2 timers are scheduled" in line(out, "install.timers")


def test_a_stopped_or_unhealthy_container_fails():
    host = container_host()
    host.answers["docker ps"] = "qdrant Up 3 hours\n"
    _, out, _ = install_only("--container", host=host)
    assert "mcp" in line(out, "install.stack") and "[FAIL]" in line(out, "install.stack")

    host.answers["docker ps"] = "mcp Up 2 minutes (unhealthy)\nqdrant Up 3 hours\n"
    _, out, _ = install_only("--container", host=host)
    assert "unhealthy" in line(out, "install.stack")


def test_the_container_stack_check_says_where_it_must_run_when_docker_is_missing():
    code, out, _ = install_only("--container", host=container_host(failing=["docker ps"]))

    assert code == 1 and "on the host" in line(out, "install.stack")


def test_a_run_that_is_not_told_the_route_fails_instead_of_passing_unchecked():
    code, out, _ = run_suite("--skip", "index", "--skip", "mcp", "--skip", "confluence")

    assert code == 1
    failed = line(out, "install.route")
    assert failed.startswith("[FAIL]") and "--systemd" in failed and "--skip install" in failed
    assert "closed-network-setup.md section 7" in failed
    assert out.rstrip().endswith("ACCEPTANCE FAILED: install.route")
    assert line(out, "gateway.embeddings").startswith("[ OK ]")  # the rest still ran


def test_the_install_group_can_be_left_out_and_is_otherwise_not_optional():
    code, out, _ = run_suite("--only", "gateway", "--only", "qdrant", "--skip", "install")
    assert code == 0 and "install." not in out



def test_asking_for_only_the_install_group_needs_a_route():
    code, out, err = run_suite("--only", "install")
    assert code == 2 and out == "" and "--systemd or --container" in err

    code, out, err = run_suite("--systemd", "--container")
    assert code == 2


# --- Confluence ---------------------------------------------------------------------

CONFLUENCE = {"CONFLUENCE_DOMAIN": "wiki.example.test", "CONFLUENCE_API_TOKEN": "confluence-secret"}


def roots_file(tmp_path, *roots):
    path = tmp_path / "roots.txt"
    path.write_text("# roots\n" + "\n".join(roots) + "\n")
    return str(path)


def confluence_only(tmp_path, *roots, host=None, environ=None):
    return run_suite("--only", "confluence", "--roots-file", roots_file(tmp_path, *roots),
                     environ={**CONFLUENCE, **(environ or {})},
                     host=host or Host({"info": '{"id": "100", "title": "Home"}'}))


def test_confluence_is_read_and_every_root_resolves(tmp_path):
    host = Host({"info": '{"id": "100", "title": "Home"}'})

    code, out, err = confluence_only(tmp_path, "100", "https://wiki.example.test/x?pageId=7", host=host)

    assert code == 0, out
    assert "reads wiki.example.test" in line(out, "confluence.read")
    assert "all 2 Roots resolve" in line(out, "confluence.roots")
    assert ["confluence", "spaces"] in host.calls
    assert all(call[1] in ("spaces", "info") for call in host.calls)  # read-only
    assert "confluence-secret" not in out + err


def test_confluence_is_skipped_when_not_configured():
    code, out, _ = run_suite("--only", "confluence")

    assert code == 0
    assert line(out, "confluence.read").startswith("[SKIP]")
    assert "CONFLUENCE_DOMAIN is not set" in line(out, "confluence.read")
    assert line(out, "confluence.roots").startswith("[SKIP]")


def test_a_confluence_that_cannot_be_read_names_the_likely_causes(tmp_path):
    code, out, _ = confluence_only(tmp_path, "100", host=Host(failing=["spaces"]))

    failed = line(out, "confluence.read")
    assert code == 1 and failed.startswith("[FAIL]") and "CONFLUENCE_BIN" in failed
    assert line(out, "confluence.roots").startswith("[SKIP]")


def test_a_root_that_does_not_resolve_is_named(tmp_path):
    host = Host({"info 100": '{"id": "100"}'}, failing=["info 999"])

    code, out, _ = confluence_only(tmp_path, "100", "999", host=host)

    failed = line(out, "confluence.roots")
    assert code == 1 and "1 of 2 Roots" in failed and "999" in failed


def test_a_missing_roots_file_skips_the_roots_check_with_the_reason(tmp_path):
    code, out, _ = run_suite("--only", "confluence", "--roots-file", str(tmp_path / "none.txt"),
                             environ=CONFLUENCE)

    assert code == 0, out
    assert "no roots file" in line(out, "confluence.roots")


# --- as a whole -----------------------------------------------------------------------

def test_all_groups_pass_together_and_the_summary_says_so(tmp_path):
    host = systemd_host()
    host.answers["info"] = '{"id": "100"}'
    with serving(served_index(PAGE), tokens=(MCP_TOKEN,)) as url:
        code, out, err = run_suite(
            "--systemd", "--mcp-url", url, "--roots-file", roots_file(tmp_path, "100"),
            environ=CONFLUENCE, host=host)

    assert code == 0, out
    assert out.rstrip().endswith("ACCEPTANCE PASSED")
    assert "0 failed" in out
    assert "[FAIL]" not in out


def test_a_failure_anywhere_gives_exit_code_1_and_the_summary_names_the_check():
    code, out, _ = run_suite("--only", "qdrant", qdrant=Qdrant(key="other"))

    assert code == 1
    assert out.rstrip().endswith("ACCEPTANCE FAILED: qdrant.reachable")


def test_unusable_settings_are_a_usage_error_naming_the_variable():
    code, out, err = run_suite(environ={"RAG_TOP_K": "many"})

    assert code == 2 and "RAG_TOP_K" in err and out == ""


def test_the_settings_used_are_printed_and_no_secret_ever_is():
    environ = {"QDRANT_URL": "http://user:url-password@qdrant.example.test:6333",
               "RAG_RERANK_MODEL": "rerank-model"}
    for failing in (Qdrant(), Qdrant(key="other")):
        code, out, err = run_suite("--only", "gateway", "--only", "qdrant", "--only", "index",
                                   "--only", "mcp", environ=environ, qdrant=failing,
                                   gateway=DownGateway(f"gateway answered 401 for {ENVIRONMENT['LITELLM_API_KEY']}"))

        assert "RAG_EMBED_MODEL" not in out  # names of settings are not values
        assert "embed-model" in out and "gateway.example.test:4000" in out
        assert "key set" in out and "mcp tokens:       2" in out
        for secret in [*SECRETS, "url-password"]:
            assert secret not in out + err, secret


def test_the_install_group_runs_on_a_host_without_the_packages_dependencies():
    """The container route's install checks run on the host, with only PYTHONPATH=src."""
    program = '''
import sys
class Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {"qdrant_client", "mcp", "httpx2", "anyio", "llama_index", "bs4",
                                  "nltk", "markdownify", "uvicorn"}:
            raise ImportError("blocked: " + name)
sys.meta_path.insert(0, Block())
from local_rag import acceptance
code = acceptance.main(["--only", "install", "--systemd"], {}, run=lambda command: "640\\n")
sys.exit(0 if code in (0, 1) else 9)
'''
    done = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True,
                          env={"PYTHONPATH": str(REPO / "src"), "PATH": os.environ["PATH"]})

    assert done.returncode == 0, done.stderr
    assert "install.units" in done.stdout


# --- what the review asked to be pinned down ------------------------------------------

def test_the_marker_sits_at_the_end_of_a_nearly_full_passage():
    from local_rag import acceptance_corpus as corpus_files
    from local_rag.chunker import Chunker

    gateway = good_gateway()
    chunker = Chunker(gateway.count_tokens, size=400, overlap=60)
    for _ in range(5):  # the random mark changes the text
        run = corpus_files.Run.new()
        text = corpus_files.long_text(run, chunk_tokens=400, count_tokens=gateway.count_tokens,
                                      split=chunker.split)
        last = chunker.split(text)[-1]

        assert gateway.count_tokens(last) >= 0.9 * 400
        assert run.tail in last and text.rstrip().endswith(f"{run.tail}.")
        assert len(chunker.split(text)) >= 10


def test_a_secret_inside_the_gateway_url_is_never_printed():
    environ = {"LITELLM_BASE_URL": "http://svc:url-secret-pw@gateway.example.test:4000/v1"}

    code, out, err = run_suite("--only", "gateway", environ=environ, gateway=DownGateway(
        "gateway http://svc:url-secret-pw@gateway.example.test:4000/v1/embeddings unreachable"))

    assert code == 1 and "gateway.example.test" in out
    assert "url-secret-pw" not in out + err


def test_a_server_that_rejects_a_missing_token_but_accepts_a_wrong_one_fails_that_check():
    inner = build_server(served_index(PAGE)).streamable_http_app(stateless_http=True)

    async def app(scope, receive, send):
        headers = dict(scope.get("headers", []))
        if scope["type"] == "http" and scope["path"] != "/healthz" and b"authorization" not in headers:
            await send({"type": "http.response.start", "status": 401, "headers": []})
            return await send({"type": "http.response.body", "body": b""})
        return await inner(scope, receive, send)

    with serving_app(app) as url:
        code, out, _ = mcp_only(url)

    assert code == 1
    assert line(out, "mcp.no-token").startswith("[ OK ]")
    assert "not 401" in line(out, "mcp.wrong-token")


def test_a_switch_back_that_does_nothing_fails():
    qdrant = Wrapped(update_collection_aliases=do_nothing_after(2))

    code, out, _ = index_only(qdrant=qdrant)

    assert code == 1
    assert line(out, "index.rebuild").startswith("[ OK ]")
    assert "after switch-back" in line(out, "index.switch-back")
    assert qdrant.names() == set()


def do_nothing_after(allowed):
    """A client method that works ``allowed`` times, then silently does nothing."""
    def broken(real):
        calls = []

        def method(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs) if len(calls) <= allowed else None
        return method
    return broken


# --- the Retrieval mode ----------------------------------------------------------------

KEYWORD_ONLY = {"RAG_RETRIEVAL": "keyword", "RAG_EMBED_MODEL": "", "LITELLM_BASE_URL": ""}
NO_STACK_EXTRAS = ("--skip", "install", "--skip", "mcp", "--skip", "confluence")


def test_a_keyword_only_suite_passes_with_no_gateway_and_skips_what_needs_embeddings():
    code, out, _ = run_suite(*NO_STACK_EXTRAS, environ=KEYWORD_ONLY, no_gateway=True)

    assert code == 0, out
    assert "ACCEPTANCE PASSED" in out and "[FAIL]" not in out
    for name in ("gateway.embeddings", "gateway.dimension", "gateway.input-limit",
                 "gateway.rerank", "index.dense", "index.tail-meaning"):
        assert line(out, name).startswith("[SKIP]"), name
    assert "keyword" in line(out, "gateway.embeddings")
    for name in ("index.sync", "index.keyword", "index.tail-keyword", "index.status",
                 "index.replace", "index.remove", "index.rebuild", "index.switch-back"):
        assert line(out, name).startswith("[ OK ]"), name


def test_a_keyword_only_suite_still_checks_a_reranker_that_is_named():
    code, out, _ = run_suite(
        *NO_STACK_EXTRAS, environ={**KEYWORD_ONLY, "RAG_RERANK_MODEL": "rerank-model",
                                   "LITELLM_BASE_URL": "http://gateway.example.test:4000/v1"})

    assert code == 0, out
    assert line(out, "gateway.embeddings").startswith("[SKIP]")
    assert line(out, "gateway.rerank").startswith("[ OK ]")


def test_a_hybrid_suite_with_a_dead_embedding_model_fails_instead_of_skipping():
    code, out, _ = run_suite(
        *NO_STACK_EXTRAS, gateway=DownGateway(GatewayError("gateway unreachable")))

    assert code == 1
    assert line(out, "gateway.embeddings").startswith("[FAIL]")


def test_the_suite_says_which_retrieval_mode_it_ran_in():
    _, keyword, _ = run_suite("--only", "qdrant", environ=KEYWORD_ONLY)
    _, hybrid, _ = run_suite("--only", "qdrant")

    assert "retrieval:        keyword" in keyword
    assert "retrieval:        hybrid" in hybrid
