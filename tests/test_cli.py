import io

import pytest

import fake_confluence
from qdrant_client import QdrantClient

from local_rag import KnowledgeIndex, Settings
from local_rag.cli import main
from local_rag.fakes import FakeModelGateway
from conftest import SYNONYMS
from fake_litellm_server import FakeLiteLLM


@pytest.fixture
def folder(tmp_path):
    root = tmp_path / "docs"
    (root / "dba").mkdir(parents=True)
    (root / "garage").mkdir()
    (root / "dba" / "ora-00942.md").write_text(
        "ORA-00942: table or view does not exist. Check the schema owner and grants.")
    (root / "dba" / "ora-12154.md").write_text(
        "ORA-12154: TNS could not resolve the connect identifier. Check tnsnames.")
    (root / "garage" / "battery.md").write_text(
        "When the car will not start, the battery is dead more often than not.")
    return root


@pytest.fixture
def cli(monkeypatch):
    """Runs the command line against one shared in-process Qdrant and the fake gateway."""
    client = QdrantClient(location=":memory:")
    gateway = FakeModelGateway(synonyms=SYNONYMS)

    def run(*argv, **environ):
        out, err = io.StringIO(), io.StringIO()
        code = main(list(argv), environ, stdout=out, stderr=err,
                    gateway=gateway, client=client)
        return code, out.getvalue(), err.getvalue()
    run.client, run.gateway = client, gateway
    return run


def test_a_synced_folder_is_found_by_search(cli, folder):
    code, out, _ = cli("sync", "--folder", str(folder))
    assert code == 0 and "3 added" in out

    code, out, _ = cli("search", "ORA-00942")

    assert code == 0
    assert "ora-00942" in out


def sync(cli, folder):
    code, out, err = cli("sync", "--folder", str(folder))
    assert (code, err) == (0, ""), out
    return out


def test_search_prints_title_link_space_and_page_id(cli, folder):
    sync(cli, folder)

    code, out, _ = cli("search", "ORA-00942")

    assert code == 0
    first = out.split("\n\n")[0]
    assert "ora-00942" in first
    assert "link: file://" in first and "ora-00942.md" in first
    assert "space: dba" in first
    assert "page id: dba/ora-00942.md" in first
    assert "table or view does not exist" in first


def test_search_can_be_narrowed_to_a_space_and_limited(cli, folder):
    sync(cli, folder)

    _, out, _ = cli("search", "check", "--space", "dba")
    assert "space: dba" in out and "space: garage" not in out

    _, out, _ = cli("search", "check", "-k", "1")
    assert out.count("page id:") == 1


def test_search_with_no_match_says_so(cli):
    code, out, _ = cli("search", "anything")

    assert (code, out.strip()) == (0, "No results.")


def test_sync_reports_what_changed(cli, folder):
    sync(cli, folder)
    (folder / "dba" / "ora-12154.md").write_text("ORA-12154 is now resolved by tnsnames.")
    (folder / "garage" / "battery.md").unlink()

    code, out, _ = cli("sync", "--folder", str(folder))

    assert code == 0
    assert "Sync: 0 added, 1 replaced, 1 removed, 1 unchanged, 0 failed" in out


def test_a_sync_with_failed_documents_prints_them_and_exits_non_zero(cli, folder):
    (folder / "garage" / "empty.md").write_text("")

    code, out, _ = cli("sync", "--folder", str(folder))

    assert code == 1
    assert "3 added" in out and "1 failed" in out
    assert "failed garage/empty.md:" in out


def test_status_before_anything_is_synced(cli):
    code, out, _ = cli("status")

    assert code == 0
    assert "generation:  none yet" in out
    assert "documents:   0" in out
    assert "last sync:   never" in out


def test_status_prints_generation_counts_format_and_last_sync(cli, folder):
    sync(cli, folder)

    code, out, _ = cli("status")

    assert code == 0
    assert "generation:  knowledge.gen-" in out
    assert "access:      read-write" in out
    assert "documents:   3" in out and "passages:    3" in out
    assert "schema 1, embedding model 'embed' (64 dimensions)" in out
    assert "chunks of 400 tokens overlapping by 60" in out
    assert "last sync:   Sync completed at " in out
    assert "3 added, 0 replaced, 0 removed, 0 unchanged, 0 failed" in out


def test_rebuild_promotes_a_complete_index(cli, folder):
    sync(cli, folder)
    before = generation(cli)

    code, out, _ = cli("rebuild", "--folder", str(folder))

    assert code == 0
    assert "Rebuild: promoted to the Live index, 3 indexed, 0 failed" in out
    assert generation(cli) != before


def test_a_refused_rebuild_exits_non_zero_and_leaves_the_live_index(cli, folder):
    sync(cli, folder)
    before = generation(cli)
    (folder / "garage" / "empty.md").write_text("")

    code, out, _ = cli("rebuild", "--folder", str(folder))

    assert code == 1
    assert "Rebuild: refused" in out and "1 of 4 Documents failed" in out
    assert "failed garage/empty.md:" in out
    assert generation(cli) == before


def test_the_rebuild_threshold_comes_from_the_environment(cli, folder):
    sync(cli, folder)
    (folder / "garage" / "empty.md").write_text("")

    code, out, _ = cli("rebuild", "--folder", str(folder),
                       RAG_REBUILD_FAILURE_THRESHOLD="0.25")

    assert code == 0
    assert "promoted" in out and "3 indexed, 1 failed" in out


def test_switch_back_returns_to_the_previous_index(cli, folder):
    sync(cli, folder)
    first = generation(cli)
    cli("rebuild", "--folder", str(folder))

    code, out, _ = cli("switch-back")

    assert code == 0 and "previous index" in out
    assert generation(cli) == first


def test_switch_back_without_a_previous_index_fails(cli, folder):
    sync(cli, folder)

    code, out, err = cli("switch-back")

    assert code == 1 and out == ""
    assert err.startswith("error: ")


def generation(cli):
    _, out, _ = cli("status")
    return next(line for line in out.splitlines() if line.startswith("generation:"))


def test_a_folder_that_does_not_exist_is_not_an_empty_corpus(cli, folder, tmp_path):
    sync(cli, folder)

    code, _, err = cli("sync", "--folder", str(tmp_path / "unmounted"))

    assert code == 2 and "is not a folder" in err
    assert "documents:   3" in cli("status")[1]


@pytest.mark.parametrize("argv", [
    ["sync"],
    ["rebuild"],
    ["sync", "--folder", "x", "--root", "100"],
    ["sync", "--roots-file", "missing.txt"],
])
def test_a_command_without_a_usable_corpus_is_a_usage_error(cli, argv):
    code, out, err = cli(*argv)

    assert code == 2 and out == "" and err.startswith("error: ")


def test_an_unknown_command_is_a_usage_error(cli):
    assert cli("frobnicate")[0] == 2


def test_an_unusable_setting_is_named_in_the_error(cli):
    code, _, err = cli("status", RAG_TOP_K="many")

    assert code == 1 and "RAG_TOP_K" in err and "'many'" in err


def test_a_threshold_out_of_range_is_an_error(cli):
    code, _, err = cli("status", RAG_REBUILD_FAILURE_THRESHOLD="2")

    assert code == 1 and "rebuild_failure_threshold" in err


def test_an_unreachable_model_gateway_fails_each_document_and_exits_non_zero(folder):
    client = QdrantClient(location=":memory:")
    out, err = io.StringIO(), io.StringIO()

    code = main(["sync", "--folder", str(folder)],
                {"LITELLM_BASE_URL": "http://127.0.0.1:1/v1"},
                stdout=out, stderr=err, client=client)

    assert code == 1
    assert "Sync: 0 added, 0 replaced, 0 removed, 0 unchanged, 3 failed" in out.getvalue()
    assert "Traceback" not in out.getvalue() + err.getvalue()


def test_the_real_model_gateway_is_built_from_the_environment(folder):
    server = FakeLiteLLM(FakeModelGateway(synonyms=SYNONYMS)).start()
    client = QdrantClient(location=":memory:")
    environ = {"LITELLM_BASE_URL": server.base_url, "LITELLM_API_KEY": server.api_key,
               "RAG_RERANK_MODEL": "rerank"}

    def run(*argv):
        out = io.StringIO()
        return main(list(argv), environ, stdout=out, stderr=io.StringIO(), client=client), \
            out.getvalue()

    try:
        assert run("sync", "--folder", str(folder))[0] == 0
        code, out = run("search", "ORA-00942")
    finally:
        server.stop()

    assert code == 0 and "page id: dba/ora-00942.md" in out
    assert "/v1/embeddings" in server.paths() and "/v1/rerank" in server.paths()


def test_a_second_run_while_one_is_running_is_refused(cli, folder):
    index = KnowledgeIndex(Settings(), cli.gateway, client=cli.client)

    with index._exclusive.hold("Sync"):
        code, _, err = cli("sync", "--folder", str(folder))

    assert code == 1 and "another run holds the lock" in err


def test_a_confluence_corpus_is_synced_from_roots(cli, tmp_path):
    site = fake_confluence.FakeSite(tmp_path)
    site.page("100", "ENG home", body="# ENG home\n\nStart here.\n")
    site.page("101", "Fixing ORA-00942", parent="100",
              body="Table or view does not exist. Check the grants.\n")
    roots = tmp_path / "roots.txt"
    roots.write_text("# roots\n100   # ENG home\n")

    code, out, _ = cli("sync", "--roots-file", str(roots), CONFLUENCE_BIN=str(site.bin))

    assert code == 0 and "2 added" in out
    _, out, _ = cli("search", "ORA-00942")
    assert "page id: 101" in out and "space: ENG" in out
    assert "link: https://wiki.example.test/pages/viewpage.action?pageId=101" in out


def test_a_listing_that_failed_makes_the_sync_fail_and_removes_nothing(cli, tmp_path):
    site = fake_confluence.FakeSite(tmp_path)
    site.page("100", "ENG home", body="Start here.\n")
    site.page("101", "Fixing ORA-00942", parent="100", body="Check the grants.\n")
    environ = {"CONFLUENCE_BIN": str(site.bin)}
    cli("sync", "--root", "100", **environ)
    site.fail("100", "children")

    code, out, _ = cli("sync", "--root", "100", **environ)

    assert code == 1 and "failed root 100:" in out
    assert "documents:   2" in cli("status")[1]


def test_a_roots_file_that_is_not_text_is_a_usage_error(cli, tmp_path):
    roots = tmp_path / "roots.bin"
    roots.write_bytes(b"\xff\xfe\x00\x80")

    code, _, err = cli("sync", "--roots-file", str(roots))

    assert code == 2 and "cannot read the roots file" in err
