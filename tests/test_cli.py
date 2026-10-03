import io

import pytest
from qdrant_client import QdrantClient

import fake_confluence
from conftest import SYNONYMS
from fake_litellm_server import FakeLiteLLM
from index_states import NAMES
from local_rag.cli import main
from local_rag.exclusive import ExclusiveRun
from local_rag.fakes import FakeModelGateway

EMBEDDING = {
    "RAG_EMBED_MODEL": "embed",  # Hybrid retrieval, the default, needs these named
    "LITELLM_BASE_URL": "http://gateway.example.test:4000/v1",
}


@pytest.fixture
def folder(tmp_path):
    root = tmp_path / "docs"
    (root / "dba").mkdir(parents=True)
    (root / "garage").mkdir()
    (root / "dba" / "ora-00942.md").write_text(
        "ORA-00942: table or view does not exist. Check the schema owner and grants."
    )
    (root / "dba" / "ora-12154.md").write_text(
        "ORA-12154: TNS could not resolve the connect identifier. Check tnsnames."
    )
    (root / "garage" / "battery.md").write_text(
        "When the car will not start, the battery is dead more often than not."
    )
    declare(tmp_path, docs=f'kind = "folder"\npath = "{root}"')
    return root


def declare(directory, **corpora):
    """Write the Corpus declaration file: each keyword is a Corpus, its value the table body."""
    path = directory / "corpora.toml"
    path.write_text("".join(f"[corpus.{name}]\n{body}\n\n" for name, body in corpora.items()))
    return path


def declared(folder):
    """The environment that points the command line at the declaration beside the folder."""
    return {"RAG_CORPORA_FILE": str(folder.parent / "corpora.toml")}


@pytest.fixture
def cli(monkeypatch, tmp_path):
    """Runs the command line against one shared in-process Qdrant and the fake gateway."""
    client = QdrantClient(location=":memory:")
    gateway = FakeModelGateway(synonyms=SYNONYMS)

    def run(*argv, **environ):
        out, err = io.StringIO(), io.StringIO()
        code = main(
            list(argv),
            {**EMBEDDING, "RAG_CORPORA_FILE": str(tmp_path / "corpora.toml"), **environ},
            stdout=out,
            stderr=err,
            gateway=gateway,
            client=client,
        )
        return code, out.getvalue(), err.getvalue()

    run.client, run.gateway = client, gateway
    return run


def test_a_synced_folder_is_found_by_search(cli, folder):
    code, out, _ = cli("sync", "docs")
    assert code == 0 and "3 added" in out

    code, out, _ = cli("search", "ORA-00942")

    assert code == 0
    assert "ora-00942" in out


def sync(cli, folder):
    code, out, err = cli("sync", "docs")
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
    assert "page id: docs:dba/ora-00942.md" in first
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

    code, out, _ = cli("sync", "docs")

    assert code == 0
    assert "Sync: 0 added, 1 replaced, 1 removed, 1 unchanged, 0 failed" in out


def test_a_sync_with_failed_documents_prints_them_and_exits_non_zero(cli, folder):
    (folder / "garage" / "empty.md").write_text("")

    code, out, _ = cli("sync", "docs")

    assert code == 1
    assert "3 added" in out and "1 failed" in out
    assert "failed docs:garage/empty.md:" in out


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

    code, out, _ = cli("rebuild")

    assert code == 0
    assert "Rebuild: promoted to the Live index, 3 indexed, 0 failed" in out
    assert generation(cli) != before


def test_a_refused_rebuild_exits_non_zero_and_leaves_the_live_index(cli, folder):
    sync(cli, folder)
    before = generation(cli)
    (folder / "garage" / "empty.md").write_text("")

    code, out, _ = cli("rebuild")

    assert code == 1
    assert "Rebuild: refused" in out and "1 of 4 Documents failed" in out
    assert "failed docs:garage/empty.md:" in out
    assert generation(cli) == before


def test_the_rebuild_threshold_comes_from_the_environment(cli, folder):
    sync(cli, folder)
    (folder / "garage" / "empty.md").write_text("")

    code, out, _ = cli("rebuild", RAG_REBUILD_FAILURE_THRESHOLD="0.25")

    assert code == 0
    assert "promoted" in out and "3 indexed, 1 failed" in out


def test_switch_back_returns_to_the_previous_index(cli, folder):
    sync(cli, folder)
    first = generation(cli)
    cli("rebuild")

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

    declare(tmp_path, docs=f'kind = "folder"\npath = "{tmp_path / "unmounted"}"')

    code, _, err = cli("sync", "docs")

    assert code == 2 and "is not a folder" in err
    assert "documents:   3" in cli("status")[1]


@pytest.mark.parametrize(
    "argv",
    [
        ["sync"],
        ["sync", "nope"],
        ["sync", "docs", "--corpora", "missing.toml"],
        ["rebuild", "--corpora", "missing.toml"],
    ],
)
def test_a_command_without_a_usable_corpus_is_a_usage_error(cli, folder, argv):
    code, out, err = cli(*argv)

    assert code == 2 and out == ""


def test_an_unknown_corpus_names_the_declared_ones(cli, folder):
    code, _, err = cli("sync", "nope")

    assert code == 2 and "'nope'" in err and "docs" in err


def test_a_declaration_that_is_not_valid_is_a_usage_error(cli, tmp_path):
    (tmp_path / "corpora.toml").write_text('[corpus.docs]\nkind = "folder"\npaht = "x"\n')

    code, _, err = cli("sync", "docs")

    assert code == 2 and "paht" in err


def test_a_declaration_file_that_is_not_text_is_a_usage_error(cli, tmp_path):
    (tmp_path / "corpora.toml").write_bytes(b"\xff\xfe\x00\x80")

    code, _, err = cli("sync", "docs")

    assert code == 2 and "not valid TOML" in err


def test_the_declaration_file_can_be_named_on_the_command_line(cli, folder, tmp_path):
    elsewhere = tmp_path / "elsewhere.toml"
    elsewhere.write_text(f'[corpus.notes]\nkind = "folder"\npath = "{folder}"\n')

    code, out, _ = cli(
        "sync", "notes", "--corpora", str(elsewhere), RAG_CORPORA_FILE=str(tmp_path / "unused.toml")
    )

    assert code == 0 and "3 added" in out


def test_syncing_one_corpus_leaves_the_documents_of_another_alone(cli, folder, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    (other / "notes.md").write_text("Notes about ZX-9000.")
    declare(
        tmp_path,
        docs=f'kind = "folder"\npath = "{folder}"',
        other=f'kind = "folder"\npath = "{other}"',
    )
    cli("sync", "docs")
    cli("sync", "other")
    (other / "notes.md").unlink()
    (other / "more.md").write_text("More about QX-5.")

    code, out, _ = cli("sync", "other")

    assert code == 0 and "1 added, 0 replaced, 1 removed" in out
    assert "docs:dba/ora-00942.md" in cli("search", "ORA-00942")[1]
    status = cli("status")[1]
    assert "documents:   4" in status
    assert "corpus docs:" in status and "corpus other:" in status


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

    code = main(
        ["sync", "docs"],
        {
            **EMBEDDING,
            **declared(folder),
            "LITELLM_BASE_URL": "http://127.0.0.1:1/v1",
            "RAG_EMBED_RETRY_BACKOFF": "0",
        },
        stdout=out,
        stderr=err,
        client=client,
    )

    assert code == 1
    assert "Sync: 0 added, 0 replaced, 0 removed, 0 unchanged, 3 failed" in out.getvalue()
    assert "Traceback" not in out.getvalue() + err.getvalue()


def test_the_real_model_gateway_is_built_from_the_environment(folder):
    server = FakeLiteLLM(FakeModelGateway(synonyms=SYNONYMS)).start()
    client = QdrantClient(location=":memory:")
    environ = {
        **EMBEDDING,
        **declared(folder),
        "LITELLM_BASE_URL": server.base_url,
        "LITELLM_API_KEY": server.api_key,
        "RAG_RERANK_MODEL": "rerank",
    }

    def run(*argv):
        out = io.StringIO()
        return main(
            list(argv), environ, stdout=out, stderr=io.StringIO(), client=client
        ), out.getvalue()

    try:
        assert run("sync", "docs")[0] == 0
        code, out = run("search", "ORA-00942")
    finally:
        server.stop()

    assert code == 0 and "page id: docs:dba/ora-00942.md" in out
    assert "/v1/embeddings" in server.paths() and "/v1/rerank" in server.paths()


def test_a_second_run_while_one_is_running_is_refused(cli, folder):
    with ExclusiveRun(cli.client, NAMES.lock).hold("Sync"):
        code, _, err = cli("sync", "docs")

    assert code == 1 and "another run holds the lock" in err


def test_a_confluence_corpus_is_synced_from_roots(cli, tmp_path):
    site = fake_confluence.FakeSite(tmp_path)
    site.page("100", "ENG home", body="# ENG home\n\nStart here.\n")
    site.page(
        "101",
        "Fixing ORA-00942",
        parent="100",
        body="Table or view does not exist. Check the grants.\n",
    )
    declare(tmp_path, wiki='kind = "confluence"\nroots = ["100"]')

    code, out, _ = cli("sync", "wiki", CONFLUENCE_BIN=str(site.bin))

    assert code == 0 and "2 added" in out
    _, out, _ = cli("search", "ORA-00942")
    assert "page id: wiki:101" in out and "space: ENG" in out
    assert "link: https://wiki.example.test/pages/viewpage.action?pageId=101" in out


def test_a_listing_that_failed_makes_the_sync_fail_and_removes_nothing(cli, tmp_path):
    site = fake_confluence.FakeSite(tmp_path)
    site.page("100", "ENG home", body="Start here.\n")
    site.page("101", "Fixing ORA-00942", parent="100", body="Check the grants.\n")
    declare(tmp_path, wiki='kind = "confluence"\nroots = ["100"]')
    environ = {"CONFLUENCE_BIN": str(site.bin)}
    cli("sync", "wiki", **environ)
    site.fail("100", "children")

    code, out, _ = cli("sync", "wiki", **environ)

    assert code == 1 and "failed root 100:" in out
    assert "documents:   2" in cli("status")[1]


# --- the Retrieval mode ----------------------------------------------------------------


def test_keyword_retrieval_runs_with_no_embedding_model_and_never_calls_a_gateway(folder):
    client = QdrantClient(location=":memory:")
    environ = {
        "RAG_RETRIEVAL": "keyword",
        **declared(folder),
        "LITELLM_BASE_URL": "http://127.0.0.1:1/v1",
    }  # nothing listens there

    def run(*argv):
        out, err = io.StringIO(), io.StringIO()
        return (
            main(list(argv), environ, stdout=out, stderr=err, client=client),
            out.getvalue(),
            err.getvalue(),
        )

    assert run("sync", "docs")[0] == 0
    code, out, _ = run("search", "ORA-00942")
    assert code == 0 and "page id: docs:dba/ora-00942.md" in out and "note:" not in out
    code, out, _ = run("status")
    assert code == 0 and "keyword-only" in out and "dimensions" not in out
    assert run("rebuild")[0] == 0


def test_a_search_that_fell_back_to_keywords_prints_the_notice(cli, folder, monkeypatch):
    cli("sync", "docs")
    cli("search", "ORA-00942")  # the dimension check passes while the gateway is up

    def down(text):
        raise ConnectionError("unreachable")

    monkeypatch.setattr(cli.gateway, "embed_query", down)

    code, out, _ = cli("search", "ORA-00942")

    assert code == 0 and "page id: docs:dba/ora-00942.md" in out
    assert "note: Results are keyword-only" in out


def test_hybrid_retrieval_without_an_embedding_model_is_a_settings_error(cli):
    code, _, err = cli("status", RAG_EMBED_MODEL="")

    assert code == 1 and "RAG_EMBED_MODEL" in err and "RAG_RETRIEVAL=keyword" in err


def test_forget_removes_a_corpuss_documents_and_reports_how_many(cli, folder, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    (other / "notes.md").write_text("Notes about ZX-9000.")
    declare(
        tmp_path,
        docs=f'kind = "folder"\npath = "{folder}"',
        other=f'kind = "folder"\npath = "{other}"',
    )
    cli("sync", "docs")
    cli("sync", "other")

    code, out, _ = cli("forget", "docs")

    assert code == 0 and "3 Documents" in out
    assert "docs:" not in cli("search", "ORA-00942", "-k", "10")[1]
    assert "other:notes.md" in cli("search", "ZX-9000")[1]


def test_forget_works_for_a_corpus_that_is_no_longer_declared(cli, folder, tmp_path):
    cli("sync", "docs")
    declare(tmp_path, other='kind = "folder"\npath = "/nowhere"')

    code, out, _ = cli("forget", "docs")

    assert code == 0 and "3 Documents" in out


def test_forget_of_a_name_with_no_documents_says_so(cli, folder):
    cli("sync", "docs")

    code, out, _ = cli("forget", "ghost")

    assert code == 0 and "0 Documents" in out


def test_status_warns_about_documents_of_an_undeclared_corpus(cli, folder, tmp_path):
    cli("sync", "docs")
    declare(tmp_path, other='kind = "folder"\npath = "/nowhere"')

    code, out, _ = cli("status")

    assert code == 0
    assert "warning:" in out and "'docs'" in out and "forget docs" in out
    assert "documents:   3" in out
    assert "docs:dba/ora-00942.md" in cli("search", "ORA-00942")[1]


def test_a_rebuild_leaves_out_an_undeclared_corpus(cli, folder, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    (other / "notes.md").write_text("Notes about ZX-9000.")
    declare(
        tmp_path,
        docs=f'kind = "folder"\npath = "{folder}"',
        other=f'kind = "folder"\npath = "{other}"',
    )
    cli("sync", "docs")
    cli("sync", "other")
    declare(tmp_path, docs=f'kind = "folder"\npath = "{folder}"')

    code, _, _ = cli("rebuild")

    assert code == 0
    assert "other:" not in cli("search", "ZX-9000", "-k", "10")[1]
    assert "undeclared" not in cli("status")[1]


def test_search_can_name_a_repository_and_a_tracked_ref_and_shows_the_ref(cli, tmp_path):
    from fake_gitlab import FakeGitLab

    remote = FakeGitLab(tmp_path / "remote")
    remote.commit("platform/pay", {"a.py": "def settle(): pass  # zebrafish\n"})
    remote.commit("platform/pay", {"a.py": "def settle(): pass  # zebrafish\n"}, branch="feat")
    remote.commit("platform/other", {"b.py": "def settle(): pass  # zebrafish\n"})
    declare(
        tmp_path,
        code=f'''kind = "gitlab"
cache = "{tmp_path / "mirrors"}"
[[corpus.code.repositories]]
path = "platform/pay"
refs = ["main", "feat"]
[[corpus.code.repositories]]
path = "platform/other"
refs = ["main"]''',
    )
    env = {"GITLAB_URL": remote.url, "GITLAB_TOKEN": "t"}
    assert cli("sync", "code", **env)[0] == 0

    _, plain, _ = cli("search", "zebrafish")
    _, on_ref, _ = cli("search", "zebrafish", "--ref", "feat")
    _, in_repository, _ = cli("search", "zebrafish", "--repository", "platform/other")

    assert plain.count("ref: main") == 2 and "ref: feat" not in plain
    assert "ref: feat" in on_ref and "ref: main" not in on_ref
    assert "platform/other" in in_repository and "platform/pay" not in in_repository
