"""The shipped skills: the admin helper script against a fake host, and each SKILL.md naming only
commands the helper has, so the skill cannot silently rot."""

import getpass
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from fake_drop import REPO

SKILL = REPO / "skills/nexus-rag-admin"
SCRIPT = SKILL / "scripts/rag_admin.py"
TOKEN = "a" * 48


def fingerprint(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()[:12]


@pytest.fixture
def host(tmp_path: Path) -> Path:
    """A systemd-route host: an env file with one token, a GitLab Corpus declared."""
    etc = tmp_path / "root/etc/nexus-rag"
    etc.mkdir(parents=True)
    (etc / "nexus-rag.env").write_text(
        "QDRANT_API_KEY=q1\n"
        f"MCP_TOKENS={TOKEN}\n"
        "# GITLAB_URL=https://gitlab.example.test\n"
        "GITLAB_TOKEN=glpat-secret\n"
    )
    (etc / "corpora.toml").write_text(
        "# the operator's comment\n"
        '[corpus.gitlab]\nkind = "gitlab"\n\n'
        '[[corpus.gitlab.repositories]]\npath = "platform/payments"\nrefs = ["main"]\n'
    )
    (tmp_path / "home").mkdir()
    return tmp_path


def admin(host: Path, *args: str, stdin: str = "") -> subprocess.CompletedProcess:
    environ = dict(
        os.environ,
        RAG_ADMIN_ROOT=str(host / "root"),
        RAG_ADMIN_HOME=str(host / "home"),
        RAG_ADMIN_DRY_RUN="1",
        SUDO_USER=getpass.getuser(),
    )
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        input=stdin,
        capture_output=True,
        text=True,
        env=environ,
    )


def env_file(host: Path) -> str:
    return (host / "root/etc/nexus-rag/nexus-rag.env").read_text()


def test_env_show_hides_every_secret(host):
    shown = admin(host, "env", "show").stdout
    assert "glpat-secret" not in shown and TOKEN not in shown and "q1" not in shown
    assert "GITLAB_TOKEN=<hidden>" in shown


def test_a_secret_is_read_from_stdin_and_never_taken_as_an_argument(host):
    refused = admin(host, "env", "set", "GITLAB_TOKEN", "glpat-new")
    assert refused.returncode == 2 and "glpat-new" not in env_file(host)
    done = admin(host, "env", "set", "GITLAB_TOKEN", stdin="glpat-new\n")
    assert done.returncode == 0 and "glpat-new" not in done.stdout + done.stderr
    assert "GITLAB_TOKEN=glpat-new" in env_file(host)


def test_env_set_fills_the_template_line_and_warns_about_rebuild_settings(host):
    admin(host, "env", "set", "GITLAB_URL", "https://gitlab.example.test")
    assert "\nGITLAB_URL=https://gitlab.example.test\n" in env_file(host)
    warned = admin(host, "env", "set", "RAG_EMBED_MODEL", "qwen3-embedding")
    assert "Rebuild" in warned.stdout


def test_settings_the_helper_manages_otherwise_are_refused(host):
    for key in ("MCP_TOKENS", "QDRANT_API_KEY"):
        assert admin(host, "env", "set", key, stdin="x").returncode == 2


def test_tokens_are_added_named_and_revoked_by_label(host):
    added = admin(host, "token", "add", "dana")
    assert added.returncode == 0, added.stderr
    new = (host / "home/nexus-rag-tokens/dana.token").read_text().strip()
    assert new not in added.stdout and new in env_file(host)
    assert oct((host / "home/nexus-rag-tokens/dana.token").stat().st_mode)[-3:] == "600"
    ledger = (host / "root/etc/nexus-rag/mcp-tokens.labels").read_text()
    assert new not in ledger and fingerprint(new) in ledger

    assert admin(host, "token", "name", fingerprint(TOKEN)[:8], "ops").returncode == 0
    listed = admin(host, "token", "list").stdout
    assert "dana" in listed and "ops" in listed and TOKEN not in listed

    assert admin(host, "token", "revoke", "dana").returncode == 0
    assert new not in env_file(host) and TOKEN in env_file(host)


def test_the_last_token_cannot_be_revoked(host):
    assert admin(host, "token", "revoke", fingerprint(TOKEN)).returncode == 2
    assert TOKEN in env_file(host)


def test_a_token_given_on_stdin_is_revoked_without_being_echoed(host):
    admin(host, "token", "add", "dana")
    wrong = admin(host, "token", "revoke", "-", stdin="not-a-token")
    assert wrong.returncode == 2 and "not-a-token" not in wrong.stderr
    assert admin(host, "token", "revoke", "-", stdin=TOKEN).returncode == 0
    assert TOKEN not in env_file(host)


def test_add_repo_stages_and_apply_installs_keeping_comments(host):
    staged = admin(host, "corpora", "add-repo", "gitlab", "tools/cli", "develop", "v1.0")
    assert staged.returncode == 0, staged.stderr
    assert admin(host, "corpora", "add-repo", "gitlab", "tools/cli", "main").returncode == 2
    applied = admin(host, "corpora", "apply")
    assert applied.returncode == 0, applied.stderr
    assert "added     gitlab: tools/cli@develop" in applied.stdout
    live = (host / "root/etc/nexus-rag/corpora.toml").read_text()
    assert "# the operator's comment" in live and 'refs = ["develop", "v1.0"]' in live
    assert list((host / "root/etc/nexus-rag/backups").glob("corpora.toml.*"))
    assert not (host / "home/nexus-rag-staging/corpora.toml").exists()


def test_add_repo_refuses_what_is_not_a_project_path_or_corpus(host):
    assert admin(host, "corpora", "add-repo", "gitlab", "payments", "main").returncode == 2
    assert admin(host, "corpora", "add-repo", "code", "a/b", "main").returncode == 2
    assert admin(host, "corpora", "add-repo", "code", "a/b", "main", "--new-corpus").returncode == 0


def test_apply_reports_a_removed_corpus_needs_forget(host):
    admin(host, "corpora", "stage")
    (host / "home/nexus-rag-staging/corpora.toml").write_text(
        '[corpus.docs]\nkind = "folder"\npath = "/srv/docs"\n'
    )
    applied = admin(host, "corpora", "apply").stdout
    assert "removed   corpus gitlab" in applied and "forget gitlab" in applied


def test_commands_run_the_way_the_route_runs_them(host):
    systemd = admin(host, "run", "nexus-rag", "status").stdout
    assert "User=ragsvc" in systemd and "/opt/nexus-rag/venv/bin/nexus-rag status" in systemd
    assert "nexus-rag-sync@gitlab.service" in admin(host, "sync", "gitlab", "--detach").stdout
    etc = host / "root/etc/nexus-rag"
    (etc / "nexus-rag.env").rename(etc / "container.env")
    container = admin(host, "run", "nexus-rag", "status").stdout
    assert "run --rm -T sync nexus-rag status" in container
    assert (
        "nexus-rag-container-sync@gitlab.timer" in admin(host, "timer", "enable", "gitlab").stdout
    )


def test_acceptance_and_run_pass_their_options_through(host):
    accepted = admin(host, "acceptance", "--only", "gitlab").stdout
    assert "nexus_rag.acceptance --systemd --only gitlab" in accepted
    searched = admin(host, "run", "nexus-rag", "search", "x", "--repository", "a/b").stdout
    assert "nexus-rag search x --repository a/b" in searched


def test_the_validator_builds_every_corpus_as_a_sync_would(tmp_path):
    validator = re.search(r'VALIDATOR = """\\\n(.*?)"""', SCRIPT.read_text(), re.DOTALL)
    assert validator
    declaration = tmp_path / "corpora.toml"
    declaration.write_text(
        '[corpus.gitlab]\nkind = "gitlab"\n[[corpus.gitlab.repositories]]\n'
        'path = "a/b"\nrefs = ["main"]\n'
    )
    environ = {"PATH": os.environ["PATH"], "HOME": str(tmp_path)}
    run = [sys.executable, "-c", validator.group(1), str(declaration)]
    missing = subprocess.run(run, capture_output=True, text=True, env=environ)
    assert missing.returncode == 1 and "GITLAB_URL" in missing.stdout
    environ |= {"GITLAB_URL": "https://gitlab.example.test", "GITLAB_TOKEN": "t"}
    ok = subprocess.run(run, capture_output=True, text=True, env=environ)
    assert ok.returncode == 0 and "ok        corpus gitlab" in ok.stdout


def test_every_command_the_skill_names_exists():
    text = (SKILL / "SKILL.md").read_text()
    script = SCRIPT.read_text()
    named = set(re.findall(r"rag-admin ([a-z][a-z-]*)(?: ([a-z][a-z-]*))?", text))
    assert named
    for command, action in named:
        assert f'sub.add_parser("{command}")' in script, command
        actions = re.search(rf'{command}\.add_argument\("action", choices=\[(.*?)\]', script)
        if actions and action and action.islower() and not action.startswith("local"):
            assert f'"{action}"' in actions.group(1), f"{command} {action}"


@pytest.mark.parametrize(
    "reference", ["docs/agents/closed-network-setup.md", "docs/configuration.md"]
)
def test_the_skill_carries_the_docs_it_points_at(reference):
    assert (SKILL / "references" / Path(reference).name).resolve() == REPO / reference


def test_the_research_skill_names_only_tools_and_arguments_the_server_has():
    text = (REPO / "skills/nexus-rag-research/SKILL.md").read_text()
    server = (REPO / "src/nexus_rag/mcp_server.py").read_text()
    tools = set(re.findall(r"`(search_docs|list_spaces|get_page|\w+_\w+)\(", text))
    assert tools == {"search_docs", "list_spaces", "get_page"}
    for tool in tools:
        assert f"async def {tool}(" in server
    signature = re.search(r"async def search_docs\((.*?)\)", server, re.DOTALL)
    assert signature
    for argument in set(
        re.findall(r"(\w+)=", " ".join(re.findall(r"`search_docs\(([^`]*)\)`", text)))
    ):
        assert f"{argument}:" in signature.group(1), argument
