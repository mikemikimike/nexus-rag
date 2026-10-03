"""The manual smoke check: does this installation work against the real Model gateway,
Qdrant, LiteParse, Confluence CLI and (the systemd route) units?

    python -m local_rag.smoke [--systemd]

Run it with the settings the services run with, as the service user. It prints one line
per check, ``[ OK ]``, ``[WARN]``, ``[SKIP]`` or ``[FAIL]``, never a secret, then a summary
ending ``SMOKE CHECK PASSED`` or ``SMOKE CHECK FAILED: <checks>``, and exits 0 unless a
check failed (1), or the command line cannot be used (2). It changes nothing: it embeds one
sentence, reranks two texts, reads the Knowledge index's status and runs ``--version`` style
commands. The README says how to run it on each route.
"""

import argparse
import functools
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import TextIO

from qdrant_client import QdrantClient

from local_rag import checks
from local_rag.checks import (
    QDRANT_DOWN,
    SYSTEMD_UNITS,
    Failure,
    Report,
    Skipped,
    Warned,
    run_command,
)
from local_rag.environment import (
    Environment,
    SettingsError,
    environment_from,
    settings_from_environment,
)
from local_rag.gateway import ModelGateway
from local_rag.gateway_litellm import gateway_from_settings
from local_rag.index import KnowledgeIndex
from local_rag.index_format import Access
from local_rag.settings import RetrievalMode

PLACEHOLDER_PREFIX = "CHANGE_ME"
EMBEDDINGS = "Model gateway (embeddings)"

type Run = Callable[[Sequence[str]], str]  # runs a command, returns its output, raises if it fails


def main(
    argv: Sequence[str],
    environ: Mapping[str, str],
    *,
    gateway: ModelGateway | None = None,
    client: QdrantClient | None = None,
    run: Run | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run the checks; returns 0 if none failed. The keyword arguments replace the Model
    gateway, the Qdrant connection and the host's commands, for callers (tests) that have
    stand-ins."""
    try:
        args = _parser().parse_args(argv)
    except SystemExit as stop:
        return int(stop.code or 0)
    run = run or run_command
    report = Report(stdout or sys.stdout, checks.secrets_in(environ), verdict="SMOKE CHECK")
    check = report.check

    try:
        environment = environment_from(environ)
    except SettingsError as error:
        check("settings", functools.partial(_settings_error, error))
        report.summary()
        return 1
    check("MCP_TOKENS", lambda: _tokens(environment.mcp_tokens))
    try:
        settings = settings_from_environment(environ)
    except SettingsError as error:
        check("settings", functools.partial(_settings_error, error))
        report.summary()
        return 1

    gateway = gateway or gateway_from_settings(settings)
    check(EMBEDDINGS, lambda: checks.embeddings(settings, gateway))
    # A reranker is checked on its own in Keyword-only retrieval, which has no embedding check.
    check(
        "Rerank",
        lambda: checks.rerank(settings, gateway),
        needs=[] if settings.retrieval is RetrievalMode.KEYWORD else [EMBEDDINGS],
    )
    check("Knowledge index", lambda: _index(settings, gateway, client))
    check("LiteParse", lambda: _liteparse(run, settings.liteparse_bin))
    check("Confluence CLI", lambda: _confluence(environment, run))
    check("GitLab (git)", lambda: _gitlab(environment, run))
    if args.systemd:
        for unit in SYSTEMD_UNITS:
            check(unit, functools.partial(_unit_active, run, unit))
    report.summary()
    return 1 if report.failed else 0


def _tokens(tokens: Sequence[str]) -> str:
    if not tokens:
        raise Failure(
            "MCP_TOKENS is empty; the MCP server refuses to start without a token",
            cause="no token was generated into the env file, or this is not the "
            "environment the services run with",
            see="section 3 E (Access control)",
        )
    if any(t.startswith(PLACEHOLDER_PREFIX) for t in tokens):
        raise Failure(
            "MCP_TOKENS still holds a placeholder",
            cause="the env file's example value was never replaced by a generated token",
            see="section 3 E (Access control)",
        )
    return f"{len(tokens)} token{'s' if len(tokens) != 1 else ''}"


def _index(settings, gateway: ModelGateway | None, client: QdrantClient | None) -> str:
    index = KnowledgeIndex(settings, gateway, client=client)
    health = index.health()
    if not health.qdrant or not health.knowledge_index:
        raise Failure(
            "; ".join(health.notes) or "not reachable",
            cause="Qdrant is down, QDRANT_URL or QDRANT_API_KEY is wrong, or an HTTP "
            "proxy sits in the way (put the host in NO_PROXY)",
            see=QDRANT_DOWN,
        )
    status = index.status()
    if status.access is Access.REFUSED:
        raise Failure(
            "; ".join(status.warnings) or "the Live index is refused",
            cause="the index cannot be searched by this code (the embedding model's "
            "dimension changed, for example) until a Rebuild",
            see='section 10 (row "embedding dimension differs")',
        )
    summary = (
        f"Qdrant reachable, {status.documents} documents, {status.passages} passages"
        if status.generation
        else "Qdrant reachable, nothing indexed yet (run a Sync)"
    )
    if status.access is not Access.READ_WRITE or status.warnings:
        raise Warned(f"{summary}; {'; '.join(status.warnings)}")
    return summary


def _confluence(environment: Environment, run: Run) -> str:
    if not environment.confluence_domain:
        raise Skipped("CONFLUENCE_DOMAIN is not set")
    command = environment.confluence_bin
    try:
        run([command, "spaces"])  # one authenticated, read-only call
    except Exception as error:
        raise Failure(
            f"{command} spaces failed: {error}",
            cause="the CLI is not on this user's PATH (set CONFLUENCE_BIN), or the "
            "domain, token or auth type is wrong",
            see='section 10 (rows "Sync: `confluence not found`" and "Sync: '
            '`confluence ... failed`")',
        ) from None
    return f"{command} reads {environment.confluence_domain}"


def _gitlab(environment: Environment, run: Run) -> str:
    if not environment.gitlab_url:
        raise Skipped("GITLAB_URL is not set")
    try:
        return run(["git", "--version"]).strip() or "git runs"
    except Exception as error:
        raise Failure(
            f"git --version failed: {error}",
            cause="git is not installed where the Sync runs (a host prerequisite on "
            "the systemd route; the container image carries it)",
            see="section 2 (pre-flight)",
        ) from None


def _liteparse(run: Run, command: str) -> str:
    try:
        return run([command, "--version"]).strip() or "runs"
    except Exception as error:
        raise Failure(
            f"{command} --version failed: {error}",
            cause="LIT_BIN does not name the installed lit command, or the Installer "
            "did not finish: run it again, it is idempotent",
            see="section 4 (Install)",
        ) from None


def _unit_active(run: Run, unit: str) -> str:
    try:
        run(["systemctl", "is-active", "--quiet", unit])
    except Exception as error:
        raise Failure(
            f"not active: {error}",
            cause="the unit failed or was stopped; `systemctl status` and `journalctl -u` name why",
            see='section 10 (row "/healthz 503, Qdrant is unreachable")',
        ) from None
    return "active"


def _settings_error(error: SettingsError) -> str:
    raise Failure(
        str(error),
        cause="a setting in the env file is missing or unusable, as said",
        see="section 5 (Configure)",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m local_rag.smoke", description=__doc__.split("\n\n")[0]
    )
    parser.add_argument(
        "--systemd",
        action="store_true",
        help="also check that the systemd route's units are active",
    )
    return parser


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:], os.environ))
