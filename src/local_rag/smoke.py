"""The manual smoke check: does this installation work against the real Model gateway,
Qdrant, LiteParse, Confluence CLI and (the systemd route) units?

    python -m local_rag.smoke [--systemd]

Run it with the settings the services run with, as the service user. It prints one line
per check, ``[ OK ]``, ``[WARN]``, ``[SKIP]`` or ``[FAIL]``, never a secret, and exits 0
unless a check failed (1), or the command line cannot be used (2). It changes nothing: it
embeds one sentence, reads the Knowledge index's status and runs ``--version`` style
commands. The README says how to run it on each route.
"""
import argparse
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TextIO

from qdrant_client import QdrantClient

from local_rag.environment import SettingsError, settings_from_environment
from local_rag.gateway import ModelGateway
from local_rag.gateway_litellm import LiteLLMGateway
from local_rag.index import KnowledgeIndex
from local_rag.index_format import Access

SYSTEMD_UNITS = ("qdrant.service", "local-rag-mcp.service",
                 "local-rag-sync.timer", "local-rag-full.timer")
COMMAND_TIMEOUT = 30.0
PLACEHOLDER_PREFIX = "CHANGE_ME"

type Run = Callable[[Sequence[str]], str]  # runs a command, returns its output, raises if it fails


@dataclass
class _Skipped(Exception):
    why: str


@dataclass
class _Warning(Exception):
    what: str


def main(argv: Sequence[str], environ: Mapping[str, str], *,
         gateway: ModelGateway | None = None, client: QdrantClient | None = None,
         run: Run | None = None, stdout: TextIO | None = None,
         stderr: TextIO | None = None) -> int:
    """Run the checks; returns 0 if none failed. The keyword arguments replace the Model
    gateway, the Qdrant connection and the host's commands, for callers (tests) that have
    stand-ins."""
    out = stdout or sys.stdout
    try:
        args = _parser().parse_args(argv)
    except SystemExit as stop:
        return int(stop.code or 0)
    run = run or _run_command
    failed = False

    def check(name: str, what: Callable[[], str]) -> None:
        nonlocal failed
        try:
            print(f"[ OK ] {name}: {what()}", file=out)
        except _Skipped as skipped:
            print(f"[SKIP] {name}: {skipped.why}", file=out)
        except _Warning as warning:
            print(f"[WARN] {name}: {warning.what}", file=out)
        except Exception as error:
            failed = True
            print(f"[FAIL] {name}: {str(error) or type(error).__name__}", file=out)

    check("MCP_TOKENS", lambda: _tokens(environ))
    try:
        settings = settings_from_environment(environ)
    except SettingsError as error:
        check("settings", lambda: _raise(error))
        return 1

    gateway = gateway or (LiteLLMGateway.from_settings(settings) if _gateway_named(environ)
                          else None)
    check("Model gateway (embeddings)", lambda: _embeddings(environ, gateway))
    check("Rerank", lambda: _rerank(settings.rerank_model, gateway))
    check("Knowledge index", lambda: _index(settings, gateway, client))
    check("LiteParse", lambda: _version(run, settings.liteparse_bin))
    check("Confluence CLI", lambda: _confluence(environ, run))
    if args.systemd:
        for unit in SYSTEMD_UNITS:
            check(unit, lambda unit=unit: _unit_active(run, unit))
    return 1 if failed else 0


def _tokens(environ: Mapping[str, str]) -> str:
    tokens = [t.strip() for t in environ.get("MCP_TOKENS", "").split(",") if t.strip()]
    if not tokens:
        raise ValueError("MCP_TOKENS is empty; the MCP server refuses to start without a token")
    if any(t.startswith(PLACEHOLDER_PREFIX) for t in tokens):
        raise ValueError("MCP_TOKENS still holds a placeholder")
    return f"{len(tokens)} token{'s' if len(tokens) != 1 else ''}"


def _gateway_named(environ: Mapping[str, str]) -> bool:
    return bool(environ.get("LITELLM_BASE_URL") and environ.get("RAG_EMBED_MODEL"))


def _embeddings(environ: Mapping[str, str], gateway: ModelGateway | None) -> str:
    # Settings would fill a default in for these, and a default is no Model gateway of ours.
    missing = [name for name in ("LITELLM_BASE_URL", "RAG_EMBED_MODEL") if not environ.get(name)]
    if missing:
        raise ValueError(f"{' and '.join(missing)} not set in the environment")
    dimension = len(gateway.embed_query("smoke test"))
    return f"model {environ['RAG_EMBED_MODEL']!r} answers, {dimension} dimensions"


def _rerank(model: str | None, gateway: ModelGateway | None) -> str:
    if not model:
        raise _Skipped("RAG_RERANK_MODEL is not set, so search does not rerank")
    scores = gateway.rerank("smoke test", ["first text", "second text"])
    if len(scores) != 2:
        raise ValueError(f"asked to score 2 texts, got {len(scores)} scores")
    return f"model {model!r} answers"


def _index(settings, gateway: ModelGateway | None, client: QdrantClient | None) -> str:
    index = KnowledgeIndex(settings, gateway, client=client)
    health = index.health()
    if not health.qdrant or not health.knowledge_index:
        raise RuntimeError("; ".join(health.notes) or "not reachable")
    status = index.status()
    if status.access is Access.REFUSED:
        raise RuntimeError("; ".join(status.warnings) or "the Live index is refused")
    summary = (f"Qdrant reachable, {status.documents} documents, {status.passages} passages"
               if status.generation else "Qdrant reachable, nothing indexed yet (run a Sync)")
    if status.access is not Access.READ_WRITE or status.warnings:
        raise _Warning(f"{summary}; {'; '.join(status.warnings)}")
    return summary


def _confluence(environ: Mapping[str, str], run: Run) -> str:
    if not environ.get("CONFLUENCE_DOMAIN"):
        raise _Skipped("CONFLUENCE_DOMAIN is not set")
    command = environ.get("CONFLUENCE_BIN") or "confluence"
    run([command, "spaces", "--json"])  # one authenticated, read-only call
    return f"{command} reads {environ['CONFLUENCE_DOMAIN']}"


def _version(run: Run, command: str) -> str:
    return run([command, "--version"]).strip() or "runs"


def _unit_active(run: Run, unit: str) -> str:
    run(["systemctl", "is-active", "--quiet", unit])
    return "active"


def _run_command(command: Sequence[str]) -> str:
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True,
                              timeout=COMMAND_TIMEOUT).stdout
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or "").strip().splitlines()
        raise RuntimeError(f"{command[0]} exited {error.returncode}"
                           + (f": {detail[-1]}" if detail else "")) from None
    except FileNotFoundError:
        raise RuntimeError(f"{command[0]} not found") from None
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"{command[0]}: {error}") from None


def _raise(error: Exception) -> str:
    raise error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m local_rag.smoke", description=__doc__.split("\n\n")[0])
    parser.add_argument("--systemd", action="store_true",
                        help="also check that the systemd route's units are active")
    return parser


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:], os.environ))
