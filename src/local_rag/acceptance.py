"""The live acceptance suite: does the whole stack work, end to end, on this network?

    python -m local_rag.acceptance [--only GROUP ...] [--systemd | --container]
                                   [--mcp-url URL] [--token TOKEN] [--roots-file FILE]

Run it with the settings the services run with (``run_rag ...`` / ``compose run --rm -T sync
...``, like the smoke check). It checks, in groups: ``gateway`` (the Model gateway embeds, and
a long text is split into Passages the model accepts), ``qdrant`` (reachable, a wrong key
refused), ``index`` (a round trip on a throwaway collection and temp folder), ``mcp`` (the
running MCP server over HTTP), ``install`` (units or containers, env file mode, timers; needs
``--systemd`` or ``--container``) and ``confluence`` (only when ``CONFLUENCE_DOMAIN`` is set).

It never touches the live index: the index group works in its own collection, named
``acceptance-<random>``, and removes it and its temp folder at the end, also after a failure.
The MCP and Confluence checks only read. It prints which settings it used (never a secret), one
line per check (``[ OK ]``, ``[FAIL]`` or ``[SKIP]`` with a reason), and a summary. Exit code 0:
nothing failed; 1: a check failed; 2: the command line or settings cannot be used.
"""
import argparse
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any, TextIO
from urllib.parse import urlsplit, urlunsplit

from local_rag.acceptance_corpus import Run
from local_rag.acceptance_report import Context, Report, run_command
from local_rag.environment import SettingsError, settings_from_environment

GROUPS = ("gateway", "qdrant", "index", "mcp", "install", "confluence")
SECRET_NAMES = ("KEY", "TOKEN", "TOKENS", "SECRET", "PASSWORD")

type Connect = Callable[[str, str | None], Any]  # (Qdrant location, API key) -> a Qdrant client


def main(argv: Sequence[str], environ: Mapping[str, str], *,
         gateway: Any = None, connect: Connect | None = None,
         run: Callable[[Sequence[str]], str] | None = None,
         stdout: TextIO | None = None, stderr: TextIO | None = None) -> int:
    """Run the suite; returns the exit code. The keyword arguments replace the Model gateway,
    the Qdrant connection and the host's commands, for callers (tests) that have stand-ins."""
    out, err = stdout or sys.stdout, stderr or sys.stderr
    try:
        args = _parser().parse_args(argv)
    except SystemExit as stop:  # argparse: --help, or a command line it rejected
        return int(stop.code or 0)
    groups = [g for g in GROUPS if not args.only or g in args.only]
    if "install" in groups and args.only and not (args.systemd or args.container):
        print("error: --only install needs --systemd or --container", file=err)
        return 2
    try:
        live = settings_from_environment(environ)
    except SettingsError as error:
        print(f"error: {error}", file=err)
        return 2

    run_id = Run.new()
    settings = replace(live, collection=run_id.collection)
    report = Report(out, _secrets(environ, args))
    context = Context(settings=settings, environ=environ, report=report, gateway=gateway,
                      connect=connect or _connect, run=run or run_command, args=args,
                      run_id=run_id)
    _describe(report, live, environ, run_id, groups)
    try:
        _prepare(context, groups)
        for group in groups:
            _runner(group)(context)
    finally:
        _clean_up(context)
    report.summary()
    return 1 if report.failed else 0


def _prepare(ctx: Context, groups: Sequence[str]) -> None:
    """The Model gateway, built only when a group will use it."""
    if ctx.gateway is not None or not {"gateway", "index"} & set(groups):
        return
    if ctx.environ.get("LITELLM_BASE_URL") and ctx.environ.get("RAG_EMBED_MODEL"):
        from local_rag.gateway_litellm import LiteLLMGateway
        try:
            ctx.gateway = LiteLLMGateway.from_settings(ctx.settings)
        except ValueError as error:
            ctx.gateway_problem = str(error)


def _runner(group: str) -> Callable[[Context], None]:
    if group in ("gateway", "qdrant"):
        from local_rag import acceptance_stack as module
    elif group == "index":
        from local_rag import acceptance_index as module
    elif group == "mcp":
        from local_rag import acceptance_mcp as module
    else:
        from local_rag import acceptance_host as module
    return getattr(module, f"run_{group}")


def _clean_up(ctx: Context) -> None:
    """Undo what the index group made, last first; each step is reported, none is skipped."""
    while ctx.cleanups:
        name, undo = ctx.cleanups.pop()
        ctx.report.check(f"cleanup.{name}", undo)


def _describe(report: Report, live, environ: Mapping[str, str], run_id: Run,
              groups: Sequence[str]) -> None:
    """Which settings the run uses; a secret is only said to be set."""
    def secret(name: str) -> str:
        return "set" if environ.get(name) else "not set"

    tokens = [t for t in environ.get("MCP_TOKENS", "").split(",") if t.strip()]
    report.say("local-rag acceptance suite")
    report.say(f"  groups:           {', '.join(groups)}")
    report.say(f"  model gateway:    {_safe_url(live.gateway_base_url if environ.get('LITELLM_BASE_URL') else None)}"
               f", embeddings {environ.get('RAG_EMBED_MODEL') or 'not set'}"
               f", rerank {live.rerank_model or 'not set'}, key {secret('LITELLM_API_KEY')}")
    report.say(f"  qdrant:           {_safe_url(live.qdrant_location)}, key {secret('QDRANT_API_KEY')}")
    report.say(f"  chunks:           {live.chunk_tokens} tokens, overlap {live.chunk_overlap}")
    report.say(f"  mcp tokens:       {len(tokens)}")
    report.say(f"  confluence:       {environ.get('CONFLUENCE_DOMAIN') or 'not set'}")
    report.say(f"  live collection:  {live.collection} (not touched)")
    if "index" in groups:
        report.say(f"  throwaway:        {run_id.collection} (removed at the end)")
    report.say()


def _safe_url(url: str | None) -> str:
    if not url:
        return "not set"
    parts = urlsplit(url)
    if "@" in parts.netloc:  # credentials in the URL are a secret
        parts = parts._replace(netloc=parts.netloc.rpartition("@")[2])
    return urlunsplit(parts)


def _secrets(environ: Mapping[str, str], args: argparse.Namespace) -> list[str]:
    values = [value for name, value in environ.items()
              if any(name.endswith(word) for word in SECRET_NAMES)]
    values += [t.strip() for value in [environ.get("MCP_TOKENS", ""), args.token or ""]
               for t in value.split(",")]
    values += [urlsplit(environ.get("QDRANT_URL", "")).password or ""]
    return [v for v in values if v]


def _connect(location: str, api_key: str | None):
    import warnings

    from qdrant_client import QdrantClient
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # an API key over plain HTTP is the design, see the README
        return QdrantClient(url=location, api_key=api_key, timeout=60)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m local_rag.acceptance", description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", action="append", choices=GROUPS, metavar="GROUP",
                        help=f"run only this group (repeatable): {', '.join(GROUPS)}")
    route = parser.add_mutually_exclusive_group()
    route.add_argument("--systemd", action="store_true",
                       help="check the systemd route's units (the install group)")
    route.add_argument("--container", action="store_true",
                       help="check the container route's stack and timers (the install group)")
    parser.add_argument("--mcp-url", metavar="URL",
                        help="where the MCP server listens (default: MCP_HOST and MCP_PORT)")
    parser.add_argument("--token", metavar="TOKEN",
                        help="the MCP token to use (default: the first of MCP_TOKENS)")
    parser.add_argument("--roots-file", metavar="FILE",
                        help="the Confluence roots file (default: /etc/local-rag/confluence_roots.txt)")
    return parser


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:], os.environ))
