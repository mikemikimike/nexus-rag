"""The live acceptance suite: does the whole stack work, end to end, on this network?

    python -m local_rag.acceptance [--only GROUP ...] [--skip GROUP ...]
                                   [--systemd | --container]
                                   [--mcp-url URL] [--token TOKEN] [--corpora FILE]

Run it with the settings the services run with (``run_rag ...`` / ``compose run --rm -T sync
...``, like the smoke check). It checks, in groups: ``gateway`` (the Model gateway embeds, and
a long text is split into Passages the model accepts), ``qdrant`` (reachable, a wrong key
refused), ``index`` (a round trip on a throwaway collection and temp folder), ``mcp`` (the
running MCP server over HTTP), ``install`` (units or containers, env file mode, timers; needs
``--systemd`` or ``--container``, else it fails, and ``--skip install`` leaves it out) and
``confluence`` (only when ``CONFLUENCE_DOMAIN`` is set) and ``gitlab`` (only when the
declaration holds a GitLab Corpus: the tool runs and every Tracked ref can be read).

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
from local_rag.acceptance_report import Context
from local_rag.checks import Report, run_command, secrets_in
from local_rag.environment import (
    Environment, SettingsError, environment_from, settings_from_environment)

GROUPS = ("gateway", "qdrant", "index", "mcp", "install", "confluence", "gitlab")

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
    groups = [g for g in GROUPS if (not args.only or g in args.only) and g not in args.skip]
    if "install" in groups and args.only and not (args.systemd or args.container):
        print("error: --only install needs --systemd or --container", file=err)
        return 2
    try:
        live = settings_from_environment(
            environ, gateway_required=bool({"gateway", "index"} & set(groups)))
        environment = environment_from(environ)
    except SettingsError as error:
        print(f"error: {error}", file=err)
        return 2

    run_id = Run.new()
    settings = replace(live, collection=run_id.collection)
    report = Report(out, [*secrets_in(environ), *(args.token or "").split(",")], verdict="ACCEPTANCE")
    context = Context(settings=settings, environment=environment, report=report, gateway=gateway,
                      connect=connect or _connect, run=run or run_command, args=args,
                      run_id=run_id)
    _describe(report, live, environment, environ, run_id, groups)
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
    from local_rag.gateway_litellm import gateway_from_settings
    try:
        ctx.gateway = gateway_from_settings(ctx.settings)
    except ValueError as error:  # an address that is set but is not an http(s) URL
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


def _describe(report: Report, live, environment: Environment, environ: Mapping[str, str],
              run_id: Run, groups: Sequence[str]) -> None:
    """Which settings the run uses; a secret is only said to be set."""
    def secret(name: str) -> str:
        return "set" if environ.get(name) else "not set"

    report.say("local-rag acceptance suite")
    report.say(f"  groups:           {', '.join(groups)}")
    report.say(f"  model gateway:    {_safe_url(live.gateway_base_url)}"
               f", embeddings {environ.get('RAG_EMBED_MODEL') or 'not set'}"
               f", rerank {live.rerank_model or 'not set'}, key {secret('LITELLM_API_KEY')}")
    report.say(f"  retrieval:        {live.retrieval}")
    report.say(f"  qdrant:           {_safe_url(live.qdrant_location)}, key {secret('QDRANT_API_KEY')}")
    report.say(f"  passages:         {live.chunk_tokens} tokens, overlap {live.chunk_overlap}")
    report.say(f"  mcp tokens:       {len(environment.mcp_tokens)}")
    report.say(f"  confluence:       {environment.confluence_domain or 'not set'}")
    report.say(f"  gitlab:           {_safe_url(environment.gitlab_url)}")
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
    parser.add_argument("--skip", action="append", choices=GROUPS, metavar="GROUP", default=[],
                        help="leave this group out (repeatable); the install group is otherwise "
                             "run, and fails without --systemd or --container")
    route = parser.add_mutually_exclusive_group()
    route.add_argument("--systemd", action="store_true",
                       help="check the systemd route's units (the install group)")
    route.add_argument("--container", action="store_true",
                       help="check the container route's stack and timers (the install group)")
    parser.add_argument("--mcp-url", metavar="URL",
                        help="where the MCP server listens (default: MCP_HOST and MCP_PORT)")
    parser.add_argument("--token", metavar="TOKEN",
                        help="the MCP token to use (default: the first of MCP_TOKENS)")
    parser.add_argument("--corpora", metavar="FILE",
                        help="the Corpus declaration file whose Confluence Roots and GitLab Corpora are checked "
                             "(default: RAG_CORPORA_FILE, else /etc/local-rag/corpora.toml)")
    return parser


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:], os.environ))
