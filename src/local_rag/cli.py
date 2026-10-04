"""Command line adapter over the Knowledge index: Sync, Rebuild, forget, switch-back, search, status.

Settings come from the environment (see ``environment``). The Corpora come from the
declaration file (see ``declaration``): ``--corpora FILE``, else ``RAG_CORPORA_FILE``, else
``/etc/local-rag/corpora.toml``. Each command prints what
happened and returns a process exit code: 0 on success, 1 on failure or refusal, 2 for a
command line that cannot be understood, and 75 (``EX_TEMPFAIL``) when another run holds the
lock, which the timer units accept as success: a skipped run is not a failed unit.
"""

import argparse
import logging
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from qdrant_client import QdrantClient

from local_rag.declaration import Declaration, DeclarationError
from local_rag.environment import (
    DEFAULT_CORPORA_FILE,
    Environment,
    environment_from,
    settings_from_environment,
)
from local_rag.errors import describe
from local_rag.exclusive import SyncInProgress
from local_rag.gateway import ModelGateway
from local_rag.gateway_litellm import gateway_from_settings
from local_rag.index import IndexStatus, KnowledgeIndex, Passage, RebuildReport, SyncReport
from local_rag.index_format import Access
from local_rag.settings import RetrievalMode

ANOTHER_RUN_HOLDS_THE_LOCK = 75


@dataclass(frozen=True)
class _Session:
    """What a command runs against: the Knowledge index, the environment, where it prints and
    how to read the Corpus declaration, which only the commands that need it do."""

    index: KnowledgeIndex
    environment: Environment
    out: TextIO
    read_declaration: Callable[[], Declaration]

    def declared[R](self, build: Callable[[Declaration], R]) -> R:
        try:
            return build(self.read_declaration())
        except DeclarationError as error:
            raise UsageError(str(error)) from None


class UsageError(Exception):
    """The command line, or the Corpus declaration it points to, cannot be used."""


def main(
    argv: Sequence[str],
    environ: Mapping[str, str],
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    gateway: ModelGateway | None = None,
    client: QdrantClient | None = None,
) -> int:
    """Run one command and return its exit code.

    ``gateway`` and ``client`` replace the LiteLLM gateway and the Qdrant connection that
    the settings name, for callers (tests) that have stand-ins.
    """
    out, err = stdout or sys.stdout, stderr or sys.stderr
    try:
        args = _parser().parse_args(argv)
    except SystemExit as stop:  # argparse: --help, or a command line it rejected
        return int(stop.code or 0)
    try:
        settings, environment = settings_from_environment(environ), environment_from(environ)
        index = KnowledgeIndex(settings, gateway or gateway_from_settings(settings), client=client)
        return args.run(
            args, _Session(index, environment, out, _declaration_reader(args, environment))
        )
    except UsageError as error:
        print(f"error: {error}", file=err)
        return 2
    except SyncInProgress as error:
        print(f"error: {describe(error)}", file=err)
        return ANOTHER_RUN_HOLDS_THE_LOCK
    except Exception as error:
        print(f"error: {describe(error)}", file=err)
        return 1


def run() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    sys.exit(main(sys.argv[1:], os.environ))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="local-rag", description="Sync, search and inspect the Knowledge index."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    declared = argparse.ArgumentParser(add_help=False)
    declared.add_argument(
        "--corpora",
        type=Path,
        metavar="FILE",
        help=f"the Corpus declaration file (default: $RAG_CORPORA_FILE, else {DEFAULT_CORPORA_FILE})",
    )

    sync = commands.add_parser(
        "sync", parents=[declared], help="bring one Corpus's Documents up to date"
    )
    sync.add_argument("corpus", help="the name of a Corpus in the declaration file")
    sync.set_defaults(run=_sync)
    rebuild = commands.add_parser(
        "rebuild",
        parents=[declared],
        help="re-embed every declared Corpus into a fresh index and make it live if complete "
        "enough",
    )
    rebuild.set_defaults(run=_rebuild)
    forget = commands.add_parser("forget", help="remove every Document of a Corpus from the index")
    forget.add_argument("corpus", help="the name of a Corpus that has Documents in the index")
    forget.set_defaults(run=_forget)
    commands.add_parser(
        "switch-back", help="make the previous index the Live index again"
    ).set_defaults(run=_switch_back)
    search = commands.add_parser("search", help="search the Live index")
    search.add_argument("query")
    search.add_argument("--space", help="only results from this Space")
    search.add_argument("--repository", help="only code from this Repository (its path)")
    search.add_argument("--ref", help="search this Tracked ref instead of the Primary refs")
    search.add_argument("-k", type=int, help="how many Passages to return")
    search.set_defaults(run=_search)
    commands.add_parser(
        "status", help="show the Live index and the last Sync or Rebuild"
    ).set_defaults(run=_status)
    return parser


def _declaration_reader(
    args: argparse.Namespace, environment: Environment
) -> Callable[[], Declaration]:
    """Reads the declaration when called: search and status never read the file."""
    path = getattr(args, "corpora", None) or environment.corpora_file
    return lambda: Declaration.read(path)


def _sync(args, session: _Session) -> int:
    corpus = session.declared(lambda d: d.corpus(args.corpus, session.environment))
    report = session.index.sync(corpus)
    _print_sync(report, session.out)
    return 1 if report.failed else 0


def _rebuild(args, session: _Session) -> int:
    corpora = session.declared(lambda d: d.corpora(session.environment))
    report = session.index.rebuild(corpora)
    _print_rebuild(report, session.out)
    return 0 if report.promoted else 1


def _forget(args, session: _Session) -> int:
    # Any name that has Documents may be forgotten: the point is to remove a Corpus that is
    # no longer declared, so the declaration is not consulted.
    removed = session.index.forget(args.corpus)
    print(f"Forget: removed {removed} Documents of Corpus {args.corpus!r}", file=session.out)
    return 0


def _switch_back(args, session: _Session) -> int:
    session.index.switch_back()
    print("Switched back to the previous index.", file=session.out)
    return 0


def _search(args, session: _Session) -> int:
    out = session.out
    passages = session.index.search(
        args.query, k=args.k, space=args.space, repository=args.repository, ref=args.ref
    )
    if not passages:
        print("No results.", file=out)
    for number, passage in enumerate(passages, 1):
        _print_passage(number, passage, out)
    if passages.notice:
        print(f"note: {passages.notice}", file=out)
    return 0


def _status(args, session: _Session) -> int:
    try:
        declared = session.read_declaration().names
    except DeclarationError:
        declared = None  # status works without a usable declaration; it just cannot compare
    status = session.index.status(declared)
    _print_status(status, session.out)
    return 1 if status.access is Access.REFUSED else 0


def _print_sync(report: SyncReport, out: TextIO) -> None:
    print(
        f"Sync: {report.added} added, {report.replaced} replaced, {report.removed} removed, "
        f"{report.unchanged} unchanged, {report.failed} failed",
        file=out,
    )
    _print_failures(report.failures, out)


def _print_rebuild(report: RebuildReport, out: TextIO) -> None:
    if report.promoted:
        print(
            f"Rebuild: promoted to the Live index, {report.indexed} indexed, "
            f"{report.failed} failed",
            file=out,
        )
    else:
        print(f"Rebuild: refused, the Live index is unchanged: {report.reason}", file=out)
    _print_failures(report.failures, out)
    for warning in report.warnings:
        print(f"warning: {warning}", file=out)


def _print_failures(failures: Mapping[str, str], out: TextIO) -> None:
    for document_id, why in failures.items():
        print(f"  failed {document_id}: {why}", file=out)


def _print_passage(number: int, passage: Passage, out: TextIO) -> None:
    print(f"{number}. {passage.title}", file=out)
    for name, value in passage.citation():
        print(f"   {name.replace('_', ' ')}: {value}", file=out)
    for line in passage.text.strip().splitlines():
        print(f"   | {line}", file=out)
    print(file=out)


def _embedding(f) -> str:
    if f.retrieval is RetrievalMode.KEYWORD:
        return "keyword-only retrieval (no embedding model)"
    return f"embedding model {f.embed_model!r} ({f.dimension} dimensions)"


def _chunks(f) -> str:
    if f.chunk_tokens is None or f.chunk_overlap is None:
        return "chunk size not recorded"
    return f"chunks of {f.chunk_tokens} tokens overlapping by {f.chunk_overlap}"


def _code_chunks(f) -> str:
    if f.code_chunk_tokens is None or f.code_splitting is None:
        return ", code chunking not recorded"
    by_syntax = f" ({', '.join(f.syntax_languages)} by syntax)" if f.syntax_languages else ""
    return f", code split by {f.code_splitting}{by_syntax} into {f.code_chunk_tokens} tokens"


def _print_status(status: IndexStatus, out: TextIO) -> None:
    def show(name: str, value) -> None:
        print(f"{name + ':':<13}{'unknown' if value is None else value}", file=out)

    show("generation", status.generation or "none yet")
    show("access", status.access)
    show("documents", status.documents)
    show("passages", status.passages)
    if status.format is None:
        show("format", "not recorded")
    else:
        f = status.format
        show(
            "format",
            f"schema {f.schema_version}, {_embedding(f)}, sparse k1={f.sparse.k1:g} "
            f"b={f.sparse.b:g} avg_doc_length={f.sparse.avg_doc_length:g}, "
            f"{_chunks(f)}{_code_chunks(f)}",
        )
    if status.last_sync is None:
        show("last sync", "never")
    else:
        last = status.last_sync
        show(
            "last sync",
            f"{last.kind} {last.outcome} at {last.at.isoformat(timespec='seconds')}: {last.detail}",
        )
    for name, last in sorted(status.corpora.items()):
        show(
            f"corpus {name}",
            f"{last.kind} {last.outcome} at {last.at.isoformat(timespec='seconds')}: {last.detail}",
        )
    for warning in status.warnings:
        print(f"warning: {warning}", file=out)
