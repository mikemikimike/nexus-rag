"""Command line adapter over the Knowledge index: Sync, Rebuild, switch-back, search, status.

Settings come from the environment (see ``environment``). Each command prints what
happened and returns a process exit code: 0 on success, 1 on failure or refusal, 2 for a
command line that cannot be understood.
"""
import argparse
import logging
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TextIO

from qdrant_client import QdrantClient

from local_rag.confluence import DEFAULT_EXCLUDED_ATTACHMENTS, ConfluenceCorpus, read_roots
from local_rag.corpus import Corpus
from local_rag.environment import settings_from_environment
from local_rag.folder import FolderCorpus
from local_rag.gateway import ModelGateway
from local_rag.settings import RetrievalMode
from local_rag.gateway_litellm import gateway_from_settings
from local_rag.index import IndexStatus, KnowledgeIndex, Passage, RebuildReport, SyncReport
from local_rag.index_format import Access


class UsageError(Exception):
    """The command line names a Corpus that cannot be used."""


def main(argv: Sequence[str], environ: Mapping[str, str], *,
         stdout: TextIO | None = None, stderr: TextIO | None = None,
         gateway: ModelGateway | None = None, client: QdrantClient | None = None) -> int:
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
        settings = settings_from_environment(environ)
        corpus = _corpus(args, environ) if args.command in ("sync", "rebuild") else None
        index = KnowledgeIndex(settings, gateway or gateway_from_settings(settings),
                               client=client)
        return args.run(args, index, corpus, out)
    except UsageError as error:
        print(f"error: {error}", file=err)
        return 2
    except Exception as error:
        print(f"error: {_describe(error)}", file=err)
        return 1


def run() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    sys.exit(main(sys.argv[1:], os.environ))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="local-rag", description="Sync, search and inspect the Knowledge index.")
    commands = parser.add_subparsers(dest="command", required=True)

    source = argparse.ArgumentParser(add_help=False)
    group = source.add_argument_group(
        "Corpus", "give a folder, or Confluence roots (page ids or URLs), not both")
    group.add_argument("--folder", type=Path, help="a folder of documents")
    group.add_argument("--root", action="append", default=[], metavar="PAGE",
                       help="a Confluence page id or URL; its whole subtree is indexed")
    group.add_argument("--roots-file", type=Path,
                       help="a file with one Confluence root per line")
    group.add_argument("--max-depth", type=int, default=0,
                       help="Confluence: how deep below a root to go (0 = all)")
    group.add_argument("--no-attachments", action="store_true",
                       help="Confluence: leave attachments out")
    group.add_argument("--exclude-attachments", metavar="GLOBS",
                       default=",".join(DEFAULT_EXCLUDED_ATTACHMENTS),
                       help="Confluence: comma-separated attachment name patterns to skip")

    sync = commands.add_parser("sync", parents=[source], help="bring the index up to date")
    sync.set_defaults(run=_sync)
    rebuild = commands.add_parser(
        "rebuild", parents=[source],
        help="re-embed everything into a fresh index and make it live if complete enough")
    rebuild.set_defaults(run=_rebuild)
    commands.add_parser(
        "switch-back", help="make the previous index the Live index again"
    ).set_defaults(run=_switch_back)
    search = commands.add_parser("search", help="search the Live index")
    search.add_argument("query")
    search.add_argument("--space", help="only results from this Space")
    search.add_argument("-k", type=int, help="how many Passages to return")
    search.set_defaults(run=_search)
    commands.add_parser(
        "status", help="show the Live index and the last Sync or Rebuild"
    ).set_defaults(run=_status)
    return parser


def _corpus(args: argparse.Namespace, environ: Mapping[str, str]) -> Corpus:
    roots = list(args.root)
    if args.roots_file is not None:
        try:
            roots += read_roots(args.roots_file)
        except (OSError, UnicodeDecodeError) as error:
            raise UsageError(f"cannot read the roots file: {error}") from None
    if args.folder is not None and roots:
        raise UsageError("give a folder or Confluence roots, not both")
    if args.folder is not None:
        if not args.folder.is_dir():
            # An absent folder (an unmounted share) must not look like an empty Corpus,
            # which a Sync would answer by removing every Document.
            raise UsageError(f"{args.folder} is not a folder")
        return FolderCorpus(args.folder)
    if not roots:
        raise UsageError("give a Corpus: --folder, or --root / --roots-file for Confluence")
    return ConfluenceCorpus(
        roots, confluence_bin=environ.get("CONFLUENCE_BIN") or "confluence",
        max_depth=args.max_depth, attachments=not args.no_attachments,
        exclude_attachments=[g.strip() for g in args.exclude_attachments.split(",")
                             if g.strip()])


def _sync(args, index: KnowledgeIndex, corpus: Corpus, out: TextIO) -> int:
    report = index.sync(corpus)
    _print_sync(report, out)
    return 1 if report.failed else 0


def _rebuild(args, index: KnowledgeIndex, corpus: Corpus, out: TextIO) -> int:
    report = index.rebuild(corpus)
    _print_rebuild(report, out)
    return 0 if report.promoted else 1


def _switch_back(args, index: KnowledgeIndex, corpus: None, out: TextIO) -> int:
    index.switch_back()
    print("Switched back to the previous index.", file=out)
    return 0


def _search(args, index: KnowledgeIndex, corpus: None, out: TextIO) -> int:
    passages = index.search(args.query, k=args.k, space=args.space)
    if not passages:
        print("No results.", file=out)
    for number, passage in enumerate(passages, 1):
        _print_passage(number, passage, out)
    if passages.notice:
        print(f"note: {passages.notice}", file=out)
    return 0


def _status(args, index: KnowledgeIndex, corpus: None, out: TextIO) -> int:
    status = index.status()
    _print_status(status, out)
    return 1 if status.access is Access.REFUSED else 0


def _print_sync(report: SyncReport, out: TextIO) -> None:
    print(f"Sync: {report.added} added, {report.replaced} replaced, {report.removed} removed, "
          f"{report.unchanged} unchanged, {report.failed} failed", file=out)
    _print_failures(report.failures, out)


def _print_rebuild(report: RebuildReport, out: TextIO) -> None:
    if report.promoted:
        print(f"Rebuild: promoted to the Live index, {report.indexed} indexed, "
              f"{report.failed} failed", file=out)
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
    print(f"   link: {passage.url}", file=out)
    print(f"   space: {passage.space}", file=out)
    print(f"   page id: {passage.parent_id or passage.document_id}", file=out)
    if passage.parent_id:
        print(f"   attachment: {passage.document_id}", file=out)
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
        show("format", f"schema {f.schema_version}, {_embedding(f)}, sparse k1={f.sparse.k1:g} "
                       f"b={f.sparse.b:g} avg_doc_length={f.sparse.avg_doc_length:g}, "
                       f"{_chunks(f)}")
    if status.last_sync is None:
        show("last sync", "never")
    else:
        last = status.last_sync
        show("last sync", f"{last.kind} {last.outcome} at "
                          f"{last.at.isoformat(timespec='seconds')}: {last.detail}")
    for warning in status.warnings:
        print(f"warning: {warning}", file=out)


def _describe(error: Exception) -> str:
    return str(error) or type(error).__name__
