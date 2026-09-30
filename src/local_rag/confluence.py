"""A Corpus over Confluence, read through the `confluence` command (pchuri/confluence-cli)."""
import fnmatch
import json
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from local_rag.corpus import DocumentRef, Listing, RawDocument
from local_rag.extraction import MEDIA_TYPES

_MAX_ERROR_LENGTH = 400
_PAGE_FILE = "page.md"
# Attachments left out by default: images (they are only readable with OCR, which is
# off unless enabled, and a page's screenshots would each fail as an empty Document),
# video and archives.
DEFAULT_EXCLUDED_ATTACHMENTS = (
    "*.png", "*.jpg", "*.jpeg", "*.gif", "*.svg", "*.bmp", "*.webp", "*.tif", "*.tiff",
    "*.mp4", "*.mov", "*.avi", "*.mkv", "*.zip", "*.gz", "*.tar", "*.7z")


class ConfluenceError(Exception):
    pass


@dataclass(frozen=True)
class _Entry:
    """What a listing learned about a Document, kept until the Sync fetches it."""

    id: str
    title: str
    url: str
    space: str
    version: str
    page_id: str
    attachment: str | None = None  # the file name, for an Attachment


class ConfluenceCorpus:
    """Every page under the roots is a Document, identified by its page id and versioned by
    its page version; every attachment of those pages is an Attachment Document, identified
    as "<page id>/attachments/<file name>" and citing its page's title, link and Space.
    Roots are page ids or page URLs; a root covers its whole subtree. Attachments of a type
    that cannot be read, or matching ``exclude_attachments`` (file name globs), are left out.
    """

    def __init__(self, roots: Sequence[str], *, confluence_bin: str = "confluence",
                 max_depth: int = 0, attachments: bool = True,
                 exclude_attachments: Sequence[str] = DEFAULT_EXCLUDED_ATTACHMENTS,
                 timeout: float = 600.0):
        self._roots = list(roots)
        self._bin = confluence_bin
        self._max_depth = max_depth
        self._attachments = attachments
        self._exclude = [pattern.lower() for pattern in exclude_attachments]
        self._timeout = timeout
        self._entries: dict[str, _Entry] = {}

    def list(self) -> Listing:
        """The pages and attachments under the roots; a root or an attachment listing that
        failed is named in the Listing's failures, and the rest is still listed."""
        self._entries = {}
        failures: dict[str, str] = {}
        for root in self._roots:
            try:
                entries = self._list_root(root, failures)
            except ConfluenceError as error:
                failures[f"root {root}"] = str(error)
            else:
                self._entries.update((e.id, e) for e in entries)
        return Listing([DocumentRef(e.id, e.version) for e in self._entries.values()],
                       failures)

    def fetch(self, document_id: str) -> RawDocument:
        entry = self._entries[document_id]
        with tempfile.TemporaryDirectory() as directory:
            if entry.attachment is None:
                name, media_type = _PAGE_FILE, "text/markdown"
                self._run("export", entry.page_id, "--format", "markdown", "--dest",
                          directory, "--file", name, "--skip-attachments")
            else:
                name, media_type = entry.attachment, _media_type(entry.attachment)
                self._run("attachments", entry.page_id, "--pattern", name, "--download",
                          "--dest", directory)
            try:
                content = (Path(directory) / name).read_bytes()
            except FileNotFoundError:
                raise ConfluenceError(f"confluence produced no file {name!r}") from None
        return RawDocument(
            id=entry.id, title=entry.title, url=entry.url, space=entry.space,
            version=entry.version, content=content, media_type=media_type,
            parent_id=None if entry.attachment is None else entry.page_id)

    def _list_root(self, root: str, failures: dict[str, str]) -> Sequence[_Entry]:
        info = self._json("info", root)
        found = [info] if info.get("type", "page") == "page" else []
        children = ["children", str(info["id"]), "--recursive"]
        if self._max_depth:
            children += ["--max-depth", str(self._max_depth)]
        found += [c for c in self._json(*children).get("children", [])
                  if c.get("type", "page") == "page"]
        entries = []
        for page in found:
            entry = _Entry(
                id=str(page["id"]), title=page["title"], url=page.get("url", ""),
                space=page.get("spaceKey", ""), version=_version(page.get("version")),
                page_id=str(page["id"]))
            entries.append(entry)
            if self._attachments:
                try:
                    entries += self._list_attachments(entry)
                except ConfluenceError as error:
                    failures[f"attachments of page {entry.id}"] = str(error)
        return entries

    def _list_attachments(self, page: _Entry) -> Sequence[_Entry]:
        files = self._json("attachments", page.page_id).get("attachments", [])
        return [_Entry(
            id=f"{page.id}/attachments/{f['title']}", title=f"{page.title} / {f['title']}",
            url=page.url, space=page.space,
            version=_version(f.get("version"), default=page.version),
            page_id=page.id, attachment=f["title"])
            for f in files if self._wanted(f["title"])]

    def _wanted(self, name: str) -> bool:
        return (Path(name).suffix.lower() in MEDIA_TYPES
                and not any(fnmatch.fnmatch(name.lower(), p) for p in self._exclude))

    def _json(self, *args: str):
        return json.loads(self._run(*args, "--json"))

    def _run(self, *args: str) -> str:
        try:
            result = subprocess.run(
                [self._bin, *args], capture_output=True, text=True, errors="replace",
                timeout=self._timeout)
        except FileNotFoundError:
            raise ConfluenceError(f"confluence not found (looked for {self._bin!r})") from None
        except subprocess.TimeoutExpired:
            raise ConfluenceError(
                f"confluence {args[0]} timed out after {self._timeout:g}s") from None
        if result.returncode != 0:
            raise ConfluenceError(
                f"confluence {args[0]} failed ({result.returncode}): "
                f"{(result.stderr or result.stdout).strip()[:_MAX_ERROR_LENGTH]}")
        return result.stdout


def _media_type(name: str) -> str:
    return MEDIA_TYPES[Path(name).suffix.lower()]


def _version(value, default: str = "") -> str:
    """A version as the CLI reports it: a number, or an object holding one."""
    if isinstance(value, dict):
        value = value.get("number")
    return default if value is None else str(value)


def read_roots(path: Path | str) -> list[str]:
    """The roots in a file: one page id or URL per line; blank lines and # comments are ignored."""
    roots = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            roots.append(line.split(" #")[0].strip())
    return roots
