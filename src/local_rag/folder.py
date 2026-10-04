"""A Corpus over a folder of files: the relative path is the Document's identity."""

import hashlib
import os
import stat
from pathlib import Path

from local_rag.corpus import DocumentRef, Listing, RawDocument, check_corpus_name
from local_rag.errors import describe
from local_rag.extraction import MEDIA_TYPES

_UNREADABLE = "unreadable"


class FolderCorpus:
    """Identity is the path relative to the folder, version a hash of the content and
    Space the top-level folder (files directly in the folder get the folder's own name,
    or "root" if it has none). Files of unknown type and anything hidden (a dot-prefixed
    name) are left out.
    """

    def __init__(self, root: Path | str, *, name: str = "folder"):
        self.name = check_corpus_name(name)
        self._root = Path(root).resolve()

    def list(self) -> Listing:
        refs, failures = _walk(self._root)
        return Listing(refs, failures)

    def fetch(self, document_id: str) -> RawDocument:
        path = self._root / document_id
        content = path.read_bytes()
        parts = Path(document_id).parts
        return RawDocument(
            id=document_id,
            title=path.stem,
            url=path.as_uri(),
            space=parts[0] if len(parts) > 1 else self._root.name or "root",
            version=_hash(content),
            content=content,
            media_type=MEDIA_TYPES[path.suffix.lower()],
        )


def _walk(root: Path) -> tuple[list[DocumentRef], dict[str, str]]:
    refs: list[DocumentRef] = []
    failures: dict[str, str] = {}

    def on_error(error: OSError) -> None:
        try:
            filename = getattr(error, "filename", None)
            part = Path(filename).relative_to(root).as_posix() if filename else "."
        except (ValueError, TypeError):
            part = str(getattr(error, "filename", None) or ".")
        failures[part] = describe(error)

    for dirpath, dirnames, filenames in os.walk(root, onerror=on_error):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for filename in sorted(filenames):
            if filename.startswith("."):
                continue
            path = Path(dirpath) / filename
            if path.suffix.lower() not in MEDIA_TYPES:
                continue
            try:
                # os.stat, not Path.is_file(), which swallows PermissionError on newer Pythons.
                if not stat.S_ISREG(os.stat(path).st_mode):
                    continue
            except (FileNotFoundError, NotADirectoryError):
                continue  # a dangling symlink, or gone since the listing
            except OSError as error:
                # Listed but not stat-able (a folder that is readable but not searchable):
                # report the folder so its indexed Documents are held back, not removed.
                failures.setdefault(Path(dirpath).relative_to(root).as_posix(), describe(error))
                continue
            relative = path.relative_to(root)
            document_id = relative.as_posix()
            parts = tuple(reversed([p.as_posix() for p in relative.parents]))
            refs.append(DocumentRef(document_id, _version(path), parts=parts))

    refs.sort(key=lambda r: r.id)
    return refs, failures


def _version(path: Path) -> str:
    try:
        return _hash(path.read_bytes())
    except OSError:
        # Not hashable now: a version that matches nothing indexed, so the Sync fetches
        # the file, fails on it and counts one failed Document instead of aborting.
        return _UNREADABLE


def _hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
