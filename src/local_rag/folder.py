"""A Corpus over a folder of files: the relative path is the Document's identity."""
import hashlib
from pathlib import Path

from local_rag.corpus import DocumentRef, RawDocument
from local_rag.extraction import MEDIA_TYPES

_UNREADABLE = "unreadable"


class FolderCorpus:
    """Identity is the path relative to the folder, version a hash of the content and
    Space the top-level folder (files directly in the folder get the folder's own name,
    or "root" if it has none). Files of unknown type and anything hidden (a dot-prefixed
    name) are left out.
    """

    def __init__(self, root: Path | str):
        self._root = Path(root).resolve()

    def list(self) -> list[DocumentRef]:
        return [DocumentRef(document_id, _version(path))
                for document_id, path in _files(self._root)]

    def fetch(self, document_id: str) -> RawDocument:
        path = self._root / document_id
        content = path.read_bytes()
        parts = Path(document_id).parts
        return RawDocument(
            id=document_id, title=path.stem, url=path.as_uri(),
            space=parts[0] if len(parts) > 1 else self._root.name or "root",
            version=_hash(content), content=content,
            media_type=MEDIA_TYPES[path.suffix.lower()])


def _files(root: Path) -> list[tuple[str, Path]]:
    found = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if (path.is_file() and path.suffix.lower() in MEDIA_TYPES
                and not any(part.startswith(".") for part in relative.parts)):
            found.append((relative.as_posix(), path))
    return found


def _version(path: Path) -> str:
    try:
        return _hash(path.read_bytes())
    except OSError:
        # Not hashable now: a version that matches nothing indexed, so the Sync fetches
        # the file, fails on it and counts one failed Document instead of aborting.
        return _UNREADABLE


def _hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()

