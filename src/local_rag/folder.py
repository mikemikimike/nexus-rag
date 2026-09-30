"""A Corpus over a folder of files: the relative path is the Document's identity."""
import hashlib
from pathlib import Path

from local_rag.corpus import DocumentRef, RawDocument

MEDIA_TYPES = {
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".rst": "text/x-rst",
    ".html": "text/html",
    ".htm": "text/html",
    ".pdf": "application/pdf",
    ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".ppt": "application/vnd.ms-powerpoint",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".xls": "application/vnd.ms-excel",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}


class FolderCorpus:
    """Identity is the path relative to the folder, version a hash of the content and
    Space the top-level folder (files directly in the folder get the folder's own name).
    Files of unknown type and anything hidden (a dot-prefixed name) are left out.
    """

    def __init__(self, root: Path | str):
        self._root = Path(root).resolve()

    def list(self) -> list[DocumentRef]:
        return [DocumentRef(document_id, _hash(path.read_bytes()))
                for document_id, path in _files(self._root)]

    def fetch(self, document_id: str) -> RawDocument:
        path = self._root / document_id
        content = path.read_bytes()
        parts = Path(document_id).parts
        return RawDocument(
            id=document_id, title=path.stem, url=path.as_uri(),
            space=parts[0] if len(parts) > 1 else self._root.name,
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


def _hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
