from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    url: str
    space: str
    version: str
    text: str


@dataclass(frozen=True)
class RawDocument:
    """A Document as a Corpus yields it: content not yet turned into text."""

    id: str
    title: str
    url: str
    space: str
    version: str
    content: bytes
    media_type: str


@dataclass(frozen=True)
class DocumentRef:
    id: str
    version: str


class Corpus(Protocol):
    """A source of Documents, independent of where they come from."""

    def list(self) -> Sequence[DocumentRef]: ...

    def fetch(self, document_id: str) -> RawDocument: ...


class InMemoryCorpus:
    """Holds Documents (ready text) or RawDocuments (content with a media type)."""

    def __init__(self, documents: Iterable[Document | RawDocument] = ()):
        self._documents = {d.id: _raw(d) for d in documents}

    def list(self) -> list[DocumentRef]:
        return [DocumentRef(d.id, d.version) for d in self._documents.values()]

    def fetch(self, document_id: str) -> RawDocument:
        return self._documents[document_id]


def _raw(document: Document | RawDocument) -> RawDocument:
    if isinstance(document, RawDocument):
        return document
    return RawDocument(
        id=document.id, title=document.title, url=document.url, space=document.space,
        version=document.version, content=document.text.encode("utf-8"),
        media_type="text/plain")
