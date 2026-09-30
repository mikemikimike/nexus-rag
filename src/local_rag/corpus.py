from collections.abc import Iterable, Mapping, Sequence
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
    parent_id: str | None = None  # the page an Attachment belongs to


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
    parent_id: str | None = None  # the page an Attachment belongs to


@dataclass(frozen=True)
class DocumentRef:
    id: str
    version: str


class Listing(list[DocumentRef]):
    """The Documents a Corpus could list, and the parts of it that it could not (name -> why).

    While any part is missing the listing is incomplete: Documents absent from it may still
    exist, so a Sync removes none, and a Rebuild, which would leave them out, is refused.
    """

    def __init__(self, refs: Iterable[DocumentRef] = (),
                 failures: Mapping[str, str] | None = None):
        super().__init__(refs)
        self.failures = dict(failures or {})


def listing_failures(refs: Sequence[DocumentRef]) -> dict[str, str]:
    """The parts of the Corpus a listing could not cover; empty for a plain list."""
    return dict(getattr(refs, "failures", {}))


class Corpus(Protocol):
    """A source of Documents, independent of where they come from."""

    def list(self) -> Sequence[DocumentRef]:
        """The Documents, versioned. A Corpus that may fail to list part of itself returns
        a Listing naming what it missed, rather than raising and losing the rest."""
        ...

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
        media_type="text/plain", parent_id=document.parent_id)
