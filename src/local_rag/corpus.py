import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

# A Corpus name prefixes the id of each of its Documents in the Knowledge index ("name:id"),
# so it must not hold the separator: the first colon of a stored id ends the name.
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


def check_corpus_name(name: str) -> str:
    """The name, if it can name a Corpus; raises ValueError otherwise."""
    if not _NAME.fullmatch(name):
        raise ValueError(
            f"{name!r} cannot name a Corpus: use letters, digits, '-' and '_', "
            f"starting with a letter or digit")
    return name


def qualified_id(corpus: str, document_id: str) -> str:
    """The id a Document has in the Knowledge index: its Corpus's name, then its own id."""
    return f"{corpus}:{document_id}"


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    url: str
    space: str
    version: str
    text: str
    parent_id: str | None = None  # the page an Attachment belongs to
    corpus: str | None = None  # the Corpus that owns it; set by the Knowledge index


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
    """A source of Documents, independent of where they come from.

    ``name`` is the operator's name for it. It owns the Documents it lists: a Sync of this
    Corpus adds, replaces and removes those and no others, and ids a Corpus yields need only
    be unique within it.
    """

    name: str

    def list(self) -> Sequence[DocumentRef]:
        """The Documents, versioned. A Corpus that may fail to list part of itself returns
        a Listing naming what it missed, rather than raising and losing the rest."""
        ...

    def fetch(self, document_id: str) -> RawDocument: ...


class InMemoryCorpus:
    """Holds Documents (ready text) or RawDocuments (content with a media type)."""

    def __init__(self, documents: Iterable[Document | RawDocument] = (), *,
                 name: str = "default"):
        self.name = check_corpus_name(name)
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
