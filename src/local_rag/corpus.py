import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import quote

# A Corpus name prefixes the id of each of its Documents in the Knowledge index ("name:id"),
# so it must not hold the separator: the first colon of a stored id ends the name.
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


def check_corpus_name(name: str) -> str:
    """The name, if it can name a Corpus; raises ValueError otherwise."""
    if not _NAME.fullmatch(name):
        raise ValueError(
            f"{name!r} cannot name a Corpus: use letters, digits, '-' and '_', "
            f"starting with a letter or digit"
        )
    return name


# The one list of prose Media types: every other text type is code. Whatever decides by
# prose or code asks ``is_code`` rather than keeping a list of its own.
PROSE_MEDIA_TYPES = frozenset({"text/plain", "text/markdown", "text/x-rst", "text/html"})


def is_code(media_type: str) -> bool:
    """Whether a Media type is source code, which is read by lines: text that is not prose."""
    return media_type.startswith("text/") and media_type not in PROSE_MEDIA_TYPES


def qualified_id(corpus: str, document_id: str) -> str:
    """The id a Document has in the Knowledge index: its Corpus's name, then its own id."""
    return f"{corpus}:{document_id}"


@dataclass(frozen=True)
class TrackedRef:
    """A Repository at one branch or tag whose files a GitLab Corpus contributes."""

    repository: str  # the Repository's full path, ``platform/payments``
    ref: str  # a branch or a tag
    primary: bool = True  # whether this is the Repository's Primary ref

    @property
    def part(self) -> str:
        """What a failure to read it is called in a listing: ``<repository>@<ref>``."""
        return f"{self.repository}@{self.ref}"


@dataclass(frozen=True)
class CodeFile:
    """Where a Document from code was read: a file at a Tracked ref."""

    tracked_ref: TrackedRef
    path: str  # the file's path within its Repository

    @property
    def document_id(self) -> str:
        """``<repository>@<ref>:<path>``: a ref and a Repository path never hold a colon."""
        return f"{self.tracked_ref.part}:{self.path}"

    @staticmethod
    def split_id(document_id: str) -> tuple[str, str]:
        """A code Document's id as its Tracked ref's ``part`` and the file's path."""
        part, _, path = document_id.partition(":")
        if "@" not in part or not path:
            raise ValueError(f"{document_id!r} is not the id of a file at a Tracked ref")
        return part, path

    def link(self, base_url: str) -> str:
        """Where GitLab shows the file at its ref."""
        ref = self.tracked_ref
        return f"{base_url.rstrip('/')}/{ref.repository}/-/blob/{quote(ref.ref)}/{quote(self.path)}"

    def placement(self, base_url: str) -> str:
        """The listing's own say on the file (see ``DocumentRef.placement``): whether its ref is
        the Primary ref, and its link."""
        return f"{'primary' if self.tracked_ref.primary else 'tracked'} {self.link(base_url)}"


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
    media_type: str = "text/plain"  # the Media type its Corpus gave it, kept past extraction
    parts: tuple[str, ...] = ()  # the parts of its Corpus it was listed under; set by the index
    code: CodeFile | None = None  # where a Document from code was read; None for other Corpora
    placement: str = ""  # see DocumentRef.placement; set by the index


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
    code: CodeFile | None = None  # where a Document from code was read; None for other Corpora


@dataclass(frozen=True)
class DocumentRef:
    id: str
    version: str
    # The parts of the Corpus this Document was listed under (a Root, a page's Attachments, a
    # Tracked ref), each named by the key a Listing's failures use for that part. If any of
    # them could not be listed, the Document is not removed. A Document may sit in several
    # parts at once (an Attachment is in its page's Root and in its page's Attachments).
    parts: tuple[str, ...] = ()
    # What else the stored copy says that comes from the listing, not the content (a code file's
    # Primary ref mark and link): a change re-adds the Document though its version is the same.
    placement: str = ""


class Listing(list[DocumentRef]):
    """The Documents a Corpus could list, and the parts of it that it could not (name -> why).

    While any part is missing the listing is incomplete: Documents absent from it may still
    exist, so a Sync removes none of those under a failed part (a ref's ``parts`` holds the
    failure's key) and a Rebuild, which would leave them out, is refused. Documents stored
    without a recorded part are held back while any part failed.
    """

    def __init__(self, refs: Iterable[DocumentRef] = (), failures: Mapping[str, str] | None = None):
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

    def __init__(self, documents: Iterable[Document | RawDocument] = (), *, name: str = "default"):
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
        id=document.id,
        title=document.title,
        url=document.url,
        space=document.space,
        version=document.version,
        content=document.text.encode("utf-8"),
        media_type="text/plain",
        parent_id=document.parent_id,
    )
