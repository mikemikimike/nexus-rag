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
class DocumentRef:
    id: str
    version: str


class Corpus(Protocol):
    """A source of Documents, independent of where they come from."""

    def list(self) -> Sequence[DocumentRef]: ...

    def fetch(self, document_id: str) -> Document: ...


class InMemoryCorpus:
    def __init__(self, documents: Iterable[Document] = ()):
        self._documents = {d.id: d for d in documents}

    def list(self) -> list[DocumentRef]:
        return [DocumentRef(d.id, d.version) for d in self._documents.values()]

    def fetch(self, document_id: str) -> Document:
        return self._documents[document_id]
