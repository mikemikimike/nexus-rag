"""How a generation of the Knowledge index lays out its Documents and Passages in Qdrant.

A generation is one collection. Each Passage is a point with a sparse ``bm25`` vector and, in
a Hybrid index, a dense one, carrying its Document's identity, version, title, link, Space,
Corpus and listing (its parts and placement), and for code the Repository, ref and lines.
Each Document also has one vectorless point holding its full text, so the text is stored once
however many Passages it has and never appears in search results.

This layout is what installed indexes hold: changing a name or a field here makes their
Documents look changed (re-added on the next Sync) or unreadable, so it changes only with the
schema version in ``index_format``. Points written by earlier versions lack some fields (a
Corpus, parts, a placement) and are read as they are. A point missing a field every version
wrote is reported as corrupt, never skipped.
"""

import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from nexus_rag.chunker import Piece
from nexus_rag.corpus import CodeFile, Document, DocumentRef, TrackedRef
from nexus_rag.generations import CorruptIndex
from nexus_rag.sparse import SparseVector

# What marks a Document's own point among its Passages.
_DOCUMENT_KIND = "document"


@dataclass(frozen=True)
class Passage:
    document_id: str
    title: str
    url: str
    space: str
    text: str
    score: float
    parent_id: str | None = None  # the page an Attachment belongs to
    corpus: str | None = None  # None for an index built before named Corpora
    code: CodeFile | None = None  # where a Passage of code was read
    start_line: int | None = None  # 1-based, inclusive
    end_line: int | None = None
    symbol: str | None = None  # the class or function it sits in, if the splitter knows

    @property
    def line_range(self) -> str | None:
        """``10-20``, or ``10`` for one line; None for a Passage that is not code."""
        if self.start_line is None:
            return None
        if self.end_line in (None, self.start_line):
            return str(self.start_line)
        return f"{self.start_line}-{self.end_line}"

    @property
    def code_location(self) -> list[tuple[str, str]]:
        """What names where a code Passage is, as (label, value): its file, lines and enclosing
        symbol when known; empty for a Passage that is not code."""
        if self.start_line is None:
            return []
        found = [
            ("file", self.code.path if self.code else self.title),
            ("lines", self.line_range),
            ("symbol", self.symbol),
        ]
        return [(name, value) for name, value in found if value]

    def citation(self) -> tuple[tuple[str, str], ...]:
        """The ordered (name, value) fields that cite it: where to read it, the page it is on
        (its parent's, for an Attachment), its space, the Tracked ref for code, the Attachment
        itself, and the code location. Renderers show these and decide only the layout."""
        fields = [
            ("link", self.link),
            ("page_id", self.parent_id or self.document_id),
            ("space", self.space),
        ]
        if self.code:
            fields.append(("ref", self.code.tracked_ref.ref))
        if self.parent_id:
            fields.append(("attachment", self.document_id))
        return (*fields, *self.code_location)

    @property
    def link(self) -> str:
        """Where to read it: the Document's URL, at its lines for code (GitLab's anchor)."""
        return self.url if self.start_line is None else f"{self.url}#L{self.line_range}"


@dataclass(frozen=True)
class Written:
    """One Passage of a Document as it is written: its piece and its vectors."""

    piece: Piece
    sparse: SparseVector
    dense: list[float] | None = None  # None in a Keyword-only index


@dataclass(frozen=True)
class Narrowing:
    """What a search keeps. Code at a Tracked ref that is not its Repository's Primary ref is
    left out unless ``ref`` names one; other Corpora's Passages carry no such mark."""

    space: str | None = None
    repository: str | None = None
    ref: str | None = None


class Stored:
    """What the index holds for one Document: its versions and the parts it was listed under."""

    def __init__(self) -> None:
        self.versions: set[str | None] = set()
        self.parts: set[str] = set()
        self.placements: set[str] = set()
        self.recorded = True  # False if any point predates recorded parts

    def add(self, payload: dict) -> None:
        self.versions.add(payload.get("version"))
        self.placements.add(payload.get("placement") or "")
        if payload.get("parts") is None:
            self.recorded = False
        else:
            self.parts.update(payload["parts"])

    def is_current(self, ref: DocumentRef, version: str | None = None) -> bool:
        """Unchanged: one stored version, the listed one (or ``version``, the one fetched),
        and the part and placement it was listed under (a Document that moved to another part,
        or whose Primary ref mark or link changed, is re-added so what is stored stays true)."""
        return (
            self.versions == {version or ref.version}
            and self.placements == {ref.placement}
            and (not self.recorded or self.parts == set(ref.parts))
        )

    def held_back_by(self, failures: Mapping[str, str]) -> bool:
        """True if a part this Document sat in could not be listed. With no part recorded,
        any failure might be its part."""
        if not failures:
            return False
        if not self.recorded or not self.parts:
            return True
        return not self.parts.isdisjoint(failures)


class PassageStore:
    def __init__(self, client: QdrantClient, *, payload_indexes: bool = True):
        """``payload_indexes`` is False for in-process Qdrant, which has none."""
        self._client = client
        self._payload_indexes = payload_indexes

    def exists(self, collection: str) -> bool:
        return self._client.collection_exists(collection)

    def create(self, collection: str, dimension: int | None) -> None:
        """``dimension`` is the size of the dense vectors, None for a Keyword-only index."""
        self._client.create_collection(
            collection_name=collection,
            vectors_config={}
            if dimension is None
            else {"dense": models.VectorParams(size=dimension, distance=models.Distance.COSINE)},
            sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        if self._payload_indexes:
            self._client.create_payload_index(
                collection, "document_id", models.PayloadSchemaType.KEYWORD
            )
            self._client.create_payload_index(
                collection, "corpus", models.PayloadSchemaType.KEYWORD
            )

    def dense_size(self, collection: str) -> int | None:
        """The size of the collection's dense vectors; None if it has none (Keyword-only).

        Raises CorruptIndex if the collection is not laid out as a generation.
        """
        params = self._client.get_collection(collection).config.params
        if not (isinstance(params.vectors, dict) and "bm25" in (params.sparse_vectors or {})):
            raise CorruptIndex(f"{collection} is not a Knowledge index collection")
        dense = params.vectors.get("dense")
        return dense.size if dense else None

    def write(self, collection: str, document: Document, passages: Sequence[Written]) -> None:
        """Store the Document with these Passages, replacing what it had."""
        document_point = _document_point(document)
        points = []
        for number, written in enumerate(passages):
            vectors: dict[str, models.Vector] = {
                "bm25": models.SparseVector(
                    indices=written.sparse.indices, values=written.sparse.values
                )
            }
            if written.dense is not None:
                vectors["dense"] = written.dense
            points.append(
                models.PointStruct(
                    id=_passage_id(document.id, number),
                    vector=vectors,
                    payload={
                        "document_id": document.id,
                        "version": document.version,
                        "title": document.title,
                        "url": document.url,
                        "space": document.space,
                        "text": written.piece.text,
                        "parent_id": document.parent_id,
                        "corpus": document.corpus,
                        "parts": list(document.parts),
                        "placement": document.placement,
                        **_of_code(document),
                        **_code_payload(document, written.piece),
                    },
                )
            )
        # Write the new Passages first, then delete this Document's leftover old ones (those
        # of an earlier, longer version included), so a failure in between never leaves the
        # Document missing from search.
        self._client.upsert(collection, points=[*points, document_point], wait=True)
        self._client.delete(
            collection,
            wait=True,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[_match("document_id", document.id)],
                    must_not=[
                        models.HasIdCondition(has_id=[*(p.id for p in points), document_point.id])
                    ],
                )
            ),
        )

    def remove(self, collection: str, document_id: str) -> None:
        self._client.delete(
            collection,
            wait=True,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=[_match("document_id", document_id)])
            ),
        )

    def forget(self, collection: str, corpus: str) -> int:
        """Remove every Document the Corpus owns; returns how many there were."""
        documents = self._client.count(
            collection,
            exact=True,
            count_filter=models.Filter(
                must=[_match("corpus", corpus), _match("kind", _DOCUMENT_KIND)]
            ),
        ).count
        self._client.delete(
            collection,
            wait=True,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=[_match("corpus", corpus)])
            ),
        )
        return documents

    def counts(self, collection: str) -> tuple[int, int]:
        """How many Documents and how many Passages the collection holds."""
        documents = self._client.count(
            collection,
            count_filter=models.Filter(must=[_match("kind", _DOCUMENT_KIND)]),
            exact=True,
        ).count
        return documents, self._client.count(collection, exact=True).count - documents

    def stored(self, collection: str, corpus: str) -> dict[str, Stored]:
        """What the collection holds per Document id, among the Documents ``corpus`` owns.

        More than one version means an interrupted replace, so the Document is re-synced.
        """
        points = self._scroll(
            collection,
            models.Filter(must=[_match("corpus", corpus)]),
            ["document_id", "version", "parts", "placement"],
        )
        found: dict[str, Stored] = {}
        with self._reading(collection):
            for point in points:
                payload = point.payload or {}
                found.setdefault(payload["document_id"], Stored()).add(payload)
        return found

    def corpora(self, collection: str) -> set[str]:
        """The names of the Corpora that own at least one Document in ``collection``."""
        points = self._scroll(
            collection, models.Filter(must=[_match("kind", _DOCUMENT_KIND)]), ["corpus"]
        )
        return {corpus for p in points if (corpus := (p.payload or {}).get("corpus"))}

    def spaces(self, collection: str) -> list[str]:
        """The named Spaces that have at least one Document, sorted."""
        points = self._scroll(collection, None, ["space"])
        with self._reading(collection):
            # A Document outside any named Space (a top-level project) has an empty one.
            return sorted({space for p in points if (space := (p.payload or {})["space"])})

    def document_text(self, collection: str, document_id: str) -> str | None:
        """The full text of a Document, or None if it is not there."""
        records, _ = self._client.scroll(
            collection,
            scroll_filter=models.Filter(
                must=[_match("kind", _DOCUMENT_KIND), _match("document_id", document_id)]
            ),
            limit=1,
            with_payload=["full_text"],
        )
        if not records:
            return None
        with self._reading(collection):
            return (records[0].payload or {})["full_text"]

    def search(
        self,
        collection: str,
        *,
        dense: list[float] | None,
        sparse: SparseVector,
        narrowing: Narrowing,
        candidates: int,
        limit: int,
    ) -> list[Passage]:
        """The Passages the dense and sparse vectors find, fused by rank, best first. With
        neither vector there is nothing to look for, and nothing is found."""
        must: list[models.Condition] = [
            _match(key, value)
            for key, value in (
                ("space", narrowing.space),
                ("repository", narrowing.repository),
                ("ref", narrowing.ref),
            )
            if value is not None
        ]
        # Only a Passage marked non-primary is left out: other Corpora's carry no mark.
        not_primary: list[models.Condition] = (
            [] if narrowing.ref is not None else [_match("primary", False)]
        )
        narrowed = models.Filter(must=must, must_not=not_primary)
        prefetch = []
        if dense is not None:
            prefetch.append(
                models.Prefetch(query=dense, using="dense", limit=candidates, filter=narrowed)
            )
        if sparse.indices:
            prefetch.append(
                models.Prefetch(
                    query=models.SparseVector(indices=sparse.indices, values=sparse.values),
                    using="bm25",
                    limit=candidates,
                    filter=narrowed,
                )
            )
        if not prefetch:
            return []
        points = self._client.query_points(
            collection,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        ).points
        with self._reading(collection):
            return [_passage(p) for p in points]

    def _scroll(
        self, collection: str, where: models.Filter | None, fields: list[str]
    ) -> list[models.Record]:
        found, offset = [], None
        while True:
            points, offset = self._client.scroll(
                collection,
                limit=1000,
                offset=offset,
                with_vectors=False,
                scroll_filter=where,
                with_payload=fields,
            )
            found.extend(points)
            if offset is None:
                return found

    @contextmanager
    def _reading(self, collection: str) -> Iterator[None]:
        """Report a payload that cannot be read as a corrupt index. Only reading payloads goes
        in here: an error from Qdrant itself is not corruption."""
        try:
            yield
        except (KeyError, TypeError, ValueError) as error:
            raise CorruptIndex(f"a point in {collection} is unreadable ({error!r})") from error


def _passage(point: models.ScoredPoint) -> Passage:
    payload = point.payload or {}
    return Passage(
        document_id=payload["document_id"],
        title=payload["title"],
        url=payload["url"],
        space=payload["space"],
        text=payload["text"],
        score=float(point.score),
        parent_id=payload.get("parent_id"),
        corpus=payload.get("corpus"),
        code=_code_file(payload),
        start_line=payload.get("start_line"),
        end_line=payload.get("end_line"),
        symbol=payload.get("symbol"),
    )


def _match(key: str, value: str | bool) -> models.FieldCondition:
    return models.FieldCondition(key=key, match=models.MatchValue(value=value))


def _passage_id(document_id: str, number: int) -> str:
    """The same Document always gets the same Passage ids, so a re-sync overwrites in place."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}#{number}"))


def _document_point(document: Document) -> models.PointStruct:
    return models.PointStruct(
        id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document.id}#document")),
        vector={},
        payload={
            "kind": _DOCUMENT_KIND,
            "document_id": document.id,
            "version": document.version,
            "title": document.title,
            "url": document.url,
            "space": document.space,
            "full_text": document.text,
            "parent_id": document.parent_id,
            "corpus": document.corpus,
            "parts": list(document.parts),
            "placement": document.placement,
            **_of_code(document),
        },
    )


def _code_payload(document: Document, piece: Piece) -> dict:
    """What a code Passage adds to the Document's ``_of_code`` payload: its file path and lines."""
    if piece.start_line is None:
        return {}
    return {
        "path": document.code and document.code.path,
        "start_line": piece.start_line,
        "end_line": piece.end_line,
        "symbol": piece.symbol,
    }


def _of_code(document: Document) -> dict:
    """The payload of a Document from code: its Repository, Tracked ref and whether that is
    the Primary ref. Other Documents carry none, so no filter on these keeps them out."""
    if document.code is None:
        return {}
    tracked = document.code.tracked_ref
    return {"repository": tracked.repository, "ref": tracked.ref, "primary": tracked.primary}


def _code_file(payload: Mapping) -> CodeFile | None:
    """Where a stored Passage of code was read, from its ``_of_code`` payload; its path is in
    the Document's id when the Passage has no lines of its own."""
    if payload.get("repository") is None:
        return None
    tracked = TrackedRef(payload["repository"], payload["ref"], payload.get("primary", True))
    _, _, local_id = payload["document_id"].partition(":")
    return CodeFile(tracked, payload.get("path") or CodeFile.split_id(local_id)[1])
