import logging
import uuid
from dataclasses import dataclass, field

from qdrant_client import QdrantClient, models

from local_rag.corpus import Corpus, Document, RawDocument, listing_failures
from local_rag.exclusive import ExclusiveRun
from local_rag.extraction import Extractor
from local_rag.gateway import ModelGateway
from local_rag.generations import Generations
from local_rag.ranking import cap_per_document, rerank
from local_rag.settings import Settings
from local_rag.sparse import SparseEncoder

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Passage:
    document_id: str
    title: str
    url: str
    space: str
    text: str
    score: float
    parent_id: str | None = None  # the page an Attachment belongs to


@dataclass(frozen=True)
class RebuildReport:
    indexed: int
    failed: int
    reason: str | None  # why the new index was not promoted; None if it was
    failures: dict[str, str] = field(default_factory=dict)  # failed Document id -> why

    @property
    def promoted(self) -> bool:
        return self.reason is None


@dataclass(frozen=True)
class SyncReport:
    added: int
    replaced: int
    removed: int
    unchanged: int
    failed: int
    failures: dict[str, str] = field(default_factory=dict)  # failed Document id -> why


class KnowledgeIndex:
    """The searchable body of indexed Documents: accepts a Corpus, answers searches."""

    def __init__(self, settings: Settings, gateway: ModelGateway):
        self._settings = settings
        self._gateway = gateway
        self._sparse = SparseEncoder(settings.sparse)
        self._extractor = Extractor.from_settings(settings)
        self._client = _connect(settings)
        self._generations = Generations(self._client, settings.collection)
        self._exclusive = ExclusiveRun(
            self._client, f"{settings.collection}.sync-lock",
            stale_after=settings.sync_lock_stale_after)

    def sync(self, corpus: Corpus) -> SyncReport:
        """Bring the index into agreement with the Corpus.

        Raises SyncInProgress at once if another Sync or Rebuild is running.
        """
        with self._exclusive.hold("Sync") as lease:
            live = self._generations.live()
            target = live or self._generations.fresh_name()
            indexed = self._indexed_versions(live)
            added = replaced = removed = unchanged = 0
            refs = corpus.list()
            unlisted = listing_failures(refs)
            failures = dict(unlisted)
            for ref in refs:
                versions = indexed.get(ref.id)
                if versions == {ref.version}:
                    unchanged += 1
                    continue
                try:
                    self._add(self._extract(corpus.fetch(ref.id)), target)
                except Exception as error:
                    failures[ref.id] = _describe(error)
                else:
                    if live is None:  # the first Document makes a new index live
                        lease.check()
                        self._generations.promote(target)
                        live = target
                    if versions is None:
                        added += 1
                    else:
                        replaced += 1
            current = {ref.id for ref in refs}
            # A Document missing from an incomplete listing may still exist.
            stale = set() if unlisted else indexed.keys() - current
            for document_id in stale:
                try:
                    self._remove(document_id, target)
                except Exception as error:
                    failures[document_id] = _describe(error)
                else:
                    removed += 1
            return SyncReport(added=added, replaced=replaced, removed=removed,
                              unchanged=unchanged, failed=len(failures), failures=failures)

    def rebuild(self, corpus: Corpus) -> RebuildReport:
        """Re-embed every Document into a fresh index and make it live if it is complete enough.

        Searches keep reading the Live index until then. Raises SyncInProgress at once if
        another Sync or Rebuild is running.
        """
        with self._exclusive.hold("Rebuild") as lease:
            self._generations.prune()  # generations a crashed Rebuild left behind
            generation = self._generations.fresh_name()
            promoted = False
            try:
                refs = corpus.list()
                if unlisted := listing_failures(refs):
                    return RebuildReport(
                        indexed=0, failed=len(unlisted), failures=unlisted,
                        reason=f"the Corpus could not be listed completely "
                               f"({', '.join(unlisted)})")
                failures: dict[str, str] = {}
                for ref in refs:
                    try:
                        self._add(self._extract(corpus.fetch(ref.id)), generation)
                    except Exception as error:
                        failures[ref.id] = _describe(error)
                failed = len(failures)
                reason = self._refusal(len(refs), failed)
                if reason is None:
                    lease.check()
                    self._generations.promote(generation)
                    promoted = True
            finally:
                if not promoted:
                    self._discard(generation)
            return RebuildReport(indexed=len(refs) - failed, failed=failed, reason=reason,
                                 failures=failures)

    def switch_back(self) -> None:
        """Make the previous index the Live index again.

        Raises NoPreviousIndex if there is none, and SyncInProgress at once if a Sync or
        Rebuild is running.
        """
        with self._exclusive.hold("switch-back"):
            self._generations.switch_back()

    def _refusal(self, total: int, failed: int) -> str | None:
        """Why a Rebuild that indexed ``total - failed`` Documents must not go live."""
        # Whatever the threshold, an index with no Documents cannot be made live: its
        # collection is only created by the first Document.
        if total == 0:
            return "the Corpus has no Documents"
        if failed == total:
            return "no Document could be indexed"
        rate, threshold = failed / total, self._settings.rebuild_failure_threshold
        if rate > threshold:
            return (f"{failed} of {total} Documents failed ({rate:.1%}), "
                    f"above the {threshold:.1%} threshold")
        return None

    def _discard(self, generation: str) -> None:
        try:
            self._generations.discard(generation)
        except Exception:
            # Not worth masking the error that got us here; the next Rebuild sweeps it.
            log.warning("could not remove unpromoted index %s", generation, exc_info=True)

    def search(self, query: str, k: int | None = None,
               space: str | None = None) -> list[Passage]:
        settings = self._settings
        serving = self._generations.serving()
        if serving is None:
            return []
        limit = settings.top_k if k is None else k
        candidates = settings.hybrid_candidates
        in_space = models.Filter(must=[_match("space", space)]) if space is not None else None
        prefetch = [models.Prefetch(
            query=self._gateway.embed_query(query), using="dense", limit=candidates,
            filter=in_space)]
        sparse = self._sparse.encode_query(query)
        if sparse.indices:
            prefetch.append(models.Prefetch(
                query=models.SparseVector(indices=sparse.indices, values=sparse.values),
                using="bm25", limit=candidates, filter=in_space))
        points = self._client.query_points(
            serving, prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=max(limit, candidates), with_payload=True).points
        passages = [Passage(
            document_id=p.payload["document_id"], title=p.payload["title"],
            url=p.payload["url"], space=p.payload["space"], text=p.payload["text"],
            score=float(p.score), parent_id=p.payload.get("parent_id"))
            for p in points]
        passages = cap_per_document(passages, settings.max_passages_per_document)
        if settings.rerank:
            passages = rerank(self._gateway, query, passages)
        return passages[:limit]

    def _indexed_versions(self, collection: str | None) -> dict[str, set[str]]:
        """Versions found in the index per Document id.

        More than one version means an interrupted replace, so the Document is re-synced.
        """
        found: dict[str, set[str]] = {}
        if collection is None:
            return found
        offset = None
        while True:
            points, offset = self._client.scroll(
                collection, limit=1000, offset=offset, with_vectors=False,
                with_payload=["document_id", "version"])
            for point in points:
                found.setdefault(point.payload["document_id"], set()).add(
                    point.payload.get("version"))
            if offset is None:
                return found

    def _extract(self, raw: RawDocument) -> Document:
        return Document(
            id=raw.id, title=raw.title, url=raw.url, space=raw.space, version=raw.version,
            text=self._extractor.extract(raw), parent_id=raw.parent_id)

    def _add(self, document: Document, collection: str) -> None:
        # One Passage per Document until the chunker lands.
        content = f"{document.title}\n\n{document.text}"
        [dense] = self._gateway.embed([content])
        sparse = self._sparse.encode_document(content)
        self._ensure_collection(collection, len(dense))
        document_point = _document_point(document)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document.id}#0"))]
        # Write the new Passages first, then delete this Document's leftover old ones,
        # so a failure in between never leaves the Document missing from search.
        self._client.upsert(collection, points=[models.PointStruct(
            id=ids[0],
            vector={"dense": dense, "bm25": models.SparseVector(
                indices=sparse.indices, values=sparse.values)},
            payload={"document_id": document.id, "version": document.version,
                     "title": document.title, "url": document.url,
                     "space": document.space, "text": document.text,
                     "parent_id": document.parent_id},
        ), document_point], wait=True)
        self._client.delete(
            collection, wait=True,
            points_selector=models.FilterSelector(filter=models.Filter(
                must=_of_document(document.id),
                must_not=[models.HasIdCondition(has_id=[*ids, document_point.id])])))

    def _remove(self, document_id: str, collection: str) -> None:
        self._client.delete(
            collection, wait=True,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=_of_document(document_id))))

    def list_spaces(self) -> list[str]:
        """The Spaces that have at least one indexed Document, sorted."""
        serving = self._generations.serving()
        if serving is None:
            return []
        spaces: set[str] = set()
        offset = None
        while True:
            records, offset = self._client.scroll(
                serving, limit=256, offset=offset, with_payload=["space"])
            spaces.update(r.payload["space"] for r in records)
            if offset is None:
                return sorted(spaces)

    def get_document(self, document_id: str) -> str | None:
        """The full text of an indexed Document, or None if it is not indexed."""
        serving = self._generations.serving()
        if serving is None:
            return None
        records, _ = self._client.scroll(
            serving,
            scroll_filter=models.Filter(must=[
                _match("kind", _DOCUMENT_KIND), _match("document_id", document_id)]),
            limit=1, with_payload=["full_text"])
        return records[0].payload["full_text"] if records else None

    def _ensure_collection(self, collection: str, dimension: int) -> None:
        if self._client.collection_exists(collection):
            return
        self._generations.record(collection)  # before it exists, so a crash leaves no stray
        self._client.create_collection(
            collection_name=collection,
            vectors_config={"dense": models.VectorParams(
                size=dimension, distance=models.Distance.COSINE)},
            sparse_vectors_config={"bm25": models.SparseVectorParams(
                modifier=models.Modifier.IDF)},
        )
        if self._settings.qdrant_location != ":memory:":  # in-process Qdrant has no indexes
            self._client.create_payload_index(
                collection, "document_id", models.PayloadSchemaType.KEYWORD)


def _describe(error: Exception) -> str:
    return str(error) or type(error).__name__


def _of_document(document_id: str) -> list[models.FieldCondition]:
    return [models.FieldCondition(
        key="document_id", match=models.MatchValue(value=document_id))]


# The full text of a Document lives on its own vectorless point, so it is stored once
# however many Passages the Document has and never appears in search results.
_DOCUMENT_KIND = "document"


def _match(key: str, value: str) -> models.FieldCondition:
    return models.FieldCondition(key=key, match=models.MatchValue(value=value))


def _document_point(document: Document) -> models.PointStruct:
    return models.PointStruct(
        id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document.id}#document")), vector={},
        payload={"kind": _DOCUMENT_KIND, "document_id": document.id,
                 "version": document.version, "title": document.title, "url": document.url,
                 "space": document.space, "full_text": document.text,
                 "parent_id": document.parent_id})


def _connect(settings: Settings) -> QdrantClient:
    location = settings.qdrant_location
    if location.startswith(("http://", "https://")):
        return QdrantClient(url=location, api_key=settings.qdrant_api_key, timeout=60)
    if location == ":memory:":
        return QdrantClient(location=location)
    raise ValueError(
        f"qdrant_location must be ':memory:' or an http(s) URL, got {location!r}")
