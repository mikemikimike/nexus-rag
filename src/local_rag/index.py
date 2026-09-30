import uuid
from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from local_rag.corpus import Corpus, Document
from local_rag.exclusive import ExclusiveRun
from local_rag.gateway import ModelGateway
from local_rag.settings import Settings
from local_rag.sparse import SparseEncoder


@dataclass(frozen=True)
class Passage:
    document_id: str
    title: str
    url: str
    space: str
    text: str
    score: float


@dataclass(frozen=True)
class SyncReport:
    added: int
    replaced: int
    removed: int
    unchanged: int
    failed: int


class KnowledgeIndex:
    """The searchable body of indexed Documents: accepts a Corpus, answers searches."""

    def __init__(self, settings: Settings, gateway: ModelGateway):
        self._settings = settings
        self._gateway = gateway
        self._sparse = SparseEncoder(settings.sparse)
        self._client = _connect(settings)
        self._exclusive = ExclusiveRun(
            self._client, f"{settings.collection}.sync-lock",
            stale_after=settings.sync_lock_stale_after)

    def sync(self, corpus: Corpus) -> SyncReport:
        """Bring the index into agreement with the Corpus.

        Raises SyncInProgress at once if another Sync or Rebuild is running.
        """
        with self._exclusive.hold("Sync"):
            indexed = self._indexed_versions()
            added = replaced = removed = unchanged = failed = 0
            refs = corpus.list()
            for ref in refs:
                versions = indexed.get(ref.id)
                if versions == {ref.version}:
                    unchanged += 1
                    continue
                try:
                    self._add(corpus.fetch(ref.id))
                except Exception:
                    failed += 1
                else:
                    if versions is None:
                        added += 1
                    else:
                        replaced += 1
            current = {ref.id for ref in refs}
            for document_id in indexed.keys() - current:
                try:
                    self._remove(document_id)
                except Exception:
                    failed += 1
                else:
                    removed += 1
            return SyncReport(added=added, replaced=replaced, removed=removed,
                              unchanged=unchanged, failed=failed)

    def search(self, query: str, k: int | None = None) -> list[Passage]:
        settings = self._settings
        if not self._client.collection_exists(settings.collection):
            return []
        limit = settings.top_k if k is None else k
        candidates = settings.hybrid_candidates
        prefetch = [models.Prefetch(
            query=self._gateway.embed_query(query), using="dense", limit=candidates)]
        sparse = self._sparse.encode_query(query)
        if sparse.indices:
            prefetch.append(models.Prefetch(
                query=models.SparseVector(indices=sparse.indices, values=sparse.values),
                using="bm25", limit=candidates))
        points = self._client.query_points(
            settings.collection, prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit, with_payload=True).points
        return [Passage(
            document_id=p.payload["document_id"], title=p.payload["title"],
            url=p.payload["url"], space=p.payload["space"], text=p.payload["text"],
            score=float(p.score)) for p in points]

    def _indexed_versions(self) -> dict[str, set[str]]:
        """Versions found in the index per Document id.

        More than one version means an interrupted replace, so the Document is re-synced.
        """
        name = self._settings.collection
        found: dict[str, set[str]] = {}
        if not self._client.collection_exists(name):
            return found
        offset = None
        while True:
            points, offset = self._client.scroll(
                name, limit=1000, offset=offset, with_vectors=False,
                with_payload=["document_id", "version"])
            for point in points:
                found.setdefault(point.payload["document_id"], set()).add(
                    point.payload.get("version"))
            if offset is None:
                return found

    def _add(self, document: Document) -> None:
        # One Passage per Document until the chunker lands.
        content = f"{document.title}\n\n{document.text}"
        [dense] = self._gateway.embed([content])
        sparse = self._sparse.encode_document(content)
        self._ensure_collection(len(dense))
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document.id}#0"))]
        # Write the new Passages first, then delete this Document's leftover old ones,
        # so a failure in between never leaves the Document missing from search.
        self._client.upsert(self._settings.collection, points=[models.PointStruct(
            id=ids[0],
            vector={"dense": dense, "bm25": models.SparseVector(
                indices=sparse.indices, values=sparse.values)},
            payload={"document_id": document.id, "version": document.version,
                     "title": document.title, "url": document.url,
                     "space": document.space, "text": document.text},
        )], wait=True)
        self._client.delete(
            self._settings.collection, wait=True,
            points_selector=models.FilterSelector(filter=models.Filter(
                must=_of_document(document.id),
                must_not=[models.HasIdCondition(has_id=ids)])))

    def _remove(self, document_id: str) -> None:
        self._client.delete(
            self._settings.collection, wait=True,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=_of_document(document_id))))

    def _ensure_collection(self, dimension: int) -> None:
        name = self._settings.collection
        if self._client.collection_exists(name):
            return
        self._client.create_collection(
            collection_name=name,
            vectors_config={"dense": models.VectorParams(
                size=dimension, distance=models.Distance.COSINE)},
            sparse_vectors_config={"bm25": models.SparseVectorParams(
                modifier=models.Modifier.IDF)},
        )
        if self._settings.qdrant_location != ":memory:":  # in-process Qdrant has no indexes
            self._client.create_payload_index(
                name, "document_id", models.PayloadSchemaType.KEYWORD)


def _of_document(document_id: str) -> list[models.FieldCondition]:
    return [models.FieldCondition(
        key="document_id", match=models.MatchValue(value=document_id))]


def _connect(settings: Settings) -> QdrantClient:
    location = settings.qdrant_location
    if location.startswith(("http://", "https://")):
        return QdrantClient(url=location, api_key=settings.qdrant_api_key, timeout=60)
    if location == ":memory:":
        return QdrantClient(location=location)
    raise ValueError(
        f"qdrant_location must be ':memory:' or an http(s) URL, got {location!r}")
