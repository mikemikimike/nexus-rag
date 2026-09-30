import uuid
from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from local_rag.corpus import Corpus, Document
from local_rag.gateway import ModelGateway
from local_rag.ranking import cap_per_document, rerank
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
    failed: int


class KnowledgeIndex:
    """The searchable body of indexed Documents: accepts a Corpus, answers searches."""

    def __init__(self, settings: Settings, gateway: ModelGateway):
        self._settings = settings
        self._gateway = gateway
        self._sparse = SparseEncoder(settings.sparse)
        self._client = _connect(settings)

    def sync(self, corpus: Corpus) -> SyncReport:
        added = failed = 0
        for ref in corpus.list():
            try:
                self._add(corpus.fetch(ref.id))
            except Exception:
                failed += 1
            else:
                added += 1
        return SyncReport(added=added, failed=failed)

    def search(self, query: str, k: int | None = None,
               space: str | None = None) -> list[Passage]:
        settings = self._settings
        if not self._client.collection_exists(settings.collection):
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
            settings.collection, prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=max(limit, candidates), with_payload=True).points
        passages = [Passage(
            document_id=p.payload["document_id"], title=p.payload["title"],
            url=p.payload["url"], space=p.payload["space"], text=p.payload["text"],
            score=float(p.score)) for p in points]
        passages = cap_per_document(passages, settings.max_passages_per_document)
        if settings.rerank:
            passages = rerank(self._gateway, query, passages)
        return passages[:limit]

    def _add(self, document: Document) -> None:
        # One Passage per Document until the chunker lands.
        content = f"{document.title}\n\n{document.text}"
        [dense] = self._gateway.embed([content])
        sparse = self._sparse.encode_document(content)
        self._ensure_collection(len(dense))
        self._client.upsert(self._settings.collection, points=[models.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document.id}#0")),
            vector={"dense": dense, "bm25": models.SparseVector(
                indices=sparse.indices, values=sparse.values)},
            payload={"document_id": document.id, "title": document.title,
                     "url": document.url, "space": document.space, "text": document.text},
        ), _document_point(document)], wait=True)

    def list_spaces(self) -> list[str]:
        """The Spaces that have at least one indexed Document, sorted."""
        if not self._client.collection_exists(self._settings.collection):
            return []
        spaces: set[str] = set()
        offset = None
        while True:
            records, offset = self._client.scroll(
                self._settings.collection, limit=256, offset=offset,
                with_payload=["space"])
            spaces.update(r.payload["space"] for r in records)
            if offset is None:
                return sorted(spaces)

    def get_document(self, document_id: str) -> str | None:
        """The full text of an indexed Document, or None if it is not indexed."""
        if not self._client.collection_exists(self._settings.collection):
            return None
        records, _ = self._client.scroll(
            self._settings.collection,
            scroll_filter=models.Filter(must=[
                _match("kind", _DOCUMENT_KIND), _match("document_id", document_id)]),
            limit=1, with_payload=["full_text"])
        return records[0].payload["full_text"] if records else None

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


# The full text of a Document lives on its own vectorless point, so it is stored once
# however many Passages the Document has and never appears in search results.
_DOCUMENT_KIND = "document"


def _match(key: str, value: str) -> models.FieldCondition:
    return models.FieldCondition(key=key, match=models.MatchValue(value=value))


def _document_point(document: Document) -> models.PointStruct:
    return models.PointStruct(
        id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document.id}#document")), vector={},
        payload={"kind": _DOCUMENT_KIND, "document_id": document.id,
                 "title": document.title, "url": document.url, "space": document.space,
                 "full_text": document.text})


def _connect(settings: Settings) -> QdrantClient:
    location = settings.qdrant_location
    if location.startswith(("http://", "https://")):
        return QdrantClient(url=location, api_key=settings.qdrant_api_key, timeout=60)
    if location == ":memory:":
        return QdrantClient(location=location)
    raise ValueError(
        f"qdrant_location must be ':memory:' or an http(s) URL, got {location!r}")
