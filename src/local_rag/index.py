import uuid
from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from local_rag.corpus import Corpus, Document
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
        )], wait=True)

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


def _connect(settings: Settings) -> QdrantClient:
    if settings.qdrant_location.startswith(("http://", "https://")):
        return QdrantClient(url=settings.qdrant_location,
                            api_key=settings.qdrant_api_key, timeout=60)
    return QdrantClient(location=settings.qdrant_location)
