"""The result of the last Sync or Rebuild, kept in Qdrant so any process can report it."""

import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from qdrant_client import QdrantClient, models

_POINT = 1


@dataclass(frozen=True)
class LastSync:
    kind: str  # "Sync" or "Rebuild"
    outcome: str  # completed, completed with failures, promoted, refused, failed
    detail: str
    at: datetime
    corpus: str | None = None  # the Corpus a Sync was of; None for a Rebuild


class Journal:
    def __init__(self, client: QdrantClient, name: str):
        self._client = client
        self._name = name

    def record(self, kind: str, outcome: str, detail: str, corpus: str | None = None) -> None:
        """Keep the result as the last run, and as the last of ``corpus`` if it names one."""
        if not self._client.collection_exists(self._name):
            self._client.create_collection(
                self._name,
                vectors_config=models.VectorParams(size=1, distance=models.Distance.COSINE),
            )
        payload = {"kind": kind, "outcome": outcome, "detail": detail, "at": time.time()}
        points = [models.PointStruct(id=_POINT, vector=[1.0], payload=payload)]
        if corpus is not None:
            points.append(
                models.PointStruct(
                    id=_corpus_point(corpus), vector=[1.0], payload={**payload, "corpus": corpus}
                )
            )
        self._client.upsert(self._name, points=points, wait=True)

    def last(self) -> LastSync | None:
        """The last result; None if there is none. Raises ValueError if it is unreadable."""
        if not self._client.collection_exists(self._name):
            return None
        points = self._client.retrieve(self._name, ids=[_POINT])
        if not points:
            return None
        return _read(points[0].payload)

    def last_of_each_corpus(self) -> dict[str, LastSync]:
        """The last Sync of each Corpus that has had one. Raises ValueError if one is unreadable."""
        if not self._client.collection_exists(self._name):
            return {}
        found: dict[str, LastSync] = {}
        offset = None
        while True:
            points, offset = self._client.scroll(self._name, limit=256, offset=offset)
            for point in points:
                if point.id != _POINT:
                    found[point.payload.get("corpus")] = _read(point.payload)
            if offset is None:
                return found


def _corpus_point(corpus: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"last-sync-of-{corpus}"))


def _read(p: dict) -> LastSync:
    try:
        return LastSync(
            kind=p["kind"],
            outcome=p["outcome"],
            detail=p["detail"],
            at=datetime.fromtimestamp(p["at"], UTC),
            corpus=p.get("corpus"),
        )
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("the record of the last Sync is unreadable") from error
