"""The result of the last Sync or Rebuild, kept in Qdrant so any process can report it."""
import time
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


class Journal:
    def __init__(self, client: QdrantClient, name: str):
        self._client = client
        self._name = name

    def record(self, kind: str, outcome: str, detail: str) -> None:
        if not self._client.collection_exists(self._name):
            self._client.create_collection(
                self._name, vectors_config=models.VectorParams(
                    size=1, distance=models.Distance.COSINE))
        self._client.upsert(self._name, points=[models.PointStruct(
            id=_POINT, vector=[1.0],
            payload={"kind": kind, "outcome": outcome, "detail": detail,
                     "at": time.time()})], wait=True)

    def last(self) -> LastSync | None:
        """The last result; None if there is none. Raises ValueError if it is unreadable."""
        if not self._client.collection_exists(self._name):
            return None
        points = self._client.retrieve(self._name, ids=[_POINT])
        if not points:
            return None
        p = points[0].payload
        try:
            return LastSync(kind=p["kind"], outcome=p["outcome"], detail=p["detail"],
                            at=datetime.fromtimestamp(p["at"], UTC))
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError("the record of the last Sync is unreadable") from error
