"""Exclusive-run lock shared by Sync and Rebuild: only one of them runs at a time.

The lock lives in Qdrant, as a tiny marker collection, so that it also holds across
processes (overlapping timers) that share one Qdrant server. The holder refreshes a
heartbeat while it runs; a marker whose heartbeat has stopped (the holder crashed) is
taken over once it is older than ``stale_after`` seconds.
"""
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager

from qdrant_client import QdrantClient, models

_HEARTBEAT_POINT = 1


class SyncInProgress(RuntimeError):
    """Another Sync or Rebuild already holds the Knowledge index."""


class ExclusiveRun:
    def __init__(self, client: QdrantClient, name: str, *,
                 stale_after: float = 120.0, heartbeat_every: float | None = None):
        self._client = client
        self._name = name
        self._stale_after = stale_after
        self._heartbeat_every = stale_after / 4 if heartbeat_every is None else heartbeat_every
        self._local = threading.Lock()
        self._unstamped_since: float | None = None

    @contextmanager
    def hold(self, what: str = "Sync") -> Iterator[None]:
        """Hold the lock for the ``with`` body, or raise SyncInProgress at once."""
        owner = uuid.uuid4().hex
        if not self._local.acquire(blocking=False):
            raise SyncInProgress(f"another {what} is already running")
        try:
            if not self._acquire(owner):
                raise SyncInProgress(f"another {what} is already running")
            stop = threading.Event()
            beat = threading.Thread(
                target=self._beat, args=(owner, stop), daemon=True, name="exclusive-run")
            beat.start()
            try:
                yield
            finally:
                stop.set()
                beat.join()
                self._release()
        finally:
            self._local.release()

    def _acquire(self, owner: str) -> bool:
        if self._try_create():
            self._stamp(owner)
            return True
        if self._age() < self._stale_after:
            return False
        self._client.delete_collection(self._name)
        if not self._try_create():
            return False
        self._stamp(owner)
        return True

    def _try_create(self) -> bool:
        if self._client.collection_exists(self._name):
            return False
        try:
            self._client.create_collection(
                self._name, vectors_config=models.VectorParams(
                    size=1, distance=models.Distance.COSINE))
        except Exception:
            if self._client.collection_exists(self._name):
                return False  # lost the race to another process
            raise
        return True

    def _stamp(self, owner: str) -> None:
        self._client.upsert(self._name, points=[models.PointStruct(
            id=_HEARTBEAT_POINT, vector=[1.0],
            payload={"owner": owner, "heartbeat": time.time()})], wait=True)

    def _age(self) -> float:
        points = self._client.retrieve(self._name, ids=[_HEARTBEAT_POINT])
        if points:
            self._unstamped_since = None
            return time.time() - points[0].payload["heartbeat"]
        # Created but not yet stamped: the creator is starting up, or died in between.
        if self._unstamped_since is None:
            self._unstamped_since = time.monotonic()
        return time.monotonic() - self._unstamped_since

    def _beat(self, owner: str, stop: threading.Event) -> None:
        while not stop.wait(self._heartbeat_every):
            try:
                self._stamp(owner)
            except Exception:
                pass  # a missed beat only matters if the holder is truly gone

    def _release(self) -> None:
        try:
            self._client.delete_collection(self._name)
        except Exception:
            pass  # a leftover marker goes stale and is taken over
