"""Exclusive-run lock: only one holder of a given lock runs at a time.

The Knowledge index uses one lock per Corpus for its Syncs, and one index-wide lock for a
Rebuild (or switch-back), checking the other kind before it proceeds.

The lock lives in Qdrant, as a tiny marker collection, so that it also holds across
processes (overlapping timers) that share one Qdrant server. The holder refreshes a
heartbeat while it runs; a marker whose heartbeat has stopped (the holder crashed) is
taken over once it is older than ``stale_after`` seconds.

A marker is created before it is stamped, so its creator may die in between. Whoever first
finds a marker unstamped records that moment in the marker itself (a "witness" point), so
every run and process judges its age against the same start, and an abandoned marker is
taken over ``stale_after`` seconds after it was first seen. A creator that is merely slow
to stamp is therefore never taken over early. This uses only ordinary points, so it works
on any Qdrant server the project supports.
"""

import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from qdrant_client import QdrantClient, models

_HEARTBEAT_POINT = 1
_WITNESS_POINT = 2


class SyncInProgress(RuntimeError):
    """Another Sync of the same Corpus, or a Rebuild, already holds the Knowledge index."""


class LockLost(RuntimeError):
    """The holder stalled long enough for another run to take the lock over."""


class Lease:
    """Proof of holding the lock, to re-check before an irreversible step."""

    def __init__(self, run: "ExclusiveRun", owner: str):
        self._run = run
        self._owner = owner

    def check(self) -> None:
        if not self._run._owned_by(self._owner):
            raise LockLost("the lock was taken over by another run")


class ExclusiveRun:
    def __init__(
        self,
        client: QdrantClient,
        name: str,
        *,
        stale_after: float = 120.0,
        heartbeat_every: float | None = None,
        now: Callable[[], float] | None = None,
    ):
        """``now`` replaces the clock every age decision reads, for tests.

        By default the wall clock is read, which other processes share: heartbeats and
        witness points are stamped with it.
        """
        self._client = client
        self._name = name
        self._stale_after = stale_after
        self._heartbeat_every = stale_after / 4 if heartbeat_every is None else heartbeat_every
        self._clock = now or time.time
        self._local = threading.Lock()

    @contextmanager
    def hold(self, what: str = "Sync") -> Iterator[Lease]:
        """Hold the lock for the ``with`` body, or raise SyncInProgress at once.

        Yields a Lease whose ``check()`` raises LockLost if the lock was taken over.
        """
        owner = uuid.uuid4().hex
        if not self._local.acquire(blocking=False):
            raise SyncInProgress(f"cannot start {what}: another run holds the lock")
        try:
            if not self._acquire(owner):
                raise SyncInProgress(f"cannot start {what}: another run holds the lock")
            stop = threading.Event()
            beat = threading.Thread(
                target=self._beat, args=(owner, stop), daemon=True, name="exclusive-run"
            )
            beat.start()
            try:
                yield Lease(self, owner)
            finally:
                stop.set()
                beat.join()
                self._release(owner)
        finally:
            self._local.release()

    def is_held(self) -> bool:
        """Whether some run holds the lock now (a marker gone stale does not count)."""
        return self._client.collection_exists(self._name) and self._age() < self._stale_after

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
                self._name,
                vectors_config=models.VectorParams(size=1, distance=models.Distance.COSINE),
            )
        except Exception:
            if self._client.collection_exists(self._name):
                return False  # lost the race to another process
            raise
        return True

    def _stamp(self, owner: str) -> None:
        self._client.upsert(
            self._name,
            points=[
                models.PointStruct(
                    id=_HEARTBEAT_POINT,
                    vector=[1.0],
                    payload={"owner": owner, "heartbeat": self._clock()},
                )
            ],
            wait=True,
        )

    def _age(self) -> float:
        points = {
            point.id: point.payload or {}
            for point in self._client.retrieve(self._name, ids=[_HEARTBEAT_POINT, _WITNESS_POINT])
        }
        if _HEARTBEAT_POINT in points:
            return self._clock() - points[_HEARTBEAT_POINT]["heartbeat"]
        # Created but not yet stamped: the creator is starting up, or died in between.
        if _WITNESS_POINT in points:
            return self._clock() - points[_WITNESS_POINT]["first_seen"]
        try:
            self._witness()
        except Exception:
            if self._client.collection_exists(self._name):
                raise
            # Taken over and deleted meanwhile: report it held, the caller retries.
        return 0.0

    def _witness(self) -> None:
        self._client.upsert(
            self._name,
            points=[
                models.PointStruct(
                    id=_WITNESS_POINT, vector=[1.0], payload={"first_seen": self._clock()}
                )
            ],
            wait=True,
        )

    def _beat(self, owner: str, stop: threading.Event) -> None:
        while not stop.wait(self._heartbeat_every):
            try:
                if self._owned_by(owner):  # never stamp over a run that took the lock over
                    self._stamp(owner)
            except Exception:
                pass  # a missed beat only matters if the holder is truly gone

    def _owned_by(self, owner: str) -> bool:
        if not self._client.collection_exists(self._name):
            return False
        points = self._client.retrieve(self._name, ids=[_HEARTBEAT_POINT])
        return bool(points) and (points[0].payload or {})["owner"] == owner

    def _release(self, owner: str) -> None:
        try:
            if self._owned_by(owner):  # a holder that was taken over must not free the new one
                self._client.delete_collection(self._name)
        except Exception:
            pass  # a leftover marker goes stale and is taken over
