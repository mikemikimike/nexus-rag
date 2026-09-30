"""Index generations behind the Live index, and the cleanup of the ones no longer needed.

Each generation is its own Qdrant collection. The Live index is the generation the alias
``<name>.live`` points at; the one before it is kept behind ``<name>.previous`` so it can
be switched back to. Both aliases change in a single atomic request, so a search never
sees an in-between state.

Before the first Rebuild the Live index may be a plain collection named ``<name>``, as
written by earlier versions. It is served as it is, and adopted as a generation when a
Rebuild replaces it.

Every generation this module creates is recorded by name in a registry collection before
it exists, and cleanup deletes only recorded names that neither alias points at. Other
collections in the same Qdrant, however their names start, are never touched.
"""
import uuid

from qdrant_client import QdrantClient, models

_REGISTRY_POINT = 1


class NoPreviousIndex(RuntimeError):
    """There is no previous generation to switch back to."""


class Generations:
    def __init__(self, client: QdrantClient, name: str):
        self._client = client
        self._name = name
        self._live_alias = f"{name}.live"
        self._previous_alias = f"{name}.previous"
        self._registry = f"{name}.generations"

    def live(self) -> str | None:
        """The collection the Live index is, or None before anything was indexed."""
        live, _ = self._live_and_previous()
        return live

    def serving(self) -> str | None:
        """What to query for the Live index: the alias, which a promotion moves atomically."""
        aliases = self._aliases()
        return self._live_alias if self._live_alias in aliases else self._resolve(aliases)[0]

    def fresh_name(self) -> str:
        """A name for a new generation. It is recorded when the collection is created."""
        return f"{self._name}.gen-{uuid.uuid4().hex[:12]}"

    def record(self, generation: str) -> None:
        """Note that the system is about to create this generation, so cleanup may remove it."""
        recorded = self._recorded()
        if generation not in recorded:
            self._set_recorded([*recorded, generation])

    def promote(self, generation: str) -> None:
        """Make the generation the Live index, keep the one it replaces as previous."""
        aliases = self._aliases()
        live, previous = self._resolve(aliases)
        changes = []
        if live is not None:
            if live == self._name:
                self.record(live)  # the plain collection of earlier versions becomes cleanable
            if previous is not None:
                changes.append(_drop_alias(self._previous_alias))
            changes.append(_create_alias(self._previous_alias, live))
            if self._live_alias in aliases:
                changes.append(_drop_alias(self._live_alias))
        changes.append(_create_alias(self._live_alias, generation))
        self._client.update_collection_aliases(change_aliases_operations=changes)
        self.prune()

    def switch_back(self) -> None:
        """Swap the Live index and the previous generation."""
        live, previous = self._live_and_previous()
        if live is None or previous is None:
            raise NoPreviousIndex("there is no previous index to switch back to")
        self._client.update_collection_aliases(change_aliases_operations=[
            _drop_alias(self._live_alias), _drop_alias(self._previous_alias),
            _create_alias(self._live_alias, previous),
            _create_alias(self._previous_alias, live)])

    def discard(self, generation: str) -> None:
        """Delete a generation that was never promoted."""
        self._drop_collection(generation)
        self._set_recorded([n for n in self._recorded() if n != generation])

    def prune(self) -> None:
        """Delete recorded generations that are neither live nor previous."""
        keep = set(self._live_and_previous())
        recorded = self._recorded()
        for generation in recorded:
            if generation not in keep:
                self._drop_collection(generation)
        self._set_recorded([generation for generation in recorded if generation in keep])

    def _live_and_previous(self) -> tuple[str | None, str | None]:
        return self._resolve(self._aliases())

    def _resolve(self, aliases: dict[str, str]) -> tuple[str | None, str | None]:
        live = aliases.get(self._live_alias)
        if live is None and self._client.collection_exists(self._name):
            live = self._name
        return live, aliases.get(self._previous_alias)

    def _drop_collection(self, name: str) -> None:
        if self._client.collection_exists(name):
            self._client.delete_collection(name)

    def _aliases(self) -> dict[str, str]:
        return {a.alias_name: a.collection_name for a in self._client.get_aliases().aliases}

    def _recorded(self) -> list[str]:
        if not self._client.collection_exists(self._registry):
            return []
        points = self._client.retrieve(self._registry, ids=[_REGISTRY_POINT])
        return list(points[0].payload["generations"]) if points else []

    def _set_recorded(self, generations: list[str]) -> None:
        if not self._client.collection_exists(self._registry):
            self._client.create_collection(
                self._registry, vectors_config=models.VectorParams(
                    size=1, distance=models.Distance.COSINE))
        self._client.upsert(self._registry, points=[models.PointStruct(
            id=_REGISTRY_POINT, vector=[1.0], payload={"generations": generations})], wait=True)


def _create_alias(alias: str, collection: str) -> models.CreateAliasOperation:
    return models.CreateAliasOperation(
        create_alias=models.CreateAlias(collection_name=collection, alias_name=alias))


def _drop_alias(alias: str) -> models.DeleteAliasOperation:
    return models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias))
