"""Index generations behind the Live index, and the cleanup of the ones no longer needed.

Each generation is its own Qdrant collection. The Live index is the generation the alias
``<name>`` points at; the one before it is kept behind the alias ``<name>.previous`` so
it can be switched back to. Both aliases change in a single atomic request, so a search
never sees an in-between state.

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
        self._live_alias = name
        self._previous_alias = f"{name}.previous"
        self._registry = f"{name}.generations"

    def live(self) -> str | None:
        """The collection the Live index is, or None before anything was indexed."""
        return self._aliases().get(self._live_alias)

    def new(self) -> str:
        """Name a fresh generation and record it. The caller creates the collection."""
        name = f"{self._live_alias}.gen-{uuid.uuid4().hex[:12]}"
        self._record([*self._created(), name])
        return name

    def promote(self, generation: str) -> None:
        """Make the generation the Live index, keep the one it replaces as previous."""
        aliases = self._aliases()
        live, previous = aliases.get(self._live_alias), aliases.get(self._previous_alias)
        changes = []
        if live is not None:
            if previous is not None:
                changes.append(_delete(self._previous_alias))
            changes += [_delete(self._live_alias), _create(self._previous_alias, live)]
        changes.append(_create(self._live_alias, generation))
        self._client.update_collection_aliases(change_aliases_operations=changes)
        self._prune()

    def switch_back(self) -> None:
        """Swap the Live index and the previous generation."""
        aliases = self._aliases()
        live, previous = aliases.get(self._live_alias), aliases.get(self._previous_alias)
        if live is None or previous is None:
            raise NoPreviousIndex("there is no previous index to switch back to")
        self._client.update_collection_aliases(change_aliases_operations=[
            _delete(self._live_alias), _delete(self._previous_alias),
            _create(self._live_alias, previous), _create(self._previous_alias, live)])

    def discard(self, generation: str) -> None:
        """Delete a generation that was never promoted."""
        self._delete(generation)
        self._record([n for n in self._created() if n != generation])

    def _prune(self) -> None:
        aliases = self._aliases()
        keep = {aliases.get(self._live_alias), aliases.get(self._previous_alias)}
        created = self._created()
        for name in created:
            if name not in keep:
                self._delete(name)
        self._record([name for name in created if name in keep])

    def _delete(self, name: str) -> None:
        if self._client.collection_exists(name):
            self._client.delete_collection(name)

    def _aliases(self) -> dict[str, str]:
        return {a.alias_name: a.collection_name for a in self._client.get_aliases().aliases}

    def _created(self) -> list[str]:
        if not self._client.collection_exists(self._registry):
            return []
        points = self._client.retrieve(self._registry, ids=[_REGISTRY_POINT])
        return list(points[0].payload["generations"]) if points else []

    def _record(self, generations: list[str]) -> None:
        if not self._client.collection_exists(self._registry):
            self._client.create_collection(
                self._registry, vectors_config=models.VectorParams(
                    size=1, distance=models.Distance.COSINE))
        self._client.upsert(self._registry, points=[models.PointStruct(
            id=_REGISTRY_POINT, vector=[1.0], payload={"generations": generations})], wait=True)


def _create(alias: str, collection: str) -> models.CreateAliasOperation:
    return models.CreateAliasOperation(
        create_alias=models.CreateAlias(collection_name=collection, alias_name=alias))


def _delete(alias: str) -> models.DeleteAliasOperation:
    return models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias))
