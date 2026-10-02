"""Index generations behind the Live index, and the cleanup of the ones no longer needed.

Each generation is its own Qdrant collection. The Live index is the generation the alias
``<name>.live`` points at; the one before it is kept behind ``<name>.previous`` so it can
be switched back to. Both aliases change in a single atomic request, so a search never
sees an in-between state.

Before the first Rebuild the Live index may be a plain collection named ``<name>``, as
written by earlier versions. It is served as it is, and adopted as a generation when a
Rebuild replaces it.

Every generation this module creates is recorded by name in a registry collection before
it exists, together with its format, and cleanup deletes only recorded names that neither
alias points at. Other collections in the same Qdrant, however their names start, are
never touched. The registry also notes which generation was last made live, so that a
Live index lost from Qdrant is reported as corrupt rather than read as a new, empty one.
"""
from collections.abc import Callable
from dataclasses import dataclass, field

from qdrant_client import QdrantClient, models

from local_rag.index_format import IndexFormat
from local_rag.index_names import IndexNames

_REGISTRY_POINT = 1


class NoPreviousIndex(RuntimeError):
    """There is no previous generation to switch back to."""


class CorruptIndex(RuntimeError):
    """The index's own records disagree with Qdrant or cannot be read."""

    def __init__(self, what: str):
        super().__init__(f"{what}; run a Rebuild to replace it (none is started automatically)")


@dataclass
class _Registry:
    generations: list[str] = field(default_factory=list)  # every generation the system created
    formats: dict[str, dict] = field(default_factory=dict)  # generation -> recorded format
    live: str | None = None  # the generation last made live, to tell "lost" from "never had one"


class Generations:
    def __init__(self, client: QdrantClient, names: IndexNames):
        self._client = client
        self._name = names.plain
        self._live_alias = names.live
        self._previous_alias = names.previous
        self._registry = names.registry
        self._names = names

    def live(self) -> str | None:
        """The collection the Live index is, or None before anything was indexed."""
        live, _ = self._live_and_previous()
        return live

    def serving(self) -> tuple[str, str] | None:
        """What to query for the Live index, and the collection that is now.

        The first is the alias, which a promotion moves atomically; None before anything
        was indexed.
        """
        aliases = self._aliases()
        live, _ = self._resolve(aliases)
        if live is None:
            return None
        return (self._live_alias if self._live_alias in aliases else live), live

    def fresh_name(self) -> str:
        """A name for a new generation. It is recorded when the collection is created."""
        return self._names.fresh_generation()

    def record(self, generation: str, index_format: IndexFormat | None = None) -> None:
        """Note that the system is about to create this generation, so cleanup may remove it.

        The format is kept with the name, so it exists before the collection does.
        """
        registry = self._load()
        if generation not in registry.generations:
            registry.generations.append(generation)
        if index_format is not None:
            registry.formats[generation] = index_format.to_payload()
        self._save(registry)

    def create(self, generation: str, index_format: IndexFormat,
               make: Callable[[], None]) -> None:
        """Record the generation with its format, then ``make`` its collection. Recorded first,
        so a crash part way leaves nothing that cleanup does not know of."""
        self.record(generation, index_format)
        make()

    def format_of(self, generation: str) -> IndexFormat | None:
        """The format recorded for a generation; None if none was (an earlier version's).

        Raises CorruptIndex if the registry does not know the generation at all.
        """
        registry = self._load()
        self._require_known(registry, generation)
        payload = registry.formats.get(generation)
        if payload is None:
            return None
        try:
            return IndexFormat.from_payload(payload)
        except ValueError as error:
            raise CorruptIndex(f"the recorded format of {generation} is unreadable") from error

    def promote(self, generation: str) -> None:
        """Make the generation the Live index, keep the one it replaces as previous."""
        aliases = self._aliases()
        live, previous = self._resolve(aliases, strict=False)
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
        self._set_live(generation)
        self.prune()

    def switch_back(self) -> None:
        """Swap the Live index and the previous generation."""
        live, previous = self._live_and_previous()
        if live is None or previous is None:
            raise NoPreviousIndex("there is no previous index to switch back to")
        registry = self._load()  # before the swap: an unreadable record is reported untouched
        self._client.update_collection_aliases(change_aliases_operations=[
            _drop_alias(self._live_alias), _drop_alias(self._previous_alias),
            _create_alias(self._live_alias, previous),
            _create_alias(self._previous_alias, live)])
        registry.live = previous
        self._save(registry)

    def discard(self, generation: str) -> None:
        """Delete a generation that was never promoted."""
        self._drop_collection(generation)
        registry = self._load()
        registry.generations = [n for n in registry.generations if n != generation]
        registry.formats.pop(generation, None)
        self._save(registry)

    def prune(self) -> None:
        """Delete recorded generations that are neither live nor previous."""
        keep = set(self._resolve(self._aliases(), strict=False))
        registry = self._load()
        for generation in registry.generations:
            if generation not in keep:
                self._drop_collection(generation)
        registry.generations = [g for g in registry.generations if g in keep]
        registry.formats = {g: f for g, f in registry.formats.items() if g in keep}
        self._save(registry)

    def previous(self) -> str | None:
        """The collection kept for switching back to, if any."""
        _, previous = self._live_and_previous()
        return previous

    def validate_records(self) -> None:
        """Raise CorruptIndex unless the registry is readable and knows the live generations."""
        registry = self._load()
        for generation in self._resolve(self._aliases(), strict=False):
            if generation is not None:
                self._require_known(registry, generation)

    def _require_known(self, registry: _Registry, generation: str) -> None:
        # A plain collection is an earlier version's and has no registry; any other
        # generation was recorded by this module before it was created.
        if generation != self._name and generation not in registry.generations:
            raise CorruptIndex(f"{generation} is missing from the record of index generations")

    def reset_records(self) -> None:
        """Start a fresh registry when the old one cannot be used, for a Rebuild to repair.

        Only the generations the aliases point at are adopted into it, without a format;
        any others it named can no longer be told from collections that are not ours, so
        they are left in place.
        """
        if self._client.collection_exists(self._registry):
            self._client.delete_collection(self._registry)
        for generation in self._resolve(self._aliases(), strict=False):
            if generation is not None and generation != self._name:
                self.record(generation)

    def _live_and_previous(self) -> tuple[str | None, str | None]:
        return self._resolve(self._aliases())

    def _resolve(self, aliases: dict[str, str],
                 strict: bool = True) -> tuple[str | None, str | None]:
        """The live and previous collections. Strict also reports a Live index that was lost."""
        live = aliases.get(self._live_alias)
        if live is None and self._client.collection_exists(self._name):
            live = self._name
        previous = aliases.get(self._previous_alias)
        if strict and live is None:
            # Deleting a collection takes its aliases with it, so a lost Live index
            # leaves the previous alias, or the registry's note of it, behind.
            lost = "a previous index exists" if previous is not None else None
            if lost is None and (recorded := self._load().live) is not None:
                lost = f"{recorded} was the Live index"
            if lost is not None:
                raise CorruptIndex(f"there is no Live index although {lost}")
        return live, previous

    def _drop_collection(self, name: str) -> None:
        if self._client.collection_exists(name):
            self._client.delete_collection(name)

    def _aliases(self) -> dict[str, str]:
        return {a.alias_name: a.collection_name for a in self._client.get_aliases().aliases}

    def _set_live(self, generation: str) -> None:
        registry = self._load()
        registry.live = generation
        self._save(registry)

    def _load(self) -> _Registry:
        if not self._client.collection_exists(self._registry):
            return _Registry()
        points = self._client.retrieve(self._registry, ids=[_REGISTRY_POINT])
        if not points:
            return _Registry()
        payload = points[0].payload
        generations = payload.get("generations")
        # Registries written before formats were recorded only have the names.
        formats, live = payload.get("formats", {}), payload.get("live")
        if not (isinstance(generations, list) and all(isinstance(g, str) for g in generations)
                and isinstance(formats, dict) and isinstance(live, str | None)):
            raise CorruptIndex("the record of index generations is unreadable")
        return _Registry(generations, formats, live)

    def _save(self, registry: _Registry) -> None:
        if not self._client.collection_exists(self._registry):
            self._client.create_collection(
                self._registry, vectors_config=models.VectorParams(
                    size=1, distance=models.Distance.COSINE))
        self._client.upsert(self._registry, points=[models.PointStruct(
            id=_REGISTRY_POINT, vector=[1.0],
            payload={"generations": registry.generations, "formats": registry.formats,
                     "live": registry.live})], wait=True)


def _create_alias(alias: str, collection: str) -> models.CreateAliasOperation:
    return models.CreateAliasOperation(
        create_alias=models.CreateAlias(collection_name=collection, alias_name=alias))


def _drop_alias(alias: str) -> models.DeleteAliasOperation:
    return models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias))
