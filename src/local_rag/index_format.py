"""What an index records about how it was built, and whether the running code can use it.

The format is written when a generation is created. After an upgrade or a settings change
the recorded format is compared with what the running code would write: an index that
differs but can still be searched is served read-only, one that cannot is refused. Neither
is ever rebuilt automatically.
"""
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from local_rag.settings import Settings
from local_rag.sparse import SparseParams

# Bump when the collection layout or payload shape changes in a way old code cannot read.
SCHEMA_VERSION = 1


class Access(StrEnum):
    READ_WRITE = "read-write"
    READ_ONLY = "read-only"  # searchable, but a Sync would mix old and new formats
    REFUSED = "refused"  # not usable by the running code


class IncompatibleIndex(RuntimeError):
    """The index was built in a format the running code will not write to."""


class IndexReadOnly(IncompatibleIndex):
    """The index can still be searched, but a Sync would mix formats: Rebuild instead."""


class IndexRefused(IncompatibleIndex):
    """The index cannot be used by the running code at all: Rebuild."""


@dataclass(frozen=True)
class IndexFormat:
    schema_version: int
    sparse: SparseParams
    embed_model: str
    dimension: int

    @classmethod
    def written_by(cls, settings: Settings, dimension: int) -> "IndexFormat":
        """The format of an index the running code creates."""
        return cls(SCHEMA_VERSION, settings.sparse, settings.embed_model, dimension)

    def to_payload(self) -> dict:
        return {"schema_version": self.schema_version, "embed_model": self.embed_model,
                "dimension": self.dimension, "sparse": {
                    "k1": self.sparse.k1, "b": self.sparse.b,
                    "avg_doc_length": self.sparse.avg_doc_length}}

    @classmethod
    def from_payload(cls, payload: dict) -> "IndexFormat":
        """Raises ValueError if the payload is not a recorded format."""
        try:
            sparse = payload["sparse"]
            return cls(
                schema_version=int(payload["schema_version"]),
                sparse=SparseParams(k1=float(sparse["k1"]), b=float(sparse["b"]),
                                    avg_doc_length=float(sparse["avg_doc_length"])),
                embed_model=str(payload["embed_model"]), dimension=int(payload["dimension"]))
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            raise ValueError(f"unreadable index format {payload!r}") from error


@dataclass(frozen=True)
class Assessment:
    access: Access
    warnings: tuple[str, ...] = ()


_REBUILD_ADVICE = "run a Rebuild to bring it up to date (none is started automatically)"


def assess(recorded: IndexFormat | None, stored_dimension: int, settings: Settings,
           running_dimension: Callable[[], int | None]) -> Assessment:
    """Whether the running code can use an index, given the format it recorded.

    ``stored_dimension`` is the size of the index's dense vectors as the collection itself
    says. ``recorded`` is None for an index written before formats were recorded.
    ``running_dimension`` asks the Model gateway for the size of the vectors it returns
    now, or None if it cannot be reached.
    """
    if recorded is not None and recorded.schema_version != SCHEMA_VERSION:
        # A release that still reads an older schema would return READ_ONLY for it here.
        return Assessment(Access.REFUSED, (
            f"index schema version {recorded.schema_version} is not the version "
            f"{SCHEMA_VERSION} this code reads; {_REBUILD_ADVICE}",))
    differences = []
    if recorded is None:
        differences.append("its format was not recorded (written by an earlier version)")
    else:
        if recorded.sparse != settings.sparse:
            differences.append(
                f"sparse encoder parameters differ: index built with {recorded.sparse}, "
                f"running with {settings.sparse}")
        if recorded.embed_model != settings.embed_model:
            differences.append(
                f"embedding model differs: index built with {recorded.embed_model!r}, "
                f"running with {settings.embed_model!r}")
    if not differences:
        return Assessment(Access.READ_WRITE)
    if recorded is None or recorded.embed_model != settings.embed_model:
        # Another model may still return vectors of the same size, which can be searched.
        running = running_dimension()
        if running is None:
            differences.append("the embedding dimension could not be checked: "
                               "the Model gateway is unreachable")
        elif running != stored_dimension:
            return Assessment(Access.REFUSED, (
                *differences, f"embedding dimension differs: index has {stored_dimension}, "
                f"the Model gateway returns {running}, so queries cannot be compared with "
                f"it; {_REBUILD_ADVICE}"))
    return Assessment(Access.READ_ONLY, (*differences, _REBUILD_ADVICE))
