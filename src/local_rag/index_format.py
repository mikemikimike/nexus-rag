"""What an index records about how it was built, and whether the running code can use it.

The format is written when a generation is created. After an upgrade or a settings change
the recorded format is compared with what the running code would write: an index that
differs but can still be searched is served read-only, one that cannot is refused. Neither
is ever rebuilt automatically.
"""
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from local_rag.languages import syntax_languages
from local_rag.settings import RetrievalMode, Settings
from local_rag.sparse import SparseParams

# Bump when the collection layout or payload shape changes in a way old code cannot read.
SCHEMA_VERSION = 1

# What the running code's keyword encoding does, recorded so a change shows as a Read-only index.
IDENTIFIER_SPLITTING = True

# How the running code splits code into Passages, recorded so a change shows as a Read-only
# index. A different splitting method changes it.
CODE_SPLITTING = "lines"


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
    embed_model: str | None  # None for a Keyword-only index, which has no embedding model
    dimension: int | None  # likewise
    # Not recorded (None) by an index built before Documents were split into Passages.
    chunk_tokens: int | None = None
    chunk_overlap: int | None = None
    # Indexes built before the Retrieval mode existed were all Hybrid.
    retrieval: RetrievalMode = RetrievalMode.HYBRID
    # Whether every Document records the Corpus that owns it and has an id namespaced by it.
    # Indexes built before named Corpora held one anonymous Corpus.
    corpus_ownership: bool = False
    # Whether keyword encoding adds the parts of camelCase, dotted and kebab-case identifiers.
    # Indexes built before it was recorded did not.
    identifier_splitting: bool = False
    # The size limit of a code Passage and the method that splits code; None when not recorded.
    code_chunk_tokens: int | None = None
    code_splitting: str | None = None
    # The languages whose code is split along its syntax; None when not recorded.
    syntax_languages: tuple[str, ...] | None = None

    @classmethod
    def written_by(cls, settings: Settings, dimension: int | None) -> "IndexFormat":
        """The format of an index the running code creates."""
        keyword = settings.retrieval is RetrievalMode.KEYWORD
        return cls(
            schema_version=SCHEMA_VERSION, sparse=settings.sparse,
            embed_model=None if keyword else settings.embed_model,
            dimension=None if keyword else dimension,
            chunk_tokens=settings.chunk_tokens, chunk_overlap=settings.chunk_overlap,
            retrieval=settings.retrieval, corpus_ownership=True,
            identifier_splitting=IDENTIFIER_SPLITTING,
            code_chunk_tokens=settings.code_chunk_tokens, code_splitting=CODE_SPLITTING,
            syntax_languages=syntax_languages())

    def to_payload(self) -> dict:
        return {"schema_version": self.schema_version, "retrieval": self.retrieval.value,
                "embed_model": self.embed_model,
                "dimension": self.dimension, "chunk_tokens": self.chunk_tokens,
                "chunk_overlap": self.chunk_overlap,
                "identifier_splitting": self.identifier_splitting,
                "code_chunk_tokens": self.code_chunk_tokens,
                "code_splitting": self.code_splitting,
                "syntax_languages": None if self.syntax_languages is None
                else list(self.syntax_languages),
                "corpus_ownership": self.corpus_ownership, "sparse": {
                    "k1": self.sparse.k1, "b": self.sparse.b,
                    "avg_doc_length": self.sparse.avg_doc_length}}

    @classmethod
    def from_payload(cls, payload: dict) -> "IndexFormat":
        """Raises ValueError if the payload is not a recorded format."""
        try:
            sparse = payload["sparse"]
            retrieval = RetrievalMode(payload.get("retrieval", RetrievalMode.HYBRID))
            embed_model, dimension = payload["embed_model"], payload["dimension"]
            if retrieval is RetrievalMode.HYBRID and (embed_model is None or dimension is None):
                raise ValueError("a Hybrid index records its embedding model and dimension")
            return cls(
                schema_version=int(payload["schema_version"]),
                sparse=SparseParams(k1=float(sparse["k1"]), b=float(sparse["b"]),
                                    avg_doc_length=float(sparse["avg_doc_length"])),
                embed_model=None if embed_model is None else str(embed_model),
                dimension=_optional_int(dimension),
                chunk_tokens=_optional_int(payload.get("chunk_tokens")),
                chunk_overlap=_optional_int(payload.get("chunk_overlap")),
                retrieval=retrieval,
                identifier_splitting=bool(payload.get("identifier_splitting", False)),
                code_chunk_tokens=_optional_int(payload.get("code_chunk_tokens")),
                code_splitting=None if payload.get("code_splitting") is None
                else str(payload["code_splitting"]),
                syntax_languages=None if payload.get("syntax_languages") is None
                else tuple(str(name) for name in payload["syntax_languages"]),
                corpus_ownership=bool(payload.get("corpus_ownership", False)))
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            raise ValueError(f"unreadable index format {payload!r}") from error


def _optional_int(value) -> int | None:
    return None if value is None else int(value)


@dataclass(frozen=True)
class Assessment:
    access: Access
    warnings: tuple[str, ...] = ()
    # Whether a search may use the index's dense vectors: both it and the running code are Hybrid.
    dense: bool = False
    # The Model gateway could not be reached to check the embedding dimension.
    embedding_unreachable: bool = False


_REBUILD_ADVICE = "run a Rebuild to bring it up to date (none is started automatically)"


def assess(recorded: IndexFormat | None, stored_dimension: int | None, settings: Settings,
           running_dimension: Callable[[], int | None] | None) -> Assessment:
    """Whether the running code can use an index, given the format it recorded.

    ``stored_dimension`` is the size of the index's dense vectors as the collection itself
    says, None if it has none (a Keyword-only index). ``recorded`` is None for an index
    written before formats were recorded.
    ``running_dimension`` asks the Model gateway for the size of the vectors it returns
    now (None if it cannot be reached); pass None instead to leave that unchecked.
    """
    if recorded is not None and recorded.schema_version != SCHEMA_VERSION:
        # A release that still reads an older schema would return READ_ONLY for it here.
        return Assessment(Access.REFUSED, (
            f"index schema version {recorded.schema_version} is not the version "
            f"{SCHEMA_VERSION} this code reads; {_REBUILD_ADVICE}",))
    stored = RetrievalMode.KEYWORD if stored_dimension is None else RetrievalMode.HYBRID
    both_hybrid = stored is RetrievalMode.HYBRID and settings.retrieval is RetrievalMode.HYBRID
    differences = []
    if recorded is None:
        differences.append("its format was not recorded (written by an earlier version)")
    else:
        if recorded.sparse != settings.sparse:
            differences.append(
                f"sparse encoder parameters differ: index built with {recorded.sparse}, "
                f"running with {settings.sparse}")
        if stored is not settings.retrieval:
            differences.append(
                f"retrieval mode differs: index built {stored.value}, "
                f"running {settings.retrieval.value}; searches use keywords alone")
        elif both_hybrid and recorded.embed_model != settings.embed_model:
            differences.append(
                f"embedding model differs: index built with {recorded.embed_model!r}, "
                f"running with {settings.embed_model!r}")
        if not recorded.corpus_ownership:
            differences.append(
                "its Documents do not record the Corpus that owns them (built before "
                "named Corpora)")
        if recorded.chunk_tokens is None or recorded.chunk_overlap is None:
            differences.append(
                "its chunk size and overlap were not recorded (built before Documents "
                "were split into Passages)")
        else:
            if recorded.chunk_tokens != settings.chunk_tokens:
                differences.append(
                    f"chunk size differs: index built with {recorded.chunk_tokens} tokens, "
                    f"running with {settings.chunk_tokens}")
            if recorded.chunk_overlap != settings.chunk_overlap:
                differences.append(
                    f"chunk overlap differs: index built with {recorded.chunk_overlap} "
                    f"tokens, running with {settings.chunk_overlap}")
    if recorded is not None:
        if recorded.code_chunk_tokens is None or recorded.code_splitting is None:
            differences.append(
                "its code size limit and splitting method were not recorded (built before "
                "code was split by lines)")
        else:
            if recorded.code_chunk_tokens != settings.code_chunk_tokens:
                differences.append(
                    f"code size limit differs: index built with {recorded.code_chunk_tokens} "
                    f"tokens, running with {settings.code_chunk_tokens}")
            if recorded.code_splitting != CODE_SPLITTING:
                differences.append(
                    f"code splitting differs: index built by {recorded.code_splitting!r}, "
                    f"running {CODE_SPLITTING!r}")
        if recorded.syntax_languages is None:
            differences.append(
                "the languages split along their syntax were not recorded (built before "
                "code was split by syntax)")
        elif recorded.syntax_languages != syntax_languages():
            differences.append(
                f"syntax-split languages differ: index built with "
                f"{list(recorded.syntax_languages)}, running {list(syntax_languages())}")
        if recorded.identifier_splitting != IDENTIFIER_SPLITTING:
            differences.append(
                "keyword search does not split identifiers into their parts in this index "
                "(built by an earlier version)")
    unchecked = []
    unreachable = False
    if running_dimension is not None and both_hybrid:
        # Whatever the model's name, vectors of another size cannot be compared with these.
        running = running_dimension()
        if running is None:
            unreachable = True
            unchecked.append("the embedding dimension could not be checked: "
                             "the Model gateway is unreachable")
        elif running != stored_dimension:
            return Assessment(Access.REFUSED, (
                *differences, f"embedding dimension differs: index has {stored_dimension}, "
                f"the Model gateway returns {running}, so queries cannot be compared with "
                f"it; {_REBUILD_ADVICE}"))
    if not differences:
        return Assessment(Access.READ_WRITE, tuple(unchecked), both_hybrid, unreachable)
    return Assessment(Access.READ_ONLY, (*differences, *unchecked, _REBUILD_ADVICE),
                      both_hybrid, unreachable)
