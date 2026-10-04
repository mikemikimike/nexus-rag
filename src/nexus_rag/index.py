import logging
import time
from collections.abc import Callable, Collection, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace

from qdrant_client import QdrantClient

from nexus_rag.chunker import Chunker, Piece
from nexus_rag.corpus import (
    Corpus,
    Document,
    DocumentRef,
    RawDocument,
    check_corpus_name,
    listing_failures,
    qualified_id,
)
from nexus_rag.errors import describe
from nexus_rag.exclusive import ExclusiveRun, Lease, SyncInProgress
from nexus_rag.extraction import Extractor
from nexus_rag.gateway import ModelGateway
from nexus_rag.generations import CorruptIndex, Generations
from nexus_rag.index_format import (
    Access,
    Assessment,
    IndexFormat,
    IndexReadOnly,
    IndexRefused,
    assess,
)
from nexus_rag.index_names import IndexNames
from nexus_rag.journal import Journal, LastSync
from nexus_rag.passage_store import Narrowing, Passage, PassageStore, Written
from nexus_rag.ranking import cap_per_document, rerank
from nexus_rag.settings import RetrievalMode, Settings
from nexus_rag.sparse import SparseEncoder

log = logging.getLogger(__name__)


class SearchResults(list[Passage]):
    """The Passages a search found, best first, and a notice for the caller when the search
    could not use the embedding model (a Keyword fallback)."""

    def __init__(self, passages=(), notice: str | None = None):
        super().__init__(passages)
        self.notice = notice


_FALLBACK_NOTICE = (
    "Results are keyword-only: the embedding model could not be reached for "
    "this search, so matches by meaning are missing."
)
_NO_EMBEDDINGS_NOTICE = (
    "Results are keyword-only: this index was built without an embedding "
    "model; an operator must run a Rebuild to add one."
)


@dataclass(frozen=True)
class RebuildReport:
    indexed: int
    failed: int
    reason: str | None  # why the new index was not promoted; None if it was
    failures: dict[str, str] = field(default_factory=dict)  # failed Document id -> why
    warnings: tuple[str, ...] = ()  # damage to the index's records the Rebuild had to work around

    @property
    def promoted(self) -> bool:
        return self.reason is None


@dataclass(frozen=True)
class SyncReport:
    added: int
    replaced: int
    removed: int
    unchanged: int
    failed: int
    failures: dict[str, str] = field(default_factory=dict)  # failed Document id -> why


@dataclass(frozen=True)
class IndexStatus:
    generation: str | None  # the Live index; None before anything was indexed
    access: Access
    documents: int | None  # None when the index cannot be read
    passages: int | None
    format: IndexFormat | None  # as recorded when the generation was created
    warnings: tuple[str, ...]  # why the index is read-only or refused
    last_sync: LastSync | None  # None until a Sync or Rebuild has run
    corpora: dict[str, LastSync] = field(default_factory=dict)  # each Corpus's last Sync


@dataclass(frozen=True)
class Health:
    qdrant: bool  # the Qdrant server answers
    knowledge_index: bool  # the Live index, if any, can be searched
    notes: tuple[str, ...]  # why something is not fine, or what to be aware of

    @property
    def ok(self) -> bool:
        return self.qdrant and self.knowledge_index


class KnowledgeIndex:
    """The searchable body of indexed Documents: accepts a Corpus, answers searches."""

    def __init__(
        self,
        settings: Settings,
        gateway: ModelGateway | None,
        *,
        client: QdrantClient | None = None,
        lock_clock: Callable[[], float] | None = None,
    ):
        """``client`` lets several instances (processes, in effect) share one Qdrant.

        ``lock_clock`` replaces the clock the exclusive-run locks age by, for tests.

        ``gateway`` may be None for Keyword-only retrieval without a reranker: nothing then
        reaches the Model gateway.
        """
        if gateway is None and settings.needs_model_gateway:
            raise ValueError("this Retrieval mode and these settings need a Model gateway")
        self._settings = settings
        self._gateway = gateway
        self._lock_clock = lock_clock
        self._sparse = SparseEncoder(settings.sparse)
        self._extractor = Extractor.from_settings(settings)
        self._chunker = Chunker.from_settings(settings, gateway)
        self._client = client or _connect(settings)
        self._store = PassageStore(
            self._client, payload_indexes=settings.qdrant_location != ":memory:"
        )
        # By generation: a generation's format, once written, does not change.
        self._assessments: dict[str, Assessment] = {}
        # Why the embedding model failed the last time it was used; None if it worked.
        self._embedding_failure: str | None = None
        self._names = IndexNames(settings.collection)
        self._journal = Journal(self._client, self._names.last_sync)
        self._generations = Generations(
            self._client, self._names, self._lock(self._names.registry_lock)
        )
        # Held by a Rebuild or switch-back; a Sync holds only its own Corpus's lock.
        self._exclusive = self._lock(self._names.lock)

    def _lock(self, name: str) -> ExclusiveRun:
        return ExclusiveRun(
            self._client,
            name,
            stale_after=self._settings.sync_lock_stale_after,
            now=self._lock_clock,
        )

    @contextmanager
    def _syncing(self, corpus: str) -> Iterator[Lease]:
        """Hold one Corpus's Sync lock, unless a Rebuild holds the whole index."""
        with self._lock(self._names.corpus_lock(corpus)).hold("Sync") as lease:
            # The lock is taken before this check, and a Rebuild takes its lock before
            # checking for Syncs, so at least one of two racing runs sees the other.
            if self._exclusive.is_held():
                raise SyncInProgress("cannot start Sync: another run holds the lock")
            yield lease

    @contextmanager
    def _whole_index(self, what: str) -> Iterator[Lease]:
        """Hold the index-wide lock, unless some Corpus is being synced."""
        with self._exclusive.hold(what) as lease:
            for collection in self._client.get_collections().collections:
                corpus = self._names.corpus_of_lock(collection.name)
                if corpus is not None and self._lock(collection.name).is_held():
                    raise SyncInProgress(f"cannot start {what}: a Sync of {corpus} is running")
            yield lease

    def sync(self, corpus: Corpus) -> SyncReport:
        """Bring the index into agreement with the Corpus.

        Only the Documents the Corpus owns are compared, replaced or removed; every other
        Corpus's Documents are left alone. Document ids in the report are the ids the index
        stores (the Corpus's name, a colon, then the Corpus's own id).

        Raises SyncInProgress at once if a Sync of the same Corpus or a Rebuild is running.
        """
        check_corpus_name(corpus.name)
        with self._syncing(corpus.name) as lease:
            return self._journaled("Sync", lambda: self._sync(corpus, lease), corpus=corpus.name)

    def _sync(self, corpus: Corpus, lease: Lease) -> SyncReport:
        live = self._generations.live()
        if live is not None:
            self._require_writable(live)
        target = live or self._generations.fresh_name()
        indexed = {} if live is None else self._store.stored(live, corpus.name)
        added = replaced = removed = unchanged = 0
        refs = corpus.list()
        unlisted = listing_failures(refs)
        failures = dict(unlisted)
        for ref in refs:
            stored = qualified_id(corpus.name, ref.id)
            stored_as = indexed.get(stored)
            if stored_as is not None and stored_as.is_current(ref):
                unchanged += 1
                continue
            try:
                raw = corpus.fetch(ref.id)
                # A listing may not know the version (an unversioned Confluence page), but
                # what was fetched does: if that is what is stored, nothing changed.
                if stored_as is not None and stored_as.is_current(ref, raw.version):
                    unchanged += 1
                    continue
                document = self._extract(corpus, raw, ref)
                if live is None:  # a Sync of another Corpus may have made an index live meanwhile
                    live = self._generations.live()
                    if live is not None:
                        self._discard(target)
                        self._require_writable(live)
                        target = live
                self._add(document, target)
            except Exception as error:
                failures[stored] = describe(error)
            else:
                if live is None:  # the first Document makes a new index live
                    lease.check()
                    live = self._promote_first(target)
                    if live != target:  # another Corpus's Sync was first: join its index
                        self._discard(target)
                        self._require_writable(live)
                        target = live
                        try:
                            self._add(document, target)
                        except Exception as error:
                            failures[stored] = describe(error)
                            continue
                if stored_as is None:
                    added += 1
                else:
                    replaced += 1
        current = {qualified_id(corpus.name, ref.id) for ref in refs}
        # A Document missing from an incomplete listing may still exist if it sat in a part
        # that failed to list; one with no recorded part may have sat in any of them.
        stale = {
            document_id
            for document_id in indexed.keys() - current
            if not indexed[document_id].held_back_by(unlisted)
        }
        for document_id in stale:
            try:
                self._store.remove(target, document_id)
            except Exception as error:
                failures[document_id] = describe(error)
            else:
                removed += 1
        return SyncReport(
            added=added,
            replaced=replaced,
            removed=removed,
            unchanged=unchanged,
            failed=len(failures),
            failures=failures,
        )

    def _promote_first(self, generation: str) -> str:
        """Make the generation live unless another Sync's first generation already is.

        Returns the Live index. The check and the promotion are one short critical section,
        so two first Syncs of different Corpora cannot both promote. Nothing is pruned: the
        other first Syncs' generations are still being written, and each discards its own.
        """
        with self._lock(self._names.promote_lock).wait("the first promotion"):
            live = self._generations.live()
            if live is None:
                self._generations.promote(generation, prune=False)
                return generation
            return live

    def rebuild(self, corpora: Sequence[Corpus]) -> RebuildReport:
        """Re-embed every Document of every Corpus into a fresh index and make it live if
        complete enough. Pass all the declared Corpora: the fresh index holds only these.

        Searches keep reading the Live index until then. Raises SyncInProgress at once if
        another Sync or Rebuild is running.
        """
        with self._whole_index("Rebuild") as lease:
            return self._journaled("Rebuild", lambda: self._rebuild(corpora, lease))

    def _rebuild(self, corpora: Sequence[Corpus], lease: Lease) -> RebuildReport:
        names = [check_corpus_name(c.name) for c in corpora]
        if len(set(names)) != len(names):
            raise ValueError(f"Corpus names must be unique, got {', '.join(names)}")
        warnings = self._sweep()  # generations a crashed Rebuild left behind
        generation = self._generations.fresh_name()
        promoted = False
        try:
            listed = [(corpus, corpus.list()) for corpus in corpora]
            unlisted = {
                f"{corpus.name}: {part}": why
                for corpus, refs in listed
                for part, why in listing_failures(refs).items()
            }
            if unlisted:
                return RebuildReport(
                    indexed=0,
                    failed=len(unlisted),
                    failures=unlisted,
                    reason=f"a Corpus could not be listed completely ({', '.join(unlisted)})",
                    warnings=warnings,
                )
            failures: dict[str, str] = {}
            total = 0
            for corpus, refs in listed:
                total += len(refs)
                for ref in refs:
                    try:
                        self._add(self._extract(corpus, corpus.fetch(ref.id), ref), generation)
                    except Exception as error:
                        failures[qualified_id(corpus.name, ref.id)] = describe(error)
            failed = len(failures)
            reason = self._refusal(total, failed)
            if reason is None:
                lease.check()
                self._generations.promote(generation)
                promoted = True
        finally:
            if not promoted:
                self._discard(generation)
        return RebuildReport(
            indexed=total - failed,
            failed=failed,
            reason=reason,
            failures=failures,
            warnings=warnings,
        )

    def forget(self, corpus: str) -> int:
        """Remove every Document the named Corpus owns; returns how many there were.

        This is the only way a Corpus's Documents leave the index apart from a Rebuild that
        is not given the Corpus: dropping a Corpus from the declaration removes nothing.
        Raises SyncInProgress at once if a Sync of this Corpus or a Rebuild is running.
        """
        check_corpus_name(corpus)
        with self._exclusive.hold("Forget"):
            if self._lock(self._names.corpus_lock(corpus)).is_held():
                raise SyncInProgress(f"cannot start Forget: a Sync of {corpus} is running")
            live = self._generations.live()
            if live is None:
                return 0
            self._require_writable(live)
            return self._store.forget(live, corpus)

    def _sweep(self) -> tuple[str, ...]:
        """Clean up after earlier runs, and say if the records it relies on had to be replaced."""
        try:
            self._generations.validate_records()
            self._generations.prune()
        except CorruptIndex as error:
            # A Rebuild is how corrupt records get replaced.
            warning = f"{error}; started new records, leaving other generations in place"
            log.warning("%s", warning)
            self._generations.reset_records()
            return (warning,)
        return ()

    def switch_back(self) -> None:
        """Make the previous index the Live index again.

        Raises NoPreviousIndex if there is none, and SyncInProgress at once if a Sync or
        Rebuild is running.
        """
        with self._whole_index("switch-back"):
            previous = self._generations.previous()
            if previous is not None and self._assessment(previous).access is Access.REFUSED:
                raise IndexRefused(
                    _explain("the previous index cannot be used", self._assessment(previous))
                )
            self._generations.switch_back()

    def _journaled[R: (SyncReport, RebuildReport)](
        self, kind: str, run: Callable[[], R], corpus: str | None = None
    ) -> R:
        """Run a Sync or Rebuild and keep its result for ``status``."""
        try:
            report = run()
        except Exception as error:
            self._note(kind, "failed", describe(error), corpus)
            raise
        outcome, detail = _outcome(report)
        self._note(kind, outcome, detail, corpus)
        return report

    def _note(self, kind: str, outcome: str, detail: str, corpus: str | None = None) -> None:
        try:
            self._journal.record(kind, outcome, detail, corpus)
        except Exception:
            # The run itself is what matters; status just keeps showing the one before.
            log.warning("could not record the result of the %s", kind, exc_info=True)

    def _refusal(self, total: int, failed: int) -> str | None:
        """Why a Rebuild that indexed ``total - failed`` Documents must not go live."""
        # Whatever the threshold, an index with no Documents cannot be made live: its
        # collection is only created by the first Document.
        if total == 0:
            return "the Corpora have no Documents"
        if failed == total:
            return "no Document could be indexed"
        rate, threshold = failed / total, self._settings.rebuild_failure_threshold
        if rate > threshold:
            return (
                f"{failed} of {total} Documents failed ({rate:.1%}), "
                f"above the {threshold:.1%} threshold"
            )
        return None

    def _discard(self, generation: str) -> None:
        try:
            self._generations.discard(generation)
        except Exception:
            # Not worth masking the error that got us here; the next Rebuild sweeps it.
            log.warning("could not remove unpromoted index %s", generation, exc_info=True)

    @property
    def retrieval(self) -> RetrievalMode:
        """The Retrieval mode this code runs in (the Live index may have been built in another)."""
        return self._settings.retrieval

    def search(
        self,
        query: str,
        k: int | None = None,
        space: str | None = None,
        repository: str | None = None,
        ref: str | None = None,
    ) -> SearchResults:
        """The best Passages for the query. ``repository`` keeps only that Repository's code.
        With no ``ref``, code at a Tracked ref that is not its Repository's Primary ref is
        left out (other Corpora are unaffected); a ``ref`` keeps only Passages at that ref.
        In Hybrid mode, if the embedding model cannot be
        reached the search is a Keyword fallback, and the results' ``notice`` says so."""
        settings = self._settings
        found = self._serving()
        if found is None:
            return SearchResults()
        serving, assessment = found
        limit = settings.top_k if k is None else k
        candidates = settings.hybrid_candidates
        vector = notice = None
        if assessment.dense:
            vector = None if assessment.embedding_unreachable else self._embed_query(query)
            if vector is None:
                notice = _FALLBACK_NOTICE
        elif settings.retrieval is RetrievalMode.HYBRID:
            notice = _NO_EMBEDDINGS_NOTICE
        passages = self._store.search(
            serving,
            dense=vector,
            sparse=self._sparse.encode_query(query),
            narrowing=Narrowing(space, repository, ref),
            candidates=candidates,
            limit=max(limit, candidates),
        )
        passages = cap_per_document(passages, settings.max_passages_per_document)
        if settings.rerank:
            passages = rerank(self._model_gateway, query, passages)
        return SearchResults(passages[:limit], notice)

    @property
    def _model_gateway(self) -> ModelGateway:
        """The Model gateway, for work only settings that need one do (checked when made)."""
        assert self._gateway is not None
        return self._gateway

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a Document's texts, trying again after a failure that could be transient."""
        settings = self._settings
        for attempt in range(1, settings.embed_attempts + 1):
            try:
                vectors = self._model_gateway.embed(texts)
            except Exception as error:
                self._embedding_failure = describe(error)
                # Only an error that says another try could help is tried again: a bug, or a
                # wrong key, would just wait and fail the same way.
                if attempt == settings.embed_attempts or not getattr(error, "retryable", False):
                    raise
                wait = settings.embed_retry_backoff * 2 ** (attempt - 1)
                log.warning(
                    "embedding failed (attempt %d of %d), trying again in %.1fs: %s",
                    attempt,
                    settings.embed_attempts,
                    wait,
                    self._embedding_failure,
                )
                time.sleep(wait)
            else:
                self._embedding_failure = None
                return vectors
        raise AssertionError("embed_attempts is at least 1")

    def _embed_query(self, query: str) -> list[float] | None:
        """The query's vector, or None, noting why, if the embedding model cannot be reached."""
        try:
            vector = self._model_gateway.embed_query(query)
        except Exception as error:
            self._embedding_failure = describe(error)
            log.warning("the embedding model could not be reached: %s", self._embedding_failure)
            return None
        self._embedding_failure = None
        return vector

    def status(self, declared: Collection[str] | None = None) -> IndexStatus:
        """The Live index generation, its size and format, and whether it can be written to.

        Corrupt records are reported here rather than raised. Given the names of the declared
        Corpora, the warnings also name any Corpus that has Documents but is not among them.
        """
        status = self._status(verify_dimension=True)
        if declared is None or status.generation is None:
            return status
        undeclared = sorted(self._store.corpora(status.generation) - set(declared))
        return replace(
            status,
            warnings=(
                *status.warnings,
                *(
                    f"Corpus {name!r} has Documents in the index but is undeclared; they stay "
                    f"searchable until `forget {name}` removes them, or a Rebuild leaves them out"
                    for name in undeclared
                ),
            ),
        )

    def health(self) -> Health:
        """Whether Qdrant and the Knowledge index can be reached. Never raises.

        Asks Qdrant only, so a monitor polling it does not depend on the Model gateway.
        """
        try:
            self._client.get_collections()
        except Exception as error:
            return Health(False, False, (f"Qdrant is unreachable: {describe(error)}",))
        try:
            status = self._status(verify_dimension=False)
        except Exception as error:
            return Health(True, False, (f"the Knowledge index cannot be read: {describe(error)}",))
        notes = status.warnings
        if self._embedding_failure:
            notes = (
                *notes,
                f"the embedding model was unreachable the last time it was used "
                f"({self._embedding_failure}); a hybrid search falls back to "
                f"keywords until it answers",
            )
        return Health(True, status.access is not Access.REFUSED, notes)

    def _status(self, *, verify_dimension: bool) -> IndexStatus:
        last_sync, corpora, unreadable = self._last_sync()
        try:
            live = self._generations.live()
            if live is None:
                return IndexStatus(
                    None, Access.READ_WRITE, 0, 0, None, unreadable, last_sync, corpora
                )
            assessment = self._assessment(live, verify_dimension=verify_dimension)
            documents, passages = self._store.counts(live)
            return IndexStatus(
                generation=live,
                access=assessment.access,
                documents=documents,
                passages=passages,
                format=self._generations.format_of(live),
                warnings=(*assessment.warnings, *unreadable),
                last_sync=last_sync,
                corpora=corpora,
            )
        except CorruptIndex as error:
            return IndexStatus(
                None,
                Access.REFUSED,
                None,
                None,
                None,
                (str(error), *unreadable),
                last_sync,
                corpora,
            )

    def _last_sync(self) -> tuple[LastSync | None, dict[str, LastSync], tuple[str, ...]]:
        try:
            return self._journal.last(), self._journal.last_of_each_corpus(), ()
        except ValueError as error:
            # Only a report: the next Sync or Rebuild writes a readable record again.
            return None, {}, (str(error),)

    def _readable(self) -> str | None:
        """What to query for the Live index, or None if there is none; refuses an unusable one."""
        found = self._serving()
        return None if found is None else found[0]

    def _serving(self) -> tuple[str, Assessment] | None:
        """What to query for the Live index and how it may be searched, or None if there is none."""
        found = self._generations.serving()
        if found is None:
            return None
        query, generation = found
        assessment = self._assessment(generation)
        if assessment.access is Access.REFUSED:
            raise IndexRefused(_explain("the index cannot be used", assessment))
        return query, assessment

    def _require_writable(self, generation: str) -> None:
        assessment = self._assessment(generation)
        if assessment.access is Access.READ_ONLY:
            raise IndexReadOnly(_explain("the index is read-only", assessment))
        if assessment.access is Access.REFUSED:
            raise IndexRefused(_explain("the index cannot be used", assessment))

    def _assessment(self, generation: str, *, verify_dimension: bool = True) -> Assessment:
        cached = self._assessments.get(generation)
        if cached is not None:
            return cached
        reachable = True

        def running_dimension() -> int | None:
            nonlocal reachable
            vector = self._embed_query("dimension probe")
            if vector is None:
                reachable = False
                return None
            return len(vector)

        dimension = self._store.dense_size(generation)  # a Keyword-only index has none
        recorded = self._generations.format_of(generation)
        dense = dimension is not None
        if recorded is not None and (recorded.retrieval is RetrievalMode.HYBRID) != dense:
            raise CorruptIndex(
                f"{generation} records {recorded.retrieval.value} retrieval but its collection "
                f"{'has' if dense else 'has no'} dense vectors"
            )
        assessment = assess(
            recorded, dimension, self._settings, running_dimension if verify_dimension else None
        )
        if verify_dimension and reachable:  # an unverified answer is worth asking for again
            self._assessments[generation] = assessment
            if assessment.access is not Access.READ_WRITE:
                log.warning(
                    "%s", _explain(f"index {generation} is {assessment.access}", assessment)
                )
        return assessment

    def _extract(self, corpus: Corpus, raw: RawDocument, ref: DocumentRef) -> Document:
        """The Document as the index stores it: owned by the Corpus, its ids namespaced."""
        return Document(
            id=qualified_id(corpus.name, raw.id),
            title=raw.title,
            url=raw.url,
            space=raw.space,
            version=raw.version,
            text=self._extractor.extract(raw),
            parent_id=None if raw.parent_id is None else qualified_id(corpus.name, raw.parent_id),
            media_type=raw.media_type,
            corpus=corpus.name,
            parts=ref.parts,
            placement=ref.placement,
            code=raw.code,
        )

    def _add(self, document: Document, collection: str) -> None:
        # A Document with no text is still found by its title: it gets one empty Passage.
        pieces = self._chunker.pieces(document.text, document.media_type) or [Piece("")]
        contents = [f"{_context(document, piece)}\n\n{piece.text}" for piece in pieces]
        dense: Sequence[list[float] | None]
        if self._settings.retrieval is RetrievalMode.HYBRID:
            dense = self._embed(contents)
            if len(dense) != len(contents):
                raise ValueError(
                    f"the Model gateway returned {len(dense)} vectors for {len(contents)} texts"
                )
        else:
            dense = [None] * len(contents)
        self._ensure_collection(collection, None if dense[0] is None else len(dense[0]))
        self._store.write(
            collection,
            document,
            [
                Written(piece, self._sparse.encode_document(content), vector)
                for piece, content, vector in zip(pieces, contents, dense, strict=True)
            ],
        )

    def list_spaces(self) -> list[str]:
        """The named Spaces (Confluence Spaces, GitLab groups, folders) that have at least one indexed Document, sorted."""
        serving = self._readable()
        if serving is None:
            return []
        return self._store.spaces(serving)

    def get_document(self, document_id: str) -> str | None:
        """The full text of an indexed Document, or None if it is not indexed."""
        serving = self._readable()
        if serving is None:
            return None
        return self._store.document_text(serving, document_id)

    def _ensure_collection(self, collection: str, dimension: int | None) -> None:
        """``dimension`` is the size of the dense vectors, None for a Keyword-only index."""
        if self._store.exists(collection):
            return
        self._generations.create(
            collection,
            IndexFormat.written_by(self._settings, dimension),
            lambda: self._store.create(collection, dimension),
        )


def _explain(what: str, assessment: Assessment) -> str:
    return f"{what}: {'; '.join(assessment.warnings)}"


def _outcome(report: SyncReport | RebuildReport) -> tuple[str, str]:
    """The outcome and detail ``status`` shows for a finished Sync or Rebuild."""
    if isinstance(report, SyncReport):
        outcome = "completed with failures" if report.failed else "completed"
        return outcome, (
            f"{report.added} added, {report.replaced} replaced, "
            f"{report.removed} removed, {report.unchanged} unchanged, "
            f"{report.failed} failed"
        )
    if not report.promoted:
        return "refused", report.reason or ""
    detail = f"{report.indexed} indexed, {report.failed} failed"
    return "promoted", "; ".join([detail, *report.warnings])


def _context(document: Document, piece: Piece) -> str:
    """What is put before a Passage's text when it is embedded and keyword-indexed: a
    Document's title, or for code its Repository, path and symbol."""
    if piece.start_line is None or document.code is None:
        return document.title
    return "/".join(
        filter(None, (document.code.tracked_ref.repository, document.code.path, piece.symbol))
    )


def _connect(settings: Settings) -> QdrantClient:
    location = settings.qdrant_location
    if location.startswith(("http://", "https://")):
        return QdrantClient(url=location, api_key=settings.qdrant_api_key, timeout=60)
    if location == ":memory:":
        return QdrantClient(location=location)
    raise ValueError(f"qdrant_location must be ':memory:' or an http(s) URL, got {location!r}")
