import logging
import time
import uuid
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace

from qdrant_client import QdrantClient, models

from local_rag.chunker import Chunker, Piece
from local_rag.corpus import (
    Corpus, Document, DocumentRef, RawDocument, check_corpus_name, listing_failures, qualified_id)
from local_rag.exclusive import ExclusiveRun, Lease, SyncInProgress
from local_rag.extraction import Extractor
from local_rag.gateway import ModelGateway
from local_rag.generations import CorruptIndex, Generations
from local_rag.index_format import (
    Access, Assessment, IndexFormat, IndexReadOnly, IndexRefused, assess)
from local_rag.journal import Journal, LastSync
from local_rag.ranking import cap_per_document, rerank
from local_rag.settings import RetrievalMode, Settings
from local_rag.sparse import SparseEncoder

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Passage:
    document_id: str
    title: str
    url: str
    space: str
    text: str
    score: float
    parent_id: str | None = None  # the page an Attachment belongs to
    corpus: str | None = None  # None for an index built before named Corpora
    # Set for a code Passage: where it is in its file.
    repository: str | None = None
    ref: str | None = None
    path: str | None = None
    start_line: int | None = None  # 1-based, inclusive
    end_line: int | None = None
    symbol: str | None = None  # the class or function it sits in, if the splitter knows

    @property
    def line_range(self) -> str | None:
        """``10-20``, or ``10`` for one line; None for a Passage that is not code."""
        if self.start_line is None:
            return None
        if self.end_line in (None, self.start_line):
            return str(self.start_line)
        return f"{self.start_line}-{self.end_line}"

    @property
    def link(self) -> str:
        """Where to read it: the Document's URL, at its lines for code (GitLab's anchor)."""
        return self.url if self.start_line is None else f"{self.url}#L{self.line_range}"


class SearchResults(list[Passage]):
    """The Passages a search found, best first, and a notice for the caller when the search
    could not use the embedding model (a Keyword fallback)."""

    def __init__(self, passages=(), notice: str | None = None):
        super().__init__(passages)
        self.notice = notice


_FALLBACK_NOTICE = ("Results are keyword-only: the embedding model could not be reached for "
                    "this search, so matches by meaning are missing.")
_NO_EMBEDDINGS_NOTICE = ("Results are keyword-only: this index was built without an embedding "
                         "model; an operator must run a Rebuild to add one.")


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

    def __init__(self, settings: Settings, gateway: ModelGateway | None, *,
                 client: QdrantClient | None = None):
        """``client`` lets several instances (processes, in effect) share one Qdrant.

        ``gateway`` may be None for Keyword-only retrieval without a reranker: nothing then
        reaches the Model gateway.
        """
        if gateway is None and settings.needs_model_gateway:
            raise ValueError("this Retrieval mode and these settings need a Model gateway")
        self._settings = settings
        self._gateway = gateway
        self._sparse = SparseEncoder(settings.sparse)
        self._extractor = Extractor.from_settings(settings)
        self._chunker = Chunker.from_settings(settings, gateway)
        self._client = client or _connect(settings)
        # By generation: a generation's format, once written, does not change.
        self._assessments: dict[str, Assessment] = {}
        # Why the embedding model failed the last time it was used; None if it worked.
        self._embedding_failure: str | None = None
        self._journal = Journal(self._client, f"{settings.collection}.last-sync")
        self._generations = Generations(self._client, settings.collection)
        # Held by a Rebuild or switch-back; a Sync holds only its own Corpus's lock.
        self._exclusive = self._lock(f"{settings.collection}.sync-lock")

    def _lock(self, name: str) -> ExclusiveRun:
        return ExclusiveRun(
            self._client, name, stale_after=self._settings.sync_lock_stale_after)

    @contextmanager
    def _syncing(self, corpus: str) -> Iterator[Lease]:
        """Hold one Corpus's Sync lock, unless a Rebuild holds the whole index."""
        prefix = f"{self._settings.collection}.sync-lock."
        with self._lock(prefix + corpus).hold("Sync") as lease:
            # The lock is taken before this check, and a Rebuild takes its lock before
            # checking for Syncs, so at least one of two racing runs sees the other.
            if self._exclusive.is_held():
                raise SyncInProgress("cannot start Sync: another run holds the lock")
            yield lease

    @contextmanager
    def _whole_index(self, what: str) -> Iterator[Lease]:
        """Hold the index-wide lock, unless some Corpus is being synced."""
        prefix = f"{self._settings.collection}.sync-lock."
        with self._exclusive.hold(what) as lease:
            for collection in self._client.get_collections().collections:
                if collection.name.startswith(prefix) and self._lock(collection.name).is_held():
                    raise SyncInProgress(
                        f"cannot start {what}: a Sync of "
                        f"{collection.name.removeprefix(prefix)} is running")
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
            return self._journaled(
                "Sync", lambda: self._sync(corpus, lease), corpus=corpus.name)

    def _sync(self, corpus: Corpus, lease: Lease) -> SyncReport:
        live = self._generations.live()
        if live is not None:
            self._require_writable(live)
        target = live or self._generations.fresh_name()
        indexed = self._indexed(live, corpus.name)
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
                document = self._extract(corpus, corpus.fetch(ref.id), ref.parts)
                if live is None:  # a Sync of another Corpus may have made an index live meanwhile
                    live = self._generations.live()
                    if live is not None:
                        self._discard(target)
                        self._require_writable(live)
                        target = live
                self._add(document, target)
            except Exception as error:
                failures[stored] = _describe(error)
            else:
                if live is None:  # the first Document makes a new index live
                    lease.check()
                    live = self._promote_first(target)
                    if live != target:  # another Corpus's Sync was first: join its index
                        self._discard(target)
                        self._require_writable(live)
                        target = live
                        self._add(document, target)
                if stored_as is None:
                    added += 1
                else:
                    replaced += 1
        current = {qualified_id(corpus.name, ref.id) for ref in refs}
        # A Document missing from an incomplete listing may still exist if it sat in a part
        # that failed to list; one with no recorded part may have sat in any of them.
        stale = {document_id for document_id in indexed.keys() - current
                 if not indexed[document_id].held_back_by(unlisted)}
        for document_id in stale:
            try:
                self._remove(document_id, target)
            except Exception as error:
                failures[document_id] = _describe(error)
            else:
                removed += 1
        return SyncReport(added=added, replaced=replaced, removed=removed,
                          unchanged=unchanged, failed=len(failures), failures=failures)

    def _promote_first(self, generation: str) -> str:
        """Make the generation live unless another Sync's first generation already is.

        Returns the Live index. The check and the promotion are one short critical section,
        so two first Syncs of different Corpora cannot both promote.
        """
        guard = self._lock(f"{self._settings.collection}.promote-lock")
        while True:
            try:
                with guard.hold("promotion"):
                    live = self._generations.live()
                    if live is None:
                        self._generations.promote(generation)
                        return generation
                    return live
            except SyncInProgress:
                time.sleep(0.01)

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
            unlisted = {f"{corpus.name}: {part}": why
                        for corpus, refs in listed for part, why in listing_failures(refs).items()}
            if unlisted:
                return RebuildReport(
                    indexed=0, failed=len(unlisted), failures=unlisted,
                    reason=f"a Corpus could not be listed completely "
                           f"({', '.join(unlisted)})", warnings=warnings)
            failures: dict[str, str] = {}
            total = 0
            for corpus, refs in listed:
                total += len(refs)
                for ref in refs:
                    try:
                        self._add(self._extract(corpus, corpus.fetch(ref.id), ref.parts),
                                  generation)
                    except Exception as error:
                        failures[qualified_id(corpus.name, ref.id)] = _describe(error)
            failed = len(failures)
            reason = self._refusal(total, failed)
            if reason is None:
                lease.check()
                self._generations.promote(generation)
                promoted = True
        finally:
            if not promoted:
                self._discard(generation)
        return RebuildReport(indexed=total - failed, failed=failed, reason=reason,
                             failures=failures, warnings=warnings)

    def forget(self, corpus: str) -> int:
        """Remove every Document the named Corpus owns; returns how many there were.

        This is the only way a Corpus's Documents leave the index apart from a Rebuild that
        is not given the Corpus: dropping a Corpus from the declaration removes nothing.
        Raises SyncInProgress at once if a Sync of this Corpus or a Rebuild is running.
        """
        check_corpus_name(corpus)
        with self._exclusive.hold("Forget"):
            if self._lock(f"{self._settings.collection}.sync-lock.{corpus}").is_held():
                raise SyncInProgress(f"cannot start Forget: a Sync of {corpus} is running")
            live = self._generations.live()
            if live is None:
                return 0
            self._require_writable(live)
            owned = models.Filter(must=[_match("corpus", corpus)])
            documents = self._client.count(
                live, exact=True, count_filter=models.Filter(
                    must=[_match("corpus", corpus), _match("kind", _DOCUMENT_KIND)])).count
            self._client.delete(
                live, wait=True, points_selector=models.FilterSelector(filter=owned))
            return documents

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
                raise IndexRefused(_explain(
                    "the previous index cannot be used", self._assessment(previous)))
            self._generations.switch_back()

    def _journaled[R: (SyncReport, RebuildReport)](
            self, kind: str, run: Callable[[], R], corpus: str | None = None) -> R:
        """Run a Sync or Rebuild and keep its result for ``status``."""
        try:
            report = run()
        except Exception as error:
            self._note(kind, "failed", _describe(error), corpus)
            raise
        if isinstance(report, SyncReport):
            outcome = "completed with failures" if report.failed else "completed"
            detail = (f"{report.added} added, {report.replaced} replaced, "
                      f"{report.removed} removed, {report.unchanged} unchanged, "
                      f"{report.failed} failed")
        elif report.promoted:
            outcome, detail = "promoted", f"{report.indexed} indexed, {report.failed} failed"
            detail = "; ".join([detail, *report.warnings])
        else:
            outcome, detail = "refused", report.reason
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
            return (f"{failed} of {total} Documents failed ({rate:.1%}), "
                    f"above the {threshold:.1%} threshold")
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

    def search(self, query: str, k: int | None = None,
               space: str | None = None) -> SearchResults:
        """The best Passages for the query. In Hybrid mode, if the embedding model cannot be
        reached the search is a Keyword fallback, and the results' ``notice`` says so."""
        settings = self._settings
        found = self._serving()
        if found is None:
            return SearchResults()
        serving, assessment = found
        limit = settings.top_k if k is None else k
        candidates = settings.hybrid_candidates
        in_space = models.Filter(must=[_match("space", space)]) if space is not None else None
        prefetch = []
        notice = None
        if assessment.dense:
            vector = None if assessment.embedding_unreachable else self._embed_query(query)
            if vector is None:
                notice = _FALLBACK_NOTICE
            else:
                prefetch.append(models.Prefetch(
                    query=vector, using="dense", limit=candidates, filter=in_space))
        elif settings.retrieval is RetrievalMode.HYBRID:
            notice = _NO_EMBEDDINGS_NOTICE
        sparse = self._sparse.encode_query(query)
        if sparse.indices:
            prefetch.append(models.Prefetch(
                query=models.SparseVector(indices=sparse.indices, values=sparse.values),
                using="bm25", limit=candidates, filter=in_space))
        if not prefetch:
            return SearchResults(notice=notice)
        points = self._client.query_points(
            serving, prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=max(limit, candidates), with_payload=True).points
        passages = [Passage(
            document_id=p.payload["document_id"], title=p.payload["title"],
            url=p.payload["url"], space=p.payload["space"], text=p.payload["text"],
            score=float(p.score), parent_id=p.payload.get("parent_id"),
            corpus=p.payload.get("corpus"), repository=p.payload.get("repository"),
            ref=p.payload.get("ref"), path=p.payload.get("path"),
            start_line=p.payload.get("start_line"), end_line=p.payload.get("end_line"),
            symbol=p.payload.get("symbol"))
            for p in points]
        passages = cap_per_document(passages, settings.max_passages_per_document)
        if settings.rerank:
            passages = rerank(self._gateway, query, passages)
        return SearchResults(passages[:limit], notice)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a Document's texts, trying again after a failure that could be transient."""
        settings = self._settings
        for attempt in range(1, settings.embed_attempts + 1):
            try:
                vectors = self._gateway.embed(texts)
            except Exception as error:
                self._embedding_failure = _describe(error)
                # Only an error that says another try could help is tried again: a bug, or a
                # wrong key, would just wait and fail the same way.
                if attempt == settings.embed_attempts or not getattr(error, "retryable", False):
                    raise
                wait = settings.embed_retry_backoff * 2 ** (attempt - 1)
                log.warning("embedding failed (attempt %d of %d), trying again in %.1fs: %s",
                            attempt, settings.embed_attempts, wait, self._embedding_failure)
                time.sleep(wait)
            else:
                self._embedding_failure = None
                return vectors
        raise AssertionError("embed_attempts is at least 1")

    def _embed_query(self, query: str) -> list[float] | None:
        """The query's vector, or None, noting why, if the embedding model cannot be reached."""
        try:
            vector = self._gateway.embed_query(query)
        except Exception as error:
            self._embedding_failure = _describe(error)
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
        undeclared = sorted(self._indexed_corpora(status.generation) - set(declared))
        return replace(status, warnings=(*status.warnings, *(
            f"Corpus {name!r} has Documents in the index but is undeclared; they stay "
            f"searchable until `forget {name}` removes them, or a Rebuild leaves them out"
            for name in undeclared)))

    def health(self) -> Health:
        """Whether Qdrant and the Knowledge index can be reached. Never raises.

        Asks Qdrant only, so a monitor polling it does not depend on the Model gateway.
        """
        try:
            self._client.get_collections()
        except Exception as error:
            return Health(False, False, (f"Qdrant is unreachable: {_describe(error)}",))
        try:
            status = self._status(verify_dimension=False)
        except Exception as error:
            return Health(
                True, False, (f"the Knowledge index cannot be read: {_describe(error)}",))
        notes = status.warnings
        if self._embedding_failure:
            notes = (*notes, f"the embedding model was unreachable the last time it was used "
                             f"({self._embedding_failure}); a hybrid search falls back to "
                             f"keywords until it answers")
        return Health(True, status.access is not Access.REFUSED, notes)

    def _status(self, *, verify_dimension: bool) -> IndexStatus:
        last_sync, corpora, unreadable = self._last_sync()
        try:
            live = self._generations.live()
            if live is None:
                return IndexStatus(
                    None, Access.READ_WRITE, 0, 0, None, unreadable, last_sync, corpora)
            assessment = self._assessment(live, verify_dimension=verify_dimension)
            documents = self._client.count(
                live, count_filter=models.Filter(must=[_match("kind", _DOCUMENT_KIND)]),
                exact=True).count
            total = self._client.count(live, exact=True).count
            return IndexStatus(
                generation=live, access=assessment.access, documents=documents,
                passages=total - documents, format=self._generations.format_of(live),
                warnings=(*assessment.warnings, *unreadable), last_sync=last_sync,
                corpora=corpora)
        except CorruptIndex as error:
            return IndexStatus(
                None, Access.REFUSED, None, None, None, (str(error), *unreadable), last_sync,
                corpora)

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

        params = self._client.get_collection(generation).config.params
        if not (isinstance(params.vectors, dict)
                and "bm25" in (params.sparse_vectors or {})):
            raise CorruptIndex(f"{generation} is not a Knowledge index collection")
        dense = params.vectors.get("dense")  # a Keyword-only index has none
        recorded = self._generations.format_of(generation)
        if recorded is not None and (recorded.retrieval is RetrievalMode.HYBRID) != bool(dense):
            raise CorruptIndex(
                f"{generation} records {recorded.retrieval.value} retrieval but its collection "
                f"{'has' if dense else 'has no'} dense vectors")
        assessment = assess(
            recorded, dense.size if dense else None,
            self._settings, running_dimension if verify_dimension else None)
        if verify_dimension and reachable:  # an unverified answer is worth asking for again
            self._assessments[generation] = assessment
            if assessment.access is not Access.READ_WRITE:
                log.warning("%s", _explain(
                    f"index {generation} is {assessment.access}", assessment))
        return assessment

    def _indexed(self, collection: str | None, corpus: str) -> dict[str, "_Stored"]:
        """What the index holds per Document id, among the Documents ``corpus`` owns.

        More than one version means an interrupted replace, so the Document is re-synced.
        """
        found: dict[str, _Stored] = {}
        if collection is None:
            return found
        offset = None
        while True:
            points, offset = self._client.scroll(
                collection, limit=1000, offset=offset, with_vectors=False,
                scroll_filter=models.Filter(must=[_match("corpus", corpus)]),
                with_payload=["document_id", "version", "parts"])
            for point in points:
                found.setdefault(point.payload["document_id"], _Stored()).add(point.payload)
            if offset is None:
                return found

    def _indexed_corpora(self, collection: str) -> set[str]:
        """The names of the Corpora that own at least one Document in ``collection``."""
        found: set[str] = set()
        offset = None
        while True:
            points, offset = self._client.scroll(
                collection, limit=1000, offset=offset, with_vectors=False,
                scroll_filter=models.Filter(must=[_match("kind", _DOCUMENT_KIND)]),
                with_payload=["corpus"])
            found.update(p.payload["corpus"] for p in points if p.payload.get("corpus"))
            if offset is None:
                return found

    def _extract(self, corpus: Corpus, raw: RawDocument,
                 parts: tuple[str, ...] = ()) -> Document:
        """The Document as the index stores it: owned by the Corpus, its ids namespaced."""
        return Document(
            id=qualified_id(corpus.name, raw.id), title=raw.title, url=raw.url,
            space=raw.space, version=raw.version, text=self._extractor.extract(raw),
            parent_id=None if raw.parent_id is None else qualified_id(corpus.name, raw.parent_id),
            media_type=raw.media_type, corpus=corpus.name, parts=parts,
            repository=raw.repository, ref=raw.ref, path=raw.path)

    def _add(self, document: Document, collection: str) -> None:
        # A Document with no text is still found by its title: it gets one empty Passage.
        pieces = self._chunker.pieces(document.text, document.media_type) or [Piece("")]
        contents = [f"{_context(document, piece)}\n\n{piece.text}" for piece in pieces]
        if self._settings.retrieval is RetrievalMode.HYBRID:
            dense = self._embed(contents)
            if len(dense) != len(contents):
                raise ValueError(
                    f"the Model gateway returned {len(dense)} vectors for {len(contents)} texts")
        else:
            dense = [None] * len(contents)
        self._ensure_collection(collection, None if dense[0] is None else len(dense[0]))
        document_point = _document_point(document)
        passages = []
        for number, (piece, content, vector) in enumerate(zip(pieces, contents, dense)):
            sparse = self._sparse.encode_document(content)
            vectors = {"bm25": models.SparseVector(indices=sparse.indices, values=sparse.values)}
            if vector is not None:
                vectors["dense"] = vector
            passages.append(models.PointStruct(
                id=_passage_id(document.id, number),
                vector=vectors,
                payload={"document_id": document.id, "version": document.version,
                         "title": document.title, "url": document.url,
                         "space": document.space, "text": piece.text,
                         "parent_id": document.parent_id, "corpus": document.corpus,
                         "parts": list(document.parts),
                         **_code_payload(document, piece)}))
        # Write the new Passages first, then delete this Document's leftover old ones (those
        # of an earlier, longer version included), so a failure in between never leaves the
        # Document missing from search.
        self._client.upsert(collection, points=[*passages, document_point], wait=True)
        self._client.delete(
            collection, wait=True,
            points_selector=models.FilterSelector(filter=models.Filter(
                must=_of_document(document.id),
                must_not=[models.HasIdCondition(
                    has_id=[*(p.id for p in passages), document_point.id])])))

    def _remove(self, document_id: str, collection: str) -> None:
        self._client.delete(
            collection, wait=True,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=_of_document(document_id))))

    def list_spaces(self) -> list[str]:
        """The Spaces that have at least one indexed Document, sorted."""
        serving = self._readable()
        if serving is None:
            return []
        spaces: set[str] = set()
        offset = None
        while True:
            records, offset = self._client.scroll(
                serving, limit=256, offset=offset, with_payload=["space"])
            spaces.update(r.payload["space"] for r in records)
            if offset is None:
                return sorted(spaces)

    def get_document(self, document_id: str) -> str | None:
        """The full text of an indexed Document, or None if it is not indexed."""
        serving = self._readable()
        if serving is None:
            return None
        records, _ = self._client.scroll(
            serving,
            scroll_filter=models.Filter(must=[
                _match("kind", _DOCUMENT_KIND), _match("document_id", document_id)]),
            limit=1, with_payload=["full_text"])
        return records[0].payload["full_text"] if records else None

    def _ensure_collection(self, collection: str, dimension: int | None) -> None:
        """``dimension`` is the size of the dense vectors, None for a Keyword-only index."""
        if self._client.collection_exists(collection):
            return
        # Before it exists, so a crash leaves no stray.
        self._generations.record(
            collection, IndexFormat.written_by(self._settings, dimension))
        self._client.create_collection(
            collection_name=collection,
            vectors_config={} if dimension is None else {"dense": models.VectorParams(
                size=dimension, distance=models.Distance.COSINE)},
            sparse_vectors_config={"bm25": models.SparseVectorParams(
                modifier=models.Modifier.IDF)},
        )
        if self._settings.qdrant_location != ":memory:":  # in-process Qdrant has no indexes
            self._client.create_payload_index(
                collection, "document_id", models.PayloadSchemaType.KEYWORD)
            self._client.create_payload_index(
                collection, "corpus", models.PayloadSchemaType.KEYWORD)


def _explain(what: str, assessment: Assessment) -> str:
    return f"{what}: {'; '.join(assessment.warnings)}"


def _describe(error: Exception) -> str:
    return str(error) or type(error).__name__


def _of_document(document_id: str) -> list[models.FieldCondition]:
    return [models.FieldCondition(
        key="document_id", match=models.MatchValue(value=document_id))]


# The full text of a Document lives on its own vectorless point, so it is stored once
# however many Passages the Document has and never appears in search results.
_DOCUMENT_KIND = "document"


def _match(key: str, value: str) -> models.FieldCondition:
    return models.FieldCondition(key=key, match=models.MatchValue(value=value))


def _passage_id(document_id: str, number: int) -> str:
    """The same Document always gets the same Passage ids, so a re-sync overwrites in place."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}#{number}"))


class _Stored:
    """What the index holds for one Document: its versions and the parts it was listed under."""

    def __init__(self) -> None:
        self.versions: set[str | None] = set()
        self.parts: set[str] = set()
        self.recorded = True  # False if any point predates recorded parts

    def add(self, payload: dict) -> None:
        self.versions.add(payload.get("version"))
        if payload.get("parts") is None:
            self.recorded = False
        else:
            self.parts.update(payload["parts"])

    def is_current(self, ref: DocumentRef) -> bool:
        """Unchanged: one stored version, the listed one, and the part it was listed under
        (a Document that moved to another part is re-added so its part stays true)."""
        return self.versions == {ref.version} and (
            not self.recorded or self.parts == set(ref.parts))

    def held_back_by(self, failures: Mapping[str, str]) -> bool:
        """True if a part this Document sat in could not be listed. With no part recorded,
        any failure might be its part."""
        if not failures:
            return False
        if not self.recorded or not self.parts:
            return True
        return not self.parts.isdisjoint(failures)


def _context(document: Document, piece: Piece) -> str:
    """What is put before a Passage's text when it is embedded and keyword-indexed: a
    Document's title, or for code its Repository, path and symbol."""
    if piece.start_line is None or not (document.repository or document.path):
        return document.title
    return "/".join(filter(None, (document.repository, document.path, piece.symbol)))


def _code_payload(document: Document, piece: Piece) -> dict:
    if piece.start_line is None:
        return {}
    return {"repository": document.repository, "ref": document.ref, "path": document.path,
            "start_line": piece.start_line, "end_line": piece.end_line, "symbol": piece.symbol}


def _document_point(document: Document) -> models.PointStruct:
    return models.PointStruct(
        id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document.id}#document")), vector={},
        payload={"kind": _DOCUMENT_KIND, "document_id": document.id,
                 "version": document.version, "title": document.title, "url": document.url,
                 "space": document.space, "full_text": document.text,
                 "parent_id": document.parent_id, "corpus": document.corpus,
                 "parts": list(document.parts)})


def _connect(settings: Settings) -> QdrantClient:
    location = settings.qdrant_location
    if location.startswith(("http://", "https://")):
        return QdrantClient(url=location, api_key=settings.qdrant_api_key, timeout=60)
    if location == ":memory:":
        return QdrantClient(location=location)
    raise ValueError(
        f"qdrant_location must be ':memory:' or an http(s) URL, got {location!r}")
