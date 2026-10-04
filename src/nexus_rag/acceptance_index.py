"""The acceptance checks of the Knowledge index: a round trip on a throwaway collection.

The suite makes its own collection and a temp folder Corpus of generated Documents, Syncs
them, searches, changes and removes a Document, runs a Rebuild and a switch-back, and then
removes everything it made (the cleanup runs even when a check failed, see ``acceptance``).
Nothing here names the live collection.
"""

import shutil
import tempfile
from pathlib import Path

from nexus_rag import acceptance_corpus as corpus_files
from nexus_rag.acceptance_corpus import KEYWORD_DOCUMENT, LONG_DOCUMENT, SEMANTIC_DOCUMENT
from nexus_rag.acceptance_report import Context, is_keyword_only
from nexus_rag.checks import KEYWORD_ONLY, Failure, Skipped
from nexus_rag.chunker import Chunker
from nexus_rag.corpus import qualified_id
from nexus_rag.folder import FolderCorpus
from nexus_rag.index import KnowledgeIndex
from nexus_rag.index_format import Access
from nexus_rag.index_names import IndexNames
from nexus_rag.tokens import count_tokens

SEARCH_K = 3  # few enough that a Document is found by being near the query, not by being one of few
NEEDS = ["gateway.embeddings", "qdrant.reachable"]


class _RoundTrip:
    """The throwaway index, its folder and the texts the Documents currently have."""

    def __init__(self, ctx: Context):
        self.ctx, self.settings, self.run_id = ctx, ctx.settings, ctx.run_id
        self.client = ctx.connect(self.settings.qdrant_location, self.settings.qdrant_api_key)
        self.folder = Path(tempfile.mkdtemp(prefix="nexus-rag-acceptance-"))
        ctx.cleanups.append(("folder", self._remove_folder))
        ctx.cleanups.append(("collections", self._remove_collections))
        self.index = KnowledgeIndex(self.settings, ctx.gateway, client=self.client)
        self.corpus = FolderCorpus(self.folder)
        self.chunker = Chunker.from_settings(self.settings, ctx.gateway)
        self.texts: dict[str, str] = {}
        self.generation: str | None = None

    def _remove_folder(self) -> str:
        shutil.rmtree(self.folder)
        return f"removed {self.folder}"

    def _remove_collections(self) -> str:
        name = self.settings.collection
        names = IndexNames(name)
        ours = [c.name for c in self.client.get_collections().collections if names.owns(c.name)]
        for collection in ours:  # an alias goes with its collection
            self.client.delete_collection(collection)
        left = [c.name for c in self.client.get_collections().collections if names.owns(c.name)]
        if left:
            raise Failure(
                f"could not remove {', '.join(left)} from Qdrant",
                cause="the Qdrant key may not allow deleting collections; remove them "
                "by hand, they are named after the throwaway collection "
                f"{name} and hold no live data",
                see="section 9 (Operate)",
            )
        return f"removed {len(ours)} collections of {name} from Qdrant"

    def search(self, query: str, k: int = SEARCH_K):
        return self.index.search(query, k=k)

    def stored_id(self, document_id: str) -> str:
        return qualified_id(self.corpus.name, document_id)

    def found(self, query: str, document_id: str) -> bool:
        stored = self.stored_id(document_id)
        return any(p.document_id == stored for p in self.search(query))

    def mentions(self, query: str, term: str) -> bool:
        """Whether any Passage the search returns holds the term (the dense side always
        returns the nearest Passages, so a term's absence is only seen in their text)."""
        return any(term in p.text for p in self.search(query))

    def need(
        self,
        query: str,
        document_id: str,
        what: str,
        cause: str,
        *,
        containing: str | None = None,
        k: int = SEARCH_K,
    ) -> None:
        passages = self.search(query, k)
        stored = self.stored_id(document_id)
        if not any(
            p.document_id == stored and (containing is None or containing in p.text)
            for p in passages
        ):
            seen = ", ".join(sorted({p.document_id for p in passages})) or "nothing"
            raise Failure(
                f"searching for {what} found {seen}, not {document_id}",
                cause=cause,
                see="section 10",
            )

    def expected_passages(self) -> int:
        return sum(len(self.chunker.split(text)) or 1 for text in self.texts.values())

    def sync(self):
        return self.index.sync(self.corpus)


def run_index(ctx: Context) -> None:
    check = ctx.report.check
    trip: list[_RoundTrip] = []

    def need_trip() -> _RoundTrip:
        return trip[0]

    def sync_first() -> str:
        trip.append(_RoundTrip(ctx))
        return _sync(trip[0])

    # Keyword-only retrieval has no embedding model to have checked first.
    check("index.sync", sync_first, needs=["qdrant.reachable"] if is_keyword_only(ctx) else NEEDS)
    after = ["index.sync"]
    check("index.keyword", lambda: _keyword(need_trip()), needs=after)
    check("index.dense", lambda: _dense(need_trip()), needs=after)
    check("index.tail-keyword", lambda: _tail_keyword(need_trip()), needs=after)
    check("index.tail-meaning", lambda: _tail_meaning(need_trip()), needs=after)
    check("index.status", lambda: _status(need_trip()), needs=after)
    check("index.replace", lambda: _replace(need_trip()), needs=after)
    check("index.remove", lambda: _remove(need_trip()), needs=after)
    check("index.rebuild", lambda: _rebuild(need_trip()), needs=after)
    check("index.switch-back", lambda: _switch_back(need_trip()), needs=["index.rebuild"])


def _sync(trip: _RoundTrip) -> str:
    trip.texts = corpus_files.write_corpus(
        trip.folder,
        trip.run_id,
        chunk_tokens=trip.settings.chunk_tokens,
        count_tokens=trip.ctx.gateway.count_tokens if trip.ctx.gateway else count_tokens,
        split=trip.chunker.split,
    )
    report = trip.sync()
    if report.failed or report.added != len(trip.texts):
        failures = "; ".join(f"{name}: {why}" for name, why in report.failures.items())
        raise Failure(
            f"Sync of {len(trip.texts)} generated Documents: {report.added} added, "
            f"{report.failed} failed" + (f" ({failures})" if failures else ""),
            cause="the first failure's reason names it: the Model gateway (a Passage "
            "above its input limit means RAG_CHUNK_TOKENS is too high), Qdrant, "
            "or text extraction",
            see="section 10",
        )
    trip.generation = trip.index.status().generation
    return f"{report.added} generated Documents synced into the throwaway collection"


def _keyword(trip: _RoundTrip) -> str:
    trip.need(
        trip.run_id.keyword,
        KEYWORD_DOCUMENT,
        f"the term {trip.run_id.keyword}",
        "the keyword (BM25) side of search does not find a term that is in a Document",
        containing=trip.run_id.keyword,
    )
    return "a distinctive term finds its Document"


def _dense(trip: _RoundTrip) -> str:
    if is_keyword_only(trip.ctx):
        raise Skipped(KEYWORD_ONLY)
    trip.need(
        corpus_files.SEMANTIC_QUERY,
        SEMANTIC_DOCUMENT,
        "a paraphrase sharing no word",
        "the embedding model does not place a paraphrase near its Document: "
        "RAG_EMBED_MODEL is not an embedding model, or vectors are not used",
        k=2,
    )
    return "a paraphrase that shares no word finds its Document, by meaning alone"


def _tail_keyword(trip: _RoundTrip) -> str:
    trip.need(
        trip.run_id.tail,
        LONG_DOCUMENT,
        f"the term {trip.run_id.tail}, from the last sentence of a long Document",
        "the tail of a long Document was not indexed: it was cut off, not split into Passages",
        containing=trip.run_id.tail,
    )
    count = len(trip.chunker.split(trip.texts[LONG_DOCUMENT]))
    return f"a term from the tail of a long Document ({count} Passages) is found"


def _tail_meaning(trip: _RoundTrip) -> str:
    if is_keyword_only(trip.ctx):
        raise Skipped(KEYWORD_ONLY)
    trip.need(
        corpus_files.TAIL_QUERY,
        LONG_DOCUMENT,
        "a paraphrase of the tail of a long Document",
        "the embedding model reads only the start of a Passage: RAG_CHUNK_TOKENS is above "
        "its real input limit; lower it, then run a Rebuild",
        containing="lighthouse",
    )
    return "a paraphrase of the tail of a long Document finds that Passage: the model reads it all"


def _status(trip: _RoundTrip) -> str:
    status = trip.index.status()
    expected = trip.expected_passages()
    if (status.documents, status.passages) != (len(trip.texts), expected):
        raise Failure(
            f"status counts {status.documents} Documents and {status.passages} "
            f"Passages, expected {len(trip.texts)} and {expected}",
            cause="Passages were lost or doubled while indexing",
            see="section 10",
        )
    if status.access is not Access.READ_WRITE or status.warnings:
        raise Failure(
            f"status says {status.access}: {'; '.join(status.warnings)}",
            cause="the index just built does not match what is running",
            see="section 10",
        )
    return f"{status.documents} Documents and {status.passages} Passages counted, read-write"


def _replace(trip: _RoundTrip) -> str:
    old, new = trip.run_id.keyword, trip.run_id.replacement
    trip.texts[KEYWORD_DOCUMENT] = corpus_files.keyword_text(new)
    corpus_files.write_document(trip.folder, KEYWORD_DOCUMENT, trip.texts[KEYWORD_DOCUMENT])
    report = trip.sync()
    if (report.replaced, report.added, report.removed, report.failed) != (1, 0, 0, 0):
        raise Failure(
            f"after a Document changed, Sync reported {report.replaced} replaced, "
            f"{report.added} added, {report.removed} removed, {report.failed} failed",
            cause="the Corpus's version of the Document did not change",
            see="section 10",
        )
    if trip.mentions(old, old):
        raise Failure(
            f"the old text of a changed Document is still found ({old})",
            cause="the Passages of the old version were not deleted from Qdrant",
            see="section 10",
        )
    trip.need(
        new,
        KEYWORD_DOCUMENT,
        f"the new term {new}",
        "the new version was not indexed",
        containing=new,
    )
    return "a changed Document is replaced: the new text is found, the old is gone"


def _remove(trip: _RoundTrip) -> str:
    (trip.folder / SEMANTIC_DOCUMENT).unlink()
    del trip.texts[SEMANTIC_DOCUMENT]
    report = trip.sync()
    if report.removed != 1 or report.failed:
        raise Failure(
            f"after a Document was deleted, Sync reported {report.removed} removed, "
            f"{report.failed} failed",
            cause="the Sync did not see the deletion",
            see="section 10",
        )
    if (
        trip.found(corpus_files.SEMANTIC_QUERY, SEMANTIC_DOCUMENT)
        or trip.index.get_document(SEMANTIC_DOCUMENT) is not None
    ):
        raise Failure(
            "a removed Document is still found",
            cause="its Passages were not deleted from Qdrant",
            see="section 10",
        )
    status = trip.index.status()
    if status.documents != len(trip.texts) or status.passages != trip.expected_passages():
        raise Failure(
            f"status counts {status.documents} Documents and {status.passages} "
            f"Passages after a removal, expected {len(trip.texts)} and "
            f"{trip.expected_passages()}",
            cause="Passages were left behind",
            see="section 10",
        )
    return "a removed Document disappears from search and from the counts"


def _rebuild(trip: _RoundTrip) -> str:
    before = trip.index.status().generation
    report = trip.index.rebuild([trip.corpus])
    if not report.promoted:
        raise Failure(
            f"Rebuild was refused: {report.reason}",
            cause="a Rebuild of two generated Documents should always be complete; "
            "the reason names what failed",
            see='section 10 (row "A Rebuild is refused")',
        )
    now = trip.index.status()
    if now.generation == before or now.documents != len(trip.texts):
        raise Failure(
            f"after a Rebuild the Live index is {now.generation} with "
            f"{now.documents} Documents (before: {before}, expected "
            f"{len(trip.texts)})",
            cause="the Rebuild did not go live",
            see="section 10",
        )
    trip.need(
        trip.run_id.replacement,
        KEYWORD_DOCUMENT,
        "a term after the Rebuild",
        "the rebuilt index lost a Document",
        containing=trip.run_id.replacement,
    )
    trip.generation = before
    return f"rebuilt {report.indexed} Documents into a new generation and made it live"


def _switch_back(trip: _RoundTrip) -> str:
    trip.index.switch_back()
    now = trip.index.status().generation
    if now != trip.generation:
        raise Failure(
            f"after switch-back the Live index is {now}, expected {trip.generation}",
            cause="the previous generation was not kept",
            see="section 9 (Operate)",
        )
    trip.need(
        trip.run_id.replacement,
        KEYWORD_DOCUMENT,
        "a term after switch-back",
        "the previous generation cannot be searched",
        containing=trip.run_id.replacement,
    )
    return "switch-back makes the previous generation live again, and it is searchable"
