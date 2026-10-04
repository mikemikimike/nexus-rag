import time

import pytest

from conftest import DOCUMENTS
from index_states import collections, overwrite_the_last_sync
from nexus_rag import Access, IndexFormat, IndexReadOnly, KnowledgeIndex, Settings
from nexus_rag.corpus import InMemoryCorpus
from nexus_rag.sparse import SparseParams
from test_index_format import after_an_upgrade
from test_sync import FlakyCorpus


def test_status_of_an_index_nothing_was_synced_into_is_empty_and_writable(index):
    status = index.status()

    assert status.generation is None
    assert (status.documents, status.passages) == (0, 0)
    assert status.format is None and status.last_sync is None
    assert status.access == "read-write" and status.warnings == ()


def test_status_reports_the_live_generation_counts_and_format(index, client):
    index.sync(InMemoryCorpus(DOCUMENTS))

    status = index.status()

    assert status.generation in collections(client)
    assert status.generation.startswith("knowledge.gen-")
    assert (status.documents, status.passages) == (3, 3)
    assert status.format == IndexFormat(
        schema_version=1,
        sparse=SparseParams(),
        embed_model="embed",
        dimension=64,
        chunk_tokens=400,
        chunk_overlap=60,
        corpus_ownership=True,
        identifier_splitting=True,
        code_chunk_tokens=800,
        code_splitting="lines",
        syntax_languages=("c", "python"),
        token_counter=2,
    )
    assert status.access == "read-write" and status.warnings == ()


def test_status_follows_the_live_index_through_a_rebuild(index):
    index.sync(InMemoryCorpus(DOCUMENTS))
    first = index.status().generation

    index.rebuild([InMemoryCorpus(DOCUMENTS[:2])])

    status = index.status()
    assert status.generation != first
    assert status.documents == 2


def test_status_reports_the_last_sync(index):
    index.sync(InMemoryCorpus(DOCUMENTS))
    index.sync(InMemoryCorpus(DOCUMENTS))

    last = index.status().last_sync

    assert last.kind == "Sync" and last.outcome == "completed"
    assert last.detail == "0 added, 0 replaced, 0 removed, 3 unchanged, 0 failed"
    assert abs(time.time() - last.at.timestamp()) < 60


def test_status_reports_a_sync_with_failed_documents(index):
    index.sync(FlakyCorpus(DOCUMENTS, failing={"dba/ora-12154"}))

    last = index.status().last_sync

    assert last.outcome == "completed with failures"
    assert last.detail == "2 added, 0 replaced, 0 removed, 0 unchanged, 1 failed"


def test_status_reports_a_sync_that_could_not_finish(index):
    class BrokenListing(InMemoryCorpus):
        def list(self):
            raise OSError("listing failed")

    index.sync(InMemoryCorpus(DOCUMENTS))
    with pytest.raises(OSError):
        index.sync(BrokenListing(DOCUMENTS))

    last = index.status().last_sync

    assert (last.kind, last.outcome, last.detail) == ("Sync", "failed", "listing failed")


def test_status_reports_the_last_rebuild_promoted_or_refused(index):
    index.rebuild([InMemoryCorpus(DOCUMENTS)])
    promoted = index.status().last_sync
    index.rebuild([FlakyCorpus(DOCUMENTS, failing={"dba/ora-12154"})])
    refused = index.status().last_sync

    assert (promoted.kind, promoted.outcome) == ("Rebuild", "promoted")
    assert promoted.detail == "3 indexed, 0 failed"
    assert (refused.kind, refused.outcome) == ("Rebuild", "refused")
    assert "1 of 3 Documents failed" in refused.detail


def test_the_last_sync_is_kept_with_the_index_not_with_the_process(index, client):
    index.sync(InMemoryCorpus(DOCUMENTS))

    other_process = KnowledgeIndex(Settings(), index._gateway, client=client)

    assert other_process.status().last_sync.outcome == "completed"


def test_a_sync_refused_because_the_index_is_read_only_shows_in_status(index, client):
    index.sync(InMemoryCorpus(DOCUMENTS))
    upgraded = after_an_upgrade(index, client, embed_model="embed-v2")
    with pytest.raises(IndexReadOnly):
        upgraded.sync(InMemoryCorpus(DOCUMENTS))

    last = upgraded.status().last_sync

    assert (last.outcome, "read-only" in last.detail) == ("failed", True)


def test_an_unreadable_last_sync_record_is_reported_without_failing_status(index, client):
    index.sync(InMemoryCorpus(DOCUMENTS))
    overwrite_the_last_sync(client, {"kind": "Sync"})

    status = index.status()

    assert status.last_sync is None
    assert any("last Sync" in warning for warning in status.warnings)
    assert status.access == Access.READ_WRITE and status.documents == 3
    assert index.health().ok
    index.sync(InMemoryCorpus(DOCUMENTS))  # the next run writes a readable record again
    assert index.status().last_sync.outcome == "completed"
