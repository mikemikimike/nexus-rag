import dataclasses
import time
import pytest

from local_rag import KnowledgeIndex, Settings
from local_rag.corpus import Document, InMemoryCorpus
from local_rag.fakes import FakeModelGateway
from conftest import DOCUMENTS


class CountingGateway(FakeModelGateway):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.embedded: list[str] = []

    def embed(self, texts):
        self.embedded.extend(texts)
        return super().embed(texts)


def revised(document: Document, version: str, text: str | None = None) -> Document:
    return dataclasses.replace(
        document, version=version, text=text if text is not None else document.text)


def test_syncing_an_unchanged_corpus_again_reports_everything_unchanged(index, corpus):
    index.sync(corpus)

    report = index.sync(corpus)

    assert (report.added, report.replaced, report.removed,
            report.unchanged, report.failed) == (0, 0, 0, 3, 0)


def test_unchanged_versions_are_not_re_embedded():
    gateway = CountingGateway()
    index = KnowledgeIndex(Settings(), gateway)
    corpus = InMemoryCorpus(DOCUMENTS)
    index.sync(corpus)
    embedded_before = len(gateway.embedded)

    index.sync(corpus)

    assert len(gateway.embedded) == embedded_before


def test_a_changed_version_replaces_the_document_and_only_that_one(index):
    battery = DOCUMENTS[2]
    index.sync(InMemoryCorpus(DOCUMENTS))
    changed = InMemoryCorpus([
        *DOCUMENTS[:2], revised(battery, "2", "Jump the alternator with ZX-9000 cables.")])

    report = index.sync(changed)

    assert (report.added, report.replaced, report.unchanged) == (0, 1, 2)
    passages = index.search("ZX-9000")
    assert passages[0].document_id == "default:garage/battery"
    assert "ZX-9000" in passages[0].text
    assert [p for p in index.search("battery dead") if "dead more often" in p.text] == []


def test_a_replaced_document_leaves_no_duplicate_passages(index):
    index.sync(InMemoryCorpus(DOCUMENTS))
    index.sync(InMemoryCorpus([*DOCUMENTS[:2], revised(DOCUMENTS[2], "2")]))

    passages = index.search("battery", k=10)

    assert [p.document_id for p in passages].count("default:garage/battery") == 1


def test_documents_removed_from_the_corpus_disappear_from_search(index):
    index.sync(InMemoryCorpus(DOCUMENTS))

    report = index.sync(InMemoryCorpus(DOCUMENTS[:2]))

    assert (report.removed, report.unchanged) == (1, 2)
    assert "default:garage/battery" not in {p.document_id for p in index.search("car battery", k=10)}


def test_a_document_added_later_is_reported_as_added(index):
    index.sync(InMemoryCorpus(DOCUMENTS[:2]))

    report = index.sync(InMemoryCorpus(DOCUMENTS))

    assert (report.added, report.unchanged) == (1, 2)


def test_syncing_an_empty_corpus_removes_everything(index):
    index.sync(InMemoryCorpus(DOCUMENTS))

    report = index.sync(InMemoryCorpus([]))

    assert report.removed == 3
    assert index.search("ORA-00942") == []


class FlakyCorpus(InMemoryCorpus):
    def __init__(self, documents, failing=()):
        super().__init__(documents)
        self.failing = set(failing)

    def fetch(self, document_id):
        if document_id in self.failing:
            raise OSError("export failed")
        return super().fetch(document_id)


def test_a_failed_fetch_of_a_changed_document_keeps_its_old_passages_searchable(index):
    index.sync(InMemoryCorpus(DOCUMENTS))
    flaky = FlakyCorpus(
        [*DOCUMENTS[:2], revised(DOCUMENTS[2], "2", "Totally new ZX-9000 text.")],
        failing={"garage/battery"})

    report = index.sync(flaky)

    assert (report.failed, report.replaced, report.unchanged) == (1, 0, 2)
    assert index.search("car will not start")[0].document_id == "default:garage/battery"
    assert not any("ZX-9000" in p.text for p in index.search("ZX-9000"))


def test_a_failed_document_is_retried_on_the_next_sync(index):
    index.sync(InMemoryCorpus(DOCUMENTS))
    changed = [*DOCUMENTS[:2], revised(DOCUMENTS[2], "2", "Totally new ZX-9000 text.")]
    index.sync(FlakyCorpus(changed, failing={"garage/battery"}))

    report = index.sync(InMemoryCorpus(changed))

    assert report.replaced == 1
    assert "ZX-9000" in index.search("ZX-9000")[0].text


def test_a_failed_new_document_counts_as_failed_not_added(index):
    report = index.sync(FlakyCorpus(DOCUMENTS, failing={"dba/ora-12154"}))

    assert (report.added, report.failed) == (2, 1)


class FailingEmbedGateway(FakeModelGateway):
    fail = False

    def embed(self, texts):
        if self.fail:
            raise ConnectionError("gateway down")
        return super().embed(texts)


def test_a_failed_embed_of_a_changed_document_keeps_its_old_passages_searchable():
    gateway = FailingEmbedGateway()
    index = KnowledgeIndex(Settings(), gateway)
    index.sync(InMemoryCorpus(DOCUMENTS))
    gateway.fail = True

    report = index.sync(InMemoryCorpus([*DOCUMENTS[:2], revised(DOCUMENTS[2], "2", "New.")]))

    assert (report.failed, report.replaced) == (1, 0)
    gateway.fail = False
    assert index.search("car battery dead")[0].document_id == "default:garage/battery"


def test_search_keeps_returning_results_while_a_sync_runs(index):
    index.sync(InMemoryCorpus(DOCUMENTS))
    seen = []

    class SearchingCorpus(InMemoryCorpus):
        def fetch(self, document_id):
            seen.append([p.document_id for p in index.search("ORA-00942")])
            return super().fetch(document_id)

    index.sync(SearchingCorpus([*DOCUMENTS[:2], revised(DOCUMENTS[2], "2")]))

    assert seen and all(ids and ids[0] == "dba/ora-00942" for ids in seen)


def test_a_second_sync_while_one_is_running_is_refused(index):
    from local_rag import SyncInProgress

    outcome = []

    class ReentrantCorpus(InMemoryCorpus):
        def fetch(self, document_id):
            try:
                index.sync(InMemoryCorpus(DOCUMENTS))
            except SyncInProgress as error:
                outcome.append(error)
            return super().fetch(document_id)

    index.sync(ReentrantCorpus(DOCUMENTS))

    assert outcome and "Sync" in str(outcome[0])


def test_the_index_accepts_a_new_sync_after_a_previous_one_failed(index):
    class BrokenListing(InMemoryCorpus):
        def list(self):
            raise OSError("listing failed")

    with pytest.raises(OSError):
        index.sync(BrokenListing(DOCUMENTS))

    assert index.sync(InMemoryCorpus(DOCUMENTS)).added == 3


def test_a_failed_listing_removes_nothing(index):
    index.sync(InMemoryCorpus(DOCUMENTS))

    class BrokenListing(InMemoryCorpus):
        def list(self):
            raise OSError("listing failed")

    with pytest.raises(OSError):
        index.sync(BrokenListing(DOCUMENTS))

    assert index.search("ORA-00942")[0].document_id == "default:dba/ora-00942"


def test_exclusive_run_is_shared_by_holders_of_the_same_qdrant():
    from qdrant_client import QdrantClient

    from local_rag import SyncInProgress
    from local_rag.exclusive import ExclusiveRun

    client = QdrantClient(location=":memory:")
    first, second = ExclusiveRun(client, "lock"), ExclusiveRun(client, "lock")

    with first.hold("Sync"):
        with pytest.raises(SyncInProgress):
            with second.hold("Rebuild"):
                pytest.fail("second holder must be refused")

    with second.hold("Rebuild"):
        pass


def test_exclusive_run_takes_over_the_lock_of_a_crashed_holder():
    from qdrant_client import QdrantClient

    from local_rag.exclusive import ExclusiveRun

    client = QdrantClient(location=":memory:")
    crashed = ExclusiveRun(client, "lock", stale_after=0.05, heartbeat_every=3600)
    survivor = ExclusiveRun(client, "lock", stale_after=0.05)
    holder = crashed.hold()
    holder.__enter__()  # never exits: the process died, and no heartbeat is left
    time.sleep(0.1)

    with survivor.hold():
        pass


def test_a_holder_that_was_taken_over_neither_frees_nor_stamps_over_the_new_lock():
    from qdrant_client import QdrantClient

    from local_rag import SyncInProgress
    from local_rag.exclusive import ExclusiveRun, LockLost

    client = QdrantClient(location=":memory:")
    stalled = ExclusiveRun(client, "lock", stale_after=0.05, heartbeat_every=3600)
    newcomer = ExclusiveRun(client, "lock", stale_after=0.05, heartbeat_every=3600)
    holder = stalled.hold()
    lease = holder.__enter__()
    time.sleep(0.1)

    with newcomer.hold() as taken_over:
        with pytest.raises(LockLost):
            lease.check()
        taken_over.check()
        holder.__exit__(None, None, None)  # the stalled holder wakes up and finishes
        taken_over.check()
        with pytest.raises(SyncInProgress):
            with ExclusiveRun(client, "lock", stale_after=0.05).hold():
                pytest.fail("the new holder's lock must still stand")
