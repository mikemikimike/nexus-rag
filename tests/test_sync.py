import dataclasses

import pytest

from conftest import DOCUMENTS
from nexus_rag import KnowledgeIndex, Settings
from nexus_rag.corpus import Document, DocumentRef, InMemoryCorpus, Listing
from nexus_rag.fakes import FakeModelGateway


class CountingGateway(FakeModelGateway):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.embedded: list[str] = []

    def embed(self, texts):
        self.embedded.extend(texts)
        return super().embed(texts)


def revised(document: Document, version: str, text: str | None = None) -> Document:
    return dataclasses.replace(
        document, version=version, text=text if text is not None else document.text
    )


def test_syncing_an_unchanged_corpus_again_reports_everything_unchanged(index, corpus):
    index.sync(corpus)

    report = index.sync(corpus)

    assert (report.added, report.replaced, report.removed, report.unchanged, report.failed) == (
        0,
        0,
        0,
        3,
        0,
    )


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
    changed = InMemoryCorpus(
        [*DOCUMENTS[:2], revised(battery, "2", "Jump the alternator with ZX-9000 cables.")]
    )

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
    assert "default:garage/battery" not in {
        p.document_id for p in index.search("car battery", k=10)
    }


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
        failing={"garage/battery"},
    )

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

    assert seen and all(ids and ids[0] == "default:dba/ora-00942" for ids in seen)


def test_a_second_sync_while_one_is_running_is_refused(index):
    from nexus_rag import SyncInProgress

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


class FakeClock:
    """A clock that moves only when told to, starting at a plausible wall-clock time."""

    def __init__(self):
        self.now = 1_700_000_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def crashed_holder():
    """Starts a holder that never exits, as if its process died; stops its heartbeat
    thread when the test ends."""
    holders = []

    def crash(run):
        holder = run.hold()
        holder.__enter__()
        holders.append(holder)
        return holder

    yield crash
    for holder in holders:
        holder.__exit__(None, None, None)  # no-op on a lock that was taken over


def test_exclusive_run_is_shared_by_holders_of_the_same_qdrant():
    from qdrant_client import QdrantClient

    from nexus_rag import SyncInProgress
    from nexus_rag.exclusive import ExclusiveRun

    client = QdrantClient(location=":memory:")
    first, second = ExclusiveRun(client, "lock"), ExclusiveRun(client, "lock")

    with first.hold("Sync"):
        with pytest.raises(SyncInProgress):
            with second.hold("Rebuild"):
                pytest.fail("second holder must be refused")

    with second.hold("Rebuild"):
        pass


def test_exclusive_run_takes_over_the_lock_of_a_crashed_holder(clock, crashed_holder):
    from qdrant_client import QdrantClient

    from nexus_rag import SyncInProgress
    from nexus_rag.exclusive import ExclusiveRun

    client = QdrantClient(location=":memory:")
    crashed = ExclusiveRun(client, "lock", heartbeat_every=3600, now=clock)
    survivor = ExclusiveRun(client, "lock", now=clock)
    crashed_holder(crashed)  # no heartbeat is left

    clock.advance(119)
    with pytest.raises(SyncInProgress):
        with survivor.hold():
            pytest.fail("a lock not yet stale must hold")

    clock.advance(2)
    with survivor.hold():
        pass


def test_a_marker_never_stamped_goes_stale_by_the_same_clock(clock):
    from qdrant_client import QdrantClient

    from nexus_rag import SyncInProgress
    from nexus_rag.exclusive import ExclusiveRun

    client = QdrantClient(location=":memory:")
    ExclusiveRun(client, "lock", now=clock)._try_create()  # its creator died before stamping
    run = ExclusiveRun(client, "lock", now=clock)

    with pytest.raises(SyncInProgress):
        with run.hold():
            pytest.fail("a lock not yet stale must hold")
    clock.advance(121)
    with run.hold():
        pass


def test_an_abandoned_unstamped_marker_is_taken_over_by_a_fresh_run_each_time(clock):
    from qdrant_client import QdrantClient

    from nexus_rag import SyncInProgress
    from nexus_rag.exclusive import ExclusiveRun

    client = QdrantClient(location=":memory:")
    ExclusiveRun(client, "lock", now=clock)._try_create()  # its creator died before stamping

    for _ in range(3):  # every CLI run and every Corpus lock builds a new ExclusiveRun
        with pytest.raises(SyncInProgress):
            with ExclusiveRun(client, "lock", now=clock).hold():
                pytest.fail("an unstamped marker not yet stale must hold")
        clock.advance(40)
    clock.advance(1)

    with ExclusiveRun(client, "lock", now=clock).hold():
        pass


def test_an_unstamped_marker_is_not_taken_over_before_stale_after(clock):
    from qdrant_client import QdrantClient

    from nexus_rag import SyncInProgress
    from nexus_rag.exclusive import ExclusiveRun

    client = QdrantClient(location=":memory:")
    ExclusiveRun(client, "lock", now=clock)._try_create()
    assert ExclusiveRun(client, "lock", now=clock).is_held()

    clock.advance(119)

    with pytest.raises(SyncInProgress):
        with ExclusiveRun(client, "lock", now=clock).hold():
            pytest.fail("a creator that is merely slow to stamp must keep its marker")
    assert ExclusiveRun(client, "lock", now=clock).is_held()


def test_a_creator_that_stamps_late_keeps_its_lock(clock):
    from qdrant_client import QdrantClient

    from nexus_rag import SyncInProgress
    from nexus_rag.exclusive import ExclusiveRun

    client = QdrantClient(location=":memory:")
    slow = ExclusiveRun(client, "lock", heartbeat_every=3600, now=clock)
    slow._try_create()
    with pytest.raises(SyncInProgress):
        with ExclusiveRun(client, "lock", now=clock).hold():  # sees it unstamped, starts timing
            pytest.fail("held")
    clock.advance(100)
    slow._stamp("slow")  # the creator was alive after all

    clock.advance(100)  # long past when the first observer saw it unstamped
    with pytest.raises(SyncInProgress):
        with ExclusiveRun(client, "lock", now=clock).hold():
            pytest.fail("a stamped, live lock must hold")


def test_a_rebuild_is_not_refused_forever_by_an_abandoned_unstamped_corpus_lock(clock):
    from qdrant_client import QdrantClient

    from nexus_rag import SyncInProgress
    from nexus_rag.exclusive import ExclusiveRun

    client = QdrantClient(location=":memory:")
    settings = Settings()
    index = KnowledgeIndex(settings, FakeModelGateway(), client=client, lock_clock=clock)
    index.sync(named("confluence", DOCUMENTS))
    ExclusiveRun(client, f"{settings.collection}.sync-lock.code", now=clock)._try_create()

    with pytest.raises(SyncInProgress):
        index.rebuild([named("confluence", DOCUMENTS)])
    clock.advance(settings.sync_lock_stale_after + 1)

    assert index.rebuild([named("confluence", DOCUMENTS)]).promoted


def test_a_first_sync_takes_over_an_abandoned_unstamped_promote_lock(clock, monkeypatch):
    from qdrant_client import QdrantClient

    from nexus_rag.exclusive import ExclusiveRun

    client = QdrantClient(location=":memory:")
    settings = Settings()
    index = KnowledgeIndex(settings, FakeModelGateway(), client=client, lock_clock=clock)
    ExclusiveRun(client, f"{settings.collection}.promote-lock", now=clock)._try_create()
    monkeypatch.setattr("nexus_rag.index.time.sleep", lambda _: clock.advance(10))

    assert index.sync(named("code", DOCUMENTS)).added == 3


def test_a_first_sync_gives_up_on_a_promote_lock_that_stays_held(
    clock, crashed_holder, monkeypatch
):
    from qdrant_client import QdrantClient

    from nexus_rag import SyncInProgress
    from nexus_rag.exclusive import ExclusiveRun

    client = QdrantClient(location=":memory:")
    settings = Settings()
    index = KnowledgeIndex(settings, FakeModelGateway(), client=client, lock_clock=clock)
    name = f"{settings.collection}.promote-lock"
    holder = ExclusiveRun(client, name, heartbeat_every=3600, now=clock)
    crashed_holder(holder)

    def wait(_):  # time passes, and the live holder keeps beating
        clock.advance(10)
        holder._stamp("holder")

    monkeypatch.setattr("nexus_rag.index.time.sleep", wait)

    with pytest.raises(SyncInProgress, match="promote lock"):
        index.sync(named("code", DOCUMENTS))


def test_a_holder_that_was_taken_over_neither_frees_nor_stamps_over_the_new_lock(clock):
    from qdrant_client import QdrantClient

    from nexus_rag import SyncInProgress
    from nexus_rag.exclusive import ExclusiveRun, LockLost

    client = QdrantClient(location=":memory:")
    stalled = ExclusiveRun(client, "lock", heartbeat_every=3600, now=clock)
    newcomer = ExclusiveRun(client, "lock", heartbeat_every=3600, now=clock)
    holder = stalled.hold()
    lease = holder.__enter__()
    clock.advance(121)

    with newcomer.hold() as taken_over:
        with pytest.raises(LockLost):
            lease.check()
        taken_over.check()
        holder.__exit__(None, None, None)  # the stalled holder wakes up and finishes
        taken_over.check()
        with pytest.raises(SyncInProgress):
            with ExclusiveRun(client, "lock", now=clock).hold():
                pytest.fail("the new holder's lock must still stand")


class PartedCorpus(InMemoryCorpus):
    """Lists each Document under its parts; the parts in ``failing`` could not be listed,
    so Documents under any of them are missing from the listing."""

    def __init__(self, documents, parts, failing=()):
        super().__init__(documents)
        self.parts = parts
        self.failing = set(failing)

    def list(self):
        refs = [DocumentRef(r.id, r.version, self.parts.get(r.id, ())) for r in super().list()]
        listed = [r for r in refs if self.failing.isdisjoint(r.parts)]
        return Listing(listed, {part: "no answer" for part in self.failing})


ORA, ORA_2, BATTERY = DOCUMENTS
PARTS = {ORA.id: ("dba",), ORA_2.id: ("dba",), BATTERY.id: ("garage",)}


def test_an_incomplete_listing_holds_back_removals_only_in_the_failed_part(index):
    index.sync(PartedCorpus(DOCUMENTS, PARTS))

    report = index.sync(PartedCorpus([ORA, BATTERY], PARTS, failing={"garage"}))

    assert (report.removed, report.failed) == (1, 1)
    assert list(report.failures) == ["garage"]
    assert index.get_document("default:dba/ora-12154") is None
    assert index.get_document("default:garage/battery") is not None


def test_a_document_that_moved_to_another_part_is_held_back_by_its_new_part(index):
    index.sync(PartedCorpus(DOCUMENTS, PARTS))
    moved = {**PARTS, ORA.id: ("garage",)}
    assert index.sync(PartedCorpus(DOCUMENTS, moved)).replaced == 1  # its part is re-recorded

    index.sync(PartedCorpus([ORA_2, BATTERY], moved, failing={"garage"}))

    assert index.get_document("default:dba/ora-00942") is not None


def test_a_document_in_several_parts_is_held_back_when_any_of_them_failed(index):
    parts = {ORA.id: ("root", "attachments"), ORA_2.id: ("root",)}
    index.sync(PartedCorpus([ORA, ORA_2], parts))

    report = index.sync(PartedCorpus([], parts, failing={"attachments"}))

    assert (report.removed, report.failed) == (1, 1)
    assert index.get_document("default:dba/ora-00942") is not None
    assert index.get_document("default:dba/ora-12154") is None


def test_documents_listed_under_no_part_are_held_back_by_any_failure(index):
    index.sync(PartedCorpus(DOCUMENTS, {}))

    report = index.sync(PartedCorpus([], {}, failing={"anywhere"}))

    assert (report.removed, report.failed) == (0, 1)
    assert index.get_document("default:garage/battery") is not None


def named(name, documents):
    return InMemoryCorpus(documents, name=name)


def test_a_sync_of_one_corpus_runs_to_completion_while_another_corpus_syncs(index):
    outcome = []

    class LongCorpus(InMemoryCorpus):
        def fetch(self, document_id):
            outcome.append(index.sync(named("code", DOCUMENTS[1:2])))
            return super().fetch(document_id)

    index.sync(LongCorpus(DOCUMENTS[:1], name="confluence"))

    assert [r.added for r in outcome] == [1]
    assert index.search("ORA-12154")[0].document_id == "code:dba/ora-12154"
    assert index.search("ORA-00942")[0].document_id == "confluence:dba/ora-00942"


def test_a_second_sync_of_the_same_corpus_is_refused_while_other_corpora_sync_freely(index):
    from nexus_rag import SyncInProgress

    outcome = []

    class ReentrantCorpus(InMemoryCorpus):
        def fetch(self, document_id):
            with pytest.raises(SyncInProgress):
                index.sync(named("confluence", DOCUMENTS))
            outcome.append(index.sync(named("code", DOCUMENTS[1:2])).added)
            return super().fetch(document_id)

    index.sync(ReentrantCorpus(DOCUMENTS[:1], name="confluence"))

    assert outcome == [1]


def test_a_corpus_whose_sync_crashed_can_be_synced_once_its_lock_is_stale(clock, crashed_holder):
    from qdrant_client import QdrantClient

    from nexus_rag import SyncInProgress
    from nexus_rag.exclusive import ExclusiveRun

    client = QdrantClient(location=":memory:")
    settings = Settings()
    index = KnowledgeIndex(settings, FakeModelGateway(), client=client, lock_clock=clock)
    crashed = ExclusiveRun(
        client, f"{settings.collection}.sync-lock.code", heartbeat_every=3600, now=clock
    )
    crashed_holder(crashed)  # no heartbeat is left

    with pytest.raises(SyncInProgress):
        index.sync(named("code", DOCUMENTS))
    assert index.sync(named("confluence", DOCUMENTS)).added == 3  # another Corpus is unaffected
    clock.advance(settings.sync_lock_stale_after + 1)

    assert index.sync(named("code", DOCUMENTS)).added == 3
