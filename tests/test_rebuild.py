import dataclasses

import pytest

from local_rag import (
    IndexReadOnly, KnowledgeIndex, NoPreviousIndex, Settings, SyncInProgress)
from local_rag.corpus import InMemoryCorpus
from local_rag.fakes import FakeModelGateway
from conftest import DOCUMENTS
from index_states import (
    NAMES, collections, generations_in, legacy_index, take_over_the_lock)
from test_sync import CountingGateway, FlakyCorpus, revised


def test_a_rebuild_above_the_threshold_is_refused_and_the_live_index_is_untouched(index):
    index.sync(InMemoryCorpus(DOCUMENTS))
    changed = [revised(d, "2", f"{d.text} Now mentions ZX-9000.") for d in DOCUMENTS]

    report = index.rebuild([FlakyCorpus(changed, failing={"dba/ora-12154"})])

    assert not report.promoted
    assert (report.indexed, report.failed) == (2, 1)
    assert "1 of 3" in report.reason
    assert not any("ZX-9000" in p.text for p in index.search("ZX-9000", k=10))
    assert index.search("ORA-12154")[0].document_id == "default:dba/ora-12154"


def test_a_rebuild_within_the_threshold_is_promoted():
    index = KnowledgeIndex(Settings(rebuild_failure_threshold=0.5), FakeModelGateway())
    index.sync(InMemoryCorpus(DOCUMENTS))
    changed = [revised(d, "2", f"{d.text} Now mentions ZX-9000.") for d in DOCUMENTS]

    report = index.rebuild([FlakyCorpus(changed, failing={"dba/ora-12154"})])

    assert report.promoted and report.reason is None
    assert (report.indexed, report.failed) == (2, 1)
    assert "ZX-9000" in index.search("ZX-9000")[0].text
    assert index.get_document("default:dba/ora-12154") is None


def test_a_failure_rate_exactly_at_the_threshold_is_promoted():
    index = KnowledgeIndex(Settings(rebuild_failure_threshold=1 / 3), FakeModelGateway())

    report = index.rebuild([FlakyCorpus(DOCUMENTS, failing={"dba/ora-12154"})])

    assert report.promoted


def test_the_default_threshold_accepts_no_failed_document(index):
    report = index.rebuild([FlakyCorpus(DOCUMENTS, failing={"dba/ora-12154"})])

    assert not report.promoted
    assert index.search("ORA-00942") == []


def test_a_rebuild_promotes_a_complete_index_when_nothing_was_indexed_before(index):
    report = index.rebuild([InMemoryCorpus(DOCUMENTS)])

    assert (report.promoted, report.indexed, report.failed) == (True, 3, 0)
    assert index.search("ORA-00942")[0].document_id == "default:dba/ora-00942"


def test_a_rebuild_never_replaces_the_index_with_nothing():
    index = KnowledgeIndex(Settings(rebuild_failure_threshold=1.0), FakeModelGateway())
    index.sync(InMemoryCorpus(DOCUMENTS))

    everything_fails = index.rebuild([FlakyCorpus(DOCUMENTS, failing={d.id for d in DOCUMENTS})])
    empty = index.rebuild([InMemoryCorpus([])])

    assert not everything_fails.promoted and not empty.promoted
    assert empty.reason and everything_fails.reason
    assert index.search("ORA-00942")[0].document_id == "default:dba/ora-00942"


def test_a_rebuild_re_embeds_every_document_even_when_versions_are_unchanged():
    gateway = CountingGateway()
    index = KnowledgeIndex(Settings(), gateway)
    index.sync(InMemoryCorpus(DOCUMENTS))
    embedded_before = len(gateway.embedded)

    index.rebuild([InMemoryCorpus(DOCUMENTS)])

    assert len(gateway.embedded) == embedded_before + len(DOCUMENTS)


def test_a_rebuild_drops_documents_that_left_the_corpus(index):
    index.sync(InMemoryCorpus(DOCUMENTS))

    index.rebuild([InMemoryCorpus(DOCUMENTS[:2])])

    assert index.get_document("default:garage/battery") is None
    assert index.list_spaces() == ["DBA"]


def test_sync_after_a_rebuild_continues_from_the_rebuilt_index(index):
    index.sync(InMemoryCorpus(DOCUMENTS))
    index.rebuild([InMemoryCorpus(DOCUMENTS)])

    report = index.sync(InMemoryCorpus([*DOCUMENTS[:2], revised(DOCUMENTS[2], "2")]))

    assert (report.replaced, report.unchanged, report.added) == (1, 2, 0)


def test_search_keeps_returning_results_throughout_a_rebuild(index):
    index.sync(InMemoryCorpus(DOCUMENTS))
    seen = []

    class SearchingCorpus(InMemoryCorpus):
        def fetch(self, document_id):
            seen.append([p.document_id for p in index.search("ORA-00942")])
            return super().fetch(document_id)

    index.rebuild([SearchingCorpus(DOCUMENTS)])

    assert len(seen) == 3 and all(ids and ids[0] == "default:dba/ora-00942" for ids in seen)
    assert index.search("ORA-00942")[0].document_id == "default:dba/ora-00942"


def test_a_sync_or_second_rebuild_while_a_rebuild_runs_is_refused(index):
    outcome = []

    class ReentrantCorpus(InMemoryCorpus):
        def fetch(self, document_id):
            for start in (index.sync, index.rebuild):
                with pytest.raises(SyncInProgress) as refused:
                    start(InMemoryCorpus(DOCUMENTS))
                outcome.append(str(refused.value))
            return super().fetch(document_id)

    index.rebuild([ReentrantCorpus(DOCUMENTS[:1])])

    assert len(outcome) == 2


def test_a_rebuild_while_a_sync_runs_is_refused(index):
    outcome = []

    class ReentrantCorpus(InMemoryCorpus):
        def fetch(self, document_id):
            with pytest.raises(SyncInProgress) as refused:
                index.rebuild([InMemoryCorpus(DOCUMENTS)])
            outcome.append(str(refused.value))
            return super().fetch(document_id)

    index.sync(ReentrantCorpus(DOCUMENTS[:1]))

    assert outcome and "Rebuild" in outcome[0]


def test_a_rebuild_that_crashes_leaves_the_live_index_and_accepts_the_next_run(index):
    index.sync(InMemoryCorpus(DOCUMENTS))

    class BrokenListing(InMemoryCorpus):
        def list(self):
            raise OSError("listing failed")

    with pytest.raises(OSError):
        index.rebuild([BrokenListing(DOCUMENTS)])

    assert index.search("ORA-00942")[0].document_id == "default:dba/ora-00942"
    assert index.rebuild([InMemoryCorpus(DOCUMENTS)]).promoted


def corpus_saying(text: str) -> InMemoryCorpus:
    return InMemoryCorpus([dataclasses.replace(d, text=text) for d in DOCUMENTS])


def test_switching_back_restores_the_previous_index(index):
    index.sync(InMemoryCorpus(DOCUMENTS))
    index.rebuild([corpus_saying("Now about ZX-9000.")])
    assert "ZX-9000" in index.search("ZX-9000")[0].text

    index.switch_back()

    assert index.search("ORA-00942")[0].document_id == "default:dba/ora-00942"
    assert "ZX-9000" not in index.get_document("default:dba/ora-00942")


def test_switching_back_with_no_previous_index_is_refused(index):
    with pytest.raises(NoPreviousIndex):
        index.switch_back()

    index.rebuild([InMemoryCorpus(DOCUMENTS)])
    with pytest.raises(NoPreviousIndex):
        index.switch_back()


def test_only_one_previous_generation_is_kept(index):
    index.rebuild([corpus_saying("first")])
    index.rebuild([corpus_saying("second")])
    index.rebuild([corpus_saying("third")])

    index.switch_back()
    assert index.get_document("default:dba/ora-00942") == "second"

    index.switch_back()  # the index it switched away from, not the discarded "first"
    assert index.get_document("default:dba/ora-00942") == "third"


def test_a_refused_rebuild_keeps_the_previous_generation(index):
    index.rebuild([corpus_saying("first")])
    index.rebuild([corpus_saying("second")])

    refused = index.rebuild([FlakyCorpus(DOCUMENTS, failing={"dba/ora-12154"})])
    index.switch_back()

    assert not refused.promoted
    assert index.get_document("default:dba/ora-00942") == "first"


def test_switching_back_during_a_sync_is_refused(index):
    index.rebuild([InMemoryCorpus(DOCUMENTS)])
    index.rebuild([InMemoryCorpus(DOCUMENTS)])
    refused = []

    class SwitchingCorpus(InMemoryCorpus):
        def fetch(self, document_id):
            with pytest.raises(SyncInProgress):
                index.switch_back()
            refused.append(document_id)
            return super().fetch(document_id)

    index.sync(SwitchingCorpus([revised(DOCUMENTS[0], "2")]))

    assert refused


def test_cleanup_only_removes_generations_the_system_created(index, client):
    from qdrant_client import models

    # Collections in the same Qdrant that merely look like ours by name.
    lookalikes = {"knowledge.gen-0123456789ab", "knowledge.previous-backup",
                  "knowledge_old", "knowledge-2024", "knowledge.generations-old"}
    for name in lookalikes:
        client.create_collection(
            name, vectors_config=models.VectorParams(size=1, distance=models.Distance.COSINE))

    for text in ("first", "second", "third"):
        index.rebuild([corpus_saying(text)])
    index.rebuild([FlakyCorpus(DOCUMENTS, failing={"dba/ora-12154"})])

    assert lookalikes <= collections(client)


def test_generations_left_behind_do_not_accumulate(index, client):
    for text in ("first", "second", "third", "fourth"):
        index.rebuild([corpus_saying(text)])
    index.rebuild([FlakyCorpus(DOCUMENTS, failing={"dba/ora-12154"})])

    assert len(generations_in(client)) == 2


def test_a_failed_cleanup_does_not_mask_the_error_and_is_swept_by_the_next_rebuild(
        index, client, monkeypatch):
    class Crash(BaseException):
        pass

    class CrashingCorpus(InMemoryCorpus):
        def fetch(self, document_id):
            if document_id == "dba/ora-12154":
                raise Crash()
            return super().fetch(document_id)

    real_delete = client.delete_collection

    def broken_delete(collection):
        if collection in generations_in(client):
            raise OSError("qdrant went away")
        return real_delete(collection)

    index.rebuild([InMemoryCorpus(DOCUMENTS)])
    with monkeypatch.context() as broken:
        broken.setattr(client, "delete_collection", broken_delete)
        with pytest.raises(Crash):
            index.rebuild([CrashingCorpus(DOCUMENTS)])
    assert len(generations_in(client)) == 2  # live + the stray

    index.rebuild([InMemoryCorpus(DOCUMENTS)])

    assert len(generations_in(client)) == 2  # live + previous


def test_a_sync_of_nothing_leaves_no_generation_behind(index, client):
    index.sync(InMemoryCorpus([]))
    index.sync(FlakyCorpus(DOCUMENTS, failing={d.id for d in DOCUMENTS}))

    assert not generations_in(client) and NAMES.registry not in collections(client)
    assert index.sync(InMemoryCorpus(DOCUMENTS)).added == 3


def test_an_index_written_by_an_earlier_version_keeps_being_served_but_not_synced(
        index, client, gateway):
    legacy_index(client, gateway, DOCUMENTS)

    with pytest.raises(IndexReadOnly, match="not recorded"):
        index.sync(InMemoryCorpus([*DOCUMENTS[:2], revised(DOCUMENTS[2], "2", "New.")]))

    assert index.search("ORA-00942")[0].document_id == "dba/ora-00942"
    assert index.get_document("garage/battery") == DOCUMENTS[2].text
    assert index.list_spaces() == ["DBA", "GARAGE"]


def test_a_rebuild_replaces_an_index_written_by_an_earlier_version_and_cleans_it_up(
        index, client, gateway):
    legacy_index(client, gateway, DOCUMENTS)

    index.rebuild([corpus_saying("first")])
    index.switch_back()
    assert index.search("ORA-00942")[0].document_id == "dba/ora-00942"
    assert "first" != index.get_document("default:dba/ora-00942")
    index.switch_back()
    index.rebuild([corpus_saying("second")])

    assert NAMES.legacy not in collections(client)
    assert index.get_document("default:dba/ora-00942") == "second"


def test_a_search_started_before_a_promotion_still_answers():
    class RebuildingGateway(FakeModelGateway):
        index = None
        rebuilt = False

        def embed_query(self, text):
            if not self.rebuilt:  # two promotions, between finding the Live index and querying it
                self.rebuilt = True
                self.index.rebuild([corpus_saying("second")])
                self.index.rebuild([corpus_saying("third")])
            return super().embed_query(text)

    gateway = RebuildingGateway()
    index = gateway.index = KnowledgeIndex(Settings(), gateway)
    index.sync(InMemoryCorpus(DOCUMENTS))

    passages = index.search("ORA-00942")

    assert passages and passages[0].text == "third"


@pytest.mark.parametrize("threshold", [-0.1, 1.5, float("nan")])
def test_a_rebuild_threshold_outside_zero_to_one_is_rejected(threshold):
    with pytest.raises(ValueError, match="rebuild_failure_threshold"):
        Settings(rebuild_failure_threshold=threshold)


def test_a_rebuild_that_lost_the_lock_does_not_promote(index, client):
    from local_rag.exclusive import LockLost

    index.sync(InMemoryCorpus(DOCUMENTS))

    class StallingCorpus(InMemoryCorpus):
        def fetch(self, document_id):
            take_over_the_lock(client)
            return super().fetch(document_id)

    with pytest.raises(LockLost):
        index.rebuild([StallingCorpus(DOCUMENTS)])

    assert index.get_document("default:dba/ora-00942") == DOCUMENTS[0].text
    assert len(generations_in(client)) == 1


def test_a_rebuild_is_refused_while_any_part_of_a_listing_is_incomplete():
    from test_sync import PARTS, PartedCorpus
    index = KnowledgeIndex(Settings(rebuild_failure_threshold=1.0), FakeModelGateway())
    index.sync(PartedCorpus(DOCUMENTS, PARTS))

    report = index.rebuild([PartedCorpus(DOCUMENTS, PARTS, failing={"garage"})])

    assert not report.promoted and "garage" in report.reason
    assert list(report.failures) == ["default: garage"]
