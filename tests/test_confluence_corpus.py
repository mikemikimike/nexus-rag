import pytest

import fake_confluence
import fake_lit
from conftest import DOCUMENTS
from nexus_rag import KnowledgeIndex, Settings
from nexus_rag.confluence import ConfluenceCorpus, read_roots
from nexus_rag.corpus import InMemoryCorpus
from nexus_rag.fakes import FakeModelGateway
from test_sync import CountingGateway

ROOT = "100"


@pytest.fixture
def lit(tmp_path_factory):
    return fake_lit.install(tmp_path_factory.mktemp("bin"))


@pytest.fixture
def site(tmp_path_factory):
    """A space with a home page, two child pages and a grandchild, plus an unrelated page."""
    site = fake_confluence.FakeSite(tmp_path_factory.mktemp("confluence"))
    site.page("100", "ENG home", body="# ENG home\n\nStart here.\n")
    site.page(
        "101",
        "Fixing ORA-00942",
        parent="100",
        body="# Fixing ORA-00942\n\nTable or view does not exist. Check the grants.\n",
    )
    site.page("102", "Failover runbook", parent="100", body="Promote the standby database.\n")
    site.page("103", "Failover drill", parent="102", body="Rehearse the ZX-9000 failover.\n")
    site.page("200", "Elsewhere", space="OPS", body="Not under any root.\n")
    return site


@pytest.fixture
def make_index(lit):
    def make(**settings):
        return KnowledgeIndex(Settings(liteparse_bin=str(lit), **settings), FakeModelGateway())

    return make


@pytest.fixture
def index(make_index):
    return make_index()


def corpus(site, *roots, **options):
    return ConfluenceCorpus(list(roots) or [ROOT], confluence_bin=str(site.bin), **options)


def test_a_root_page_and_all_its_descendants_are_synced_and_searched(index, site):
    report = index.sync(corpus(site))

    assert (report.added, report.failed) == (4, 0)
    found = index.search("ORA-00942")[0]
    assert (found.document_id, found.title, found.space) == (
        "confluence:101",
        "Fixing ORA-00942",
        "ENG",
    )
    assert found.url == "https://wiki.example.test/pages/viewpage.action?pageId=101"
    assert index.search("ZX-9000")[0].document_id == "confluence:103"
    assert index.get_document("confluence:200") is None
    assert "Promote the standby database." in index.get_document("confluence:102")


def downloads(site):
    return [call for call in site.calls("attachments") if "--download" in call]


def test_an_attachment_is_its_own_document_citing_its_parent_page(index, site):
    site.attach(
        "101",
        "grants.pdf",
        fake_lit.pdf("Grant the SELECT privilege to the REPORTING role.").decode(),
    )
    site.attach("101", "notes.txt", "Escalate to the DBA on call.")

    report = index.sync(corpus(site))

    assert (report.added, report.failed) == (6, 0)
    found = index.search("REPORTING role")[0]
    assert found.document_id == "confluence:101/attachments/grants.pdf"
    assert found.parent_id == "confluence:101"
    assert found.title == "Fixing ORA-00942 / grants.pdf"
    assert found.url == "https://wiki.example.test/pages/viewpage.action?pageId=101"
    assert found.space == "ENG"
    assert (
        index.get_document("confluence:101/attachments/notes.txt") == "Escalate to the DBA on call."
    )
    page = next(p for p in index.search("ORA-00942") if p.document_id == "confluence:101")
    assert page.parent_id is None


def test_an_unchanged_page_version_is_neither_exported_nor_embedded_again(index, site):
    site.attach("101", "notes.txt", "Escalate to the DBA on call.")
    index.sync(corpus(site))
    fetches = len(site.calls("export")) + len(downloads(site))

    report = index.sync(corpus(site))

    assert (report.unchanged, report.added, report.replaced) == (5, 0, 0)
    assert len(site.calls("export")) + len(downloads(site)) == fetches == 5


def test_a_new_page_version_replaces_only_that_page(index, site):
    index.sync(corpus(site))
    site.page(
        "102",
        "Failover runbook",
        parent="100",
        version=2,
        body="Promote the standby, then verify with the QX-5 check.\n",
    )

    report = index.sync(corpus(site))

    assert (report.replaced, report.unchanged, report.failed) == (1, 3, 0)
    assert [c[1] for c in site.calls("export")][-1:] == ["102"]
    assert index.search("QX-5 check")[0].document_id == "confluence:102"
    assert "QX-5" in index.get_document("confluence:102")


def test_an_attachment_is_replaced_when_its_own_version_changes(index, site):
    site.attach("101", "notes.txt", "Escalate to the DBA on call.")
    index.sync(corpus(site))
    site.attach("101", "notes.txt", "Escalate to the platform team.", version=2)

    report = index.sync(corpus(site))

    assert (report.replaced, report.unchanged) == (1, 4)
    assert (
        index.get_document("confluence:101/attachments/notes.txt")
        == "Escalate to the platform team."
    )


def test_a_new_page_version_does_not_replace_an_attachment_that_did_not_change(index, site):
    site.attach("101", "notes.txt", "Escalate to the DBA on call.")
    index.sync(corpus(site))
    site.page("101", "Fixing ORA-00942", parent="100", version=2, body="Rewritten.\n")

    report = index.sync(corpus(site))

    assert (report.replaced, report.unchanged) == (1, 4)
    assert len(downloads(site)) == 1  # from the first Sync only


def test_a_deleted_page_and_its_attachments_are_removed(index, site):
    site.attach("102", "runbook.txt", "Call the standby owner.")
    index.sync(corpus(site))
    site.delete("103")
    site.delete("102")

    report = index.sync(corpus(site))

    assert (report.removed, report.unchanged, report.failed) == (3, 2, 0)
    assert index.get_document("confluence:102") is None
    assert index.get_document("confluence:102/attachments/runbook.txt") is None
    assert "confluence:103" not in {p.document_id for p in index.search("ZX-9000")}


def test_a_deleted_attachment_is_removed_and_its_page_kept(index, site):
    site.attach("101", "notes.txt", "Escalate to the DBA on call.")
    index.sync(corpus(site))
    site.detach("101", "notes.txt")

    report = index.sync(corpus(site))

    assert (report.removed, report.unchanged) == (1, 4)
    assert index.get_document("confluence:101/attachments/notes.txt") is None


def test_a_failed_root_listing_reports_a_failure_and_the_other_roots_are_still_synced(index, site):
    index.sync(corpus(site, "101", "200"))
    site.fail("200", "info")
    site.page("101", "Fixing ORA-00942", parent="100", version=2, body="Rewritten ORA.\n")

    report = index.sync(corpus(site, "101", "200"))

    assert (report.replaced, report.failed) == (1, 1)
    assert "failed (1)" in report.failures["root 200"] and "HTTP 500" in report.failures["root 200"]


def test_a_failed_root_holds_back_removals_only_under_that_root(index, site):
    index.sync(corpus(site, "100", "200"))
    site.fail("200", "info")
    site.delete("101")

    report = index.sync(corpus(site, "100", "200"))

    assert (report.removed, report.failed) == (1, 1)
    assert list(report.failures) == ["root 200"]
    assert index.get_document("confluence:200") is not None  # its root could not be listed
    assert index.get_document("confluence:101") is None  # under a root that listed fine

    site.fail("200")
    site.delete("200")
    report = index.sync(corpus(site, "100", "200"))
    assert (report.removed, report.failed) == (0, 1)  # a deleted root page cannot be listed
    assert index.get_document("confluence:200") is not None


def test_a_failed_children_listing_holds_back_the_pages_under_that_root(index, site):
    site.attach("102", "runbook.txt", "Call the standby owner.")
    site.page("201", "Ops notes", space="OPS", parent="200", body="Page the on-call.\n")
    index.sync(corpus(site, "100", "200"))
    site.fail("100", "children")
    site.delete("103")
    site.delete("201")

    report = index.sync(corpus(site, "100", "200"))

    assert (report.removed, report.failed) == (1, 1)
    assert index.get_document("confluence:103") is not None
    assert index.get_document("confluence:102/attachments/runbook.txt") is not None
    assert index.get_document("confluence:201") is None  # under the other root


def test_a_failed_attachment_listing_holds_back_only_that_pages_attachments(index, site):
    site.attach("102", "runbook.txt", "Call the standby owner.")
    site.attach("101", "notes.txt", "Escalate to the DBA on call.")
    index.sync(corpus(site))
    site.fail("102", "attachments")
    site.delete("103")
    site.detach("101", "notes.txt")

    report = index.sync(corpus(site))

    assert (report.removed, report.failed) == (2, 1)
    assert list(report.failures) == ["attachments of page 102"]
    assert index.get_document("confluence:102/attachments/runbook.txt") is not None
    assert index.get_document("confluence:103") is None
    assert index.get_document("confluence:101/attachments/notes.txt") is None


def test_a_failed_root_also_holds_back_its_pages_attachments(index, site):
    site.attach("102", "runbook.txt", "Call the standby owner.")
    index.sync(corpus(site, "100", "200"))
    site.fail("100", "info")

    report = index.sync(corpus(site, "100", "200"))

    assert (report.removed, report.failed) == (0, 1)
    assert index.get_document("confluence:102/attachments/runbook.txt") is not None


def test_an_export_failure_is_reported_for_that_document_and_the_rest_is_synced(index, site):
    site.attach("101", "notes.txt", "Escalate to the DBA on call.")
    site.fail("102", "export")
    site.fail("101", "download")

    report = index.sync(corpus(site))

    assert (report.added, report.failed) == (3, 2)
    assert "HTTP 500" in report.failures["confluence:102"]
    assert "could not download" in report.failures["confluence:101/attachments/notes.txt"]
    assert index.get_document("confluence:102") is None
    assert index.get_document("confluence:103") is not None


def test_a_failed_export_keeps_the_old_version_searchable_and_is_retried(index, site):
    index.sync(corpus(site))
    site.page("102", "Failover runbook", parent="100", version=2, body="Promote, QX-5.\n")
    site.fail("102", "export")

    report = index.sync(corpus(site))

    assert (report.failed, report.replaced) == (1, 0)
    assert "standby" in index.get_document("confluence:102")

    site.fail("102")
    report = index.sync(corpus(site))
    assert (report.failed, report.replaced) == (0, 1)
    assert "QX-5" in index.get_document("confluence:102")


def test_an_attachment_that_cannot_be_read_fails_only_itself(index, site):
    site.attach("101", "scan.pdf", fake_lit.pdf("CORRUPT").decode())

    report = index.sync(corpus(site))

    assert (report.added, report.failed) == (4, 1)
    assert "could not read the document" in report.failures["confluence:101/attachments/scan.pdf"]


def test_export_failures_count_toward_the_rebuild_threshold(make_index, site):
    site.fail("102", "export")

    refused = make_index().rebuild([corpus(site)])
    assert not refused.promoted and list(refused.failures) == ["confluence:102"]
    assert "1 of 4 Documents failed" in refused.reason

    promoted = make_index(rebuild_failure_threshold=0.25).rebuild([corpus(site)])
    assert promoted.promoted and (promoted.indexed, promoted.failed) == (3, 1)


def test_a_rebuild_is_refused_when_a_root_listing_failed(make_index, site):
    index = make_index(rebuild_failure_threshold=1.0)
    index.sync(corpus(site, "100", "200"))
    site.fail("200", "info")

    report = index.rebuild([corpus(site, "100", "200")])

    assert not report.promoted and "root 200" in report.reason
    assert "confluence: root 200" in report.failures
    assert index.get_document("confluence:200") is not None  # the Live index is untouched
    assert len(site.calls("export")) == 5  # the refused Rebuild fetched nothing


def test_overlapping_roots_and_url_roots_list_each_page_once(index, site):
    url = "https://wiki.example.test/pages/viewpage.action?pageId=102"

    report = index.sync(corpus(site, "100", url, "103"))

    assert (report.added, report.failed) == (4, 0)


def test_images_media_and_unreadable_types_are_left_out_of_the_attachments(index, site):
    for name in ("shot.png", "demo.MP4", "backup.zip", "data.bin", "notes.txt"):
        site.attach("101", name, "x")

    report = index.sync(corpus(site))

    assert (report.added, report.failed) == (5, 0)
    assert index.get_document("confluence:101/attachments/notes.txt") == "x"


def test_attachments_can_be_turned_off_or_filtered_by_name(index, site):
    site.attach("101", "notes.txt", "Escalate.")
    site.attach("101", "draft-plan.txt", "Not yet.")

    assert index.sync(corpus(site, attachments=False)).added == 4
    assert site.calls("attachments") == []

    report = index.sync(corpus(site, exclude_attachments=["draft-*"]))

    assert report.added == 1
    assert index.get_document("confluence:101/attachments/notes.txt") == "Escalate."
    assert index.get_document("confluence:101/attachments/draft-plan.txt") is None


def test_max_depth_is_passed_to_the_cli(index, site):
    index.sync(corpus(site, max_depth=2))

    assert site.calls("children")[0][:4] == ["children", "100", "--recursive", "--max-depth"]


def test_a_missing_confluence_command_is_a_listing_failure_not_a_crash(index, tmp_path):
    index.sync(InMemoryCorpus(DOCUMENTS))

    report = index.sync(ConfluenceCorpus(["100"], confluence_bin=str(tmp_path / "nope")))

    assert (report.removed, report.failed) == (0, 1)
    assert "confluence not found" in report.failures["root 100"]


def test_a_hung_confluence_command_fails_after_the_timeout(index, site):
    site.bin.write_text("#!/bin/sh\nexec sleep 5\n")

    report = index.sync(corpus(site, timeout=0.2))

    assert "timed out" in report.failures["root 100"]


def test_roots_are_read_from_a_file_of_ids_and_urls_with_comments(tmp_path):
    roots = tmp_path / "roots.txt"
    roots.write_text(
        "# one root per line\n\n123456789   # ENG home\n"
        "https://wiki.example.test/pages/viewpage.action?pageId=987654321 # Ops\n"
    )

    assert read_roots(roots) == [
        "123456789",
        "https://wiki.example.test/pages/viewpage.action?pageId=987654321",
    ]


def test_a_malformed_listing_is_a_root_failure_not_a_crash(index, site):
    index.sync(corpus(site, "100", "200"))
    site.bin.write_text("#!/bin/sh\necho 'not json'\n")

    report = index.sync(corpus(site, "100", "200"))

    assert (report.removed, report.failed) == (0, 2)
    assert set(report.failures) == {"root 100", "root 200"}
    assert index.get_document("confluence:200") is not None


def unversioned(index, site, gateway):
    """An index holding an unversioned page with an unversioned Attachment, whose Model
    gateway records what it embeds."""
    site.page("300", "Unversioned", space="ENG", version=None, body="Some text.\n")
    site.attach("300", "notes.txt", "Escalate to the DBA on call.", version=None)
    index.sync(corpus(site, "300"))
    gateway.embedded.clear()


@pytest.fixture
def gateway():
    return CountingGateway()


@pytest.fixture
def counted(lit, gateway):
    return KnowledgeIndex(Settings(liteparse_bin=str(lit)), gateway)


def test_an_unversioned_page_is_fetched_but_not_embedded_again_when_its_content_is_unchanged(
    counted, site, gateway
):
    unversioned(counted, site, gateway)
    fetches = len(site.calls("export")) + len(downloads(site))

    report = counted.sync(corpus(site, "300"))

    assert (report.unchanged, report.replaced, report.failed) == (2, 0, 0)
    assert len(site.calls("export")) + len(downloads(site)) == 2 * fetches  # still fetched
    assert gateway.embedded == []
    assert counted.get_document("confluence:300") == "Some text."


def test_an_unversioned_page_whose_content_changed_is_replaced(counted, site, gateway):
    unversioned(counted, site, gateway)
    site.page("300", "Unversioned", space="ENG", version=None, body="Run the QX-5 check.\n")

    report = counted.sync(corpus(site, "300"))

    assert (report.replaced, report.unchanged, report.failed) == (1, 1, 0)
    assert counted.search("QX-5 check")[0].document_id == "confluence:300"
    assert "Some text" not in counted.get_document("confluence:300")


def test_an_unversioned_attachment_whose_content_changed_is_replaced(counted, site, gateway):
    unversioned(counted, site, gateway)
    site.attach("300", "notes.txt", "Escalate to the platform team.", version=None)

    report = counted.sync(corpus(site, "300"))

    assert (report.replaced, report.unchanged, report.failed) == (1, 1, 0)
    assert (
        counted.get_document("confluence:300/attachments/notes.txt")
        == "Escalate to the platform team."
    )


def test_an_unchanged_unversioned_page_listed_under_another_root_is_still_re_added(
    counted, site, gateway
):
    site.page("301", "Unversioned child", parent="300", version=None, body="Child text.\n")
    unversioned(counted, site, gateway)

    report = counted.sync(corpus(site, "301"))

    assert (report.replaced, report.unchanged, report.removed) == (1, 0, 2)
    assert gateway.embedded != []
