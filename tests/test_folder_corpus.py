import os
import shutil

import pytest

import fake_lit
from local_rag import KnowledgeIndex, Settings
from local_rag.corpus import InMemoryCorpus, RawDocument
from local_rag.fakes import FakeModelGateway
from local_rag.folder import FolderCorpus


@pytest.fixture
def lit(tmp_path_factory):
    return fake_lit.install(tmp_path_factory.mktemp("bin"))


@pytest.fixture
def folder(tmp_path):
    root = tmp_path / "docs"
    (root / "dba").mkdir(parents=True)
    (root / "security").mkdir()
    (root / "dba" / "ora-00942.md").write_text(
        "# Fixing ORA-00942\n\nTable or view does not exist. Check the grants.\n")
    (root / "dba" / "failover.pdf").write_bytes(
        fake_lit.pdf("Quarterly failover runbook for the standby database."))
    (root / "security" / "rotation.docx").write_bytes(
        fake_lit.docx("Rotate the ZX-9000 credentials every quarter."))
    return root


@pytest.fixture
def make_index(lit):
    def make(**settings):
        return KnowledgeIndex(Settings(liteparse_bin=str(lit), **settings), FakeModelGateway())
    return make


@pytest.fixture
def index(make_index):
    return make_index()


def test_a_folder_of_markdown_pdf_and_docx_is_synced_and_searched(index, folder):
    report = index.sync(FolderCorpus(folder))

    assert (report.added, report.failed) == (3, 0)
    assert index.search("ORA-00942")[0].document_id == "folder:dba/ora-00942.md"
    assert index.search("failover runbook")[0].document_id == "folder:dba/failover.pdf"
    assert index.search("ZX-9000")[0].document_id == "folder:security/rotation.docx"


def test_identity_is_the_relative_path_and_space_the_top_level_folder(index, folder):
    index.sync(FolderCorpus(folder))

    assert index.list_spaces() == ["dba", "security"]
    found = index.search("ZX-9000")[0]
    assert (found.document_id, found.space) == ("security/rotation.docx", "security")
    assert "ZX-9000" in index.get_document("folder:security/rotation.docx")
    assert {p.space for p in index.search("standby", space="security")} == {"security"}


def test_files_directly_in_the_folder_belong_to_the_folder_named_space(index, folder):
    (folder / "overview.md").write_text("Overview of the estate.\n")

    index.sync(FolderCorpus(folder))

    assert index.search("estate")[0].space == "docs"


def test_version_is_a_content_hash_so_touching_a_file_changes_nothing(index, folder):
    index.sync(FolderCorpus(folder))
    note = folder / "dba" / "ora-00942.md"
    note.write_bytes(note.read_bytes())  # new mtime, same content

    report = index.sync(FolderCorpus(folder))

    assert (report.unchanged, report.replaced) == (3, 0)


def test_a_changed_file_is_replaced_and_a_deleted_file_removed(index, folder):
    index.sync(FolderCorpus(folder))
    (folder / "dba" / "ora-00942.md").write_text("Now about ORA-99999 instead.\n")
    (folder / "security" / "rotation.docx").unlink()

    report = index.sync(FolderCorpus(folder))

    assert (report.replaced, report.removed, report.unchanged) == (1, 1, 1)
    assert index.search("ORA-99999")[0].document_id == "folder:dba/ora-00942.md"
    assert index.get_document("folder:security/rotation.docx") is None


def test_an_html_file_is_indexed_as_markdown_without_scripts_or_styles(index, folder):
    (folder / "dba" / "page.html").write_text(
        "<html><head><title>T</title><style>p {color: red}</style></head><body>"
        "<script>var leaked = 1;</script><h1>Standby setup</h1>"
        "<p>Enable <b>redo transport</b>.</p></body></html>")

    index.sync(FolderCorpus(folder))

    text = index.get_document("folder:dba/page.html")
    assert "# Standby setup" in text and "**redo transport**" in text
    assert "leaked" not in text and "color" not in text


def test_unsupported_and_hidden_files_are_left_out(index, folder):
    (folder / "dba" / "data.bin").write_bytes(b"\x00\x01")
    (folder / ".git").mkdir()
    (folder / ".git" / "notes.md").write_text("hidden")
    (folder / "dba" / ".draft.md").write_text("hidden")

    report = index.sync(FolderCorpus(folder))

    assert (report.added, report.failed) == (3, 0)


def test_a_document_that_fails_extraction_counts_as_failed_and_the_rest_is_indexed(
        index, folder):
    (folder / "dba" / "scan.pdf").write_bytes(fake_lit.pdf("CORRUPT"))

    report = index.sync(FolderCorpus(folder))

    assert (report.added, report.failed) == (3, 1)
    assert "could not read the document" in report.failures["folder:dba/scan.pdf"]
    assert index.get_document("folder:dba/scan.pdf") is None


def test_a_failed_extraction_keeps_the_old_version_searchable(index, folder):
    index.sync(FolderCorpus(folder))
    (folder / "dba" / "failover.pdf").write_bytes(fake_lit.pdf("CORRUPT"))

    report = index.sync(FolderCorpus(folder))

    assert (report.failed, report.replaced) == (1, 0)
    assert index.search("failover runbook")[0].document_id == "folder:dba/failover.pdf"


def test_an_empty_extraction_is_reported_not_indexed(index, folder):
    (folder / "dba" / "scanned.pdf").write_bytes(fake_lit.pdf(""))
    (folder / "dba" / "blank.md").write_text("  \n\n")

    report = index.sync(FolderCorpus(folder))

    assert (report.added, report.failed) == (3, 2)
    assert {"dba/scanned.pdf", "dba/blank.md"} == set(report.failures)
    assert all(reason.startswith("empty extraction") for reason in report.failures.values())
    assert index.get_document("folder:dba/scanned.pdf") is None
    assert index.get_document("folder:dba/blank.md") is None


def test_a_document_that_became_empty_keeps_its_old_version(index, folder):
    index.sync(FolderCorpus(folder))
    (folder / "dba" / "ora-00942.md").write_text("")

    report = index.sync(FolderCorpus(folder))

    assert (report.failed, report.replaced) == (1, 0)
    assert index.search("ORA-00942")[0].document_id == "folder:dba/ora-00942.md"


def test_a_missing_liteparse_fails_only_the_documents_that_need_it(folder, tmp_path):
    index = KnowledgeIndex(
        Settings(liteparse_bin=str(tmp_path / "no-such-lit")), FakeModelGateway())

    report = index.sync(FolderCorpus(folder))

    assert (report.added, report.failed) == (1, 2)
    assert "LiteParse" in report.failures["folder:dba/failover.pdf"]


def test_a_hung_liteparse_fails_the_document_after_the_timeout(make_index, folder, lit):
    lit.write_text("#!/bin/sh\nexec sleep 5\n")
    index = make_index(liteparse_timeout=0.2)

    report = index.sync(FolderCorpus(folder))

    assert report.failed == 2
    assert "timed out" in report.failures["folder:dba/failover.pdf"]


def test_ocr_is_off_for_documents_unless_enabled(make_index, folder, lit, tmp_path):
    seen = tmp_path / "args"
    lit.write_text(f'#!/bin/sh\necho "$@" >> {seen}\necho text\n')

    make_index().sync(FolderCorpus(folder))
    make_index(ocr=True).sync(FolderCorpus(folder))

    calls = seen.read_text().splitlines()
    assert [("--no-ocr" in call) for call in calls] == [True, True, False, False]


def test_an_image_is_read_with_ocr_whatever_the_setting(make_index, tmp_path, lit):
    seen = tmp_path / "args"
    lit.write_text(f'#!/bin/sh\necho "$@" >> {seen}\necho "Whiteboard: restart the ZX-9000"\n')
    shots = tmp_path / "shots"
    shots.mkdir()
    (shots / "whiteboard.png").write_bytes(b"\x89PNG-fake")
    index = make_index()

    report = index.sync(FolderCorpus(tmp_path))

    assert (report.added, report.failed) == (1, 0)
    assert "--no-ocr" not in seen.read_text()
    assert index.search("ZX-9000")[0].document_id == "folder:shots/whiteboard.png"


def test_a_utf8_byte_order_mark_is_not_part_of_the_text(index, folder):
    (folder / "dba" / "bom.md").write_bytes(b"\xef\xbb\xbfHello from Windows")

    index.sync(FolderCorpus(folder))

    assert index.get_document("folder:dba/bom.md") == "Hello from Windows"


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read any file")
def test_an_unreadable_file_counts_as_one_failed_document_and_the_rest_is_synced(
        index, folder):
    locked = folder / "dba" / "locked.md"
    locked.write_text("secret")
    locked.chmod(0)

    report = index.sync(FolderCorpus(folder))

    assert (report.added, report.failed) == (3, 1)
    assert "dba/locked.md" in report.failures


def test_an_unknown_media_type_fails_the_document(index):
    corpus = InMemoryCorpus([RawDocument(
        id="x", title="x", url="", space="s", version="1",
        content=b"\x00", media_type="application/x-mystery")])

    report = index.sync(corpus)

    assert report.failed == 1
    assert "application/x-mystery" in report.failures["folder:x"]


@pytest.mark.skipif(shutil.which("lit") is None, reason="LiteParse (`lit`) is not installed")
def test_a_real_pdf_goes_through_the_real_liteparse(tmp_path):
    (tmp_path / "dba").mkdir()
    (tmp_path / "dba" / "runbook.pdf").write_bytes(_minimal_pdf("Quarterly failover runbook"))
    index = KnowledgeIndex(Settings(), FakeModelGateway())

    report = index.sync(FolderCorpus(tmp_path))

    assert (report.added, report.failed) == (1, 0)
    assert index.search("failover")[0].document_id == "folder:dba/runbook.pdf"


def _minimal_pdf(line: str) -> bytes:
    stream = f"BT /F1 18 Tf 72 700 Td ({line}) Tj ET".encode()
    objects = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R"
        b"/Resources<</Font<</F1 5 0 R>>>>>>",
        b"<</Length %d>>\nstream\n%s\nendstream" % (len(stream), stream),
        b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
    ]
    out, offsets = b"%PDF-1.4\n", []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (number, body)
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    return out + b"trailer<</Size %d/Root 1 0 R>>\nstartxref\n%d\n%%%%EOF" % (
        len(objects) + 1, xref)


def test_a_rebuild_extracts_and_reports_which_documents_failed(make_index, folder):
    (folder / "dba" / "scan.pdf").write_bytes(fake_lit.pdf(""))
    index = make_index()

    report = index.rebuild([FolderCorpus(folder)])

    assert not report.promoted
    assert list(report.failures) == ["dba/scan.pdf"]
    assert report.failures["folder:dba/scan.pdf"].startswith("empty extraction")
    assert index.search("ORA-00942") == []  # nothing went live


def test_a_rebuild_within_the_threshold_serves_extracted_text(make_index, folder):
    (folder / "dba" / "scan.pdf").write_bytes(fake_lit.pdf(""))
    index = make_index(rebuild_failure_threshold=0.5)

    report = index.rebuild([FolderCorpus(folder)])

    assert report.promoted and report.indexed == 3
    assert index.search("ZX-9000")[0].document_id == "folder:security/rotation.docx"
