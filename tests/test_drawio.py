import base64
import urllib.parse
import zlib

import pytest

from nexus_rag import KnowledgeIndex, Settings
from nexus_rag.fakes import FakeModelGateway
from nexus_rag.folder import FolderCorpus

MODEL = """<mxGraphModel><root>
  <mxCell id="0"/><mxCell id="1" parent="0"/>
  <mxCell id="2" value="Primary database" style="rounded=1;" vertex="1" parent="1"/>
  <mxCell id="3" value="Standby ORA-00942" style="rounded=1;" vertex="1" parent="1"/>
  <mxCell id="4" value="redo shipping" edge="1" source="2" target="3" parent="1"/>
  <mxCell id="5" value="" vertex="1" parent="1"/>
</root></mxGraphModel>"""


def compressed(model: str) -> str:
    """How draw.io stores a page by default: URL-encoded, raw-deflated, base64."""
    deflate = zlib.compressobj(wbits=-15)
    raw = deflate.compress(urllib.parse.quote(model).encode()) + deflate.flush()
    return base64.b64encode(raw).decode()


def file(*pages: tuple[str, str]) -> str:
    diagrams = "".join(
        f'<diagram id="d{n}" name="{name}">{body}</diagram>' for n, (name, body) in enumerate(pages)
    )
    return f'<mxfile host="app.diagrams.net">{diagrams}</mxfile>'


@pytest.fixture
def index():
    return KnowledgeIndex(Settings(), FakeModelGateway())


def sync(index, tmp_path, name, content):
    (tmp_path / "dba").mkdir(exist_ok=True)
    (tmp_path / "dba" / name).write_text(content)
    return index.sync(FolderCorpus(tmp_path))


def test_the_labels_of_a_drawio_diagram_are_searchable(index, tmp_path):
    report = sync(index, tmp_path, "topology.drawio", file(("Overview", MODEL)))

    assert (report.added, report.failed) == (1, 0)
    text = index.get_document("folder:dba/topology.drawio")
    assert "Primary database" in text and "redo shipping" in text
    assert index.search("ORA-00942")[0].document_id == "folder:dba/topology.drawio"


def test_a_compressed_diagram_is_read_too(index, tmp_path):
    sync(index, tmp_path, "topology.drawio", file(("Overview", compressed(MODEL))))

    assert "Standby ORA-00942" in index.get_document("folder:dba/topology.drawio")


def test_every_page_is_read_under_its_name(index, tmp_path):
    other = MODEL.replace("Primary database", "Load balancer")
    sync(
        index,
        tmp_path,
        "topology.drawio",
        file(("Overview", MODEL), ("Network", compressed(other))),
    )

    text = index.get_document("folder:dba/topology.drawio")
    assert "Overview" in text and "Network" in text
    assert "Primary database" in text and "Load balancer" in text


def test_html_labels_and_wrapped_objects_are_reduced_to_their_text(index, tmp_path):
    model = """<mxGraphModel><root><mxCell id="0"/><mxCell id="1" parent="0"/>
      <mxCell id="2" value="&lt;b&gt;Gateway&lt;/b&gt;&lt;br&gt;port 4000" style="html=1;"
              vertex="1" parent="1"/>
      <object label="Vault &lt;i&gt;cluster&lt;/i&gt;" id="3"><mxCell vertex="1" parent="1"/></object>
      <UserObject label="Bastion" id="4"><mxCell vertex="1" parent="1"/></UserObject>
    </root></mxGraphModel>"""
    sync(index, tmp_path, "gw.drawio", file(("Page-1", model)))

    text = index.get_document("folder:dba/gw.drawio")
    assert "Gateway" in text and "port 4000" in text
    assert "Vault cluster" in text and "Bastion" in text
    assert "<b>" not in text and "<i>" not in text


def test_a_bare_model_without_the_file_wrapper_is_read(index, tmp_path):
    sync(index, tmp_path, "bare.drawio", MODEL)

    assert "Primary database" in index.get_document("folder:dba/bare.drawio")


def test_a_diagram_with_no_labels_is_an_empty_extraction(index, tmp_path):
    empty = '<mxGraphModel><root><mxCell id="0"/><mxCell id="1" parent="0"/></root></mxGraphModel>'

    report = sync(index, tmp_path, "blank.drawio", file(("Page-1", empty)))

    assert (report.added, report.failed) == (0, 1)
    assert report.failures["folder:dba/blank.drawio"].startswith("empty extraction")


def test_a_malformed_diagram_fails_only_that_document(index, tmp_path):
    (tmp_path / "dba").mkdir()
    (tmp_path / "dba" / "good.md").write_text("Fine.\n")

    report = sync(index, tmp_path, "broken.drawio", "<mxfile><diagram>not base64 !!")

    assert (report.added, report.failed) == (1, 1)
    assert "drawio" in report.failures["folder:dba/broken.drawio"].lower()
