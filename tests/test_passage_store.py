"""A Passage store reads back what it wrote, and reports a point it cannot read as corrupt."""

import pytest
from qdrant_client import QdrantClient, models

from local_rag import CorruptIndex
from local_rag.chunker import Piece
from local_rag.corpus import CodeFile, Document, TrackedRef
from local_rag.passage_store import Narrowing, Passage, PassageStore, Written
from local_rag.sparse import SparseVector

COLLECTION = "knowledge.gen-0123456789ab"
DOCUMENT = Document(
    id="docs:dba/ora-00942",
    title="Fixing ORA-00942",
    url="https://wiki.example.test/dba/ora-00942",
    space="DBA",
    version="1",
    text="ORA-00942: table or view does not exist.",
    corpus="docs",
    parts=("dba",),
    placement="",
)
WORD = SparseVector([7], [1.0])


@pytest.fixture
def client():
    return QdrantClient(location=":memory:")


@pytest.fixture
def store(client):
    store = PassageStore(client, payload_indexes=False)
    store.create(COLLECTION, dimension=None)
    store.write(COLLECTION, DOCUMENT, [Written(Piece(DOCUMENT.text), WORD)])
    return store


def search(store):
    return store.search(
        COLLECTION, dense=None, sparse=WORD, narrowing=Narrowing(), candidates=10, limit=10
    )


def test_a_written_document_reads_back(store):
    assert [p.document_id for p in search(store)] == [DOCUMENT.id]
    assert store.document_text(COLLECTION, DOCUMENT.id) == DOCUMENT.text
    assert store.spaces(COLLECTION) == ["DBA"]
    assert store.corpora(COLLECTION) == {"docs"}
    assert store.counts(COLLECTION) == (1, 1)
    assert store.stored(COLLECTION, "docs")[DOCUMENT.id].versions == {"1"}


def lose_a_field(client, field):
    """Drop ``field`` from every point, as damage or a hand edit might."""
    client.delete_payload(COLLECTION, keys=[field], points=models.Filter())


@pytest.mark.parametrize(
    "read",
    [
        search,
        lambda store: store.stored(COLLECTION, "docs"),
        lambda store: store.spaces(COLLECTION),
    ],
    ids=["search", "stored", "spaces"],
)
def test_a_point_without_its_document_id_or_space_is_reported_as_corrupt(client, store, read):
    lose_a_field(client, "document_id")
    lose_a_field(client, "space")

    with pytest.raises(CorruptIndex, match=COLLECTION):
        read(store)


def test_a_document_point_without_its_text_is_reported_as_corrupt(client, store):
    lose_a_field(client, "full_text")

    with pytest.raises(CorruptIndex, match=COLLECTION):
        store.document_text(COLLECTION, DOCUMENT.id)


def test_a_collection_that_is_not_a_passage_store_is_reported_as_corrupt(client, store):
    client.create_collection(
        "not-ours", vectors_config=models.VectorParams(size=1, distance=models.Distance.COSINE)
    )

    assert store.dense_size(COLLECTION) is None
    with pytest.raises(CorruptIndex, match="not-ours"):
        store.dense_size("not-ours")


@pytest.mark.parametrize(
    "read",
    [
        lambda store: store.stored("missing", "docs"),
        lambda store: store.spaces("missing"),
    ],
    ids=["stored", "spaces"],
)
def test_a_failing_qdrant_is_not_mistaken_for_a_corrupt_point(store, read):
    with pytest.raises(Exception) as raised:
        read(store)

    assert not isinstance(raised.value, CorruptIndex)


def _passage(**fields):
    return Passage(
        **{
            "document_id": "wiki:10",
            "title": "T",
            "url": "https://w/10",
            "space": "OPS",
            "text": "x",
            "score": 1.0,
            **fields,
        }
    )


def test_a_confluence_pages_citation_is_its_link_page_and_space():
    assert _passage().citation() == (
        ("link", "https://w/10"),
        ("page_id", "wiki:10"),
        ("space", "OPS"),
    )


def test_an_attachments_citation_names_its_page_and_itself():
    cited = _passage(document_id="wiki:10/attachments/a.pdf", parent_id="wiki:10").citation()

    assert cited[1:] == (
        ("page_id", "wiki:10"),
        ("space", "OPS"),
        ("attachment", "wiki:10/attachments/a.pdf"),
    )


def test_a_code_passages_citation_adds_its_ref_and_location():
    code = CodeFile(TrackedRef("platform/pay", "main"), "src/pay.py")
    cited = _passage(
        document_id=code.document_id,
        url="https://g/pay.py",
        code=code,
        start_line=10,
        end_line=20,
        symbol="Pay.run",
    ).citation()

    assert cited == (
        ("link", "https://g/pay.py#L10-20"),
        ("page_id", "platform/pay@main:src/pay.py"),
        ("space", "OPS"),
        ("ref", "main"),
        ("file", "src/pay.py"),
        ("lines", "10-20"),
        ("symbol", "Pay.run"),
    )
