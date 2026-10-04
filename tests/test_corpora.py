"""Named Corpora in one Knowledge index: each owns its Documents, a Sync touches only its own."""

import pytest

from nexus_rag.corpus import Document, InMemoryCorpus
from nexus_rag.declaration import Declaration, DeclarationError


def doc(id, text, version="1"):
    return Document(id, id, f"https://wiki.example.test/{id}", "S", version, text)


def test_a_sync_of_one_corpus_leaves_another_corpuss_documents_searchable(index):
    index.sync(InMemoryCorpus([doc("a1", "alpha ZX-1")], name="a"))
    index.sync(InMemoryCorpus([doc("b1", "bravo QX-2")], name="b"))

    report = index.sync(InMemoryCorpus([doc("a2", "alpha ZX-3")], name="a"))

    assert (report.added, report.removed, report.unchanged) == (1, 1, 0)
    assert [p.document_id for p in index.search("QX-2")][:1] == ["b:b1"]
    assert index.get_document("b:b1") == "bravo QX-2"
    assert index.get_document("a:a1") is None


def test_the_same_id_in_two_corpora_does_not_collide(index):
    index.sync(InMemoryCorpus([doc("page", "from the first")], name="a"))
    index.sync(InMemoryCorpus([doc("page", "from the second")], name="b"))

    assert index.get_document("a:page") == "from the first"
    assert index.get_document("b:page") == "from the second"
    assert {p.corpus for p in index.search("from the", k=10)} == {"a", "b"}


def test_a_sync_counts_only_its_own_corpuss_documents(index):
    index.sync(InMemoryCorpus([doc("x", "same")], name="a"))

    report = index.sync(InMemoryCorpus([doc("x", "same")], name="b"))

    assert (report.added, report.unchanged) == (1, 0)
    assert index.sync(InMemoryCorpus([doc("x", "same")], name="a")).unchanged == 1


def test_status_reports_the_last_sync_of_each_corpus(index):
    index.sync(InMemoryCorpus([doc("a1", "alpha")], name="a"))
    index.sync(InMemoryCorpus([doc("b1", "bravo")], name="b"))
    index.sync(InMemoryCorpus([doc("b1", "bravo")], name="b"))

    corpora = index.status().corpora

    assert set(corpora) == {"a", "b"}
    assert "1 added" in corpora["a"].detail and "1 unchanged" in corpora["b"].detail
    assert corpora["a"].corpus == "a"


def test_a_rebuild_covers_every_corpus_it_is_given(index):
    a = InMemoryCorpus([doc("a1", "alpha ZX-1")], name="a")
    b = InMemoryCorpus([doc("b1", "bravo QX-2")], name="b")
    index.sync(a)
    index.sync(b)

    report = index.rebuild([a, b])

    assert report.promoted and report.indexed == 2
    assert index.get_document("a:a1") and index.get_document("b:b1")


def test_a_name_that_would_break_the_namespace_is_refused():
    with pytest.raises(ValueError, match="cannot name a Corpus"):
        InMemoryCorpus([], name="a:b")


# --- the declaration file ---------------------------------------------------------------


def read(tmp_path, text):
    path = tmp_path / "corpora.toml"
    path.write_text(text)
    return Declaration.read(path)


def test_a_declaration_builds_each_corpus_by_name(tmp_path):
    (tmp_path / "docs").mkdir()
    declaration = read(
        tmp_path,
        """
[corpus.docs]
kind = "folder"
path = "docs"

[corpus.wiki]
kind = "confluence"
roots = ["100", "https://wiki.example.test/pages/viewpage.action?pageId=5"]
max_depth = 2
""",
    )

    assert declaration.names == ["docs", "wiki"]
    assert [c.name for c in declaration.corpora()] == ["docs", "wiki"]


@pytest.mark.parametrize(
    "text, message",
    [
        ('[corpus.a]\nkind = "gopher"', "kind must be one of"),
        ('[corpus.a]\nkind = "folder"\npath = "x"\nroots = []', "unknown roots"),
        ('[corpus.a]\nkind = "confluence"\nroots = "100"', "list of non-empty strings"),
        ('[corpus.a]\nkind = "confluence"\nroots = ["1"]\nmax_depth = -1', "max_depth"),
        ('[corpus."a:b"]\nkind = "folder"\npath = "x"', "cannot name a Corpus"),
        ('[other.a]\nkind = "folder"', "unknown top-level"),
        ('[corpus.a]\nkind = "gitlab"', "repositories must be"),
        (
            '[corpus.a]\nkind = "gitlab"\n[[corpus.a.repositories]]\npath = "g/p"\nrefs = ["main"]',
            "GITLAB_URL and GITLAB_TOKEN must be set",
        ),
        (
            '[corpus.a]\nkind = "gitlab"\n[[corpus.a.repositories]]\npath = "g/p"\nrefs = []',
            "at least one ref",
        ),
        (
            '[corpus.a]\nkind = "gitlab"\n[[corpus.a.repositories]]\npath = "g/p"\n'
            'refs = ["main", "main"]',
            "lists a ref twice",
        ),
        (
            '[corpus.a]\nkind = "gitlab"\n[[corpus.a.repositories]]\npath = "g/p"\nrefs = ["m"]\n'
            'branch = "x"',
            "unknown branch",
        ),
    ],
)
def test_a_wrong_declaration_is_rejected_when_it_is_read(tmp_path, text, message):
    with pytest.raises(DeclarationError, match=message):
        declaration = read(tmp_path, text)
        declaration.corpora()  # a wrong value is found at the latest when it is built


def test_a_folder_that_is_not_there_is_not_an_empty_corpus(tmp_path):
    declaration = read(tmp_path, '[corpus.docs]\nkind = "folder"\npath = "gone"')

    with pytest.raises(DeclarationError, match="is not a folder"):
        declaration.corpus("docs")


def test_an_undeclared_corpus_is_named_in_the_error(tmp_path):
    declaration = read(tmp_path, '[corpus.wiki]\nkind = "confluence"\nroots = ["1"]')

    with pytest.raises(DeclarationError, match=r"'nope'.*wiki"):
        declaration.corpus("nope")


def test_a_rebuild_leaves_out_a_corpus_it_is_not_given(index):
    a = InMemoryCorpus([doc("a1", "alpha ZX-1")], name="a")
    index.sync(a)
    index.sync(InMemoryCorpus([doc("b1", "bravo QX-2")], name="b"))

    report = index.rebuild([a])

    assert report.promoted and report.indexed == 1
    assert index.get_document("b:b1") is None
    assert index.get_document("a:a1")


def test_forgetting_a_corpus_removes_its_documents_and_counts_them(index):
    index.sync(InMemoryCorpus([doc("a1", "alpha ZX-1"), doc("a2", "alpha ZX-2")], name="a"))
    index.sync(InMemoryCorpus([doc("b1", "bravo QX-2")], name="b"))

    removed = index.forget("a")

    assert removed == 2
    assert index.get_document("a:a1") is None
    assert index.get_document("b:b1") == "bravo QX-2"
    assert not any(p.corpus == "a" for p in index.search("alpha ZX-1", k=10))
    assert index.status().documents == 1
    assert index.forget("a") == 0


def test_forgetting_before_anything_is_indexed_removes_nothing(index):
    assert index.forget("a") == 0


def test_documents_of_an_undeclared_corpus_stay_searchable_and_status_names_it(index):
    index.sync(InMemoryCorpus([doc("a1", "alpha ZX-1")], name="a"))
    index.sync(InMemoryCorpus([doc("b1", "bravo QX-2")], name="b"))

    status = index.status(declared=["a"])

    assert any("'b'" in w and "undeclared" in w and "forget" in w for w in status.warnings)
    assert not any("'a'" in w for w in status.warnings)
    assert index.get_document("b:b1") == "bravo QX-2"
    assert not any("undeclared" in w for w in index.status(declared=["a", "b"]).warnings)
    assert not any("undeclared" in w for w in index.status().warnings)


def test_forget_is_refused_while_a_sync_of_that_corpus_runs_and_removes_nothing(index):
    from nexus_rag import SyncInProgress

    index.sync(InMemoryCorpus([doc("a1", "alpha ZX-1")], name="a"))
    refused = []

    class ForgettingCorpus(InMemoryCorpus):
        def fetch(self, document_id):
            with pytest.raises(SyncInProgress):
                index.forget("a")
            refused.append(document_id)
            return super().fetch(document_id)

    index.sync(ForgettingCorpus([doc("a1", "alpha ZX-1", version="2")], name="a"))

    assert refused
    assert index.get_document("a:a1") == "alpha ZX-1"
    assert index.forget("a") == 1


def test_forget_is_refused_while_a_rebuild_runs(index):
    from nexus_rag import SyncInProgress

    index.sync(InMemoryCorpus([doc("a1", "alpha ZX-1")], name="a"))
    refused = []

    class ForgettingCorpus(InMemoryCorpus):
        def fetch(self, document_id):
            with pytest.raises(SyncInProgress):
                index.forget("a")
            refused.append(document_id)
            return super().fetch(document_id)

    index.rebuild([ForgettingCorpus([doc("a1", "alpha ZX-1")], name="a")])

    assert refused
