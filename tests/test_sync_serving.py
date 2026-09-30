from dataclasses import replace

from local_rag.corpus import InMemoryCorpus


def test_unchanged_documents_stay_unchanged_with_full_text_stored(index, corpus):
    index.sync(corpus)

    report = index.sync(corpus)

    assert (report.unchanged, report.added, report.replaced) == (3, 0, 0)


def test_replaced_document_serves_its_new_full_text(index, corpus):
    index.sync(corpus)
    changed = replace(corpus.fetch("dba/ora-00942"), version="2", text="Brand new body.")

    report = index.sync(InMemoryCorpus([changed, corpus.fetch("dba/ora-12154"),
                                        corpus.fetch("garage/battery")]))

    assert report.replaced == 1
    assert index.get_document("dba/ora-00942") == "Brand new body."


def test_removed_document_and_space_disappear_from_serving(index, corpus):
    index.sync(corpus)

    report = index.sync(InMemoryCorpus([corpus.fetch("dba/ora-00942"),
                                        corpus.fetch("dba/ora-12154")]))

    assert report.removed == 1
    assert index.get_document("garage/battery") is None
    assert index.list_spaces() == ["DBA"]
