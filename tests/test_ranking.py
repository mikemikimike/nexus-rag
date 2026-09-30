from dataclasses import replace

from local_rag.index import Passage
from local_rag.ranking import cap_per_document

BASE = Passage(document_id="", title="", url="", space="S", text="", score=0.0)


def passages(*ids):
    return [replace(BASE, document_id=i, text=f"{n}") for n, i in enumerate(ids)]


def test_cap_keeps_the_best_ranked_passages_of_each_document_in_order():
    ranked = passages("a", "a", "b", "a", "b", "c")

    capped = cap_per_document(ranked, 2)

    assert [(p.document_id, p.text) for p in capped] == [
        ("a", "0"), ("a", "1"), ("b", "2"), ("b", "4"), ("c", "5")]


def test_cap_of_one_returns_one_passage_per_document():
    capped = cap_per_document(passages("a", "b", "a", "b"), 1)

    assert [p.document_id for p in capped] == ["a", "b"]
