from local_rag import KnowledgeIndex, Settings
from local_rag.fakes import FakeModelGateway


def test_get_document_returns_the_full_text_by_document_id(index, corpus):
    index.sync(corpus)

    assert index.get_document("default:garage/battery") == (
        "When the car will not start, the battery is dead more often than not."
    )


def test_get_document_returns_none_for_an_unknown_document_id(index, corpus):
    index.sync(corpus)

    assert index.get_document("default:nope/missing") is None


def test_get_document_on_an_empty_index_returns_none(index):
    assert index.get_document("default:dba/ora-00942") is None


def test_list_spaces_lists_exactly_the_indexed_spaces(index, corpus):
    assert index.list_spaces() == []

    index.sync(corpus)

    assert index.list_spaces() == ["DBA", "GARAGE"]


def test_space_filter_returns_only_passages_from_that_space(index, corpus):
    index.sync(corpus)

    unfiltered = index.search("check", k=10)
    filtered = index.search("check", k=10, space="GARAGE")

    assert {p.space for p in unfiltered} == {"DBA", "GARAGE"}
    assert filtered and {p.space for p in filtered} == {"GARAGE"}


def test_space_filter_for_an_unindexed_space_returns_nothing(index, corpus):
    index.sync(corpus)

    assert index.search("check", space="NOPE") == []


class SpyGateway(FakeModelGateway):
    """A fake gateway whose rerank is scripted and recorded."""

    def __init__(self, rerank_fn=None, **kwargs):
        super().__init__(**kwargs)
        self.rerank_calls = []
        self._rerank_fn = rerank_fn

    def rerank(self, query, texts):
        self.rerank_calls.append((query, list(texts)))
        if self._rerank_fn is None:
            return super().rerank(query, texts)
        return self._rerank_fn(query, texts)


def prefer_battery(query, texts):
    return [1.0 if "battery" in text else 0.0 for text in texts]


def broken(query, texts):
    raise ConnectionError("gateway unreachable")


def test_rerank_is_off_by_default(corpus):
    gateway = SpyGateway(prefer_battery)
    index = KnowledgeIndex(Settings(), gateway)
    index.sync(corpus)

    index.search("check")

    assert gateway.rerank_calls == []


def test_enabled_rerank_reorders_results_by_the_gateway_scores(corpus):
    fused = KnowledgeIndex(Settings(), FakeModelGateway())
    fused.sync(corpus)
    fused_ids = [p.document_id for p in fused.search("check", k=10)]
    assert fused_ids[0] != "garage/battery"

    gateway = SpyGateway(prefer_battery)
    index = KnowledgeIndex(Settings(rerank=True), gateway)
    index.sync(corpus)
    passages = index.search("check", k=10)

    assert passages[0].document_id == "default:garage/battery"
    assert passages[0].score == 1.0
    assert sorted(p.document_id for p in passages) == sorted(fused_ids)


def test_rerank_failure_falls_back_to_the_fused_order(corpus):
    fused = KnowledgeIndex(Settings(), FakeModelGateway())
    fused.sync(corpus)
    expected = fused.search("check", k=10)

    gateway = SpyGateway(broken)
    index = KnowledgeIndex(Settings(rerank=True), gateway)
    index.sync(corpus)

    assert index.search("check", k=10) == expected
    assert len(gateway.rerank_calls) == 1


def test_fake_gateway_rerank_scores_by_query_term_overlap():
    gateway = FakeModelGateway()

    scores = gateway.rerank(
        "battery dead", ["the battery is dead", "the battery is fine", "unrelated words"]
    )

    assert scores[0] > scores[1] > scores[2] == 0.0
