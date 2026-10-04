"""The names of the collections and aliases one Knowledge index keeps in Qdrant."""

from nexus_rag.index_names import IndexNames


def test_every_collection_an_index_makes_is_named_after_it():
    names = IndexNames("knowledge")

    made = [
        names.plain,
        names.registry,
        names.last_sync,
        names.lock,
        names.promote_lock,
        names.corpus_lock("docs"),
        names.fresh_generation(),
    ]

    assert all(names.owns(collection) for collection in made)


def test_collections_that_only_look_like_an_index_are_not_its():
    names = IndexNames("knowledge")

    lookalikes = [
        "knowledge.previous-backup",
        "knowledge_old",
        "knowledge-2024",
        "knowledge.generations-old",
        "knowledge.gen-short",
        "knowledge.sync-lock.",
        "knowledge.live",
        "knowledge.previous",
        "knowledge2.generations",
        "other.gen-0123456789ab",
    ]

    assert [name for name in lookalikes if names.owns(name)] == []


def test_a_corpus_lock_is_told_from_the_index_wide_lock():
    names = IndexNames("knowledge")

    assert names.corpus_of_lock(names.corpus_lock("docs")) == "docs"
    assert names.corpus_of_lock(names.lock) is None
    assert names.corpus_of_lock("other.sync-lock.docs") is None


def test_a_fresh_generation_is_never_the_same_twice():
    names = IndexNames("knowledge")

    assert names.fresh_generation() != names.fresh_generation()
