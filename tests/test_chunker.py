import pytest

from nexus_rag import Settings
from nexus_rag.chunker import Chunker
from nexus_rag.fakes import FakeModelGateway


def words(text: str) -> int:
    return len(text.split())


def sentences(n: int, prefix: str = "s") -> str:
    return " ".join(f"{prefix}{i} alpha beta gamma delta." for i in range(n))  # 5 words each


def test_short_text_is_one_passage():
    chunker = Chunker(words, size=100, overlap=10)

    assert chunker.split("A short note. Nothing more.") == ["A short note. Nothing more."]


def test_empty_text_gives_no_passages():
    assert Chunker(words, size=100, overlap=10).split("  \n ") == []


def test_passages_respect_the_size_measured_by_the_supplied_counter():
    chunker = Chunker(words, size=30, overlap=5)

    passages = chunker.split(sentences(40))

    assert len(passages) > 1
    assert all(words(p) <= 30 for p in passages)


def test_a_larger_size_gives_fewer_passages():
    text = sentences(60)

    small = Chunker(words, size=20, overlap=0).split(text)
    large = Chunker(words, size=80, overlap=0).split(text)

    assert len(large) < len(small)


def test_the_supplied_counter_decides_what_a_token_is():
    text = sentences(40)

    by_word = Chunker(words, size=30, overlap=0).split(text)
    by_char = Chunker(len, size=30, overlap=0).split(text)  # 30 characters, ~1 sentence

    assert len(by_char) > len(by_word) * 3


def test_neighbouring_passages_share_the_configured_overlap():
    text = sentences(40)

    without = Chunker(words, size=30, overlap=0).split(text)
    with_overlap = Chunker(words, size=30, overlap=10).split(text)

    shared = set(with_overlap[0].split()[-10:]) & set(with_overlap[1].split()[:10])
    assert shared, "expected the end of one passage to reappear at the start of the next"
    assert len(with_overlap) >= len(without)


def test_no_text_is_lost():
    text = sentences(40)

    passages = Chunker(words, size=30, overlap=5).split(text)

    for i in range(40):
        assert any(f"s{i} " in p for p in passages)


def test_a_sentence_longer_than_the_size_is_still_split():
    text = " ".join(f"w{i}" for i in range(200)) + "."

    passages = Chunker(words, size=30, overlap=0).split(text)

    assert len(passages) > 1
    assert all(words(p) <= 30 for p in passages)


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError, match="overlap"):
        Chunker(words, size=10, overlap=10)


def test_chunker_takes_size_overlap_and_counter_from_settings_and_gateway():
    gateway = FakeModelGateway()
    chunker = Chunker.from_settings(Settings(chunk_tokens=20, chunk_overlap=0), gateway)

    passages = chunker.split(sentences(40))

    assert len(passages) > 1
    assert all(gateway.count_tokens(p) <= 20 for p in passages)


def test_chunking_neither_needs_nor_touches_global_llamaindex_settings():
    from llama_index.core import Settings as LlamaSettings

    before = LlamaSettings._tokenizer, LlamaSettings._node_parser, LlamaSettings._embed_model

    Chunker(words, size=30, overlap=5).split(sentences(20))

    assert (
        (LlamaSettings._tokenizer, LlamaSettings._node_parser, LlamaSettings._embed_model)
        == before
        == (None, None, None)
    )


def test_every_prose_or_binary_media_type_is_split_as_prose():
    text = sentences(40)
    prose = Chunker(words, size=30, overlap=5).split(text)

    for media_type in ("text/plain", "text/markdown", "text/x-rst", "application/pdf"):
        assert Chunker(words, size=30, overlap=5).split(text, media_type) == prose
