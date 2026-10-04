import pytest

from conftest import DOCUMENTS
from nexus_rag import Access, IndexFormat, KnowledgeIndex, Settings
from nexus_rag.chunker import Chunker
from nexus_rag.corpus import InMemoryCorpus
from nexus_rag.index_format import TOKEN_COUNTER
from nexus_rag.settings import RetrievalMode
from nexus_rag.tokens import count_tokens

CJK = "这是一个测试句子" * 40  # 320 characters, no separators
HEX = "ab12" * 1250  # 5,000 characters in one run
TEXTS = {"cjk": CJK, "hex": HEX, "both": f"{CJK}\n{HEX}\n"}


def test_words_and_punctuation_count_one_each():
    assert count_tokens("table missing.") == 3
    assert count_tokens("") == 0


@pytest.mark.parametrize(
    "text",
    [
        "这是一个测试句子",  # Chinese
        "これはテストです",  # Japanese, kana and kanji
        "이것은시험입니다",  # Korean
        "นี่คือการทดสอบ",  # Thai: no separators either
        "ﾃｽﾄﾃｽﾄﾃｽﾄ",  # halfwidth Katakana
        "ㅌㅔㅅㅡㅌㅡㅌㅔㅅㅡㅌㅡ",  # Hangul compatibility Jamo
    ],
)
def test_each_character_of_a_script_without_word_separators_counts(text):
    assert count_tokens(text) >= len(text) - 1


def test_a_cjk_run_is_not_one_token():
    assert count_tokens(CJK) >= len(CJK)


def test_a_long_run_of_letters_and_digits_counts_by_its_length():
    assert count_tokens(HEX) >= len(HEX) // 3


def test_a_long_word_counts_as_several_tokens_but_an_ordinary_one_as_one():
    assert count_tokens("internationalization") > 1
    assert count_tokens("table missing configuration") == 3


def test_latin_words_beside_cjk_count_separately():
    assert count_tokens("hello世界") == 3


@pytest.mark.parametrize("text", TEXTS.values(), ids=TEXTS)
def test_prose_passages_of_unseparated_text_stay_bounded_in_characters(text):
    limit = 50
    chunker = Chunker(count_tokens, size=limit, overlap=0)

    passages = chunker.split(text)

    assert len(passages) > 1
    assert all(len(passage) <= limit * 3 for passage in passages)


@pytest.mark.parametrize("text", TEXTS.values(), ids=TEXTS)
def test_code_passages_of_unseparated_text_stay_bounded_in_characters(text):
    chunker = Chunker(count_tokens, size=400, overlap=0, code_size=50)

    passages = chunker.split(text, "text/x-python")

    assert len(passages) > 1
    assert all(len(passage) <= 50 * 3 for passage in passages)


def test_an_index_built_with_another_token_counter_is_read_only_until_a_rebuild(
    monkeypatch, client
):
    corpus = InMemoryCorpus(DOCUMENTS)
    settings = Settings(retrieval=RetrievalMode.KEYWORD)
    original = IndexFormat.to_payload

    def without(self):  # what the version that counted a run as one token wrote
        payload = original(self)
        del payload["token_counter"]
        return payload

    monkeypatch.setattr(IndexFormat, "to_payload", without)
    KnowledgeIndex(settings, None, client=client).sync(corpus)
    monkeypatch.undo()
    upgraded = KnowledgeIndex(settings, None, client=client)

    status = upgraded.status()

    assert status.access is Access.READ_ONLY
    assert any("token counter" in warning for warning in status.warnings)
    assert upgraded.rebuild([corpus]).promoted
    assert upgraded.status().access is Access.READ_WRITE
    assert upgraded.status().format.token_counter == TOKEN_COUNTER
