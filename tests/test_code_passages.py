"""Code is split into Passages by lines, each knowing its line range."""

import dataclasses
import io
import itertools

import pytest
from qdrant_client import QdrantClient

from nexus_rag import Access, KnowledgeIndex, Settings
from nexus_rag.chunker import Chunker
from nexus_rag.cli import main
from nexus_rag.corpus import CodeFile, InMemoryCorpus, RawDocument, TrackedRef
from nexus_rag.environment import SettingsError, settings_from_environment
from nexus_rag.settings import RetrievalMode
from nexus_rag.tokens import count_tokens
from test_mcp_server import call, serving

URL = "https://gitlab.example.test/platform/payments/-/blob/main/src/ledger.py"
SMALL = Settings(
    retrieval=RetrievalMode.KEYWORD,
    code_chunk_tokens=60,
    max_passages_per_document=1000,
    hybrid_candidates=200,
)


def functions(count: int) -> str:
    """A Python file of ``count`` three-line functions separated by blank lines."""
    return "\n\n".join(f"def step_{i}(value):\n    return value + {i}\n" for i in range(count))


def code(
    text: str, *, path: str = "src/ledger.py", media_type: str = "text/x-python", eol: str = "\n"
) -> RawDocument:
    return RawDocument(
        id=f"platform/payments@main:{path}",
        title=f"payments/{path}",
        url=URL,
        space="platform",
        version="1",
        content=text.replace("\n", eol).encode(),
        media_type=media_type,
        code=CodeFile(TrackedRef("platform/payments", "main"), path),
    )


def indexed(text: str, settings: Settings = SMALL, **kwargs) -> KnowledgeIndex:
    index = KnowledgeIndex(settings, None)
    index.sync(InMemoryCorpus([code(text, **kwargs)]))
    return index


def passages(index: KnowledgeIndex, query: str = "step"):
    return sorted(index.search(query, k=100), key=lambda p: p.start_line)


def test_every_code_passage_knows_the_lines_it_covers():
    text = functions(30)
    lines = text.split("\n")

    found = passages(indexed(text))

    assert len(found) > 3
    for passage in found:
        assert passage.start_line >= 1
        assert passage.text == "\n".join(lines[passage.start_line - 1 : passage.end_line])


def test_code_passages_cover_the_whole_file_with_a_small_overlap():
    text = functions(30)
    found = passages(indexed(text, media_type="text/x-ruby"))

    covered = set()
    for passage in found:
        covered.update(range(passage.start_line, passage.end_line + 1))
    nonblank = {n for n, line in enumerate(text.split("\n"), 1) if line.strip()}
    assert nonblank <= covered
    assert any(a.end_line >= b.start_line for a, b in itertools.pairwise(found))


def test_a_long_file_is_split_within_the_code_size_limit():
    found = passages(indexed(functions(60)))

    assert len(found) > 3
    assert all(count_tokens(p.text) <= 60 for p in found)


def test_passages_break_at_blank_lines_not_inside_a_function():
    found = passages(indexed(functions(30)))

    for passage in found:
        assert passage.text.startswith("def ") or passage.text.startswith("    return")
        assert passage.text.rstrip().split("\n")[-1].startswith("    return")


def test_indentation_survives_and_the_first_line_is_not_stripped():
    text = "    indented_first = 1\n\n\n\n    indented_last = 2\n"

    (only,) = passages(indexed(text), "indented_first")

    assert only.text == text.rstrip("\n")
    assert (only.start_line, only.end_line) == (1, 5)


def test_crlf_files_count_lines_the_same_way():
    found = passages(indexed(functions(30), eol="\r\n"))

    assert found[0].start_line == 1 and "\r" not in found[0].text
    assert max(p.end_line for p in found) == len(functions(30).rstrip("\n").split("\n"))


def test_a_single_line_longer_than_the_limit_is_split_within_it():
    text = "data = [" + ", ".join(str(n) for n in range(400)) + "]\n"

    found = passages(indexed(text), "platform")  # in every Passage's prefix

    assert len(found) > 1
    assert all(count_tokens(p.text) <= 60 for p in found)
    assert {(p.start_line, p.end_line) for p in found} == {(1, 1)}


def test_prose_passages_have_no_line_range(gateway, corpus):
    index = KnowledgeIndex(Settings(), gateway)
    index.sync(corpus)

    (first, *_) = index.search("ORA-00942")

    assert (first.start_line, first.end_line, first.symbol) == (None, None, None)
    assert first.link == first.url


def test_markdown_in_a_code_repository_is_still_prose():
    index = KnowledgeIndex(SMALL, None)
    index.sync(
        InMemoryCorpus(
            [code("# Title\n\nSome words here.\n", path="README.md", media_type="text/markdown")]
        )
    )

    (found, *_) = index.search("words")

    assert found.start_line is None


def test_the_indexed_text_starts_with_the_repository_and_path():
    index = indexed("x = 1\n", path="src/ledger.py")

    assert index.search("platform payments")  # in no title, no text: only in the prefix
    assert index.search("ledger")


def test_a_passage_links_to_its_lines():
    (passage,) = indexed("alpha = 1\nbeta = 2\n").search("alpha")
    assert passage.link == f"{URL}#L1-2"

    (single,) = indexed("only = 1\n").search("only")
    assert single.link == f"{URL}#L1"


def test_the_chunker_exposes_pieces_with_lines_and_a_symbol_slot():
    chunker = Chunker(count_tokens, size=400, overlap=60, code_size=60)

    pieces = chunker.pieces("a = 1\n\nb = 2\n", "text/x-python")

    assert [(p.text, p.start_line, p.end_line, p.symbol) for p in pieces] == [
        ("a = 1\n\nb = 2", 1, 3, None)
    ]
    assert chunker.pieces("Hello there.", "text/plain")[0].start_line is None


def test_the_code_size_limit_is_read_from_the_environment():
    assert Settings().code_chunk_tokens == 800
    given = {"RAG_RETRIEVAL": "keyword", "RAG_CODE_CHUNK_TOKENS": "300"}
    assert settings_from_environment(given).code_chunk_tokens == 300


@pytest.mark.parametrize("value", ["0", "many"])
def test_an_unusable_code_size_limit_is_a_settings_error(value):
    with pytest.raises(SettingsError, match="code_chunk_tokens|RAG_CODE_CHUNK_TOKENS"):
        settings_from_environment({"RAG_RETRIEVAL": "keyword", "RAG_CODE_CHUNK_TOKENS": value})


def test_a_changed_code_size_limit_makes_the_index_read_only():
    client = QdrantClient(location=":memory:")
    KnowledgeIndex(SMALL, None, client=client).sync(InMemoryCorpus([code("a = 1\n")]))

    other = KnowledgeIndex(dataclasses.replace(SMALL, code_chunk_tokens=90), None, client=client)

    status = other.status()
    assert status.access is Access.READ_ONLY
    assert any("code" in warning for warning in status.warnings)


def test_the_code_size_limit_and_splitting_method_are_recorded():
    status = indexed("a = 1\n").status()

    assert status.format.code_chunk_tokens == 60
    assert status.format.code_splitting == "lines"


def test_an_index_that_did_not_record_code_splitting_is_read_only(monkeypatch):
    from nexus_rag.index_format import IndexFormat

    original = IndexFormat.to_payload

    def without(self):
        payload = original(self)
        del payload["code_chunk_tokens"], payload["code_splitting"]
        return payload

    monkeypatch.setattr(IndexFormat, "to_payload", without)
    index = indexed("a = 1\n")
    monkeypatch.undo()

    assert index.status().access is Access.READ_ONLY


def search_cli(text: str, query: str) -> str:
    client = QdrantClient(location=":memory:")
    KnowledgeIndex(SMALL, None, client=client).sync(InMemoryCorpus([code(text)]))
    out = io.StringIO()
    environ = {"RAG_RETRIEVAL": "keyword", "RAG_CODE_CHUNK_TOKENS": "60"}
    assert main(["search", query], environ, stdout=out, stderr=io.StringIO(), client=client) == 0
    return out.getvalue()


def test_the_cli_shows_path_ref_and_line_range_with_an_anchored_link():
    out = search_cli("def ledger():\n    return 1\n", "ledger")

    assert "src/ledger.py" in out
    assert "ref: main" in out
    assert "lines: 1-2" in out
    assert f"link: {URL}#L1-2" in out
    assert "| def ledger():" in out


def test_the_mcp_result_header_shows_path_ref_and_line_range():
    index = indexed("def ledger():\n    return 1\n")
    with serving(index) as base_url:
        text = call(base_url, "search_docs", {"query": "ledger"})

    assert f"{URL}#L1-2" in text
    assert "ref: main" in text and "lines: 1-2" in text and "src/ledger.py" in text
