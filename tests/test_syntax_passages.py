"""Python and C code is split along its syntax, each Passage naming the symbol it sits in."""

import dataclasses
import io

from qdrant_client import QdrantClient

from local_rag import Access, KnowledgeIndex
from local_rag.cli import main
from local_rag.corpus import InMemoryCorpus
from local_rag.tokens import count_tokens
from test_code_passages import SMALL, code, indexed, passages
from test_mcp_server import call, serving

BIG = dataclasses.replace(SMALL, code_chunk_tokens=400)
MID = dataclasses.replace(SMALL, code_chunk_tokens=40)

PYTHON = '''\
import os

LIMIT = 10


# Posts entries.
class LedgerClient:
    """Talks to the ledger."""

    def __init__(self, url):
        self.url = url

    @staticmethod
    def balance(account):
        return account * 2

    async def post_entry(self, entry):
        total = entry + LIMIT
        return total


def settle(day):
    return day + 1
'''

C = """\
#include <stdio.h>

struct point {
    int x;
    int y;
};

typedef struct {
    int id;
} record_t;

static int *find_slot(int index)
{
    return 0;
}

int main(void)
{
    printf("hi");
    return 0;
}
"""


def by_symbol(found):
    return {p.symbol: p for p in found}


def test_python_functions_and_classes_are_whole_passages_with_their_symbols():
    found = passages(indexed(PYTHON, MID), "LedgerClient settle import")

    symbols = by_symbol(found)
    assert set(symbols) == {
        None,
        "LedgerClient",
        "LedgerClient.__init__",
        "LedgerClient.balance",
        "LedgerClient.post_entry",
        "settle",
    }
    assert symbols["settle"].text == "def settle(day):\n    return day + 1"
    post = symbols["LedgerClient.post_entry"]
    assert post.text == (
        "    async def post_entry(self, entry):\n"
        "        total = entry + LIMIT\n        return total"
    )
    assert (post.start_line, post.end_line) == (17, 19)


def test_decorators_and_leading_comments_stay_with_their_definition():
    symbols = by_symbol(passages(indexed(PYTHON, MID), "LedgerClient balance"))

    assert symbols["LedgerClient.balance"].text.startswith("    @staticmethod\n    def balance")
    assert symbols["LedgerClient"].text.startswith("# Posts entries.\nclass LedgerClient:")


def test_the_symbol_is_part_of_the_indexed_prefix():
    index = indexed(PYTHON, MID)

    found = index.search("LedgerClient", k=10)

    method = next(p for p in found if p.symbol == "LedgerClient.balance")
    assert "LedgerClient" not in method.text  # found through the prefix alone


def test_a_function_longer_than_the_limit_is_split_along_its_statements():
    body = "".join(f"    total = total + step_{i}(value)\n" for i in range(40))
    text = f"def run(value):\n    total = 0\n{body}    return total\n"

    found = passages(indexed(text, SMALL), "total")

    assert len(found) > 2
    assert {p.symbol for p in found} == {"run"}
    assert all(count_tokens(p.text) <= 60 for p in found)
    assert found[0].text.startswith("def run(value):")
    assert all(p.text.startswith("    total") or p.text.startswith("def ") for p in found)
    assert found[-1].text.endswith("return total")


def test_a_long_class_is_split_between_its_methods():
    methods = "\n".join(f"    def m{i}(self):\n        return {i}\n" for i in range(12))
    text = f"class Big:\n{methods}"

    found = passages(indexed(text, SMALL), "return")

    assert {p.symbol for p in found} >= {f"Big.m{i}" for i in range(12)}
    for p in found:
        if p.symbol != "Big":
            assert p.text.lstrip().startswith("def ")
            assert p.text.rstrip().endswith(f"return {p.symbol[5:]}")


def test_a_python_file_that_does_not_parse_falls_back_to_lines():
    text = "def broken(:\n    return 1\n\ndef fine():\n    return 2\n"

    found = passages(indexed(text, BIG), "return")

    assert [(p.start_line, p.end_line, p.symbol) for p in found] == [(1, 5, None)]


def test_c_functions_structs_and_typedefs_are_whole_passages_with_their_symbols():
    found = passages(indexed(C, BIG, path="src/main.c", media_type="text/x-c"), "int include")

    symbols = by_symbol(found)
    assert set(symbols) == {None, "point", "record_t", "find_slot", "main"}
    assert symbols["find_slot"].text == "static int *find_slot(int index)\n{\n    return 0;\n}"
    assert symbols["main"].text.endswith("    return 0;\n}")
    assert symbols["point"].text.startswith("struct point {")
    assert symbols["record_t"].text.startswith("typedef struct {")
    assert symbols[None].text == "#include <stdio.h>"


def test_a_c_file_that_does_not_parse_falls_back_to_lines():
    text = "int main( {\n    return 0;\n"

    found = passages(indexed(text, BIG, path="a.c", media_type="text/x-c"), "return")

    assert [(p.start_line, p.symbol) for p in found] == [(1, None)]


def test_other_languages_keep_line_based_splitting():
    found = passages(
        indexed(
            "function a() {\n  return 1;\n}\n", BIG, path="a.js", media_type="text/x-javascript"
        ),
        "return",
    )

    assert [(p.start_line, p.symbol) for p in found] == [(1, None)]


def test_the_syntax_split_languages_are_recorded_in_the_index_format():
    status = indexed("a = 1\n").status()

    assert status.format.syntax_languages == ("c", "python")


def test_an_index_that_did_not_record_syntax_languages_is_read_only(monkeypatch):
    from local_rag.index_format import IndexFormat

    original = IndexFormat.to_payload

    def without(self):
        payload = original(self)
        del payload["syntax_languages"]
        return payload

    monkeypatch.setattr(IndexFormat, "to_payload", without)
    index = indexed("a = 1\n")
    monkeypatch.undo()

    status = index.status()
    assert status.access is Access.READ_ONLY
    assert any("syntax" in warning for warning in status.warnings)


def test_adding_a_language_makes_an_index_read_only_until_a_rebuild(monkeypatch):
    from local_rag import languages

    client = QdrantClient(location=":memory:")
    monkeypatch.setattr(languages, "MEDIA_TYPE_LANGUAGES", {"text/x-python": "python"})
    KnowledgeIndex(SMALL, None, client=client).sync(InMemoryCorpus([code("a = 1\n")]))
    monkeypatch.undo()

    status = KnowledgeIndex(SMALL, None, client=client).status()

    assert status.access is Access.READ_ONLY
    assert any("syntax" in warning for warning in status.warnings)


def test_the_cli_and_mcp_headers_show_the_symbol():
    client = QdrantClient(location=":memory:")
    KnowledgeIndex(BIG, None, client=client).sync(InMemoryCorpus([code(PYTHON)]))
    out = io.StringIO()
    environ = {"RAG_RETRIEVAL": "keyword", "RAG_CODE_CHUNK_TOKENS": "400"}
    assert main(["search", "settle"], environ, stdout=out, stderr=io.StringIO(), client=client) == 0
    assert "symbol: settle" in out.getvalue()

    index = indexed(PYTHON, BIG)
    with serving(index) as base_url:
        text = call(base_url, "search_docs", {"query": "settle"})
    assert "symbol: settle" in text
