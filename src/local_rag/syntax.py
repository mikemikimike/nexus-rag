"""Splits Python and C code along its syntax (tree-sitter), naming the symbol each piece sits in.

Whole functions, classes and structs stay in one piece where they fit the size limit. A
definition that does not is split along its own children (a class between its methods, a
function between its statements); what cannot be split further is cut by lines. Each piece
of a definition names it (``LedgerClient.post_entry``). Source that does not parse is left to
the line splitter.
"""

import dataclasses
import threading
from collections.abc import Callable
from dataclasses import dataclass

import tree_sitter_c
import tree_sitter_python
from tree_sitter import Language, Node, Parser

from local_rag.languages import MEDIA_TYPE_LANGUAGES

_GRAMMARS = {"python": tree_sitter_python.language, "c": tree_sitter_c.language}
_LANGUAGES: dict[str, Language] = {}
_local = threading.local()  # a Parser is not thread-safe; Syncs may run concurrently


def _parser(language: str) -> Parser:
    parsers = _local.__dict__.setdefault("parsers", {})
    if language not in parsers:
        if language not in _LANGUAGES:
            _LANGUAGES[language] = Language(_GRAMMARS[language]())
        parsers[language] = Parser(_LANGUAGES[language])
    return parsers[language]


# Node types that define a symbol, and the type that holds the children of one that is too big.
_DEFINITIONS = {
    "python": {"function_definition", "class_definition"},
    "c": {
        "function_definition",
        "struct_specifier",
        "union_specifier",
        "enum_specifier",
        "type_definition",
    },
}


@dataclass
class _Item:
    start: int  # 0-based rows, inclusive
    end: int
    symbol: str | None
    pieces: list | None = None  # already cut, when it did not fit and cannot be split further


class SyntaxSplitter:
    """Splits one language's code. ``fallback`` splits (Piece-wise) what is not parsed."""

    def __init__(self, language: str, count_tokens: Callable[[str], int], *, size: int, fallback):
        self._language = language
        self._count = count_tokens
        self._size = size
        self._fallback = fallback

    def split(self, text: str) -> list:
        try:
            root = _parser(self._language).parse(text.encode("utf-8")).root_node
            if root.has_error:
                return self._fallback.split(text)
            lines = text.split("\n")
            if lines and lines[-1] == "":
                lines.pop()
            if not lines:
                return []
            items = self._items(lines, root, 0, len(lines) - 1, None)
            return self._pack(lines, items)
        except Exception:  # a file the parser cannot handle must not fail its Document
            return self._fallback.split(text)

    # -- building items ---------------------------------------------------------------

    def _items(
        self,
        lines,
        node: Node | None,
        start: int,
        end: int,
        symbol: str | None,
        children: list[Node] | None = None,
    ) -> list[_Item]:
        """The items covering rows start..end, one per child of the container (plus a head
        before the first child). Items that do not fit are expanded in place."""
        if children is None:
            children = list(node.named_children)
        starts = self._starts(children)
        items = []
        if not starts or starts[0][0] > start:
            items.append(_Item(start, (starts[0][0] - 1) if starts else end, symbol))
        for index, (first, child) in enumerate(starts):
            last = starts[index + 1][0] - 1 if index + 1 < len(starts) else end
            items += self._expand(lines, _Item(first, last, self._symbol(child, symbol)), child)
        if items and items[0].start > start:
            items[0].start = start
        return items

    def _starts(self, children: list[Node]) -> list[tuple[int, Node]]:
        """(first row, node) per child; comments directly above a child join it, and a
        child that starts on a row an earlier one ended on (a trailing comment) is absorbed."""
        starts, pending, covered = [], None, -1
        for child in children:
            first, last = child.start_point.row, _last_row(child)
            if pending is None and first <= covered:
                continue
            if pending is not None and first > covered + 1:
                starts.append((pending, None))  # a comment apart from what follows
                pending = None
            if child.type == "comment":
                pending = first if pending is None else pending
            else:
                starts.append((first if pending is None else pending, child))
                pending = None
            covered = max(covered, last)
        if pending is not None:
            starts.append((pending, None))
        return starts

    def _symbol(self, node: Node | None, parent: str | None) -> str | None:
        if node is None:
            return parent
        target = (
            node.child_by_field_name("definition") if node.type == "decorated_definition" else node
        )
        if target is None or target.type not in _DEFINITIONS[self._language]:
            return parent
        name = self._name(target)
        if name is None:
            return parent
        return name if parent is None else f"{parent}.{name}"

    def _name(self, node: Node) -> str | None:
        if node.type == "function_definition" and self._language == "c":
            declarator = node.child_by_field_name("declarator")
            while declarator is not None and declarator.type != "identifier":
                declarator = declarator.child_by_field_name("declarator") or _first_identifier(
                    declarator
                )
            return None if declarator is None else declarator.text.decode("utf-8")
        if node.type == "type_definition":
            declarator = node.child_by_field_name("declarator")
            return None if declarator is None else declarator.text.decode("utf-8")
        if node.type.endswith("_specifier") and node.child_by_field_name("body") is None:
            return None  # a forward declaration or a use, not a definition
        name = node.child_by_field_name("name")
        return None if name is None else name.text.decode("utf-8")

    def _expand(self, lines, item: _Item, node: Node | None) -> list[_Item]:
        """The item itself if it fits, else the items of its children, else its lines cut."""
        text = _trimmed_text(lines, item.start, item.end)
        if self._count(text) <= self._size:
            return [item]
        if node is not None:
            inner = node
            if node.type == "decorated_definition":
                inner = node.child_by_field_name("definition") or node
            body = inner.child_by_field_name("body")
            container = body if body is not None else inner
            children = list(container.named_children)
            # Rows before the first child are the definition's head (decorators included).
            if children and (
                len(children) > 1
                or children[0].start_point.row > item.start
                or _last_row(children[0]) < item.end
            ):
                sub = self._items(lines, None, item.start, item.end, item.symbol, children)
                if len(sub) > 1 or sub[0].start != item.start or sub[0].end != item.end:
                    return sub
        return [dataclasses.replace(item, pieces=self._cut(lines, item))]

    def _cut(self, lines, item: _Item) -> list:
        from local_rag.chunker import Piece

        parts = self._fallback.split("\n".join(lines[item.start : item.end + 1]))
        return [
            Piece(p.text, p.start_line + item.start, p.end_line + item.start, item.symbol)
            for p in parts
        ]

    # -- packing items into pieces ------------------------------------------------------

    def _pack(self, lines, items: list[_Item]) -> list:
        from local_rag.chunker import Piece, _trimmed

        pieces, group = [], None  # group: the run of fitting items being joined

        def flush():
            nonlocal group
            if group is not None:
                piece = _trimmed(lines, group.start, group.end + 1)
                if piece.text:
                    pieces.append(Piece(piece.text, piece.start_line, piece.end_line, group.symbol))
                group = None

        for item in items:
            if item.pieces is not None:
                flush()
                pieces += item.pieces
            elif (
                group is not None
                and group.symbol == item.symbol
                and self._count(_trimmed_text(lines, group.start, item.end)) <= self._size
            ):
                group.end = item.end
            else:
                flush()
                group = _Item(item.start, item.end, item.symbol)
        flush()
        return pieces


def _trimmed_text(lines, start: int, end: int) -> str:
    return "\n".join(lines[start : end + 1]).strip("\n")


def _last_row(node: Node) -> int:
    """The row a node's last character is on (a node ending after a newline ends above it)."""
    row, column = node.end_point
    return row - 1 if column == 0 and row > node.start_point.row else row


def _first_identifier(node: Node) -> Node | None:
    for child in node.named_children:
        if child.type == "identifier":
            return child
        found = _first_identifier(child)
        if found is not None:
            return found
    return None


def splitter_for(media_type: str, count_tokens, *, size: int, fallback) -> SyntaxSplitter | None:
    """The syntax splitter for a Media type, None if its language is not split by syntax."""
    language = MEDIA_TYPE_LANGUAGES.get(media_type)
    if language is None:
        return None
    return SyntaxSplitter(language, count_tokens, size=size, fallback=fallback)
