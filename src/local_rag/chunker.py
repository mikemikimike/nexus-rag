"""Splits a Document's text into Passages.

Prose is split by sentences, code (see ``is_code``) by syntax where the language has
a splitter and by lines otherwise, each piece knowing the lines it
covers. Sizes are measured by a token counter supplied by the caller (normally the Model
gateway's), so nothing here reads LlamaIndex's global settings, downloads a tokenizer
or touches its embedding client.
"""
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.node_parser.text.utils import split_by_sentence_tokenizer_internal
from nltk.tokenize import PunktSentenceTokenizer

from local_rag.corpus import is_code
from local_rag.gateway import ModelGateway
from local_rag.settings import Settings
from local_rag.syntax import splitter_for
from local_rag.tokens import count_tokens


CODE_OVERLAP_LINES = 3  # lines shared by neighbouring code Passages


@dataclass(frozen=True)
class Piece:
    """One Passage's text. Code pieces know their lines (1-based, inclusive, counted in the
    file as it is) and, when the splitter knows it, the symbol they sit in."""

    text: str
    start_line: int | None = None
    end_line: int | None = None
    symbol: str | None = None  # e.g. ``LedgerClient.post_entry``


class Splitter(Protocol):
    def split(self, text: str) -> list[Piece]: ...


class _ProseSplitter:
    def __init__(self, splitter: SentenceSplitter):
        self._splitter = splitter

    def split(self, text: str) -> list[Piece]:
        return [Piece(passage) for passage in self._splitter.split_text(text)]


class LineSplitter:
    """Splits code into runs of whole lines of at most ``size`` tokens.

    A run ends at the last blank line or drop in indentation that leaves it at least half
    full, and the next begins ``overlap`` lines before that end. A line larger than ``size``
    on its own is cut within itself, its pieces all citing that line.
    """

    def __init__(self, count_tokens: Callable[[str], int], *, size: int,
                 overlap: int = CODE_OVERLAP_LINES):
        self._count = count_tokens
        self._size = size
        self._overlap = overlap

    def split(self, text: str) -> list[Piece]:
        lines = text.split("\n")
        if lines and lines[-1] == "":
            lines.pop()  # the newline that ends the last line starts no line
        pieces, start = [], 0
        while start < len(lines):
            end = self._end(lines, start)
            if end - start == 1 and self._count(lines[start]) > self._size:
                pieces += [Piece(part, start + 1, start + 1)
                           for part in self._cut(lines[start]) if part.strip()]
            else:
                pieces.append(_trimmed(lines, start, end))
            if end >= len(lines):
                break
            start = max(end - self._overlap, start + 1)
        return [piece for piece in pieces if piece.text]

    def _end(self, lines: list[str], start: int) -> int:
        """The exclusive end of the run that begins at ``start``."""
        tokens, end = 0, start
        while end < len(lines):
            cost = self._count(lines[end]) + 1  # the line break
            if tokens + cost > self._size and end > start:
                break
            tokens, end = tokens + cost, end + 1
        if end >= len(lines):
            return end
        for candidate in range(end, start, -1):
            if self._count("\n".join(lines[start:candidate])) < self._size // 2:
                break
            if _is_boundary(lines, candidate):
                return candidate
        return end

    def _cut(self, line: str) -> list[str]:
        parts = []
        while line:
            low, high = 1, len(line)
            while low < high:  # the longest prefix that fits
                middle = (low + high + 1) // 2
                if self._count(line[:middle]) <= self._size:
                    low = middle
                else:
                    high = middle - 1
            parts.append(line[:low])
            line = line[low:]
        return parts


def _is_boundary(lines: list[str], index: int) -> bool:
    """Whether a run may end just before ``lines[index]``: after a blank line, or where
    indentation drops."""
    if not lines[index - 1].strip():
        return True
    return bool(lines[index].strip()) and _indent(lines[index]) < _indent(lines[index - 1])


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _trimmed(lines: list[str], start: int, end: int) -> Piece:
    """The run of lines[start:end] without its blank first and last lines."""
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return Piece("\n".join(lines[start:end]), start + 1, end)


class Chunker:
    def __init__(self, count_tokens: Callable[[str], int], *, size: int, overlap: int,
                 code_size: int = 800):
        if overlap >= size:
            raise ValueError(f"chunk overlap ({overlap}) must be smaller than size ({size})")
        punkt = PunktSentenceTokenizer()  # rule-based: needs no downloaded model
        self._count, self._code_size = count_tokens, code_size
        self._code = LineSplitter(count_tokens, size=code_size)
        self._syntax = {}  # Media type -> its splitter, made when first needed
        self._prose = _ProseSplitter(SentenceSplitter(
            chunk_size=size, chunk_overlap=overlap,
            # The splitter only takes len() of what its tokenizer returns.
            tokenizer=lambda text: range(count_tokens(text)),
            chunking_tokenizer_fn=lambda text: split_by_sentence_tokenizer_internal(text, punkt)))

    @classmethod
    def from_settings(cls, settings: Settings, gateway: ModelGateway | None) -> "Chunker":
        """Sized by the gateway's token count, or the local one when there is no gateway."""
        return cls(count_tokens if gateway is None else gateway.count_tokens,
                   size=settings.chunk_tokens, overlap=settings.chunk_overlap,
                   code_size=settings.code_chunk_tokens)

    def split(self, text: str, media_type: str = "text/plain") -> list[str]:
        """The text of each Passage of a Document, split the way its Media type calls for."""
        return [piece.text for piece in self.pieces(text, media_type)]

    def pieces(self, text: str, media_type: str = "text/plain") -> list[Piece]:
        """The Passages of a Document's text, with the lines each covers if it is code."""
        return self._splitter_for(media_type).split(text)

    def _splitter_for(self, media_type: str) -> Splitter:
        """Where a Media type is mapped to how it is split: code in a language with a
        syntax-aware splitter (see ``languages``) by that, other code by lines, the rest as
        prose."""
        if not is_code(media_type):
            return self._prose
        if media_type not in self._syntax:
            self._syntax[media_type] = splitter_for(
                media_type, self._count, size=self._code_size, fallback=self._code) or self._code
        return self._syntax[media_type]
