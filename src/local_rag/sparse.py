"""BM25-style sparse encoder: the one place sparse parameters are defined and applied.

Document side: saturated term frequency with length normalisation. Query side: 1.0 per
unique term. IDF is applied by Qdrant (sparse vector modifier=IDF).
"""
import re
import zlib
from collections import Counter
from dataclasses import dataclass

_WORD = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class SparseParams:
    k1: float = 1.2
    b: float = 0.75
    avg_doc_length: float = 250.0


@dataclass(frozen=True)
class SparseVector:
    indices: list[int]
    values: list[float]


def tokenize(text: str) -> list[str]:
    tokens = []
    for word in _WORD.findall(text.lower()):
        if len(word) < 2:
            continue
        tokens.append(word)
        if "_" in word:  # snake_case identifiers also match their parts
            tokens.extend(part for part in word.split("_") if len(part) > 1)
    return tokens


def _index(token: str) -> int:
    return zlib.crc32(token.encode("utf-8")) & 0x7FFFFFFF


class SparseEncoder:
    def __init__(self, params: SparseParams):
        self.params = params

    def encode_document(self, text: str) -> SparseVector:
        tokens = tokenize(text)
        length = max(len(tokens), 1)
        k1, b, avg = self.params.k1, self.params.b, self.params.avg_doc_length
        indices, values = [], []
        for index, count in Counter(_index(t) for t in tokens).items():
            indices.append(index)
            values.append(count * (k1 + 1) / (count + k1 * (1 - b + b * length / avg)))
        return SparseVector(indices, values)

    def encode_query(self, text: str) -> SparseVector:
        indices = sorted({_index(t) for t in tokenize(text)})
        return SparseVector(indices, [1.0] * len(indices))
