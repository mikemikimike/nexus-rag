"""Tiny BM25-style sparse encoder (no downloads, multilingual).
Doc side: saturated TF with length normalisation. Query side: 1.0 per unique term.
IDF is applied server-side by Qdrant (sparse vector modifier=IDF)."""
import re
import zlib
from collections import Counter

import config

_WORD = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    out = []
    for w in _WORD.findall(text.lower()):
        if len(w) < 2:
            continue
        out.append(w)
        if "_" in w:  # snake_case identifiers also match their parts
            out.extend(p for p in w.split("_") if len(p) > 1)
    return out


def _idx(tok: str) -> int:
    return zlib.crc32(tok.encode("utf-8")) & 0x7FFFFFFF


def encode_doc(text: str):
    toks = tokenize(text)
    dl = max(len(toks), 1)
    tf = Counter(_idx(t) for t in toks)
    k1, b, avg = config.BM25_K1, config.BM25_B, config.BM25_AVG_DL
    idx, val = [], []
    for i, c in tf.items():
        idx.append(i)
        val.append(c * (k1 + 1) / (c + k1 * (1 - b + b * dl / avg)))
    return idx, val


def encode_query(text: str):
    idx = sorted({_idx(t) for t in tokenize(text)})
    return idx, [1.0] * len(idx)
