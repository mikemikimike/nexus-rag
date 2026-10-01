"""The Documents the acceptance suite generates for its temp folder Corpus.

Their text is invented, so nothing in a real Knowledge index can be mistaken for them, and
each run gets its own identifiers, so one run cannot find what an earlier one left behind.
Two of the probes are searched by meaning, with a query that shares no word with the
Document: the suite's proof that the dense vectors work (and, for the tail of a long
Document, that the embedding model read all of a Passage).
"""
import random
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

COLLECTION_PREFIX = "acceptance-"
KEYWORD_DOCUMENT = "ops/keyword-note.md"
SEMANTIC_DOCUMENT = "ops/semantic-note.md"
LONG_DOCUMENT = "lab/long-note.md"

# A query that shares no word with the Document it must find, so that only the dense vectors
# can find it. The words listed are the ones an embedding model treats as one meaning.
SEMANTIC_TEXT = "The vehicle will not start because its battery is dead."
SEMANTIC_QUERY = "automobile refuses ignition"
SEMANTIC_MEANING = ("vehicle", "automobile", "start", "ignition", "refuses", "battery", "dead")
TAIL_TEXT = "Finally, the lighthouse keeper repaired the broken lamp before the storm."
TAIL_QUERY = "beacon mended tempest"
TAIL_MEANING = ("lighthouse", "beacon", "keeper", "repaired", "mended", "lamp", "storm",
                "tempest")
MEANINGS = (SEMANTIC_MEANING, TAIL_MEANING)

# How many times longer than one Passage the long Document is: far longer, so that a
# Document is split into many Passages.
LONG_FACTOR = 10

_ADJECTIVES = ("quarterly", "annual", "draft", "internal", "weekly", "revised", "pending",
               "regional", "summary", "provisional")
_NOUNS = ("ledger", "invoice", "budget", "agenda", "forecast", "roster", "inventory",
          "schedule", "voucher", "minutes", "estimate", "memo")
_VERBS = ("covers", "lists", "updates", "replaces", "mentions", "tracks", "follows")


@dataclass(frozen=True)
class Run:
    """The identifiers of one run, all made from one random mark."""

    mark: str

    @classmethod
    def new(cls) -> "Run":
        return cls(secrets.token_hex(5))

    keyword = property(lambda self: f"acc{self.mark}k")  # in the keyword Document
    replacement = property(lambda self: f"acc{self.mark}r")  # there, once it has been changed
    tail = property(lambda self: f"acc{self.mark}t")  # in the last sentence of the long Document
    collection = property(lambda self: f"{COLLECTION_PREFIX}{self.mark}")  # the throwaway index


def keyword_text(identifier: str) -> str:
    return (f"Maintenance window runbook. The reference of this runbook is {identifier}. "
            "It is a generated Document of the acceptance suite.")


def long_text(run: Run, chunk_tokens: int, count_tokens: Callable[[str], int]) -> str:
    """Many Passages of invented sentences, and the tail's marker in the very last sentence."""
    rng = random.Random(run.tail)
    paragraphs, sentences, number = [], [], 0
    while count_tokens("\n\n".join(paragraphs)) < LONG_FACTOR * chunk_tokens:
        number += 1
        sentences.append(
            f"The {rng.choice(_ADJECTIVES)} {rng.choice(_NOUNS)} {rng.choice(_VERBS)} "
            f"the {rng.choice(_ADJECTIVES)} {rng.choice(_NOUNS)} of item {number}.")
        if len(sentences) == 5:
            paragraphs.append(" ".join(sentences))
            sentences = []
    paragraphs += [" ".join(sentences)] if sentences else []
    paragraphs.append(f"{TAIL_TEXT} The closing reference is {run.tail}.")
    return "\n\n".join(paragraphs)


def write_corpus(folder: Path, run: Run, chunk_tokens: int,
                 count_tokens: Callable[[str], int]) -> dict[str, str]:
    """Write the three Documents below ``folder``; returns each one's text by identity."""
    texts = {
        KEYWORD_DOCUMENT: keyword_text(run.keyword),
        SEMANTIC_DOCUMENT: SEMANTIC_TEXT,
        LONG_DOCUMENT: long_text(run, chunk_tokens, count_tokens),
    }
    for document_id, text in texts.items():
        write_document(folder, document_id, text)
    return texts


def write_document(folder: Path, document_id: str, text: str) -> None:
    path = folder / document_id
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
