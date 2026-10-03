"""The Documents the acceptance suite generates for its temp folder Corpus.

Their text is invented, so nothing in a real Knowledge index can be mistaken for them, and
each run gets its own identifiers, so one run cannot find what an earlier one left behind.
Two of the probes are searched by meaning, with a query that shares no word with the
Document: the suite's proof that the dense vectors work (and, for the tail of a long
Document, that the embedding model read all of a Passage).
"""

import random
import secrets
import string
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
TAIL_MEANING = ("lighthouse", "beacon", "keeper", "repaired", "mended", "lamp", "storm", "tempest")
MEANINGS = (SEMANTIC_MEANING, TAIL_MEANING)

# How many times longer than one Passage the long Document is: far longer, so that a
# Document is split into many Passages.
LONG_FACTOR = 10
# How full the last Passage of the long Document is made, as a share of the Passage size.
FULL_PASSAGE = 0.9

_ADJECTIVES = (
    "quarterly",
    "annual",
    "draft",
    "internal",
    "weekly",
    "revised",
    "pending",
    "regional",
    "summary",
    "provisional",
)
_NOUNS = (
    "ledger",
    "invoice",
    "budget",
    "agenda",
    "forecast",
    "roster",
    "inventory",
    "schedule",
    "voucher",
    "minutes",
    "estimate",
    "memo",
)
_VERBS = ("covers", "lists", "updates", "replaces", "mentions", "tracks", "follows")


@dataclass(frozen=True)
class Run:
    """The identifiers of one run, all made from one random mark."""

    mark: str

    @classmethod
    def new(cls) -> "Run":
        # Letters only: digits would split an identifier into parts that other runs share.
        return cls("".join(secrets.choice(string.ascii_lowercase) for _ in range(10)))

    keyword = property(lambda self: f"acc{self.mark}k")  # in the keyword Document
    replacement = property(lambda self: f"acc{self.mark}r")  # there, once it has been changed
    tail = property(lambda self: f"acc{self.mark}t")  # in the last sentence of the long Document
    collection = property(lambda self: f"{COLLECTION_PREFIX}{self.mark}")  # the throwaway index


def keyword_text(identifier: str) -> str:
    return (
        f"Maintenance window runbook. The reference of this runbook is {identifier}. "
        "It is a generated Document of the acceptance suite."
    )


def long_text(
    run: Run,
    *,
    chunk_tokens: int,
    count_tokens: Callable[[str], int],
    split: Callable[[str], list[str]],
) -> str:
    """Many Passages of invented sentences, and the tail's marker in the very last sentence.

    Sentences are added before the marker until the last Passage is nearly full-sized, so the
    marker sits at the far end of a Passage of the size the index really embeds: an embedding
    model that reads less than that misses it.
    """
    rng = random.Random(run.tail)
    number = 0

    def sentence() -> str:
        nonlocal number
        number += 1
        return (
            f"The {rng.choice(_ADJECTIVES)} {rng.choice(_NOUNS)} {rng.choice(_VERBS)} "
            f"the {rng.choice(_ADJECTIVES)} {rng.choice(_NOUNS)} of item {number}."
        )

    paragraphs = []
    while count_tokens("\n\n".join(paragraphs)) < LONG_FACTOR * chunk_tokens:
        paragraphs.append(" ".join(sentence() for _ in range(5)))
    padding: list[str] = []
    while True:
        closing = " ".join([*padding, TAIL_TEXT, f"The closing reference is {run.tail}."])
        text = "\n\n".join([*paragraphs, closing])
        if (
            count_tokens(split(text)[-1]) >= FULL_PASSAGE * chunk_tokens
            or len(padding) > chunk_tokens
        ):  # (the bound only guards a splitter gone wrong)
            return text
        padding.append(sentence())


def write_corpus(folder: Path, run: Run, **sizes) -> dict[str, str]:
    """Write the three Documents below ``folder``; returns each one's text by identity.

    ``sizes`` are the arguments of ``long_text``."""
    texts = {
        KEYWORD_DOCUMENT: keyword_text(run.keyword),
        SEMANTIC_DOCUMENT: SEMANTIC_TEXT,
        LONG_DOCUMENT: long_text(run, **sizes),
    }
    for document_id, text in texts.items():
        write_document(folder, document_id, text)
    return texts


def write_document(folder: Path, document_id: str, text: str) -> None:
    path = folder / document_id
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
