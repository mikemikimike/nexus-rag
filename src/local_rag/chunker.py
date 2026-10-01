"""Splits a Document's text into Passages with LlamaIndex's sentence splitter.

Sizes are measured by a token counter supplied by the caller (normally the Model
gateway's), so nothing here reads LlamaIndex's global settings, downloads a tokenizer
or touches its embedding client.
"""
from collections.abc import Callable

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.node_parser.text.utils import split_by_sentence_tokenizer_internal
from nltk.tokenize import PunktSentenceTokenizer

from local_rag.gateway import ModelGateway
from local_rag.settings import Settings
from local_rag.tokens import count_tokens


class Chunker:
    def __init__(self, count_tokens: Callable[[str], int], *, size: int, overlap: int):
        if overlap >= size:
            raise ValueError(f"chunk overlap ({overlap}) must be smaller than size ({size})")
        punkt = PunktSentenceTokenizer()  # rule-based: needs no downloaded model
        self._splitter = SentenceSplitter(
            chunk_size=size, chunk_overlap=overlap,
            # The splitter only takes len() of what its tokenizer returns.
            tokenizer=lambda text: range(count_tokens(text)),
            chunking_tokenizer_fn=lambda text: split_by_sentence_tokenizer_internal(text, punkt))

    @classmethod
    def from_settings(cls, settings: Settings, gateway: ModelGateway | None) -> "Chunker":
        """Sized by the gateway's token count, or the local one when there is no gateway."""
        return cls(gateway.count_tokens if gateway else count_tokens,
                   size=settings.chunk_tokens, overlap=settings.chunk_overlap)

    def split(self, text: str) -> list[str]:
        return self._splitter.split_text(text)
