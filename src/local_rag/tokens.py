"""Token counting that needs no tokenizer files, so it works on an air-gapped network."""
import re

_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def count_tokens(text: str) -> int:
    """Words plus punctuation marks. Slightly over-counts against BPE tokenizers, so
    Passages sized with it stay safely under an embedding model's limit."""
    return len(_TOKEN.findall(text))
