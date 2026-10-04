"""Token counting that needs no tokenizer files, so it works on an air-gapped network."""

import re

# Scripts written without spaces between words: a run of them is as many tokens as characters.
_UNSPACED = (
    "฀-໿"  # Thai, Lao
    "က-႟"  # Myanmar
    "ក-៿"  # Khmer
    "぀-ヿ"  # Hiragana, Katakana
    "㐀-䶿一-鿿豈-﫿"  # CJK ideographs
    "가-힯"  # Hangul syllables
    "\U00020000-\U0003134f"  # CJK ideographs, supplementary planes
)
_TOKEN = re.compile(rf"[{_UNSPACED}]|(?:(?![{_UNSPACED}])\w)+|[^\w\s]", re.UNICODE)

# A word is one token. A run longer than any natural word (a hex string, a hash, base64, a
# minified name) is split by a tokenizer into short pieces, so it counts one token per
# RUN_CHARACTERS characters, rounded up.
LONGEST_WORD = 16
RUN_CHARACTERS = 3


def count_tokens(text: str) -> int:
    """Punctuation marks, characters of unspaced scripts, words, and long runs by their
    length. Over-counts against BPE tokenizers, so Passages sized with it stay safely under
    an embedding model's limit however the text is written."""
    return sum(_cost(piece) for piece in _TOKEN.findall(text))


def _cost(piece: str) -> int:
    if len(piece) <= LONGEST_WORD:
        return 1
    return -(-len(piece) // RUN_CHARACTERS)
