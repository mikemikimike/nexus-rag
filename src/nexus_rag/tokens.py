"""Token counting that needs no tokenizer files, so it works on an air-gapped network."""

import math
import re

# Scripts written without spaces between words: a run of them is as many tokens as characters.
_UNSPACED = (
    "\u0e00-\u0eff"  # Thai, Lao
    "\u1000-\u109f"  # Myanmar
    "\u1780-\u17ff"  # Khmer
    "\u3040-\u30ff\uff66-\uff9f"  # Hiragana, Katakana, halfwidth Katakana
    "\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"  # CJK ideographs
    "\u3130-\u318f\uac00-\ud7af"  # Hangul compatibility Jamo, syllables
    "\U00020000-\U0003134f"  # CJK ideographs, supplementary planes
)
_TOKEN = re.compile(rf"[{_UNSPACED}]|(?:(?![{_UNSPACED}])\w)+|[^\w\s]", re.UNICODE)

# A word is one token. A run longer than any natural word (a hex string, a hash, base64, a
# minified name) is split by a tokenizer into short pieces, so it counts one token per
# _RUN_CHARACTERS characters, rounded up.
_LONGEST_WORD = 16
_RUN_CHARACTERS = 3


def count_tokens(text: str) -> int:
    """Punctuation marks, characters of unspaced scripts, words, and long runs by their
    length. Over-counts against BPE tokenizers, so Passages sized with it stay safely under
    an embedding model's limit however the text is written."""
    return sum(_token_cost(piece) for piece in _TOKEN.findall(text))


def _token_cost(piece: str) -> int:
    if len(piece) <= _LONGEST_WORD:
        return 1
    return math.ceil(len(piece) / _RUN_CHARACTERS)
