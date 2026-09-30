"""Embedding model + tokenizer via your LiteLLM gateway. No tiktoken, no internet."""
import re
from functools import lru_cache

import config

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@lru_cache(maxsize=1)
def get_embed_model():
    from llama_index.embeddings.openai_like import OpenAILikeEmbedding
    return OpenAILikeEmbedding(
        model_name=config.EMBED_MODEL,
        api_base=config.LITELLM_BASE_URL,
        api_key=config.LITELLM_API_KEY,
        embed_batch_size=config.EMBED_BATCH,
    )


def token_fn():
    """Token counter used for chunking: a conservative word/punctuation counter
    (slightly over-counts vs. BPE, so chunks stay safely small)."""
    return lambda text: _TOKEN_RE.findall(text)


def init_settings():
    """LlamaIndex defaults to tiktoken (downloads from the internet) and OpenAI.
    Override tokenizer and node parser. Embeddings are called directly; the LLM is never
    used: Claude does the answering, we only retrieve."""
    from llama_index.core import Settings, set_global_tokenizer
    from llama_index.core.node_parser import SentenceSplitter

    set_global_tokenizer(token_fn())
    Settings.node_parser = SentenceSplitter(
        chunk_size=config.CHUNK_TOKENS,
        chunk_overlap=config.CHUNK_OVERLAP,
        tokenizer=token_fn(),
    )
