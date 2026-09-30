"""Embedding model + tokenizer. Default: your LiteLLM gateway. No tiktoken, no internet."""
import re
from functools import lru_cache

import config

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@lru_cache(maxsize=1)
def get_embed_model():
    if config.EMBED_BACKEND == "local":
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        return HuggingFaceEmbedding(
            model_name=config.MODEL_PATH, device=config.DEVICE,
            embed_batch_size=config.EMBED_BATCH, max_length=config.MAX_SEQ)
    from llama_index.embeddings.openai_like import OpenAILikeEmbedding
    return OpenAILikeEmbedding(
        model_name=config.EMBED_MODEL,
        api_base=config.LITELLM_BASE_URL,
        api_key=config.LITELLM_API_KEY,
        embed_batch_size=config.EMBED_BATCH,
    )


@lru_cache(maxsize=1)
def _hf_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(config.MODEL_PATH, local_files_only=True)
    tok.model_max_length = 10**9
    return tok


def token_fn():
    """Token counter used for chunking. For the gateway backend we use a conservative
    word/punctuation counter (slightly over-counts vs. BPE, so chunks stay safely small)."""
    if config.EMBED_BACKEND == "local":
        tok = _hf_tokenizer()
        return lambda text: tok.encode(text, add_special_tokens=False)
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
