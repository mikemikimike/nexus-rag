"""The acceptance checks of the two services the system depends on: the Model gateway and Qdrant."""
import re
import secrets

from qdrant_client.http.exceptions import UnexpectedResponse

from local_rag import acceptance_corpus
from local_rag.acceptance_report import (
    KEYWORD_ONLY, QDRANT_DOWN, Context, Failure, Skipped, is_keyword_only)
from local_rag.chunker import Chunker
from local_rag.gateway_litellm import GatewayError

PROBE_TITLE = "long-note"  # what the folder Corpus makes of lab/long-note.md
_LIMIT_WORDS = re.compile(r"context length|too long|too many tokens|maximum|exceed|token limit",
                          re.IGNORECASE)


def gateway_failure(error: Exception) -> Failure:
    """A Model gateway error, with the cause its wording points to."""
    text = str(error) or type(error).__name__
    if "answered 401" in text or "answered 403" in text:
        return Failure(text, cause="LITELLM_API_KEY is wrong or missing",
                       see='section 10 (row "gateway ... answered 401")')
    if "answered 404" in text or "answered 400" in text:
        return Failure(text, cause="LITELLM_BASE_URL lacks /v1, or RAG_EMBED_MODEL is not a "
                                   "LiteLLM model_name", see='section 10 (row "gateway ... answered 404")')
    if "unreachable" in text:
        return Failure(text, cause="the address, a firewall or DNS between this host and the "
                                   "Model gateway (it is always contacted directly, ignoring proxy "
                                   "variables)", see='section 10 (row "gateway ... unreachable")')
    return Failure(text, cause="the Model gateway answered something unusable; "
                               "read its own log", see="section 3 B (Model gateway)")


def _require_gateway(ctx: Context):
    if ctx.gateway_problem:
        raise Failure(ctx.gateway_problem, cause="LITELLM_BASE_URL is not an http(s) URL",
                      see="section 5 (Configure)")
    return ctx.gateway


def run_gateway(ctx: Context) -> None:
    check = ctx.report.check
    check("gateway.embeddings", lambda: _embeddings(ctx))
    check("gateway.dimension", lambda: _dimension(ctx), needs=["gateway.embeddings"])
    check("gateway.input-limit", lambda: _input_limit(ctx), needs=["gateway.embeddings"])
    # A reranker is checked on its own in Keyword-only retrieval, which has no embedding check.
    check("gateway.rerank", lambda: _rerank(ctx),
          needs=[] if is_keyword_only(ctx) else ["gateway.embeddings"])


def _embeddings(ctx: Context) -> str:
    if is_keyword_only(ctx):
        raise Skipped(KEYWORD_ONLY)
    gateway = _require_gateway(ctx)
    try:
        vector = gateway.embed_query("acceptance probe")
    except Exception as error:
        raise gateway_failure(error) from None
    if not vector:
        raise Failure("the Model gateway answered with an empty vector",
                      cause="RAG_EMBED_MODEL names a model that does not produce embeddings",
                      see="section 3 B (Model gateway)")
    return f"model {ctx.settings.embed_model!r} answers, {len(vector)} dimensions"


def _dimension(ctx: Context) -> str:
    gateway = ctx.gateway
    try:
        query = gateway.embed_query("acceptance probe")
        documents = gateway.embed(["first acceptance text", "second acceptance text, a bit longer"])
    except Exception as error:
        raise gateway_failure(error) from None
    lengths = sorted({len(v) for v in documents})
    if len(lengths) != 1 or lengths[0] != len(query):
        raise Failure(
            f"vectors differ in length: a query has {len(query)} dimensions, "
            f"documents have {', '.join(map(str, lengths))}",
            cause="the gateway route serves different models for queries and documents, or "
                  "truncates vectors; the index is tied to one dimension",
            see="section 10 (row \"embedding dimension differs\")")
    return f"query and document vectors agree, {len(query)} dimensions"


def _input_limit(ctx: Context) -> str:
    gateway, settings = ctx.gateway, ctx.settings
    chunker = Chunker.from_settings(settings, gateway)
    text = acceptance_corpus.long_text(ctx.run_id, chunk_tokens=settings.chunk_tokens,
                                       count_tokens=gateway.count_tokens, split=chunker.split)
    passages = chunker.split(text)
    if len(passages) < 2:
        raise Failure(f"a text of {gateway.count_tokens(text)} counted tokens was not split "
                      f"into Passages of {settings.chunk_tokens}",
                      cause="a defect in the Chunker, not in the infrastructure")
    contents = [f"{PROBE_TITLE}\n\n{passage}" for passage in passages]  # as the index embeds them
    largest = max(gateway.count_tokens(content) for content in contents)
    try:
        vectors = gateway.embed(contents)
    except GatewayError as error:
        if _LIMIT_WORDS.search(str(error)):
            raise Failure(
                f"the Model gateway refused a Passage of {largest} counted tokens: {error}",
                cause=f"RAG_CHUNK_TOKENS ({settings.chunk_tokens}, counted as words and "
                      "punctuation) is above the embedding model's input limit; lower it, "
                      "then run a Rebuild",
                see='section 10 (row "A Rebuild is refused")') from None
        raise gateway_failure(error) from None
    if len(vectors) != len(contents):
        raise Failure(f"{len(contents)} Passages were sent, {len(vectors)} vectors came back",
                      cause="the gateway dropped part of a batch", see="section 3 B (Model gateway)")
    return (f"{len(passages)} Passages of a text of {gateway.count_tokens(text)} counted tokens "
            f"embedded, the largest {largest} (RAG_CHUNK_TOKENS is {settings.chunk_tokens})")


def _rerank(ctx: Context) -> str:
    model = ctx.settings.rerank_model
    if not model:
        raise Skipped("RAG_RERANK_MODEL is not set, so search does not rerank")
    _require_gateway(ctx)
    relevant = "Reset a forgotten password from the account page."
    unrelated = "The cafeteria serves soup on Tuesdays."
    try:
        scores = ctx.gateway.rerank("how do I reset my password", [unrelated, relevant])
    except Exception as error:
        raise Failure(str(error) or type(error).__name__,
                      cause="the Model gateway does not serve /v1/rerank for "
                            "RAG_RERANK_MODEL; unset it to disable rerank",
                      see='section 10 (the proxy row) and the README troubleshooting "Rerank ignored"'
                      ) from None
    if len(scores) != 2 or not scores[1] > scores[0]:
        raise Failure(f"the reranker scored the relevant text {scores[1:]} against "
                      f"{scores[:1]} for an unrelated one" if len(scores) == 2 else
                      f"asked to score 2 texts, got {len(scores)} scores",
                      cause="RAG_RERANK_MODEL is not a reranker", see="section 3 B (Model gateway)")
    return f"model {model!r} ranks the relevant text first"


def run_qdrant(ctx: Context) -> None:
    check = ctx.report.check
    check("qdrant.reachable", lambda: _reachable(ctx))
    check("qdrant.wrong-key", lambda: _wrong_key(ctx), needs=["qdrant.reachable"])


def _reachable(ctx: Context) -> str:
    settings = ctx.settings
    try:
        count = len(ctx.connect(settings.qdrant_location, settings.qdrant_api_key)
                    .get_collections().collections)
    except UnexpectedResponse as error:
        if error.status_code in (401, 403):
            raise Failure(f"Qdrant at {settings.qdrant_location} refused the configured key "
                          f"(HTTP {error.status_code})",
                          cause="QDRANT_API_KEY here differs from the one Qdrant runs with "
                                "(QDRANT__SERVICE__API_KEY)", see=QDRANT_DOWN) from None
        raise Failure(f"Qdrant at {settings.qdrant_location} answered HTTP {error.status_code}",
                      cause="something other than Qdrant answers on QDRANT_URL",
                      see="section 10") from None
    except Exception as error:
        raise Failure(f"Qdrant at {settings.qdrant_location} is unreachable: "
                      f"{type(error).__name__}: {error}",
                      cause="Qdrant is down, QDRANT_URL is wrong, or an HTTP proxy sits in the "
                            "way (put the host in NO_PROXY)",
                      see=QDRANT_DOWN) from None
    keyed = "with its key" if settings.qdrant_api_key else "without a key"
    return f"Qdrant at {settings.qdrant_location} answers {keyed}, {count} collections"


def _wrong_key(ctx: Context) -> str:
    settings = ctx.settings
    if not settings.qdrant_api_key:
        raise Skipped("QDRANT_API_KEY is not set, so there is no key to be wrong about")
    wrong = f"wrong-{secrets.token_hex(8)}"
    try:
        ctx.connect(settings.qdrant_location, wrong).get_collections()
    except UnexpectedResponse as error:
        if error.status_code in (401, 403):
            return f"a wrong key is refused (HTTP {error.status_code})"
        raise Failure(f"a wrong key was answered with HTTP {error.status_code}, not a refusal",
                      cause="something in front of Qdrant answers instead of it",
                      see="section 10") from None
    except Exception as error:
        raise Failure(f"could not tell whether a wrong key is refused: {type(error).__name__}",
                      cause="Qdrant became unreachable during the check",
                      see=QDRANT_DOWN) from None
    raise Failure("Qdrant accepted a wrong API key",
                  cause="Qdrant runs without an API key (QDRANT__SERVICE__API_KEY unset), so "
                        "anyone on the network can read and change the Knowledge index",
                  see=QDRANT_DOWN)
