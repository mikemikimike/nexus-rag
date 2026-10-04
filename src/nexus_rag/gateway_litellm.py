"""Model gateway adapter for the network's LiteLLM gateway (OpenAI-compatible HTTP)."""

import json
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Any

from nexus_rag.settings import Settings
from nexus_rag.tokens import count_tokens

_TRANSIENT_STATUSES = {408, 429}  # a timeout, or "too many requests": try again later

RERANK_MAX_CHARS = 2000  # longer texts are cut before scoring; the head carries the topic


# The gateway is on the local network: never route it through a proxy from the environment.
_DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class GatewayError(RuntimeError):
    """The Model gateway could not answer: unreachable, refused, or answered nonsense.

    ``retryable`` says whether asking again could help: an unreachable or overloaded
    gateway may recover, a wrong key or model name will not.
    """

    def __init__(self, message: str, *, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


def gateway_from_settings(settings: Settings) -> "LiteLLMGateway | None":
    """The Model gateway the settings need, or None if nothing will reach one: Keyword-only
    retrieval with no reranker."""
    if not settings.needs_model_gateway:
        return None
    return LiteLLMGateway.from_settings(settings)


class LiteLLMGateway:
    """Embeddings and optional rerank over HTTP. Token counting is local."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        embed_model: str = "embed",
        rerank_model: str | None = None,
        batch_size: int = 16,
        timeout: float = 60.0,
        query_instruction: str = "",
    ):
        if not base_url.startswith(("http://", "https://")):
            raise ValueError(f"gateway base URL must be http(s), got {base_url!r}")
        if batch_size < 1:
            raise ValueError("embed batch size must be at least 1")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._embed_model = embed_model
        self._rerank_model = rerank_model
        self._batch_size = batch_size
        self._timeout = timeout
        self._query_instruction = query_instruction

    @classmethod
    def from_settings(cls, settings: Settings) -> "LiteLLMGateway":
        if settings.gateway_base_url is None:
            raise ValueError("no Model gateway is named: gateway_base_url is not set")
        return cls(
            settings.gateway_base_url,
            api_key=settings.gateway_api_key,
            embed_model=settings.embed_model,
            rerank_model=settings.rerank_model,
            batch_size=settings.embed_batch_size,
            timeout=settings.gateway_timeout,
            query_instruction=settings.query_instruction,
        )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            vectors.extend(self._embed_batch(list(texts[start : start + self._batch_size])))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        if self._query_instruction:  # the format Qwen3-Embedding was trained with
            text = f"Instruct: {self._query_instruction}\nQuery: {text}"
        [vector] = self._embed_batch([text])
        return vector

    def rerank(self, query: str, texts: Sequence[str]) -> list[float]:
        """Relevance scores aligned with `texts`, higher is better. Raises GatewayError on
        any failure so the caller can fall back to its existing order."""
        if not self._rerank_model:
            raise GatewayError("no rerank model configured")
        if not texts:
            return []
        answer = self._post(
            "/rerank",
            {
                "model": self._rerank_model,
                "query": query,
                "documents": [text[:RERANK_MAX_CHARS] for text in texts],
                "top_n": len(texts),
            },
        )
        scores: dict[int, float] = {}
        try:
            for result in answer["results"]:
                scores[int(result["index"])] = float(result["relevance_score"])
        except (KeyError, TypeError, ValueError) as error:
            raise GatewayError(f"malformed rerank answer: {error!r}") from error
        if sorted(scores) != list(range(len(texts))):
            raise GatewayError(f"rerank answer scored {len(scores)} of {len(texts)} texts")
        return [scores[i] for i in range(len(texts))]

    def count_tokens(self, text: str) -> int:
        return count_tokens(text)

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        answer = self._post("/embeddings", {"model": self._embed_model, "input": batch})
        try:
            ordered = sorted(answer["data"], key=lambda item: item["index"])
            vectors = [[float(x) for x in item["embedding"]] for item in ordered]
        except (KeyError, TypeError, ValueError) as error:
            raise GatewayError(
                f"malformed embeddings answer: {error!r}", retryable=False
            ) from error
        if len(vectors) != len(batch):
            raise GatewayError(f"expected {len(batch)} embeddings, gateway returned {len(vectors)}")
        return vectors

    def _post(self, path: str, payload: dict[str, Any]) -> Any:
        url = f"{self._base_url}{path}"
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        request = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
        )
        try:
            with _DIRECT.open(request, timeout=self._timeout) as response:
                body = response.read()
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")[:500]
            raise GatewayError(
                f"gateway {url} answered {error.code}: {detail}",
                retryable=error.code in _TRANSIENT_STATUSES or error.code >= 500,
            ) from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise GatewayError(f"gateway {url} unreachable: {error}") from error
        try:
            return json.loads(body)
        except ValueError as error:
            raise GatewayError(
                f"gateway {url} answered something that is not valid JSON"
            ) from error
