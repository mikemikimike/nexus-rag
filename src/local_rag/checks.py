"""What the Smoke check and the Acceptance suite share: how a check reports, and the checks
both of them run.

    [ OK ] gateway.embeddings: model 'embed' answers, 1024 dimensions
    [WARN] Knowledge index: Qdrant reachable, 3 documents, 9 passages; ... run a Rebuild
    [FAIL] gateway.embeddings: ... Likely cause: ... See docs/agents/closed-network-setup.md section 10.
    [SKIP] gateway.rerank: RAG_RERANK_MODEL is not set, so search does not rerank

A check that depends on another that did not pass is skipped with that reason, so a broken
gateway shows as one failure and not a dozen. A warning never fails the run. Standard library
only: the Acceptance suite's install group runs on a host that has no Python packages.
"""

import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TextIO
from urllib.parse import urlsplit

from local_rag.errors import describe
from local_rag.gateway import ModelGateway
from local_rag.settings import RetrievalMode, Settings

GUIDE = "docs/agents/closed-network-setup.md"
COMMAND_TIMEOUT = 30.0
QDRANT_DOWN = (
    'section 10 (row "/healthz 503, Qdrant is unreachable")'  # the row for any Qdrant trouble
)
SECRET_NAMES = ("KEY", "TOKEN", "TOKENS", "SECRET", "PASSWORD")
SYSTEMD_UNITS = (
    "qdrant.service",
    "local-rag-mcp.service",
    "local-rag-sync.timer",
)  # what the systemd route runs
KEYWORD_ONLY = "RAG_RETRIEVAL=keyword, so no embedding model is used"
_LABELS = {"ok": "[ OK ]", "warn": "[WARN]", "fail": "[FAIL]", "skip": "[SKIP]"}


class Failure(Exception):
    """A check failed: what was observed, the likely cause, and where the guide helps."""

    def __init__(self, what: str, *, cause: str | None = None, see: str | None = None):
        super().__init__(what)
        self.what, self.cause, self.see = what, cause, see


class Skipped(Exception):
    """A check that does not apply here; the reason is reported."""

    def __init__(self, why: str):
        super().__init__(why)
        self.why = why


class Warned(Exception):
    """A check that passed, with something the operator should act on."""

    def __init__(self, what: str):
        super().__init__(what)
        self.what = what


class Report:
    """One line per check, and a summary ending in ``<VERDICT> PASSED`` or ``<VERDICT> FAILED``."""

    def __init__(
        self,
        out: TextIO,
        secrets: Iterable[str] = (),
        *,
        verdict: str,
        default_see: str = "section 10",
    ):
        self._out = out
        self._secrets = sorted({s for s in secrets if s}, key=len, reverse=True)
        self._verdict = verdict
        self._default_see = default_see
        self.results: dict[str, str] = {}  # check name -> "ok", "warn", "fail" or "skip"

    def say(self, text: str = "") -> None:
        for secret in self._secrets:  # whatever the text came from, a secret never leaves
            text = text.replace(secret, "<hidden>")
        print(text, file=self._out, flush=True)

    def check(self, name: str, what: Callable[[], str], *, needs: Iterable[str] = ()) -> bool:
        """Run one check and report it; True if it passed (a warning counts as passed).

        ``needs`` are checks that must have passed, if they ran, for this one to mean anything.
        """
        blocked = [n for n in needs if self.results.get(n) in ("fail", "skip")]
        if blocked:
            return self._finish(name, "skip", f"not run, {blocked[0]} did not pass")
        try:
            return self._finish(name, "ok", what())
        except Skipped as skipped:
            return self._finish(name, "skip", skipped.why)
        except Warned as warned:
            return self._finish(name, "warn", warned.what)
        except Failure as failure:
            return self._finish(name, "fail", self._explain(failure))
        except Exception as error:
            return self._finish(
                name,
                "fail",
                self._explain(
                    Failure(
                        f"{type(error).__name__}: {error}" if str(error) else type(error).__name__
                    )
                ),
            )

    def _explain(self, failure: Failure) -> str:
        text = failure.what.rstrip(". ")
        if failure.cause:
            text += f". Likely cause: {failure.cause.rstrip('. ')}"
        return f"{text}. See {GUIDE} {failure.see or self._default_see}."

    def _finish(self, name: str, result: str, text: str) -> bool:
        self.results[name] = result
        self.say(f"{_LABELS[result]} {name}: {' '.join(text.split())}")
        return result in ("ok", "warn")

    @property
    def failed(self) -> list[str]:
        return [name for name, result in self.results.items() if result == "fail"]

    def summary(self) -> None:
        counts = {kind: list(self.results.values()).count(kind) for kind in _LABELS}
        warned = f", {counts['warn']} warned" if counts["warn"] else ""
        self.say()
        self.say(
            f"{len(self.results)} checks: {counts['ok']} passed{warned}, "
            f"{counts['fail']} failed, {counts['skip']} skipped"
        )
        self.say(
            f"{self._verdict} PASSED"
            if not self.failed
            else f"{self._verdict} FAILED: {', '.join(self.failed)}"
        )


def secrets_in(environ: Mapping[str, str]) -> list[str]:
    """The values in the environment that must never be printed."""
    values = [
        value
        for name, value in environ.items()
        if any(name.endswith(word) for word in SECRET_NAMES)
    ]
    values += [t.strip() for t in environ.get("MCP_TOKENS", "").split(",")]
    values += [
        urlsplit(value).password or ""
        for value in environ.values()
        if value.startswith(("http://", "https://"))
    ]  # credentials inside a URL
    return [v for v in values if v]


def run_command(command: Sequence[str]) -> str:
    """Run a host command; returns its output, raises RuntimeError saying why it failed."""
    try:
        return subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=COMMAND_TIMEOUT
        ).stdout
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or "").strip().splitlines()
        raise RuntimeError(
            f"{command[0]} exited {error.returncode}" + (f": {detail[-1]}" if detail else "")
        ) from None
    except FileNotFoundError:
        raise RuntimeError(f"{command[0]} not found") from None
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"{command[0]}: {error}") from None


def gateway_failure(error: Exception) -> Failure:
    """A Model gateway error, with the cause its wording points to."""
    text = describe(error)
    if "answered 401" in text or "answered 403" in text:
        return Failure(
            text,
            cause="LITELLM_API_KEY is wrong or missing",
            see='section 10 (row "gateway ... answered 401")',
        )
    if "answered 404" in text or "answered 400" in text:
        return Failure(
            text,
            cause="LITELLM_BASE_URL lacks /v1, or RAG_EMBED_MODEL is not a LiteLLM model_name",
            see='section 10 (row "gateway ... answered 404")',
        )
    if "unreachable" in text:
        return Failure(
            text,
            cause="the address, a firewall or DNS between this host and the "
            "Model gateway (it is always contacted directly, ignoring proxy "
            "variables)",
            see='section 10 (row "gateway ... unreachable")',
        )
    return Failure(
        text,
        cause="the Model gateway answered something unusable; read its own log",
        see="section 3 B (Model gateway)",
    )


def embeddings(settings: Settings, gateway: ModelGateway | None) -> str:
    """The embedding model answers a query with a vector (skipped in Keyword-only retrieval)."""
    if settings.retrieval is RetrievalMode.KEYWORD:
        raise Skipped(KEYWORD_ONLY)
    assert gateway is not None, "settings that embed need a Model gateway"
    try:
        vector = gateway.embed_query("a probe sentence")
    except Exception as error:
        raise gateway_failure(error) from None
    if not vector:
        raise Failure(
            "the Model gateway answered with an empty vector",
            cause="RAG_EMBED_MODEL names a model that does not produce embeddings",
            see="section 3 B (Model gateway)",
        )
    return f"model {settings.embed_model!r} answers, {len(vector)} dimensions"


def rerank(settings: Settings, gateway: ModelGateway | None) -> str:
    """The rerank model, if one is named, ranks a relevant text above an unrelated one."""
    model = settings.rerank_model
    if not model:
        raise Skipped("RAG_RERANK_MODEL is not set, so search does not rerank")
    assert gateway is not None, "settings that rerank need a Model gateway"
    relevant = "Reset a forgotten password from the account page."
    unrelated = "The cafeteria serves soup on Tuesdays."
    try:
        scores = gateway.rerank("how do I reset my password", [unrelated, relevant])
    except Exception as error:
        raise Failure(
            describe(error),
            cause="the Model gateway does not serve /v1/rerank for "
            "RAG_RERANK_MODEL; unset it to disable rerank",
            see='section 10 (the proxy row) and docs/operations.md troubleshooting "Rerank ignored"',
        ) from None
    if len(scores) != 2 or not scores[1] > scores[0]:
        raise Failure(
            f"the reranker scored the relevant text {scores[1:]} against "
            f"{scores[:1]} for an unrelated one"
            if len(scores) == 2
            else f"asked to score 2 texts, got {len(scores)} scores",
            cause="RAG_RERANK_MODEL is not a reranker",
            see="section 3 B (Model gateway)",
        )
    return f"model {model!r} ranks the relevant text first"
