"""How the acceptance suite reports: one line per check, a summary, and no secret.

    [ OK ] gateway.embeddings: model 'embed' answers, 1024 dimensions
    [FAIL] gateway.embeddings: ... Likely cause: ... See docs/agents/closed-network-setup.md section 10.
    [SKIP] gateway.rerank: RAG_RERANK_MODEL is not set, so search does not rerank

The labels are the smoke check's. A check that depends on another that did not pass is
skipped with that reason, so a broken gateway shows as one failure and not a dozen.
Standard library only.
"""
import argparse
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TextIO

from local_rag.settings import RetrievalMode

GUIDE = "docs/agents/closed-network-setup.md"
COMMAND_TIMEOUT = 30.0
QDRANT_DOWN = 'section 10 (row "/healthz 503, Qdrant is unreachable")'  # the row for any Qdrant trouble


class Failure(Exception):
    """A check failed: what was observed, the likely cause, and where the guide helps."""

    def __init__(self, what: str, *, cause: str | None = None, see: str | None = None):
        super().__init__(what)
        self.what, self.cause, self.see = what, cause, see


KEYWORD_ONLY = "RAG_RETRIEVAL=keyword, so no embedding model is used"


def is_keyword_only(ctx: "Context") -> bool:
    """Whether the suite runs in Keyword-only retrieval, where the embedding checks do not apply."""
    return ctx.settings.retrieval is RetrievalMode.KEYWORD


class Skipped(Exception):
    """A check that does not apply here; the reason is reported."""

    def __init__(self, why: str):
        super().__init__(why)
        self.why = why


class Report:
    def __init__(self, out: TextIO, secrets: Iterable[str] = (), *, default_see: str = "section 10"):
        self._out = out
        self._secrets = sorted({s for s in secrets if s}, key=len, reverse=True)
        self._default_see = default_see
        self.results: dict[str, str] = {}  # check name -> "ok", "fail" or "skip"

    def say(self, text: str = "") -> None:
        for secret in self._secrets:  # whatever the text came from, a secret never leaves
            text = text.replace(secret, "<hidden>")
        print(text, file=self._out, flush=True)

    def check(self, name: str, what: Callable[[], str], *, needs: Iterable[str] = ()) -> bool:
        """Run one check and report it; True if it passed.

        ``needs`` are checks that must have passed, if they ran, for this one to mean anything.
        """
        blocked = [n for n in needs if self.results.get(n) in ("fail", "skip")]
        if blocked:
            return self._finish(name, "skip", f"not run, {blocked[0]} did not pass")
        try:
            return self._finish(name, "ok", what())
        except Skipped as skipped:
            return self._finish(name, "skip", skipped.why)
        except Failure as failure:
            return self._finish(name, "fail", self._explain(failure))
        except Exception as error:
            return self._finish(name, "fail", self._explain(Failure(
                f"{type(error).__name__}: {error}" if str(error) else type(error).__name__)))

    def _explain(self, failure: Failure) -> str:
        text = failure.what.rstrip(". ")
        if failure.cause:
            text += f". Likely cause: {failure.cause.rstrip('. ')}"
        return f"{text}. See {GUIDE} {failure.see or self._default_see}."

    def _finish(self, name: str, result: str, text: str) -> bool:
        label = {"ok": "[ OK ]", "fail": "[FAIL]", "skip": "[SKIP]"}[result]
        self.results[name] = result
        self.say(f"{label} {name}: {' '.join(text.split())}")
        return result == "ok"

    @property
    def failed(self) -> list[str]:
        return [name for name, result in self.results.items() if result == "fail"]

    def summary(self) -> None:
        counts = {kind: list(self.results.values()).count(kind) for kind in ("ok", "fail", "skip")}
        self.say()
        self.say(f"{len(self.results)} checks: {counts['ok']} passed, {counts['fail']} failed, "
                 f"{counts['skip']} skipped")
        self.say("ACCEPTANCE PASSED" if not self.failed else
                 f"ACCEPTANCE FAILED: {', '.join(self.failed)}")


@dataclass
class Context:
    """What every check group works from."""

    settings: Any  # the Settings the services run with, but for the throwaway collection
    environ: Mapping[str, str]
    report: Report
    gateway: Any  # the Model gateway, or None if it cannot be built
    connect: Callable[[str, str | None], Any]  # (Qdrant location, API key) -> a Qdrant client
    run: Callable[[Sequence[str]], str]  # runs a host command, returns its output, raises if it fails
    args: argparse.Namespace
    run_id: Any  # the Run whose identifiers the generated Documents carry
    gateway_problem: str | None = None  # why there is no gateway, if the settings name one
    cleanups: list[tuple[str, Callable[[], str]]] = field(default_factory=list)  # name, undo


def run_command(command: Sequence[str]) -> str:
    """Run a host command; returns its output, raises RuntimeError saying why it failed."""
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True,
                              timeout=COMMAND_TIMEOUT).stdout
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or "").strip().splitlines()
        raise RuntimeError(f"{command[0]} exited {error.returncode}"
                           + (f": {detail[-1]}" if detail else "")) from None
    except FileNotFoundError:
        raise RuntimeError(f"{command[0]} not found") from None
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"{command[0]}: {error}") from None
