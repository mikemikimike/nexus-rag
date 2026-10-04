"""What every acceptance check group works from. How a check reports is shared with the
Smoke check, in ``checks``. Standard library only."""

import argparse
import os
import pwd
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nexus_rag.checks import Report
from nexus_rag.environment import Environment
from nexus_rag.settings import RetrievalMode


def is_keyword_only(ctx: "Context") -> bool:
    """Whether the suite runs in Keyword-only retrieval, where the embedding checks do not apply."""
    return ctx.settings.retrieval is RetrievalMode.KEYWORD


def _user_named(uid: int) -> str:
    try:
        return pwd.getpwuid(uid).pw_name
    except KeyError:  # a uid with no passwd entry, as in some containers
        return f"uid {uid}"


def current_user() -> str:
    """The user this process runs as."""
    return _user_named(os.geteuid())


def owner_of(path: Path) -> str:
    """The user who owns ``path``."""
    return _user_named(path.stat().st_uid)


@dataclass
class Context:
    """What every check group works from."""

    settings: Any  # the Settings the services run with, but for the throwaway collection
    environment: Environment  # the wiring outside the Knowledge index the services run with
    report: Report
    gateway: Any  # the Model gateway, or None if it cannot be built
    connect: Callable[[str, str | None], Any]  # (Qdrant location, API key) -> a Qdrant client
    run: Callable[
        [Sequence[str]], str
    ]  # runs a host command, returns its output, raises if it fails
    args: argparse.Namespace
    run_id: Any  # the Run whose identifiers the generated Documents carry
    gateway_problem: str | None = None  # why there is no gateway, if the settings name one
    cleanups: list[tuple[str, Callable[[], str]]] = field(default_factory=list)  # name, undo
    current_user: Callable[[], str] = current_user  # the user the suite runs as
    owner_of: Callable[[Path], str] = owner_of  # the user who owns a path
