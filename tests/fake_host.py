"""A stand-in for the machine the Installer runs on: records the commands it would run."""
import sys
from pathlib import Path

from local_rag.installer import CommandFailed
from collections.abc import Sequence


class FakeHost:
    def __init__(self, *, users: set[str] = frozenset(), commands: set[str] | None = None,
                 root: bool = True, machine: str = "x86_64", python: tuple[int, int] = (3, 12),
                 modules: set[str] | None = None, fail_on: str | None = None):
        self.users = set(users)
        self.available = {"systemctl", "useradd", "chown", "python3.12"} if commands is None else set(commands)
        self.modules = {"venv", "ensurepip"} if modules is None else set(modules)
        self._root, self._machine, self._python = root, machine, python
        self.fail_on = fail_on
        self.ran: list[list[str]] = []
        self.lockfiles: list[str] = []  # content of each requirements.lock a command was given

    def run(self, command: Sequence[str]) -> None:
        self.ran.append(list(command))
        for word in command:
            if word.endswith("requirements.lock"):
                self.lockfiles.append(Path(word).read_text())
        if self.fail_on and self.fail_on in " ".join(command):
            raise CommandFailed(" ".join(command))
        if command[1:3] == ["-m", "venv"]:  # what creating a virtual environment leaves behind
            python = Path(command[3]) / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.touch()
        if command[0] == "useradd":
            self.users.add(command[-1])

    def ran_matching(self, *words: str) -> list[list[str]]:
        return [c for c in self.ran if all(w in " ".join(c) for w in words)]

    def user_exists(self, name: str) -> bool:
        return name in self.users

    def which(self, command: str) -> str | None:
        return f"/usr/bin/{command}" if command in self.available else None

    def has_module(self, name: str) -> bool:
        return name in self.modules

    def is_root(self) -> bool:
        return self._root

    def machine(self) -> str:
        return self._machine

    def python_version(self) -> tuple[int, int]:
        return self._python

    def python_executable(self) -> str:
        return sys.executable
