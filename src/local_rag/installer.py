"""The Installer: installs or upgrades the system on the Python 3.12, systemd target.

    sudo python3.12 -m local_rag.installer --drop DIR [--index-url MIRROR_URL]

Python packages come from the Mirror (pip, with the pinned ``requirements.lock``); the Qdrant
server and this system's own code come from the Binary drop in DIR:

    qdrant-x86_64-unknown-linux-gnu.tar.gz   the Qdrant release the repository records
    local-rag-<version>.tar.gz               ``git archive --prefix=local-rag/ HEAD`` of this repository
    SHA256SUMS                               ``sha256sum`` of the code archive

The Installer refuses a drop that does not match (see ``binary_drop``) before it changes
anything, and checks, but does not carry, the operating system packages it needs. It is
idempotent: a second run changes nothing, an upgrade replaces the Qdrant binary, the
code and the unit files (keep local changes to a unit in a drop-in, ``systemctl edit``), and
the env file (settings, secrets), the roots file, the documents and Qdrant's data are never
touched. The command to start with is the one in the code archive:
unpack it anywhere and run it with ``PYTHONPATH=<unpacked>/src``.

Standard library only: it runs before anything else is installed. ``--root`` installs below
another directory, for tests.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import platform
import pwd
import secrets
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TextIO

from local_rag.binary_drop import (
    QDRANT, Artifact, DropRefused, place_binary, sha256_of, verified_qdrant_binary,
    verify_against_sums)


SERVICE_USER = "ragsvc"
PREREQUISITE_COMMANDS = {"systemctl": "systemd", "useradd": "passwd / shadow-utils",
                         "chown": "coreutils"}
LOCKFILE = "requirements.lock"
UNITS_DIRECTORY = "deploy/systemd"
CODE_ARCHIVE_PATTERN = "local-rag-*.tar.gz"


class Layout:
    """Where everything goes, below ``root`` ("/" on a real machine, a temporary directory in tests)."""

    def __init__(self, root: Path):
        self.root = root

    def at(self, absolute: str) -> Path:
        return self.root / absolute.lstrip("/")

    service_directories = property(lambda self: [
        self.at("/var/lib/qdrant"), self.at("/var/lib/qdrant/storage"),
        self.at("/var/lib/qdrant/snapshots")])
    roots_file = property(lambda self: self.at("/etc/local-rag/confluence_roots.txt"))
    units_directory = property(lambda self: self.at("/etc/systemd/system"))
    venv = property(lambda self: self.at("/opt/local-rag/venv"))
    state_file = property(lambda self: self.at("/opt/local-rag/installed.json"))
    qdrant_binary = property(lambda self: self.at("/opt/qdrant/qdrant"))
    env_file = property(lambda self: self.at("/etc/local-rag/local-rag.env"))


class CommandFailed(Exception):
    """A command the Installer ran on the host did not succeed."""


class SystemHost:
    """The machine the Installer runs on: the commands it can run and what it can ask about itself."""

    def run(self, command: Sequence[str]) -> None:
        try:
            subprocess.run(command, check=True)
        except (subprocess.CalledProcessError, OSError) as error:
            raise CommandFailed(f"{' '.join(command)}: {error}") from None

    def user_exists(self, name: str) -> bool:
        try:
            pwd.getpwnam(name)
        except KeyError:
            return False
        return True

    def which(self, command: str) -> str | None:
        return shutil.which(command)

    def has_module(self, name: str) -> bool:
        return importlib.util.find_spec(name) is not None

    def is_root(self) -> bool:
        return os.geteuid() == 0

    def machine(self) -> str:
        return platform.machine()

    def python_version(self) -> tuple[int, int]:
        return sys.version_info[:2]

    def python_executable(self) -> str:
        return sys.executable


def main(argv: Sequence[str], *, host=None, qdrant: Artifact = QDRANT,
         stdout: TextIO | None = None, stderr: TextIO | None = None) -> int:
    """Install or upgrade; returns the process exit code (0 done, 1 refused or failed).

    ``host`` stands in for the machine (tests), ``qdrant`` for the repository's record of the
    Qdrant release.
    """
    out, err = stdout or sys.stdout, stderr or sys.stderr
    host = host or SystemHost()
    args = _parser().parse_args(argv)
    problems = _missing_prerequisites(host, real_root=args.root == Path("/"))
    if problems:
        print("prerequisites are not met:", *(f"  - {p}" for p in problems), sep="\n", file=err)
        return 1
    try:
        # Everything in the drop is verified before the machine is touched.
        archive = _verified_code_archive(args.drop)
        with (verified_qdrant_binary(args.drop / qdrant.filename, qdrant) as binary,
              tempfile.TemporaryDirectory(prefix="local-rag-code-") as scratch):
            source = _unpack_code(archive, Path(scratch))
            _Install(host, Layout(args.root), out, index_url=args.index_url).run(
                binary, archive, source)
    except DropRefused as error:
        print(f"refused: {error}", file=err)
        return 1
    except CommandFailed as error:
        print(f"failed: {error}", file=err)
        return 1
    return 0


class _Install:
    """One run: brings the machine to the state the drop describes, touching only what differs."""

    def __init__(self, host, layout: Layout, out: TextIO, *, index_url: str | None):
        self.host, self.layout, self.out, self.index_url = host, layout, out, index_url

    def done(self, status: str, what: object) -> None:
        print(f"{status:<10}{what}", file=self.out)

    def run(self, qdrant_binary: Path, archive: Path, source: Path) -> None:
        layout = self.layout
        self._service_account()
        self._directories()
        restart = []
        if place_binary(qdrant_binary, layout.qdrant_binary):
            self.done("installed", layout.qdrant_binary)
            restart.append("qdrant.service")
        else:
            self.done("unchanged", layout.qdrant_binary)
        self._env_file()
        if _create_text(layout.roots_file, ROOTS_FILE):
            self.done("created", layout.roots_file)
        if self._python_environment(archive, source):
            restart.append("local-rag-mcp.service")
        self._systemd(source / UNITS_DIRECTORY, restart)

    def _service_account(self) -> None:
        if not self.host.user_exists(SERVICE_USER):
            self.host.run(["useradd", "--system", "--home-dir", "/var/lib/ragsvc",
                           "--create-home", "--shell", "/usr/sbin/nologin", SERVICE_USER])
            self.done("created", f"user {SERVICE_USER}")

    def _directories(self) -> None:
        for directory in self.layout.service_directories:
            if directory.exists():
                continue
            # Built under another name and renamed once it is the service's: an interrupted
            # run leaves nothing a later run would take for finished.
            staged = directory.with_name(f"{directory.name}.new")
            shutil.rmtree(staged, ignore_errors=True)
            staged.mkdir(parents=True)
            try:
                self.host.run(["chown", f"{SERVICE_USER}:{SERVICE_USER}", str(staged)])
                staged.rename(directory)
            except BaseException:
                shutil.rmtree(staged, ignore_errors=True)
                raise
            self.done("created", directory)

    def _env_file(self) -> None:
        layout = self.layout
        layout.env_file.parent.mkdir(parents=True, exist_ok=True)
        if _create_env_file(layout.env_file, lambda staged: self.host.run(
                ["chown", f"root:{SERVICE_USER}", str(staged)])):
            self.done("created", f"{layout.env_file} (new secrets)")
        else:
            self.done("kept", layout.env_file)

    def _python_environment(self, archive: Path, source: Path) -> bool:
        """Build or update the venv; True if the code in it changed."""
        layout = self.layout
        python = layout.venv / "bin" / "python"
        pip = [str(python), "-m", "pip", "install", "--disable-pip-version-check"]
        recorded = self._state()
        wanted = {"code": sha256_of(archive), "lock": sha256_of(source / LOCKFILE)}
        if python.exists() and all(recorded.get(key) == value for key, value in wanted.items()):
            self.done("unchanged", layout.venv)
            return False
        if not python.exists():
            self.host.run([self.host.python_executable(), "-m", "venv", str(layout.venv)])
        if recorded.get("lock") != wanted["lock"] or not python.exists():
            mirror = ["--index-url", self.index_url] if self.index_url else []
            self.host.run([*pip, *mirror, "--only-binary", ":all:", "-r",
                           str(source / LOCKFILE)])
        self.host.run([*pip, "--no-deps", "--force-reinstall", "--no-build-isolation",
                       str(source)])
        self._remember(**wanted)
        self.done("installed", layout.venv)
        return True

    def _systemd(self, units: Path, restart: list[str]) -> None:
        """Install the unit files, and register them and restart what changed."""
        layout = self.layout
        sources = sorted(path for path in units.iterdir() if path.suffix in (".service", ".timer"))
        layout.units_directory.mkdir(parents=True, exist_ok=True)
        rewritten = False
        for source in sources:
            installed = layout.units_directory / source.name
            if not installed.exists() or installed.read_bytes() != source.read_bytes():
                shutil.copyfile(source, installed)
                rewritten = True
                self.done("installed", installed)
        registered = hashlib.sha256(b"".join(
            path.name.encode() + path.read_bytes() for path in sources)).hexdigest()
        register = rewritten or self._state().get("units") != registered
        if register:
            self.host.run(["systemctl", "daemon-reload"])
        for service in restart:
            self.host.run(["systemctl", "try-restart", service])  # does nothing if not running
        if register:
            self.host.run(["systemctl", "enable", "--now", "qdrant.service"])
            self.host.run(["systemctl", "enable", "--now", "local-rag-mcp.service",
                           "local-rag-sync.timer", "local-rag-full.timer"])
            self._remember(units=registered)

    def _state(self) -> dict[str, str]:
        return _read_state(self.layout.state_file)

    def _remember(self, **values: str) -> None:
        state = self._state() | values
        self.layout.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.layout.state_file.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def _read_state(path: Path) -> dict[str, str]:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return state if isinstance(state, dict) else {}


def _unpack_code(archive: Path, scratch: Path) -> Path:
    """Unpack the code archive below ``scratch``; returns the source tree (holding the lockfile)."""
    try:
        with tarfile.open(archive) as tar:
            tar.extractall(scratch, filter="data")
    except (tarfile.TarError, OSError) as error:
        raise DropRefused(f"cannot unpack {archive.name}: {error}") from None
    entries = list(scratch.iterdir())
    source = entries[0] if len(entries) == 1 and entries[0].is_dir() else scratch
    if not (source / LOCKFILE).is_file():
        raise DropRefused(f"{archive.name} has no {LOCKFILE}")
    if not (source / UNITS_DIRECTORY).is_dir():
        raise DropRefused(f"{archive.name} has no {UNITS_DIRECTORY}")
    return source


def _missing_prerequisites(host, *, real_root: bool) -> list[str]:
    """What the machine lacks. Operating system packages are checked, never carried."""
    problems = []
    if host.machine() != "x86_64":
        problems.append(f"this system supports x86_64 Linux only, not {host.machine()}")
    if real_root and not host.is_root():
        problems.append("run as root (or pass --root to install below another directory)")
    if host.python_version() < (3, 12):
        problems.append("Python 3.12 or newer is required (run the Installer with python3.12)")
    if not host.has_module("ensurepip") or not host.has_module("venv"):
        problems.append("the venv and ensurepip modules are missing (Debian: python3-venv)")
    for command, package in PREREQUISITE_COMMANDS.items():
        if host.which(command) is None:
            problems.append(f"{command} not found ({package})")
    return problems


def _verified_code_archive(drop: Path) -> Path:
    """The drop's archive of this system's code, once its checksum agrees with SHA256SUMS."""
    archives = sorted(drop.glob(CODE_ARCHIVE_PATTERN))
    if len(archives) != 1:
        raise DropRefused(
            f"expected one code archive ({CODE_ARCHIVE_PATTERN}) in {drop}, found {len(archives)}")
    verify_against_sums(archives[0], drop / "SHA256SUMS")
    return archives[0]


ENV_FILE = """\
# local-rag settings and secrets, read by every service (systemd EnvironmentFile).
# Written once by the Installer; an upgrade never touches it. Restart the services after
# an edit:  systemctl restart qdrant local-rag-mcp
# Add or revoke a user by editing MCP_TOKENS (comma-separated; every token sees all content).

# Qdrant (listens on localhost; the client key and the server key must stay equal)
QDRANT_URL=http://127.0.0.1:6333
QDRANT_API_KEY={qdrant_key}
QDRANT__SERVICE__API_KEY={qdrant_key}
QDRANT__SERVICE__HOST=127.0.0.1
QDRANT__SERVICE__HTTP_PORT=6333
QDRANT__STORAGE__STORAGE_PATH=/var/lib/qdrant/storage
QDRANT__STORAGE__SNAPSHOTS_PATH=/var/lib/qdrant/snapshots
QDRANT__TELEMETRY_DISABLED=true

# MCP server
MCP_TOKENS={mcp_token}
MCP_HOST=0.0.0.0
MCP_PORT=8765
# MCP_TLS_CERT=/etc/local-rag/tls/server.crt
# MCP_TLS_KEY=/etc/local-rag/tls/server.key

# Model gateway (LiteLLM): set these before the first Sync
# LITELLM_BASE_URL=http://GATEWAY_HOST:4000/v1
# LITELLM_API_KEY=
# RAG_EMBED_MODEL=YOUR-EMBEDDING-MODEL-NAME
# RAG_RERANK_MODEL=

# Confluence (read by confluence-cli during a Sync)
# CONFLUENCE_DOMAIN=confluence.example.test
# CONFLUENCE_API_PATH=/rest/api
# CONFLUENCE_AUTH_TYPE=bearer
# CONFLUENCE_API_TOKEN=
# CONFLUENCE_READ_ONLY=true
"""


ROOTS_FILE = """\
# Confluence roots for the scheduled Sync: one page id or URL per line.
# A Space's home page as a root covers the whole Space. Never overwritten by the Installer.
"""


def _create_env_file(path: Path, prepare: Callable[[Path], None] = lambda staged: None) -> bool:
    """Write the env file with fresh secrets unless one exists; True if it was written."""
    return _create_text(path, ENV_FILE.format(
        qdrant_key=secrets.token_hex(24), mcp_token=secrets.token_hex(24)),
        mode=0o640, prepare=prepare)


def _create_text(path: Path, text: str, mode: int = 0o644,
                 prepare: Callable[[Path], None] = lambda staged: None) -> bool:
    """Write ``path`` unless something is there already; True if it was written.

    The file is complete (and ``prepare`` has run on it, to set its owner) before it appears
    under its name, and an existing file is never replaced, even by a concurrent run.
    """
    if path.exists():
        return False  # without staging anything, so that the directory is not touched either
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.tmp-")
    staged = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            os.fchmod(stream.fileno(), mode)  # not the umask's idea of it
            stream.write(text)
        prepare(staged)
        try:
            os.link(staged, path)  # fails if the name is taken, unlike a rename
        except FileExistsError:
            return False
        return True
    finally:
        staged.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python3.12 -m local_rag.installer",
                                     description=__doc__.split("\n\n")[0])
    parser.add_argument("--drop", type=Path, required=True, help="the Binary drop directory")
    parser.add_argument("--index-url", help="the Mirror's package index (default: pip's own configuration)")
    parser.add_argument("--root", type=Path, default=Path("/"), help="install below this directory")
    return parser


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
