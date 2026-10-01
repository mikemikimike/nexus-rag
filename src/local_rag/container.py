"""The container delivery route: the system and Qdrant as images in the Binary drop.

On the connected machine, build the images and save them with the system's code:

    python3.12 -m local_rag.container save --out DIR

DIR is then the Binary drop carried into the air-gapped network:

    qdrant-image-<qdrant version>.tar       the official Qdrant image, saved
    local-rag-image-<version>.tar           the system's image, built from this repository
    local-rag-<version>.tar.gz              ``git archive`` of this repository (the compose file, the timers)
    SHA256SUMS                              ``sha256sum`` of the three files

On the air-gapped host (Docker with the compose plugin), unpack the code archive anywhere and
run its copy of this command; it refuses a drop that does not match what the repository
records before it loads or starts anything, and is idempotent (it is also the upgrade):

    sudo PYTHONPATH=<unpacked>/src python3.12 -m local_rag.container start --drop DIR

The settings and secrets are an env file outside the images, written once and never
replaced; data and the index live in Docker volumes, which an upgrade does not touch.

Standard library only (see ``installer``): it runs before anything else is installed.
"""
import argparse
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from local_rag.binary_drop import (
    QDRANT, SYSTEM_VERSION, Artifact, DropRefused, parse_reported_version, require_version,
    sha256_of, unpack_code, verified_code_archive, verify_against_sums)
from local_rag.installer import ROOTS_FILE, CommandFailed, Layout, SystemHost, create_text

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SYSTEM_REPOSITORY = "local-rag"
QDRANT_REPOSITORY = "qdrant/qdrant"
QDRANT_BINARY = "/qdrant/qdrant"


class ContainerHost(SystemHost):
    """The machine the command runs on, with Docker."""

    def output(self, command: Sequence[str]) -> str:
        try:
            return subprocess.run(command, check=True, capture_output=True, text=True).stdout
        except (subprocess.CalledProcessError, OSError) as error:
            raise CommandFailed(f"{' '.join(command)}: {error}") from None


def qdrant_tarball(record: Artifact = QDRANT) -> str:
    return f"qdrant-image-{record.version}.tar"


def system_tarball(version: str = SYSTEM_VERSION) -> str:
    return f"local-rag-image-{version}.tar"


COMPOSE_FILE = "deploy/container/compose.yaml"
UNITS_DIRECTORY = "deploy/container"
PREREQUISITE_COMMANDS = {"docker": "Docker Engine", "systemctl": "systemd"}


class ContainerLayout(Layout):
    """Where the container route puts things, below ``root`` ("/" on a real machine)."""

    env_file = property(lambda self: self.at("/etc/local-rag/container.env"))
    compose_file = property(lambda self: self.at("/opt/local-rag/container/compose.yaml"))


def system_image(version: str = SYSTEM_VERSION) -> str:
    return f"{SYSTEM_REPOSITORY}:{version}"


def qdrant_image(record: Artifact = QDRANT) -> str:
    return f"{QDRANT_REPOSITORY}:v{record.version}"


def main(argv: Sequence[str], *, host=None, qdrant: Artifact = QDRANT,
         stdout: TextIO | None = None, stderr: TextIO | None = None) -> int:
    """Run one command; returns the process exit code (0 done, 1 refused or failed)."""
    out, err = stdout or sys.stdout, stderr or sys.stderr
    host = host or ContainerHost()
    args = _parser().parse_args(argv)
    try:
        if args.command == "save":
            _need(host, ["docker", "git"], real_root=False)
            _save(host, args.out, args.repo, qdrant, out)
        else:
            _need(host, list(PREREQUISITE_COMMANDS), real_root=args.root == Path("/"))
            _start(host, args.drop, ContainerLayout(args.root), qdrant, out)
    except DropRefused as error:
        print(f"refused: {error}", file=err)
        return 1
    except CommandFailed as error:
        print(f"failed: {error}", file=err)
        return 1
    return 0


def _save(host, out_directory: Path, repository: Path, qdrant: Artifact, out: TextIO) -> None:
    """Build the system image, pull Qdrant's, check what each says it is, and save them."""
    system, official = system_image(), qdrant_image(qdrant)
    if out_directory.exists():
        raise DropRefused(f"{out_directory} exists already; save into a new directory")
    host.run(["docker", "build", "-f", str(repository / "deploy/container/Dockerfile"),
              "-t", system, str(repository)])
    host.run(["docker", "pull", official])
    _check_versions(host, qdrant)
    out_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out_directory.name}.", dir=out_directory.parent))
    try:
        files = [staging / qdrant_tarball(qdrant), staging / system_tarball(),
                 staging / f"local-rag-{SYSTEM_VERSION}.tar.gz"]
        host.run(["docker", "save", "-o", str(files[0]), official])
        host.run(["docker", "save", "-o", str(files[1]), system])
        host.run(["git", "-C", str(repository), "archive", "--prefix=local-rag/", "-o",
                  str(files[2]), "HEAD"])
        (staging / "SHA256SUMS").write_text(
            "".join(f"{sha256_of(file)}  {file.name}\n" for file in files), encoding="utf-8")
        staging.rename(out_directory)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    for file in files:
        print(f"saved     {out_directory / file.name}", file=out)


def _need(host, commands: Sequence[str], *, real_root: bool) -> None:
    """Refuse to start on a machine that lacks what the command runs. Never carried."""
    missing = [f"{command} not found" for command in commands if host.which(command) is None]
    if real_root and "systemctl" in commands and not host.is_root():
        missing.append("run as root (or pass --root to install below another directory)")
    if missing:
        raise DropRefused("prerequisites are not met: " + "; ".join(missing))


def _start(host, drop: Path, layout: ContainerLayout, qdrant: Artifact, out: TextIO) -> None:
    """Verify the drop, load its images, and bring the stack up; touches the machine only once
    everything the drop holds is known to be what the repository records."""
    archive = verified_code_archive(drop)
    tarballs = {qdrant_image(qdrant): drop / qdrant_tarball(qdrant),
                system_image(): drop / system_tarball()}
    for ref, tarball in tarballs.items():
        verify_against_sums(tarball, drop / "SHA256SUMS")
        _require_tag(tarball, ref)
    with tempfile.TemporaryDirectory(prefix="local-rag-code-") as scratch:
        source = unpack_code(archive, Path(scratch), requires=[COMPOSE_FILE, UNITS_DIRECTORY])
        _require_images_in(source / COMPOSE_FILE, tarballs)
        for tarball in tarballs.values():
            host.run(["docker", "load", "-i", str(tarball)])
        _check_versions(host, qdrant)
        _Stack(host, layout, out).bring_up(source)


def _require_tag(tarball: Path, ref: str) -> None:
    """Refuse an image tarball that does not hold ``ref`` (read from its manifest.json)."""
    try:
        with tarfile.open(tarball) as archive:
            member = archive.extractfile("manifest.json")
            if member is None:
                raise KeyError("manifest.json")
            with member:
                tags = [tag for entry in json.load(member) for tag in entry.get("RepoTags") or []]
    except (tarfile.TarError, OSError, KeyError, ValueError, TypeError, AttributeError) as error:
        raise DropRefused(f"{tarball.name} is not a saved Docker image: {error}") from None
    if ref not in tags:
        raise DropRefused(f"{tarball.name} holds {', '.join(tags) or 'no tag'}, not {ref}")


def _require_images_in(compose_file: Path, tarballs: dict[str, Path]) -> None:
    """Refuse a compose file that would run other images than the ones the drop carries."""
    wanted = set(tarballs)
    named = set(re.findall(r"^\s*image:\s*(\S+)\s*$", compose_file.read_text(encoding="utf-8"),
                           re.MULTILINE))
    if named != wanted:
        raise DropRefused(
            f"the compose file runs {', '.join(sorted(named))}, the drop carries "
            f"{', '.join(sorted(wanted))}")


class _Stack:
    """Puts the files of the stack on the host and starts it, touching only what differs."""

    def __init__(self, host, layout: ContainerLayout, out: TextIO):
        self.host, self.layout, self.out = host, layout, out

    def done(self, status: str, what: object) -> None:
        print(f"{status:<10}{what}", file=self.out)

    def bring_up(self, source: Path) -> None:
        layout = self.layout
        if create_text(layout.env_file, ENV_FILE.format(
                qdrant_key=secrets.token_hex(24), mcp_token=secrets.token_hex(24)), mode=0o600):
            self.done("created", f"{layout.env_file} (new secrets)")
        else:
            self.done("kept", layout.env_file)
        if create_text(layout.roots_file, ROOTS_FILE):
            self.done("created", layout.roots_file)
        _replace_if_different(source / COMPOSE_FILE, layout.compose_file, self.done)
        units = sorted(path for path in (source / UNITS_DIRECTORY).iterdir()
                       if path.suffix in (".service", ".timer"))
        changed = [_replace_if_different(unit, layout.units_directory / unit.name, self.done)
                   for unit in units]
        self.host.run(["docker", "compose", "--project-name", "local-rag", "--env-file",
                       str(layout.env_file), "-f", str(layout.compose_file), "up", "-d"])
        if any(changed):
            self.host.run(["systemctl", "daemon-reload"])
        self.host.run(["systemctl", "enable", "--now",
                       *(unit.name for unit in units if unit.suffix == ".timer")])


def _replace_if_different(source: Path, destination: Path, done) -> bool:
    """Write ``source`` to ``destination`` (atomically) unless it is there already."""
    if destination.is_file() and destination.read_bytes() == source.read_bytes():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = destination.with_name(f".{destination.name}.new")
    shutil.copyfile(source, staged)
    os.replace(staged, destination)
    done("installed", destination)
    return True


ENV_FILE = """\
# local-rag settings and secrets for the container stack (docker compose --env-file).
# Written once by `local_rag.container start`; an upgrade never touches it. Apply an edit with:
#   docker compose --project-name local-rag --env-file THIS_FILE -f /opt/local-rag/container/compose.yaml up -d
# Add or revoke a user by editing MCP_TOKENS (comma-separated; every token sees all content).

# Qdrant (reached by the other containers of the stack only)
QDRANT_API_KEY={qdrant_key}

# MCP server
MCP_TOKENS={mcp_token}
MCP_PORT=8765

# Retrieval mode: hybrid (the default) searches by meaning and keywords and needs an embedding
# model; keyword searches by keywords alone and needs none. Changing it later means a Rebuild.
# RAG_RETRIEVAL=hybrid

# Model gateway (LiteLLM): set these before the first Sync
# LITELLM_BASE_URL=http://GATEWAY_HOST:4000/v1
# LITELLM_API_KEY=
# RAG_EMBED_MODEL=YOUR-EMBEDDING-MODEL-NAME
# RAG_RERANK_MODEL=
# Other tuning (RAG_REBUILD_FAILURE_THRESHOLD, RAG_CHUNK_TOKENS, RAG_TOP_K, ...): the RAG_*
# names in src/local_rag/environment.py, listed in the compose file; unset means the default.

# Confluence (read by confluence-cli during a Sync)
# CONFLUENCE_DOMAIN=confluence.example.test
# CONFLUENCE_API_PATH=/rest/api
# CONFLUENCE_AUTH_TYPE=bearer
# CONFLUENCE_API_TOKEN=
# CONFLUENCE_READ_ONLY=true
"""


def _check_versions(host, qdrant: Artifact) -> None:
    """Ask each image what it is: it must be the version the repository records."""
    require_version("Qdrant", qdrant.version, parse_reported_version(host.output(
        ["docker", "run", "--rm", "--entrypoint", QDRANT_BINARY, qdrant_image(qdrant),
         "--version"])), source=f"the image {qdrant_image(qdrant)}")
    require_version("local-rag", SYSTEM_VERSION, parse_reported_version(host.output(
        ["docker", "run", "--rm", "--entrypoint", "python", system_image(), "-c",
         "import importlib.metadata as m; print(m.version('local-rag'))"])),
        source=f"the image {system_image()}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python3.12 -m local_rag.container",
                                     description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    save = commands.add_parser("save", help="build and save the images (connected machine)")
    save.add_argument("--out", type=Path, required=True, help="the Binary drop directory to write")
    save.add_argument("--repo", type=Path, default=REPOSITORY_ROOT,
                      help="the repository to build from")
    start = commands.add_parser("start", help="verify the drop, load the images, start the stack")
    start.add_argument("--drop", type=Path, required=True, help="the Binary drop directory")
    start.add_argument("--root", type=Path, default=Path("/"), help="install below this directory")
    return parser


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
