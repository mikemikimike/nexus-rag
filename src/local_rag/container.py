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

A closed network that has mirrors can build the system image itself instead of receiving it.
The drop then holds only the code archive and SHA256SUMS (``save --code-only`` writes exactly
that), and ``start --build`` builds the image on the host from the Debian, PyPI and npm
mirrors and pulls Qdrant's image from a registry the network has:

    sudo PYTHONPATH=<unpacked>/src python3.12 -m local_rag.container start --drop DIR --build \\
        --base-image REGISTRY/python:3.12-slim --qdrant-image REGISTRY/qdrant/qdrant:v<version> \\
        --apt-mirror URL --pip-index-url URL --npm-registry URL

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
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from local_rag.binary_drop import (
    QDRANT,
    SYSTEM_VERSION,
    Artifact,
    DropRefused,
    parse_reported_version,
    require_version,
    sha256_of,
    unpack_code,
    verified_code_archive,
    verify_against_sums,
)
from local_rag.installer import (
    CORPORA_FILE,
    SHARED_SETTINGS,
    CommandFailed,
    Layout,
    SystemHost,
    create_text,
    retire_units,
)

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
DOCKERFILE = "deploy/container/Dockerfile"
UNITS_DIRECTORY = "deploy/container"
RETIRED_UNITS = ("local-rag-container-full.timer", "local-rag-container-full.service")
# The flags that only apply with ``start --build``, and what each is for.
BUILD_FLAGS = {
    "--base-image": "the Python base image to build on (default python:3.12-slim)",
    "--qdrant-image": "where to pull Qdrant's image from (tagged as the compose file names it)",
    "--apt-mirror": "base URL of a Debian mirror serving /debian and /debian-security",
    "--pip-index-url": "the PyPI index to install requirements.lock from",
    "--npm-registry": "the npm registry to install confluence-cli from",
}
PREREQUISITE_COMMANDS = {"docker": "Docker Engine", "systemctl": "systemd"}


class ContainerLayout(Layout):
    """Where the container route puts things, below ``root`` ("/" on a real machine)."""

    env_file = property(lambda self: self.at("/etc/local-rag/container.env"))
    certs_directory = property(lambda self: self.at("/etc/local-rag/certs"))
    compose_file = property(lambda self: self.at("/opt/local-rag/container/compose.yaml"))


def system_image(version: str = SYSTEM_VERSION) -> str:
    return f"{SYSTEM_REPOSITORY}:{version}"


def qdrant_image(record: Artifact = QDRANT) -> str:
    return f"{QDRANT_REPOSITORY}:v{record.version}"


@dataclass(frozen=True)
class BuildOptions:
    """Where ``start --build`` gets what the images are made of; None keeps the public default.

    The names are the Dockerfile's build arguments.
    """

    base_image: str | None = None
    qdrant_image: str | None = None
    apt_mirror: str | None = None
    pip_index_url: str | None = None
    npm_registry: str | None = None

    def build_args(self) -> list[str]:
        named = {
            "BASE_IMAGE": self.base_image,
            "APT_MIRROR": self.apt_mirror,
            "PIP_INDEX_URL": self.pip_index_url,
            "NPM_CONFIG_REGISTRY": self.npm_registry,
        }
        return [f"--build-arg={name}={value}" for name, value in named.items() if value]


def main(
    argv: Sequence[str],
    *,
    host=None,
    qdrant: Artifact = QDRANT,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run one command; returns the process exit code (0 done, 1 refused or failed)."""
    out, err = stdout or sys.stdout, stderr or sys.stderr
    host = host or ContainerHost()
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "start" and not args.build:
        given = [flag for flag in BUILD_FLAGS if getattr(args, flag[2:].replace("-", "_"))]
        if given:
            parser.error(f"{', '.join(given)} only apply with --build")
    try:
        if args.command == "save":
            _need(host, ["git"] if args.code_only else ["docker", "git"], real_root=False)
            _save(host, args.out, args.repo, qdrant, out, code_only=args.code_only)
        else:
            _need(host, list(PREREQUISITE_COMMANDS), real_root=args.root == Path("/"))
            build = (
                BuildOptions(
                    args.base_image,
                    args.qdrant_image,
                    args.apt_mirror,
                    args.pip_index_url,
                    args.npm_registry,
                )
                if args.build
                else None
            )
            _start(host, args.drop, ContainerLayout(args.root), qdrant, out, build)
    except DropRefused as error:
        print(f"refused: {error}", file=err)
        return 1
    except CommandFailed as error:
        print(f"failed: {error}", file=err)
        return 1
    return 0


def _save(
    host, out_directory: Path, repository: Path, qdrant: Artifact, out: TextIO, *, code_only: bool
) -> None:
    """Build the system image, pull Qdrant's, check what each says it is, and save them
    (``code_only``: save just the code archive, for a network that builds its own image)."""
    system, official = system_image(), qdrant_image(qdrant)
    if out_directory.exists():
        raise DropRefused(f"{out_directory} exists already; save into a new directory")
    if not code_only:
        host.run(
            [
                "docker",
                "build",
                "-f",
                str(repository / DOCKERFILE),
                "-t",
                system,
                str(repository),
            ]
        )
        host.run(["docker", "pull", official])
        _check_versions(host, qdrant)
    out_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out_directory.name}.", dir=out_directory.parent))
    try:
        files = [staging / f"local-rag-{SYSTEM_VERSION}.tar.gz"]
        if not code_only:
            files = [staging / qdrant_tarball(qdrant), staging / system_tarball(), *files]
            host.run(["docker", "save", "-o", str(files[0]), official])
            host.run(["docker", "save", "-o", str(files[1]), system])
        host.run(
            [
                "git",
                "-C",
                str(repository),
                "archive",
                "--prefix=local-rag/",
                "-o",
                str(files[-1]),
                "HEAD",
            ]
        )
        (staging / "SHA256SUMS").write_text(
            "".join(f"{sha256_of(file)}  {file.name}\n" for file in files), encoding="utf-8"
        )
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


def _start(
    host,
    drop: Path,
    layout: ContainerLayout,
    qdrant: Artifact,
    out: TextIO,
    build: BuildOptions | None = None,
) -> None:
    """Verify the drop, get the images (load them, or with ``build`` build them from the
    mirrors), and bring the stack up; touches the machine only once everything the drop holds
    is known to be what the repository records."""
    archive = verified_code_archive(drop)
    tarballs = {
        qdrant_image(qdrant): drop / qdrant_tarball(qdrant),
        system_image(): drop / system_tarball(),
    }
    if build is None:
        for ref, tarball in tarballs.items():
            verify_against_sums(tarball, drop / "SHA256SUMS")
            _require_tag(tarball, ref)
    with tempfile.TemporaryDirectory(prefix="local-rag-code-") as scratch:
        source = unpack_code(
            archive,
            Path(scratch),
            requires=[COMPOSE_FILE, UNITS_DIRECTORY, *([DOCKERFILE] if build else [])],
        )
        _require_images_in(source / COMPOSE_FILE, set(tarballs))
        if build is None:
            for tarball in tarballs.values():
                host.run(["docker", "load", "-i", str(tarball)])
        else:
            _build_images(host, source, qdrant, build)
        _check_versions(host, qdrant)
        _Stack(host, layout, out).bring_up(source)


def _build_images(host, source: Path, qdrant: Artifact, options: BuildOptions) -> None:
    """Pull Qdrant's image (from the registry the network has, tagged as the compose file names
    it) and build the system image from the verified code, from the mirrors."""
    official = qdrant_image(qdrant)
    pulled = options.qdrant_image or official
    host.run(["docker", "pull", pulled])
    if pulled != official:
        host.run(["docker", "tag", pulled, official])
    host.run(
        [
            "docker",
            "build",
            "-f",
            str(source / DOCKERFILE),
            *options.build_args(),
            "-t",
            system_image(),
            str(source),
        ]
    )


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


def _require_images_in(compose_file: Path, wanted: set[str]) -> None:
    """Refuse a compose file that would run other images than the ones the drop provides."""
    named = set(
        re.findall(
            r"^\s*image:\s*(\S+)\s*$", compose_file.read_text(encoding="utf-8"), re.MULTILINE
        )
    )
    if named != wanted:
        raise DropRefused(
            f"the compose file runs {', '.join(sorted(named))}, the drop provides "
            f"{', '.join(sorted(wanted))}"
        )


class _Stack:
    """Puts the files of the stack on the host and starts it, touching only what differs."""

    def __init__(self, host, layout: ContainerLayout, out: TextIO):
        self.host, self.layout, self.out = host, layout, out

    def done(self, status: str, what: object) -> None:
        print(f"{status:<10}{what}", file=self.out)

    def bring_up(self, source: Path) -> None:
        layout = self.layout
        if create_text(
            layout.env_file,
            ENV_FILE.format(qdrant_key=secrets.token_hex(24), mcp_token=secrets.token_hex(24)),
            mode=0o600,
        ):
            self.done("created", f"{layout.env_file} (new secrets)")
        else:
            self.done("kept", layout.env_file)
        layout.certs_directory.mkdir(
            parents=True, exist_ok=True
        )  # a missing mount source is made root-owned by Docker
        if create_text(layout.corpora_file, CORPORA_FILE):
            self.done("created", layout.corpora_file)
        _replace_if_different(source / COMPOSE_FILE, layout.compose_file, self.done)
        units = sorted(
            path
            for path in (source / UNITS_DIRECTORY).iterdir()
            if path.suffix in (".service", ".timer")
        )
        changed = [
            retire_units(self.host, layout.units_directory, RETIRED_UNITS, self.done),
            *(
                _replace_if_different(unit, layout.units_directory / unit.name, self.done)
                for unit in units
            ),
        ]
        self.host.run(
            [
                "docker",
                "compose",
                "--project-name",
                "local-rag",
                "--env-file",
                str(layout.env_file),
                "-f",
                str(layout.compose_file),
                "up",
                "-d",
            ]
        )
        if any(changed):
            self.host.run(["systemctl", "daemon-reload"])
        timers = [unit.name for unit in units if unit.suffix == ".timer"]
        self.host.run(
            ["systemctl", "enable", "--now", *(name for name in timers if "@" not in name)]
        )  # a template is opted into per Corpus


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


ENV_FILE = (
    """\
# local-rag settings and secrets for the container stack (docker compose --env-file).
# Written once by `local_rag.container start`; an upgrade never touches it. Apply an edit with:
#   docker compose --project-name local-rag --env-file THIS_FILE -f /opt/local-rag/container/compose.yaml up -d
# Add or revoke a user by editing MCP_TOKENS (comma-separated; every token sees all content).

# Qdrant (reached by the other containers of the stack only)
QDRANT_API_KEY={qdrant_key}

# MCP server
MCP_TOKENS={mcp_token}
MCP_PORT=8765

# An internal CA signed Confluence's certificate ("unable to verify the first certificate"):
# put its PEM (with any intermediate certificates) in /etc/local-rag/certs and name it here.
# NODE_EXTRA_CA_CERTS=/etc/local-rag/certs/ca.pem

"""
    + SHARED_SETTINGS
)


def _check_versions(host, qdrant: Artifact) -> None:
    """Ask each image what it is: it must be the version the repository records."""
    require_version(
        "Qdrant",
        qdrant.version,
        parse_reported_version(
            host.output(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--entrypoint",
                    QDRANT_BINARY,
                    qdrant_image(qdrant),
                    "--version",
                ]
            )
        ),
        source=f"the image {qdrant_image(qdrant)}",
    )
    require_version(
        "local-rag",
        SYSTEM_VERSION,
        parse_reported_version(
            host.output(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--entrypoint",
                    "python",
                    system_image(),
                    "-c",
                    "import importlib.metadata as m; print(m.version('local-rag'))",
                ]
            )
        ),
        source=f"the image {system_image()}",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3.12 -m local_rag.container", description=__doc__.split("\n\n")[0]
    )
    commands = parser.add_subparsers(dest="command", required=True)
    save = commands.add_parser("save", help="build and save the images (connected machine)")
    save.add_argument(
        "--code-only",
        action="store_true",
        help="save only the code archive, for a network that builds the image itself",
    )
    save.add_argument("--out", type=Path, required=True, help="the Binary drop directory to write")
    save.add_argument(
        "--repo", type=Path, default=REPOSITORY_ROOT, help="the repository to build from"
    )
    start = commands.add_parser("start", help="verify the drop, load the images, start the stack")
    start.add_argument("--drop", type=Path, required=True, help="the Binary drop directory")
    start.add_argument("--root", type=Path, default=Path("/"), help="install below this directory")
    start.add_argument(
        "--build",
        action="store_true",
        help="build the system image from the code archive and the mirrors, and pull Qdrant's "
        "image, instead of loading the drop's image tarballs",
    )
    for flag, what in BUILD_FLAGS.items():
        start.add_argument(flag, help=f"with --build: {what}")
    return parser


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
