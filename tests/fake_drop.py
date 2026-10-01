"""Builds Binary drops for tests: a fake Qdrant tarball and a code archive."""
import hashlib
import io
import tarfile
from pathlib import Path

from local_rag.binary_drop import Artifact

REPO = Path(__file__).resolve().parent.parent
QDRANT_TARBALL = "qdrant-x86_64-unknown-linux-gnu.tar.gz"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def qdrant_tarball(path: Path, version: str = "1.19.1") -> Artifact:
    """A tarball holding an executable `qdrant` that reports ``version``; returns its record."""
    script = f"#!/bin/sh\necho 'qdrant {version}'\n".encode()
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo("qdrant")
        info.size, info.mode = len(script), 0o755
        archive.addfile(info, io.BytesIO(script))
    return Artifact(name="qdrant", version=version, filename=path.name, sha256=sha256(path))


def write_sums(directory: Path) -> None:
    archive = next(directory.glob("local-rag-*.tar.gz"))
    (directory / "SHA256SUMS").write_text(f"{sha256(archive)}  {archive.name}\n")


def code_archive(path: Path, *, lock: str = "pytest==9.0.0\n",
                 extra: dict[str, bytes] | None = None) -> None:
    """An archive of a tiny source tree carrying the real systemd units and a lockfile."""
    with tarfile.open(path, "w:gz") as archive:
        def add(name: str, data: bytes, mode: int = 0o644) -> None:
            info = tarfile.TarInfo(f"local-rag/{name}")
            info.size, info.mode = len(data), mode
            archive.addfile(info, io.BytesIO(data))

        add("requirements.lock", lock.encode())
        add("pyproject.toml", b"[project]\nname = 'local-rag'\n")
        for name, data in (extra or {}).items():
            add(name, data)
        for unit in sorted((REPO / "deploy" / "systemd").iterdir()):
            add(f"deploy/systemd/{unit.name}", unit.read_bytes())


def drop(directory: Path, *, qdrant_version: str = "1.19.1", lock: str = "pytest==9.0.0\n"
         ) -> Artifact:
    """A complete Binary drop in ``directory``; returns the Qdrant record that matches it."""
    directory.mkdir(parents=True, exist_ok=True)
    record = qdrant_tarball(directory / QDRANT_TARBALL, qdrant_version)
    archive = directory / "local-rag-0.1.0.tar.gz"
    code_archive(archive, lock=lock)
    write_sums(directory)
    return record
