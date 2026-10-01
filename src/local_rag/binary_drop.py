"""Binary drop verification: what the repository expects of the files carried in by hand.

The repository records the Qdrant release it expects (version and sha256 of the release
tarball). A file that does not match is refused before it is run or installed. Standard
library only, and independent of how the files are then installed, so the systemd route
(``installer``) and the container route share it.
"""
import hashlib
import os
import shutil
import subprocess
import tarfile
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


class DropRefused(Exception):
    """A file of the Binary drop is not what the repository expects."""


@dataclass(frozen=True)
class Artifact:
    """A file the repository expects in the Binary drop."""

    name: str
    version: str
    filename: str
    sha256: str


# The Qdrant release the system is tested against. To move to another release, change these
# two values together: the sha256 is of the tarball published on the Qdrant release page.
QDRANT = Artifact(
    name="qdrant",
    version="1.19.1",
    filename="qdrant-x86_64-unknown-linux-gnu.tar.gz",
    sha256="eef986e769d4d3e806dd2d546e1b4ecdd416211e54d34b4ed764fac7c58e1085",
)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checksum(path: Path, expected: str, *, what: str) -> None:
    """Refuse ``path`` unless its sha256 is ``expected``."""
    try:
        actual = sha256_of(path)
    except OSError as error:
        raise DropRefused(f"cannot read {what}: {error}") from None
    if actual != expected.lower():
        raise DropRefused(
            f"checksum of {what} does not match: expected {expected}, got {actual}")


def verify_against_sums(path: Path, sums: Path) -> None:
    """Refuse ``path`` unless the checksum file ``sums`` (``sha256sum`` format) lists it and agrees."""
    try:
        lines = sums.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise DropRefused(f"cannot read {sums.name}: {error}") from None
    for line in lines:
        checksum, _, name = line.partition(" ")
        if name.lstrip(" *") == path.name:
            verify_checksum(path, checksum, what=f"{path.name} (listed in {sums.name})")
            return
    raise DropRefused(f"{path.name} is not listed in {sums.name}")


@contextmanager
def verified_qdrant_binary(tarball: Path, record: Artifact = QDRANT) -> Iterator[Path]:
    """The Qdrant binary in ``tarball``, once the drop is known to match ``record``.

    The checksum is checked before anything is unpacked, and the unpacked binary is asked its
    version (``qdrant --version``). Anything that does not match
    raises DropRefused. The binary exists only inside the ``with`` block.
    """
    verify_checksum(tarball, record.sha256, what=f"{tarball.name} (Qdrant {record.version})")
    with tempfile.TemporaryDirectory(prefix="qdrant-drop-") as scratch:
        unpacked = Path(scratch) / "qdrant"
        _unpack_binary(tarball, unpacked)
        reported = _reported_version(unpacked)
        if reported != record.version:
            raise DropRefused(
                f"Qdrant version does not match: the repository expects {record.version}, "
                f"the binary in {tarball.name} reports {reported or 'nothing'}")
        yield unpacked


def place_binary(binary: Path, destination: Path) -> bool:
    """Put ``binary`` at ``destination``: atomically (a running server keeps its old file) if it
    differs, not at all if identical. Returns whether anything was written."""
    if destination.is_file() and sha256_of(destination) == sha256_of(binary):
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = destination.with_name(destination.name + ".new")
    shutil.copyfile(binary, staged)
    staged.chmod(0o755)
    os.replace(staged, destination)
    return True


def install_qdrant_binary(tarball: Path, destination: Path, record: Artifact = QDRANT) -> bool:
    """Verify the Qdrant tarball against ``record`` and put its binary in ``destination/qdrant``.

    A refused drop leaves nothing behind. Returns whether anything was written.
    """
    with verified_qdrant_binary(tarball, record) as binary:
        return place_binary(binary, destination / "qdrant")


def _unpack_binary(tarball: Path, target: Path) -> None:
    try:
        with tarfile.open(tarball) as archive:
            for member in archive:
                if member.isfile() and Path(member.name).name == "qdrant":
                    source = archive.extractfile(member)
                    assert source is not None
                    with source, target.open("wb") as out:
                        shutil.copyfileobj(source, out)
                    target.chmod(0o755)
                    return
    except (tarfile.TarError, OSError) as error:
        raise DropRefused(f"cannot read {tarball.name}: {error}") from None
    raise DropRefused(f"no qdrant binary in {tarball.name}")


def _reported_version(binary: Path) -> str:
    """The version a Qdrant binary reports: ``qdrant --version`` prints ``qdrant 1.19.1``."""
    try:
        result = subprocess.run([str(binary), "--version"], capture_output=True, text=True,
                                timeout=30, check=True)
    except (OSError, subprocess.SubprocessError) as error:
        raise DropRefused(f"the Qdrant binary does not run: {error}") from None
    words = result.stdout.split()
    return words[-1] if words else ""
