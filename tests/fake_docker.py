"""A stand-in for a machine with Docker: images are tarballs whose manifest names their tag."""

import io
import json
import tarfile
from collections.abc import Sequence
from pathlib import Path

import fake_drop
from fake_host import FakeHost
from local_rag.installer import CommandFailed


def image_tarball(path: Path, ref: str, content: str = "") -> None:
    """What ``docker save`` writes: a manifest.json naming the tags, and the layers."""
    manifest = json.dumps([{"Config": "config.json", "RepoTags": [ref], "Layers": []}]).encode()
    with tarfile.open(path, "w") as archive:
        for name, data in (
            ("manifest.json", manifest),
            ("config.json", f"{ref}{content}".encode()),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


class FakeDocker(FakeHost):
    """Records the commands, and answers the ones about images the way Docker would.

    ``reports`` is the version each image says it is (by tag); an image reports what its
    tag claims unless a test says it lies.
    """

    def __init__(self, *, reports: dict[str, str] | None = None, docker: bool = True, **options):
        super().__init__(**options)
        self.available |= {"docker", "git"} if docker else set()
        self.reports = reports or {}
        self.images: set[str] = set()

    def output(self, command: Sequence[str]) -> str:
        self.ran.append(list(command))
        if command[:2] == ["docker", "run"]:
            ref = command[command.index("--entrypoint") + 2]
            if ref not in self.images:
                raise CommandFailed(f"Unable to find image {ref}")
            if ref in self.reports:
                return f"{self.reports[ref]}\n"
            version = ref.rsplit(":", 1)[1].lstrip("v")
            return f"qdrant {version}\n" if ref.startswith("qdrant/") else f"{version}\n"
        raise CommandFailed(" ".join(command))

    def run(self, command: Sequence[str]) -> None:
        if command[:2] == ["docker", "load"]:
            self.ran.append(list(command))
            with tarfile.open(command[3]) as archive:
                member = archive.extractfile("manifest.json")
                assert member is not None
                for entry in json.load(member):
                    self.images.update(entry["RepoTags"])
        elif command[:2] == ["docker", "save"]:
            self.ran.append(list(command))
            image_tarball(Path(command[3]), command[4])
        elif command[:2] == ["docker", "build"]:
            self.ran.append(list(command))
            self.images.add(command[command.index("-t") + 1])
        elif command[:2] == ["docker", "pull"]:
            self.ran.append(list(command))
            self.images.add(command[2])
        elif "archive" in command and command[0] == "git":
            self.ran.append(list(command))
            fake_drop.code_archive(Path(command[command.index("-o") + 1]))
        else:
            super().run(command)
