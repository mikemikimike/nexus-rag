"""A GitLab stand-in for tests: real git repositories in a temporary directory, served as file:// remotes."""
import subprocess
from pathlib import Path

_IDENTITY = ["-c", "user.name=Test", "-c", "user.email=test@example.test"]


def _git(directory: Path, *args: str) -> str:
    return subprocess.run(["git", *_IDENTITY, "-C", str(directory), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


class FakeGitLab:
    """``url`` is the GitLab base URL: a Repository at ``platform/payments`` is fetched from
    ``<url>/platform/payments.git``."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.url = self.directory.as_uri()

    def git(self, path: str, *args: str) -> str:
        """Runs git in the Repository's remote."""
        return _git(self.directory / f"{path}.git", *args)

    def commit(self, path: str, files: dict[str, str | bytes | None], *, branch: str = "main",
               message: str = "change") -> str:
        """Commits ``files`` (a None value deletes the file) to ``branch`` of the Repository,
        creating both if needed; returns the commit."""
        remote = self.directory / f"{path}.git"
        work = self.directory / ".work" / path
        if not remote.exists():
            remote.mkdir(parents=True)
            _git(remote, "init", "--quiet", "--bare")
        if not work.exists():
            work.mkdir(parents=True)
            _git(work, "init", "--quiet")
        _git(work, "checkout", "--quiet", "-B", branch)
        for name, content in files.items():
            target = work / name
            if content is None:
                target.unlink()
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content if isinstance(content, bytes) else content.encode())
        _git(work, "add", "--all")
        _git(work, "commit", "--quiet", "--allow-empty", "-m", message)
        _git(work, "push", "--quiet", "--force", str(remote), f"{branch}:{branch}")
        return _git(work, "rev-parse", "HEAD")
