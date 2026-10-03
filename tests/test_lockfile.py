import re
import tomllib

from fake_drop import REPO


def lock_entries():
    lines = (REPO / "requirements.lock").read_text().splitlines()
    return [line for line in lines if line.strip() and not line.startswith(("#", " "))]


def test_every_package_in_the_lockfile_is_pinned_to_one_version():
    entries = lock_entries()

    assert entries
    assert all(re.fullmatch(r"[A-Za-z0-9._-]+==[^=<>~!\s]+", line) for line in entries), entries


def test_the_lockfile_covers_every_runtime_dependency_and_the_build_tool():
    pinned = {line.split("==")[0].lower().replace("_", "-") for line in lock_entries()}
    project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
    wanted = {
        re.split(r"[<>=~!\s;\[]", dependency)[0].lower().replace("_", "-")
        for dependency in project["dependencies"]
    }

    assert wanted <= pinned, wanted - pinned
    assert "setuptools" in pinned  # the installer builds the code without build isolation
    assert "local-rag" not in pinned  # the system's own code comes from the Binary drop
