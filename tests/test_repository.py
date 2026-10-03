"""The repository after the restructure: nothing of the old layout is left, and the README
and the docs it links point only at things that exist."""

import re

import pytest

from fake_drop import REPO

README = (REPO / "README.md").read_text()
DOCS = [path for path in (REPO / "docs").glob("*.md")]


def tracked_files():
    skipped = {".git", ".venv", ".pytest_cache", "__pycache__", "local_rag.egg-info"}
    return [
        path
        for path in REPO.rglob("*")
        if path.is_file() and not skipped & set(path.relative_to(REPO).parts)
    ]


def test_no_legacy_module_script_or_requirements_file_remains():
    names = {path.relative_to(REPO).as_posix() for path in tracked_files()}

    assert not (REPO / "local-rag").exists()
    assert "requirements.txt" not in names
    assert not [name for name in names if name.endswith(".sh")], "shell glue came back"
    assert not [name for name in names if name.startswith("scripts/")]


@pytest.mark.parametrize(
    "document", ["README.md", *sorted(p.relative_to(REPO).as_posix() for p in DOCS)]
)
def test_the_readme_and_the_docs_name_only_files_that_exist(document):
    text = (REPO / document).read_text()
    here = (REPO / document).parent
    paths = {
        REPO / path
        for path in re.findall(r"`((?:src|deploy|examples|docs|tests)/[\w./-]+?)`", text)
    }
    paths |= {here / link for link in re.findall(r"\]\(([\w./-]+)(?:#[\w-]*)?\)", text)}

    assert [str(path) for path in sorted(paths) if not path.exists()] == []


def test_the_readme_documents_both_delivery_routes_and_the_access_limit():
    assert "local_rag.installer" in README and "Binary drop" in README and "Mirror" in README
    assert "local_rag.container" in README
    assert "Every token sees everything" in README


def test_no_document_still_describes_the_old_layout():
    old = (
        "install_vm",
        "download_wheels",
        "download_qdrant",
        "offline_install",
        "ingest.py",
        "export_confluence",
        "wheelhouse",
        "scripts/smoke_test",
    )
    mentions = [
        (path.relative_to(REPO).as_posix(), word)
        for path in tracked_files()
        if path.suffix in {".md", ".py", ".service", ".timer", ".yaml", ".toml", ".snippet"}
        and path.name != "test_repository.py"
        for word in old
        if word in path.read_text(errors="ignore")
    ]

    assert mentions == []


def test_the_module_map_names_every_module_and_only_modules():
    development = (REPO / "docs/development.md").read_text()
    table = development.split("| Module | Role |")[1].split("\n\n")[0]
    rows = [line.split("|")[1] for line in table.splitlines() if line.startswith("| `")]
    named = {name for row in rows for name in re.findall(r"`(\w+)`", row)}
    modules = {path.stem for path in (REPO / "src/local_rag").glob("*.py")} - {"__init__"}

    assert named == modules
