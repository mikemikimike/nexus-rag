"""The repository after the restructure: nothing of the old layout is left, and the README
points only at things that exist."""

import re

from fake_drop import REPO

README = (REPO / "README.md").read_text()


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


def test_the_readme_names_only_files_that_exist():
    paths = set(re.findall(r"`((?:src|deploy|examples|docs|tests)/[\w./-]+?)`", README))
    paths |= set(re.findall(r"\]\(([\w./-]+\.md)\)", README))

    assert paths
    assert [path for path in sorted(paths) if not (REPO / path).exists()] == []


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
