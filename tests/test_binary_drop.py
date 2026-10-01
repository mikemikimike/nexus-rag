import io
import tarfile

import pytest

import fake_drop
from local_rag.binary_drop import Artifact, DropRefused, QDRANT, install_qdrant_binary, verify_against_sums


def test_a_qdrant_tarball_that_is_not_the_recorded_release_is_refused(tmp_path):
    tarball = tmp_path / fake_drop.QDRANT_TARBALL
    fake_drop.qdrant_tarball(tarball)  # a well-formed tarball, but not the release the repo records

    with pytest.raises(DropRefused, match="checksum"):
        install_qdrant_binary(tarball, tmp_path / "out")

    assert not (tmp_path / "out").exists()


def test_the_repository_records_a_qdrant_version_and_checksum():
    assert QDRANT.version and len(QDRANT.sha256) == 64


def test_the_recorded_release_is_unpacked_as_an_executable(tmp_path):
    record = fake_drop.qdrant_tarball(tmp_path / fake_drop.QDRANT_TARBALL)

    written = install_qdrant_binary(tmp_path / fake_drop.QDRANT_TARBALL, tmp_path / "out", record)

    binary = tmp_path / "out" / "qdrant"
    assert written
    assert binary.read_bytes().startswith(b"#!/bin/sh")
    assert binary.stat().st_mode & 0o111


def test_a_binary_that_reports_another_version_than_recorded_is_refused(tmp_path):
    tarball = tmp_path / fake_drop.QDRANT_TARBALL
    record = fake_drop.qdrant_tarball(tarball, version="1.18.0")  # checksum matches, version does not
    expected = Artifact(record.name, "1.19.1", record.filename, record.sha256)

    with pytest.raises(DropRefused, match="version"):
        install_qdrant_binary(tarball, tmp_path / "out", expected)

    assert not list((tmp_path / "out").glob("*"))


def test_a_tarball_without_the_qdrant_binary_is_refused(tmp_path):
    tarball = tmp_path / fake_drop.QDRANT_TARBALL
    with tarfile.open(tarball, "w:gz") as archive:
        info = tarfile.TarInfo("README")
        archive.addfile(info, io.BytesIO(b""))
    record = Artifact("qdrant", "1.19.1", tarball.name, fake_drop.sha256(tarball))

    with pytest.raises(DropRefused, match="no qdrant binary"):
        install_qdrant_binary(tarball, tmp_path / "out", record)


def test_a_file_is_checked_against_the_drops_sha256sums(tmp_path):
    archive = tmp_path / "local-rag-0.1.0.tar.gz"
    archive.write_bytes(b"code")
    sums = tmp_path / "SHA256SUMS"
    sums.write_text(f"{fake_drop.sha256(archive)}  {archive.name}\n")

    verify_against_sums(archive, sums)  # does not raise

    archive.write_bytes(b"tampered")
    with pytest.raises(DropRefused, match="checksum"):
        verify_against_sums(archive, sums)


def test_a_file_the_sha256sums_do_not_list_is_refused(tmp_path):
    archive = tmp_path / "local-rag-0.1.0.tar.gz"
    archive.write_bytes(b"code")
    sums = tmp_path / "SHA256SUMS"
    sums.write_text(f"{'0' * 64}  something-else.tar.gz\n")

    with pytest.raises(DropRefused, match="not listed"):
        verify_against_sums(archive, sums)

    sums.unlink()
    with pytest.raises(DropRefused, match="SHA256SUMS"):
        verify_against_sums(archive, sums)


def test_an_identical_binary_already_in_place_is_left_alone(tmp_path):
    tarball = tmp_path / fake_drop.QDRANT_TARBALL
    record = fake_drop.qdrant_tarball(tarball)
    install_qdrant_binary(tarball, tmp_path / "out", record)
    before = (tmp_path / "out" / "qdrant").stat().st_mtime_ns

    assert not install_qdrant_binary(tarball, tmp_path / "out", record)

    assert (tmp_path / "out" / "qdrant").stat().st_mtime_ns == before
    assert [p.name for p in (tmp_path / "out").iterdir()] == ["qdrant"]


def test_the_repository_records_the_system_version_it_ships_as_an_image():
    import tomllib
    from pathlib import Path

    from local_rag.binary_drop import SYSTEM_VERSION

    pyproject = tomllib.loads((Path(fake_drop.REPO) / "pyproject.toml").read_text())
    assert SYSTEM_VERSION == pyproject["project"]["version"]
