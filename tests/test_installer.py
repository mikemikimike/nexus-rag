import io
import subprocess
import tarfile
from pathlib import Path
import sys

import pytest

import fake_drop
from fake_host import FakeHost
from local_rag.binary_drop import Artifact
from local_rag.installer import main


class Machine:
    """A temporary root directory with a Binary drop beside it, and the Installer's command."""

    def __init__(self, tmp_path, **drop_options):
        self.root = tmp_path / "root"
        self.drop = tmp_path / "drop"
        self.qdrant_record = fake_drop.drop(self.drop, **drop_options)
        self.host = FakeHost()

    def install(self, *extra, host=None, **options):
        out, err = io.StringIO(), io.StringIO()
        options = {"qdrant": self.qdrant_record, **options}
        code = main(["--drop", str(self.drop), "--root", str(self.root), *extra],
                    host=host or self.host, stdout=out, stderr=err, **options)
        return code, out.getvalue(), err.getvalue()

    def path(self, absolute: str):
        return self.root / absolute.lstrip("/")

    def env(self) -> dict[str, str]:
        text = self.path("/etc/local-rag/local-rag.env").read_text()
        pairs = (line.split("=", 1) for line in text.splitlines()
                 if line.strip() and not line.startswith("#"))
        return {key: value for key, value in pairs}


@pytest.fixture
def machine(tmp_path):
    return Machine(tmp_path)


def test_first_install_generates_the_secrets_once_into_one_env_file(machine):
    code, out, err = machine.install()

    assert code == 0, err
    env = machine.env()
    assert len(env["QDRANT_API_KEY"]) >= 32
    assert env["QDRANT__SERVICE__API_KEY"] == env["QDRANT_API_KEY"]
    assert len(env["MCP_TOKENS"]) >= 32 and "," not in env["MCP_TOKENS"]
    assert env["QDRANT_API_KEY"] != env["MCP_TOKENS"]
    assert env["QDRANT_URL"] == "http://127.0.0.1:6333"
    assert oct(machine.path("/etc/local-rag/local-rag.env").stat().st_mode & 0o777) == "0o640"


def test_the_installer_runs_on_a_machine_with_only_the_standard_library():
    """On a bare machine nothing but Python 3.12 exists yet, so importing the package
    must not pull in the Knowledge index and its dependencies."""
    program = (
        "import sys\n"
        "for blocked in ('qdrant_client', 'mcp', 'llama_index', 'bs4'):\n"
        "    sys.modules[blocked] = None\n"  # makes importing it raise ImportError
        "import local_rag.installer\n")
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


def test_the_package_still_offers_its_public_names():
    import local_rag

    assert local_rag.KnowledgeIndex.__name__ == "KnowledgeIndex"
    assert local_rag.Settings.__name__ == "Settings"
    assert "KnowledgeIndex" in dir(local_rag)
    with pytest.raises(AttributeError):
        local_rag.NoSuchName


def snapshot(root):
    """Every path below root with its content and mtime: what "nothing changed" means."""
    return {str(p.relative_to(root)): (p.read_bytes() if p.is_file() else None, p.stat().st_mtime_ns)
            for p in sorted(root.rglob("*"))}


def test_an_existing_env_file_is_never_overwritten(machine):
    env_file = machine.path("/etc/local-rag/local-rag.env")
    env_file.parent.mkdir(parents=True)
    env_file.write_text("MCP_TOKENS=alice,bob\nQDRANT_API_KEY=mine\n")

    code, _, err = machine.install()

    assert code == 0, err
    assert env_file.read_text() == "MCP_TOKENS=alice,bob\nQDRANT_API_KEY=mine\n"


def test_rerunning_the_installer_changes_nothing(machine):
    machine.install()
    before = snapshot(machine.root)
    secrets_before = machine.env()

    code, out, err = machine.install()

    assert code == 0, err
    assert snapshot(machine.root) == before
    assert machine.env() == secrets_before


def test_a_qdrant_binary_that_is_not_the_release_the_repository_records_is_refused(machine):
    # The drop holds a well-formed tarball, but the repository's own record knows another one.
    code, _, err = machine.install(qdrant=main_default_record())

    assert code == 1
    assert "refused" in err and "checksum" in err
    assert not machine.root.exists()
    assert machine.host.ran == []


def test_a_qdrant_binary_of_the_wrong_version_is_refused_before_anything_is_installed(tmp_path):
    machine = Machine(tmp_path, qdrant_version="1.18.0")
    expected = Artifact("qdrant", "1.19.1", machine.qdrant_record.filename,
                        machine.qdrant_record.sha256)

    code, _, err = machine.install(qdrant=expected)

    assert code == 1 and "version" in err
    assert not machine.root.exists()
    assert machine.host.ran == []


def test_a_code_archive_that_does_not_match_the_drops_checksums_is_refused(machine):
    archive = machine.drop / "local-rag-0.1.0.tar.gz"
    archive.write_bytes(archive.read_bytes() + b"tampered")

    code, _, err = machine.install()

    assert code == 1 and "checksum" in err
    assert not machine.root.exists()
    assert machine.host.ran == []


def test_a_drop_without_a_code_archive_is_refused(machine):
    (machine.drop / "local-rag-0.1.0.tar.gz").unlink()

    code, _, err = machine.install()

    assert code == 1 and "code archive" in err
    assert not machine.root.exists()


def main_default_record():
    from local_rag.binary_drop import QDRANT
    return QDRANT


def test_missing_prerequisites_are_reported_together_and_nothing_is_installed(machine):
    host = FakeHost(commands={"useradd"}, modules={"venv"}, python=(3, 11))

    code, _, err = machine.install(host=host)

    assert code == 1
    assert "systemctl" in err                  # systemd itself
    assert "ensurepip" in err                  # Debian packages it apart: python3.12-venv
    assert "Python 3.12" in err
    assert not machine.root.exists()
    assert host.ran == []


def test_a_machine_that_is_not_x86_64_linux_is_refused(machine):
    code, _, err = machine.install(host=FakeHost(machine="aarch64"))

    assert code == 1 and "x86_64" in err
    assert not machine.root.exists()


def test_installing_below_the_real_root_needs_root_but_a_test_root_does_not(machine):
    host = FakeHost(root=False)

    assert machine.install(host=host)[0] == 0  # --root given: a temporary root, no privileges needed
    ran = len(host.ran)

    err = io.StringIO()
    code = main(["--drop", str(machine.drop)], host=host, qdrant=machine.qdrant_record,
                stdout=io.StringIO(), stderr=err)
    assert code == 1 and "run as root" in err.getvalue()
    assert len(host.ran) == ran


def test_the_service_account_is_created_once(machine):
    machine.install()
    created = machine.host.ran_matching("useradd")
    assert len(created) == 1 and created[0][-1] == "ragsvc"
    assert "--system" in created[0] and "/usr/sbin/nologin" in created[0]

    machine.install()  # ragsvc exists now
    assert len(machine.host.ran_matching("useradd")) == 1


def chowned(machine, owner, absolute):
    """Whether ``owner`` was set on the item, or on the staging copy that was then renamed to it."""
    target = machine.path(absolute)
    return any(c[:2] == ["chown", owner] and Path(c[2]).parent == target.parent
               and target.name in Path(c[2]).name for c in machine.host.ran)


def test_data_directories_and_the_env_file_are_given_to_the_service(machine):
    machine.install()

    for directory in ("/var/lib/qdrant/storage", "/var/lib/qdrant/snapshots"):
        assert machine.path(directory).is_dir()
        assert chowned(machine, "ragsvc:ragsvc", directory)
    # root owns the file (systemd reads it as root); the service group may read it
    assert chowned(machine, "root:ragsvc", "/etc/local-rag/local-rag.env")
    assert not any("-R" in c for c in machine.host.ran)  # never a sweep over existing data


def test_existing_data_and_documents_are_left_exactly_as_they_are(machine):
    index_file = machine.path("/var/lib/qdrant/storage/collections/live/segment.dat")
    index_file.parent.mkdir(parents=True)
    index_file.write_bytes(b"the index")
    roots = machine.path("/etc/local-rag/corpora.toml")
    roots.parent.mkdir(parents=True)
    roots.write_text("# my space\n")
    before = snapshot(machine.path("/var/lib/qdrant/storage"))

    machine.install()
    machine.install()

    assert snapshot(machine.path("/var/lib/qdrant/storage")) == before
    assert roots.read_text() == "# my space\n"
    assert not any(str(index_file.parent) in " ".join(c) for c in machine.host.ran)


def test_a_corpora_file_is_provided_for_the_unit_to_read_when_there_is_none(machine):
    machine.install()

    roots = machine.path("/etc/local-rag/corpora.toml")
    assert roots.is_file()
    assert all(line.startswith("#") or not line.strip() for line in roots.read_text().splitlines())


def lock_installs(host):
    return host.ran_matching("pip", "install", " -r ")


def code_installs(host):
    return host.ran_matching("pip", "install", "--no-deps")


def test_the_python_environment_is_built_from_the_pinned_lockfile_and_the_code_archive(tmp_path):
    machine = Machine(tmp_path, lock="alpha==1.0\nbeta==2.0\n")

    code, _, err = machine.install()

    assert code == 0, err
    venv = str(machine.path("/opt/local-rag/venv"))
    assert machine.host.ran_matching("-m", "venv", venv)
    (requested,) = lock_installs(machine.host)
    assert machine.host.lockfiles == ["alpha==1.0\nbeta==2.0\n"]  # the drop's lockfile, as carried
    assert requested[0] == f"{venv}/bin/python"
    (own_code,) = code_installs(machine.host)
    assert "--no-build-isolation" in own_code  # build requirements come from the lockfile too


def test_the_mirror_is_given_to_pip_when_named(machine):
    machine.install("--index-url", "https://mirror.example.test/simple")

    (requested,) = lock_installs(machine.host)
    assert requested[requested.index("--index-url") + 1] == "https://mirror.example.test/simple"


def test_rerunning_builds_nothing_again(machine):
    machine.install()
    ran = len(machine.host.ran)

    machine.install()

    assert not [c for c in machine.host.ran[ran:] if c[0] != "systemctl"]


def test_an_upgrade_reinstalls_the_code_and_requests_packages_only_when_the_lockfile_changed(tmp_path):
    machine = Machine(tmp_path, lock="alpha==1.0\n")
    machine.install()

    fake_drop.code_archive(machine.drop / "local-rag-0.1.0.tar.gz", lock="alpha==1.0\n",
                           extra={"src/local_rag/new.py": b"# changed code\n"})
    fake_drop.write_sums(machine.drop)
    machine.install()
    assert len(code_installs(machine.host)) == 2 and len(lock_installs(machine.host)) == 1

    fake_drop.code_archive(machine.drop / "local-rag-0.1.0.tar.gz", lock="alpha==1.1\n")
    fake_drop.write_sums(machine.drop)
    machine.install()
    assert len(code_installs(machine.host)) == 3 and len(lock_installs(machine.host)) == 2
    assert machine.host.lockfiles[-1] == "alpha==1.1\n"


def test_an_install_that_failed_part_way_is_retried_by_the_next_run(machine):
    machine.host.fail_on = "pip install"

    code, _, err = machine.install()
    assert code == 1 and "pip" in err

    machine.host.fail_on = None
    code, _, err = machine.install()
    assert code == 0, err
    assert len(lock_installs(machine.host)) == 2 and len(code_installs(machine.host)) == 1


UNITS = ["local-rag-full.service", "local-rag-full.timer", "local-rag-mcp.service",
         "local-rag-sync.service", "local-rag-sync.timer", "qdrant.service"]


def unit(machine, name):
    return machine.path(f"/etc/systemd/system/{name}").read_text()


def directives(text, key):
    return [line.split("=", 1)[1] for line in text.splitlines() if line.startswith(f"{key}=")]


def test_plain_services_and_timers_call_the_command_line(machine):
    machine.install()

    for name in UNITS:
        assert machine.path(f"/etc/systemd/system/{name}").is_file(), name
    assert directives(unit(machine, "local-rag-sync.service"), "ExecStart") == [
        "/opt/local-rag/venv/bin/local-rag sync confluence"]
    assert directives(unit(machine, "local-rag-full.service"), "ExecStart") == [
        "/opt/local-rag/venv/bin/local-rag sync confluence"]
    assert directives(unit(machine, "local-rag-mcp.service"), "ExecStart") == [
        "/opt/local-rag/venv/bin/local-rag-mcp"]
    assert directives(unit(machine, "qdrant.service"), "ExecStart") == ["/opt/qdrant/qdrant"]
    assert directives(unit(machine, "local-rag-sync.timer"), "OnCalendar") == ["hourly"]
    assert directives(unit(machine, "local-rag-full.timer"), "OnCalendar") == ["Sun 03:00"]


def test_every_service_loads_the_one_env_file_the_installer_creates(machine):
    machine.install()

    for name in ("qdrant", "local-rag-mcp", "local-rag-sync", "local-rag-full"):
        assert directives(unit(machine, f"{name}.service"), "EnvironmentFile") == [
            "/etc/local-rag/local-rag.env"], name
    assert machine.env()["QDRANT__STORAGE__STORAGE_PATH"] == "/var/lib/qdrant/storage"
    assert directives(unit(machine, "qdrant.service"), "WorkingDirectory") == ["/var/lib/qdrant"]


def test_the_units_are_registered_and_started(machine):
    machine.install()

    ran = machine.host.ran
    assert ["systemctl", "daemon-reload"] in ran
    assert ["systemctl", "enable", "--now", "qdrant.service"] in ran
    assert ["systemctl", "enable", "--now", "local-rag-mcp.service", "local-rag-sync.timer",
            "local-rag-full.timer"] in ran
    assert ran.index(["systemctl", "daemon-reload"]) < ran.index(
        ["systemctl", "enable", "--now", "qdrant.service"])


def test_rerunning_leaves_the_units_and_systemd_alone(machine):
    machine.install()
    ran = len(machine.host.ran)
    before = snapshot(machine.path("/etc/systemd/system"))

    machine.install()

    assert snapshot(machine.path("/etc/systemd/system")) == before
    assert machine.host.ran[ran:] == []


def test_an_upgrade_replaces_the_units_and_restarts_what_changed(tmp_path):
    machine = Machine(tmp_path)
    machine.install()
    unit_path = machine.path("/etc/systemd/system/local-rag-sync.timer")
    unit_path.write_text("an older unit\n")
    ran = len(machine.host.ran)

    machine.install()

    assert "OnCalendar=hourly" in unit_path.read_text()
    after = machine.host.ran[ran:]
    assert ["systemctl", "daemon-reload"] in after
    assert ["systemctl", "try-restart", "local-rag-mcp.service"] not in after  # same code: no restart


def test_an_upgrade_restarts_the_services_whose_files_changed(tmp_path):
    machine = Machine(tmp_path)
    machine.install()
    ran = len(machine.host.ran)

    fake_drop.code_archive(machine.drop / "local-rag-0.1.0.tar.gz", extra={"x": b"new code"})
    fake_drop.write_sums(machine.drop)
    new_qdrant = fake_drop.qdrant_tarball(machine.drop / fake_drop.QDRANT_TARBALL, "1.20.0")
    machine.qdrant_record = new_qdrant
    machine.install()

    after = machine.host.ran[ran:]
    assert ["systemctl", "try-restart", "local-rag-mcp.service"] in after
    assert ["systemctl", "try-restart", "qdrant.service"] in after


def test_a_failed_registration_with_systemd_is_retried_by_the_next_run(machine):
    machine.host.fail_on = "enable"
    assert machine.install()[0] == 1

    machine.host.fail_on = None
    code, _, err = machine.install()

    assert code == 0, err
    assert ["systemctl", "enable", "--now", "qdrant.service"] in machine.host.ran[-3:]


def test_a_code_archive_without_the_units_or_the_lockfile_is_refused_before_anything_changes(machine):
    archive = machine.drop / "local-rag-0.1.0.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("local-rag/requirements.lock")
        tar.addfile(info, io.BytesIO(b""))
    fake_drop.write_sums(machine.drop)

    code, _, err = machine.install()

    assert code == 1 and "deploy/systemd" in err
    assert not machine.root.exists() and machine.host.ran == []


def test_an_interrupted_chown_cannot_leave_an_env_file_or_directory_the_next_run_would_skip(machine):
    machine.host.fail_on = "chown"
    assert machine.install()[0] == 1
    # whatever the failed run left, it is not the real env file nor the real data directory
    assert not machine.path("/etc/local-rag/local-rag.env").exists()
    assert not machine.path("/var/lib/qdrant").exists()

    machine.host.fail_on = None
    assert machine.install()[0] == 0
    assert len(machine.env()["QDRANT_API_KEY"]) >= 32
    assert chowned(machine, "root:ragsvc", "/etc/local-rag/local-rag.env")
    assert chowned(machine, "ragsvc:ragsvc", "/var/lib/qdrant/storage")
    leftovers = [p for p in machine.root.rglob("*") if p.name.endswith(".new") or ".tmp" in p.name]
    assert leftovers == []


def test_packages_are_requested_as_wheels_only_so_nothing_is_built_as_root(machine):
    machine.install()

    (requested,) = lock_installs(machine.host)
    assert "--only-binary" in requested and requested[requested.index("--only-binary") + 1] == ":all:"
