import io
import re
import subprocess
import sys
from pathlib import Path

import pytest

import fake_docker
import fake_drop
from fake_docker import FakeDocker
from local_rag.binary_drop import QDRANT, SYSTEM_VERSION, sha256_of
from local_rag.container import main

QDRANT_IMAGE = f"qdrant/qdrant:v{QDRANT.version}"
SYSTEM_IMAGE = f"local-rag:{SYSTEM_VERSION}"
QDRANT_TAR = f"qdrant-image-{QDRANT.version}.tar"
SYSTEM_TAR = f"local-rag-image-{SYSTEM_VERSION}.tar"


def run(*argv, host, **options):
    out, err = io.StringIO(), io.StringIO()
    code = main([str(a) for a in argv], host=host, stdout=out, stderr=err, **options)
    return code, out.getvalue(), err.getvalue()


def checksums(drop: Path) -> dict[str, str]:
    lines = (drop / "SHA256SUMS").read_text().splitlines()
    return {line.split()[1].lstrip("*"): line.split()[0] for line in lines}


def test_save_writes_both_images_and_the_code_with_versions_and_checksums(tmp_path):
    drop = tmp_path / "drop"

    code, out, err = run("save", "--out", drop, host=FakeDocker())

    assert code == 0, err
    assert sorted(path.name for path in drop.iterdir()) == sorted(
        [QDRANT_TAR, SYSTEM_TAR, f"local-rag-{SYSTEM_VERSION}.tar.gz", "SHA256SUMS"])
    listed = checksums(drop)
    for name in (QDRANT_TAR, SYSTEM_TAR, f"local-rag-{SYSTEM_VERSION}.tar.gz"):
        assert listed[name] == sha256_of(drop / name)


def test_save_refuses_a_qdrant_image_that_is_not_the_recorded_version(tmp_path):
    host = FakeDocker(reports={QDRANT_IMAGE: "1.18.0"})

    code, out, err = run("save", "--out", tmp_path / "drop", host=host)

    assert code == 1
    assert "version" in err and QDRANT.version in err
    assert not (tmp_path / "drop").exists()


class Host:
    """A temporary root directory with a saved Binary drop beside it, and the start command."""

    def __init__(self, tmp_path, **docker_options):
        self.root = tmp_path / "root"
        self.drop = tmp_path / "drop"
        run("save", "--out", self.drop, host=FakeDocker())
        self.docker = FakeDocker(**docker_options)

    def start(self, *, host=None):
        return run("start", "--drop", self.drop, "--root", self.root, host=host or self.docker)

    def path(self, absolute: str) -> Path:
        return self.root / absolute.lstrip("/")

    def seal(self) -> None:
        """Make SHA256SUMS agree with the files, as if the drop had been made that way."""
        lines = (f"{sha256_of(file)}  {file.name}\n" for file in sorted(self.drop.iterdir())
                 if file.name != "SHA256SUMS")
        (self.drop / "SHA256SUMS").write_text("".join(lines))

    def env(self) -> dict[str, str]:
        text = self.path("/etc/local-rag/container.env").read_text()
        pairs = (line.split("=", 1) for line in text.splitlines()
                 if line.strip() and not line.startswith("#"))
        return {key: value for key, value in pairs}


@pytest.fixture
def host(tmp_path):
    return Host(tmp_path)


def test_start_loads_the_images_and_brings_the_stack_up_from_the_env_file(host):
    code, out, err = host.start()

    assert code == 0, err
    assert host.docker.images == {QDRANT_IMAGE, SYSTEM_IMAGE}
    (up,) = host.docker.ran_matching("compose", "up")
    assert up[:2] == ["docker", "compose"]
    assert up[up.index("--env-file") + 1] == str(host.path("/etc/local-rag/container.env"))
    assert up[up.index("-f") + 1] == str(host.path("/opt/local-rag/container/compose.yaml"))
    assert up[-2:] == ["up", "-d"]


def test_start_refuses_an_image_whose_checksum_is_not_the_saved_one(host):
    (host.drop / QDRANT_TAR).write_bytes(b"tampered")

    code, out, err = host.start()

    assert code == 1
    assert "checksum" in err and QDRANT_TAR in err
    assert host.docker.ran == []
    assert not host.root.exists()


def test_start_refuses_an_image_tarball_that_holds_another_tag(host):
    # A different Qdrant release, carefully sealed with a matching checksum.
    fake_docker.image_tarball(host.drop / QDRANT_TAR, "qdrant/qdrant:v1.18.0")
    host.seal()

    code, out, err = host.start()

    assert code == 1
    assert QDRANT_IMAGE in err
    assert host.docker.ran_matching("docker", "load") == []
    assert not host.root.exists()


def test_start_refuses_a_loaded_image_that_reports_another_version_than_recorded(tmp_path):
    host = Host(tmp_path, reports={QDRANT_IMAGE: "1.18.0"})  # tagged as the record says, is not

    code, out, err = host.start()

    assert code == 1
    assert "version" in err and QDRANT.version in err
    assert host.docker.ran_matching("compose") == []
    assert not host.root.exists()


def test_start_refuses_a_code_archive_that_is_not_the_listed_one(host):
    archive = next(host.drop.glob("local-rag-*.tar.gz"))
    archive.write_bytes(archive.read_bytes() + b"\0")

    code, out, err = host.start()

    assert code == 1 and "checksum" in err
    assert host.docker.ran == []


def test_first_start_generates_the_secrets_once_into_one_env_file(host):
    code, out, err = host.start()

    assert code == 0, err
    env = host.env()
    assert len(env["QDRANT_API_KEY"]) >= 32 and len(env["MCP_TOKENS"]) >= 32
    assert env["QDRANT_API_KEY"] != env["MCP_TOKENS"] and "," not in env["MCP_TOKENS"]
    assert oct(host.path("/etc/local-rag/container.env").stat().st_mode & 0o777) == "0o600"


def test_a_second_start_never_overwrites_settings_or_the_roots_file(host):
    host.start()
    env_file = host.path("/etc/local-rag/container.env")
    roots_file = host.path("/etc/local-rag/confluence_roots.txt")
    env_file.write_text(env_file.read_text() + "LITELLM_BASE_URL=http://gateway.example.test/v1\n")
    roots_file.write_text("12345\n")
    before = env_file.read_text()

    code, out, err = host.start()

    assert code == 0, err
    assert env_file.read_text() == before
    assert roots_file.read_text() == "12345\n"


def test_an_upgrade_leaves_the_volumes_alone(host):
    host.start()
    host.docker.ran.clear()

    host.start()

    commands = [" ".join(command) for command in host.docker.ran]
    assert not [c for c in commands if " down" in c or "volume" in c or " rm" in c or " -v" in c]
    assert host.docker.ran_matching("compose", "up")  # it recreates what changed, keeping volumes


def test_start_installs_the_timers_that_run_sync_through_docker_compose(host):
    code, out, err = host.start()

    assert code == 0, err
    for name in ("sync", "full"):
        service = host.path(f"/etc/systemd/system/local-rag-container-{name}.service").read_text()
        assert "docker compose" in service and " run " in service and "sync" in service
        assert host.path(f"/etc/systemd/system/local-rag-container-{name}.timer").exists()
    (enable,) = host.docker.ran_matching("systemctl", "enable")
    assert "local-rag-container-sync.timer" in enable and "local-rag-container-full.timer" in enable


def test_start_without_docker_is_refused_before_anything_is_touched(tmp_path):
    host = Host(tmp_path, docker=False)

    code, out, err = host.start()

    assert code == 1 and "docker" in err
    assert not host.root.exists()


def compose() -> dict:
    import yaml

    return yaml.safe_load((fake_drop.REPO / "deploy/container/compose.yaml").read_text())


def test_the_compose_file_runs_the_recorded_versions_of_both_images():
    services = compose()["services"]

    assert services["qdrant"]["image"] == QDRANT_IMAGE
    assert services["mcp"]["image"] == SYSTEM_IMAGE
    assert services["sync"]["image"] == SYSTEM_IMAGE  # one image, run with another command


def test_the_compose_file_keeps_data_and_the_index_in_named_volumes():
    stack = compose()
    mounted = {mount.split(":")[0] for service in stack["services"].values()
               for mount in service.get("volumes", []) if not mount.startswith("/")}

    assert {"qdrant-storage", "qdrant-snapshots", "rag-data"} <= mounted
    assert set(stack["volumes"]) >= mounted


def test_the_compose_file_holds_no_secret_and_takes_every_one_from_the_env_file():
    services = compose()["services"]

    for key in ("QDRANT_API_KEY", "MCP_TOKENS"):
        assert services["mcp"]["environment"][key].startswith("${" + key + ":?")
    assert services["qdrant"]["environment"]["QDRANT__SERVICE__API_KEY"].startswith(
        "${QDRANT_API_KEY:?")
    assert "env_file" not in str(services)  # works in Portainer, whose host paths are not ours


def test_every_setting_the_programs_read_can_be_set_from_the_env_file():
    # Compose passes a container only the variables it names, so a setting that is not named
    # here could never be changed in the env file (the Rebuild threshold, chunking, ...).
    import local_rag.environment as environment

    read = set(re.findall(r'"(RAG_[A-Z_]+|LITELLM_[A-Z_]+|LIT_BIN)"',
                          Path(environment.__file__).read_text()))
    services = compose()["services"]

    assert read
    for name in ("mcp", "sync"):
        assert read <= set(services[name]["environment"]), name


def test_only_the_mcp_server_is_reachable_from_outside_and_sync_is_not_part_of_up():
    services = compose()["services"]

    assert "ports" not in services["qdrant"] and "ports" not in services["sync"]
    assert services["mcp"]["ports"]
    assert services["sync"]["profiles"] and "profiles" not in services["mcp"]


def test_the_image_is_single_purpose_and_does_not_run_as_root():
    dockerfile = (fake_drop.REPO / "deploy/container/Dockerfile").read_text()
    lines = [line.strip() for line in dockerfile.splitlines()]

    assert [line for line in lines if line.startswith("CMD")] == ['CMD ["local-rag-mcp"]']
    assert not [line for line in lines if line.startswith("ENTRYPOINT")]
    assert "USER ragsvc" in lines
    assert "requirements.lock" in dockerfile  # the pinned packages, not a fresh resolution


def test_the_timer_units_call_compose_run_with_the_files_the_start_command_places():
    units = fake_drop.REPO / "deploy/container"

    for name in ("sync", "full"):
        service = (units / f"local-rag-container-{name}.service").read_text()
        (command,) = [line for line in service.splitlines() if line.startswith("ExecStart=")]
        assert command.split()[0] == "ExecStart=/usr/bin/docker"
        assert command.split()[1:3] == ["compose", "--project-name"]
        assert "--env-file /etc/local-rag/container.env" in command
        assert "-f /opt/local-rag/container/compose.yaml" in command
        assert command.endswith(" run --rm -T sync")
        assert "OnCalendar=" in (units / f"local-rag-container-{name}.timer").read_text()


def test_the_command_runs_on_a_machine_with_only_the_standard_library():
    program = (
        "import sys\n"
        "for blocked in ('qdrant_client', 'mcp', 'llama_index', 'bs4'):\n"
        "    sys.modules[blocked] = None\n"
        "import local_rag.container\n")
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
