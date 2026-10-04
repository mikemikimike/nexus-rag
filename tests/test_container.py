import io
import re
import subprocess
import sys
from pathlib import Path

import pytest

import fake_docker
import fake_drop
from fake_docker import FakeDocker
from nexus_rag.binary_drop import QDRANT, SYSTEM_VERSION, sha256_of
from nexus_rag.container import BuildOptions, main

QDRANT_IMAGE = f"qdrant/qdrant:v{QDRANT.version}"
SYSTEM_IMAGE = f"nexus-rag:{SYSTEM_VERSION}"
QDRANT_TAR = f"qdrant-image-{QDRANT.version}.tar"
SYSTEM_TAR = f"nexus-rag-image-{SYSTEM_VERSION}.tar"


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
        [QDRANT_TAR, SYSTEM_TAR, f"nexus-rag-{SYSTEM_VERSION}.tar.gz", "SHA256SUMS"]
    )
    listed = checksums(drop)
    for name in (QDRANT_TAR, SYSTEM_TAR, f"nexus-rag-{SYSTEM_VERSION}.tar.gz"):
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
        lines = (
            f"{sha256_of(file)}  {file.name}\n"
            for file in sorted(self.drop.iterdir())
            if file.name != "SHA256SUMS"
        )
        (self.drop / "SHA256SUMS").write_text("".join(lines))

    def env(self) -> dict[str, str]:
        text = self.path("/etc/nexus-rag/container.env").read_text()
        pairs = (
            line.split("=", 1)
            for line in text.splitlines()
            if line.strip() and not line.startswith("#")
        )
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
    assert up[up.index("--env-file") + 1] == str(host.path("/etc/nexus-rag/container.env"))
    assert up[up.index("-f") + 1] == str(host.path("/opt/nexus-rag/container/compose.yaml"))
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


def docker_commands(docker: FakeDocker, subcommand: str) -> list[list[str]]:
    return [c for c in docker.ran if c[:2] == ["docker", subcommand]]


class ClosedNetworkHost:
    """A drop that holds only the code, and a host that builds its own image from mirrors."""

    def __init__(self, tmp_path, **docker_options):
        self.root = tmp_path / "root"
        self.drop = tmp_path / "drop"
        run("save", "--code-only", "--out", self.drop, host=FakeDocker(docker=False))
        self.docker = FakeDocker(**docker_options)

    def start(self, *options):
        return run(
            "start", "--drop", self.drop, "--root", self.root, "--build", *options, host=self.docker
        )


def test_save_code_only_writes_the_code_archive_and_its_checksum_and_needs_no_docker(tmp_path):
    drop = tmp_path / "drop"
    docker = FakeDocker(docker=False)

    code, out, err = run("save", "--code-only", "--out", drop, host=docker)

    assert code == 0, err
    archive = f"nexus-rag-{SYSTEM_VERSION}.tar.gz"
    assert sorted(path.name for path in drop.iterdir()) == sorted([archive, "SHA256SUMS"])
    assert checksums(drop) == {archive: sha256_of(drop / archive)}
    assert [command for command in docker.ran if command[0] == "docker"] == []


def test_start_build_builds_the_system_image_from_the_verified_code_and_pulls_qdrants(tmp_path):
    host = ClosedNetworkHost(tmp_path)

    code, out, err = host.start()

    assert code == 0, err
    assert host.docker.images == {QDRANT_IMAGE, SYSTEM_IMAGE}
    assert docker_commands(host.docker, "load") == []
    assert docker_commands(host.docker, "pull") == [["docker", "pull", QDRANT_IMAGE]]
    (build,) = docker_commands(host.docker, "build")
    assert build[build.index("-t") + 1] == SYSTEM_IMAGE
    assert Path(build[build.index("-f") + 1]).name == "Dockerfile"
    assert not [arg for arg in build if arg.startswith("--build-arg")]  # the public defaults
    assert host.docker.ran_matching("compose", "up")
    assert (host.root / "etc/nexus-rag/container.env").is_file()


def test_start_build_points_the_build_at_the_mirrors_and_tags_qdrant_as_the_compose_file_names_it(
    tmp_path,
):
    host = ClosedNetworkHost(tmp_path)
    mirrored = f"registry.internal/mirror/qdrant:v{QDRANT.version}"

    code, out, err = host.start(
        "--base-image=registry.internal/python:3.12-slim",
        f"--qdrant-image={mirrored}",
        "--apt-mirror=http://apt.internal",
        "--pip-index-url=http://pypi.internal/simple",
        "--npm-registry=http://npm.internal/",
    )

    assert code == 0, err
    assert docker_commands(host.docker, "pull") == [["docker", "pull", mirrored]]
    assert docker_commands(host.docker, "tag") == [["docker", "tag", mirrored, QDRANT_IMAGE]]
    (build,) = docker_commands(host.docker, "build")
    assert {arg for arg in build if arg.startswith("--build-arg")} == {
        "--build-arg=BASE_IMAGE=registry.internal/python:3.12-slim",
        "--build-arg=APT_MIRROR=http://apt.internal",
        "--build-arg=PIP_INDEX_URL=http://pypi.internal/simple",
        "--build-arg=NPM_CONFIG_REGISTRY=http://npm.internal/",
    }


def test_start_build_refuses_a_code_archive_that_is_not_the_listed_one_before_touching_docker(
    tmp_path,
):
    host = ClosedNetworkHost(tmp_path)
    archive = next(host.drop.glob("nexus-rag-*.tar.gz"))
    archive.write_bytes(archive.read_bytes() + b"\0")

    code, out, err = host.start()

    assert code == 1 and "checksum" in err
    assert host.docker.ran == []
    assert not host.root.exists()


def test_start_build_refuses_images_that_report_another_version_than_recorded(tmp_path):
    host = ClosedNetworkHost(tmp_path, reports={SYSTEM_IMAGE: "0.0.9"})

    code, out, err = host.start()

    assert code == 1
    assert "version" in err and SYSTEM_VERSION in err
    assert host.docker.ran_matching("compose") == []
    assert not host.root.exists()


def test_start_build_fails_when_the_build_fails_and_leaves_the_machine_alone(tmp_path):
    host = ClosedNetworkHost(tmp_path, fail_on="docker build")

    code, out, err = host.start()

    assert code == 1 and "docker build" in err
    assert not host.root.exists()


def test_the_mirror_flags_mean_nothing_without_build(tmp_path):
    host = Host(tmp_path)

    with pytest.raises(SystemExit):
        run(
            "start",
            "--drop",
            host.drop,
            "--root",
            host.root,
            "--apt-mirror=http://x",
            host=host.docker,
        )
    assert host.docker.ran == []


def test_the_dockerfile_declares_every_build_argument_the_start_command_passes():
    dockerfile = (fake_drop.REPO / "deploy/container/Dockerfile").read_text()
    passed = [arg.split("=")[1] for arg in BuildOptions(*["x"] * 5).build_args()]

    assert len(passed) == 4
    for name in passed:
        assert re.search(rf"^ARG {name}=", dockerfile, re.MULTILINE)
    assert re.search(r"^FROM \$\{BASE_IMAGE\}$", dockerfile, re.MULTILINE)


def test_start_refuses_a_code_archive_that_is_not_the_listed_one(host):
    archive = next(host.drop.glob("nexus-rag-*.tar.gz"))
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
    assert oct(host.path("/etc/nexus-rag/container.env").stat().st_mode & 0o777) == "0o600"


def test_a_second_start_never_overwrites_settings_or_the_corpora_file(host):
    host.start()
    env_file = host.path("/etc/nexus-rag/container.env")
    roots_file = host.path("/etc/nexus-rag/corpora.toml")
    env_file.write_text(env_file.read_text() + "LITELLM_BASE_URL=http://gateway.example.test/v1\n")
    roots_file.write_text("# mine\n")
    before = env_file.read_text()

    code, out, err = host.start()

    assert code == 0, err
    assert env_file.read_text() == before
    assert roots_file.read_text() == "# mine\n"


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
    service = host.path("/etc/systemd/system/nexus-rag-container-sync.service").read_text()
    assert "docker compose" in service and " run " in service and "sync" in service
    assert host.path("/etc/systemd/system/nexus-rag-container-sync.timer").exists()
    (enable,) = host.docker.ran_matching("systemctl", "enable")
    assert "nexus-rag-container-sync.timer" in enable and "container-full" not in " ".join(enable)


def test_start_retires_the_weekly_timer_an_earlier_release_installed(host):
    host.start()
    units = host.path("/etc/systemd/system")
    for name in ("nexus-rag-container-full.service", "nexus-rag-container-full.timer"):
        (units / name).write_text("[Unit]\n")

    code, _, err = host.start()

    assert code == 0, err
    assert host.docker.ran_matching(
        "systemctl", "disable", "--now", "nexus-rag-container-full.timer"
    )
    assert not (units / "nexus-rag-container-full.service").exists()
    assert not (units / "nexus-rag-container-full.timer").exists()


def test_start_installs_the_templated_timer_but_enables_no_instance(host):
    code, out, err = host.start()

    assert code == 0, err
    for suffix in ("service", "timer"):
        assert host.path(f"/etc/systemd/system/nexus-rag-container-sync@.{suffix}").exists()
    (enable,) = host.docker.ran_matching("systemctl", "enable")
    assert (
        "@" not in enable
    )  # `systemctl enable name@.timer` is an error; the operator opts in per Corpus


def test_start_without_docker_is_refused_before_anything_is_touched(tmp_path):
    host = Host(tmp_path, docker=False)

    code, out, err = host.start()

    assert code == 1 and "docker" in err
    assert not host.root.exists()


def compose() -> dict:
    import yaml

    return yaml.safe_load((fake_drop.REPO / "deploy/container/compose.yaml").read_text())


def test_the_sync_can_trust_an_internal_ca_put_in_the_certs_directory(host):
    code, out, err = host.start()

    sync = compose()["services"]["sync"]
    assert code == 0, err
    assert host.path("/etc/nexus-rag/certs").is_dir()
    assert "/etc/nexus-rag/certs:/etc/nexus-rag/certs:ro" in sync["volumes"]
    assert "NODE_EXTRA_CA_CERTS" in sync["environment"]
    assert (
        "NODE_EXTRA_CA_CERTS=/etc/nexus-rag/certs/"
        in host.path("/etc/nexus-rag/container.env").read_text()
    )


def test_the_compose_file_runs_the_recorded_versions_of_both_images():
    services = compose()["services"]

    assert services["qdrant"]["image"] == QDRANT_IMAGE
    assert services["mcp"]["image"] == SYSTEM_IMAGE
    assert services["sync"]["image"] == SYSTEM_IMAGE  # one image, run with another command


def test_compose_can_build_the_system_image_from_a_checkout():
    services = compose()["services"]

    for name in ("mcp", "sync"):  # either may be the first to need the image
        build = services[name]["build"]
        assert build["dockerfile"] == "deploy/container/Dockerfile"
        assert (fake_drop.REPO / build["dockerfile"]).is_file()
    assert "build" not in services["qdrant"]


def test_the_compose_file_keeps_data_and_the_index_in_named_volumes():
    stack = compose()
    mounted = {
        mount.split(":")[0]
        for service in stack["services"].values()
        for mount in service.get("volumes", [])
        if not mount.startswith("/")
    }

    assert {"qdrant-storage", "qdrant-snapshots", "rag-data"} <= mounted
    assert set(stack["volumes"]) >= mounted


def test_the_compose_file_holds_no_secret_and_takes_every_one_from_the_env_file():
    services = compose()["services"]

    for key in ("QDRANT_API_KEY", "MCP_TOKENS"):
        assert services["mcp"]["environment"][key].startswith("${" + key + ":?")
    assert services["qdrant"]["environment"]["QDRANT__SERVICE__API_KEY"].startswith(
        "${QDRANT_API_KEY:?"
    )
    assert "env_file" not in str(services)  # works in Portainer, whose host paths are not ours


def test_the_stack_passes_the_gitlab_settings_to_the_sync_and_keeps_the_mirror_cache():
    services = compose()["services"]

    assert services["sync"]["environment"]["GITLAB_URL"] == "${GITLAB_URL:-https://gitlab.org}"
    assert services["sync"]["environment"]["GITLAB_TOKEN"] == "${GITLAB_TOKEN:-}"
    assert "rag-data:/data" in services["sync"]["volumes"]  # HOME=/data: the default mirror cache


def test_the_compose_file_names_no_variable_twice():
    import yaml

    class Strict(yaml.SafeLoader):
        pass

    def mapping(loader, node):
        keys = [loader.construct_object(key) for key, _ in node.value]
        repeated = sorted({key for key in keys if keys.count(key) > 1})
        assert not repeated, f"named twice: {repeated}"
        return loader.construct_mapping(node)

    Strict.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)

    yaml.load((fake_drop.REPO / "deploy/container/compose.yaml").read_text(), Strict)


def test_every_setting_the_programs_read_can_be_set_from_the_env_file():
    # Compose passes a container only the variables it names, so a setting that is not named
    # here could never be changed in the env file (the Rebuild threshold, chunking, ...).
    import nexus_rag.environment as environment

    read = set(
        re.findall(
            r'"(RAG_[A-Z_]+|LITELLM_[A-Z_]+|LIT_BIN)"', Path(environment.__file__).read_text()
        )
    )
    read -= {"RAG_CORPORA_FILE"}  # the stack mounts the declaration at its default path
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

    assert [line for line in lines if line.startswith("CMD")] == ['CMD ["nexus-rag-mcp"]']
    assert not [line for line in lines if line.startswith("ENTRYPOINT")]
    assert "USER ragsvc" in lines
    assert "requirements.lock" in dockerfile  # the pinned packages, not a fresh resolution


def test_the_timer_units_call_compose_run_with_the_files_the_start_command_places():
    units = fake_drop.REPO / "deploy/container"

    service = (units / "nexus-rag-container-sync.service").read_text()
    (command,) = [line for line in service.splitlines() if line.startswith("ExecStart=")]
    assert command.split()[0] == "ExecStart=/usr/bin/docker"
    assert command.split()[1:3] == ["compose", "--project-name"]
    assert "--env-file /etc/nexus-rag/container.env" in command
    assert "-f /opt/nexus-rag/container/compose.yaml" in command
    assert command.endswith(" run --rm -T sync")
    assert "OnCalendar=" in (units / "nexus-rag-container-sync.timer").read_text()


def test_the_templated_unit_syncs_the_corpus_named_by_its_instance():
    units = fake_drop.REPO / "deploy/container"
    service = (units / "nexus-rag-container-sync@.service").read_text()

    (command,) = [line for line in service.splitlines() if line.startswith("ExecStart=")]
    assert "--env-file /etc/nexus-rag/container.env" in command
    assert command.endswith(" run --rm -T sync nexus-rag sync %i")
    assert "OnCalendar=*:0/15" in (units / "nexus-rag-container-sync@.timer").read_text()


def test_the_container_sync_units_accept_the_exit_code_of_a_run_that_was_skipped():
    units = fake_drop.REPO / "deploy/container"

    for name in ("nexus-rag-container-sync", "nexus-rag-container-sync@"):
        assert "SuccessExitStatus=75" in (units / f"{name}.service").read_text().splitlines()


def test_the_command_runs_on_a_machine_with_only_the_standard_library():
    program = (
        "import sys\n"
        "for blocked in ('qdrant_client', 'mcp', 'llama_index', 'bs4'):\n"
        "    sys.modules[blocked] = None\n"
        "import nexus_rag.container\n"
    )
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


def test_both_routes_offer_the_same_settings_after_their_own_qdrant_and_mcp_parts(host, tmp_path):
    # What follows the route's own Qdrant and MCP server parts (retrieval, the Model gateway,
    # Confluence, GitLab) is one text, so a setting added for one route is offered on both.
    from test_installer import Machine

    machine = Machine(tmp_path / "systemd")
    assert host.start()[0] == 0 and machine.install()[0] == 0

    def shared(text: str) -> str:
        return text[text.index("# Retrieval mode") :]

    container = shared(host.path("/etc/nexus-rag/container.env").read_text())
    assert container == shared(machine.path("/etc/nexus-rag/nexus-rag.env").read_text())
    assert "RAG_REBUILD_FAILURE_THRESHOLD" in container and "GITLAB_TOKEN" in container
