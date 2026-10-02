"""The closed-network setup guide names only things that exist: environment variables, paths,
units, commands and flags are checked against the code, so the guide cannot silently rot."""
import argparse
import re
import subprocess

import pytest

from fake_drop import REPO
from local_rag import acceptance, cli, container, installer, smoke

GUIDE_PATH = REPO / "docs/agents/closed-network-setup.md"
GUIDE = GUIDE_PATH.read_text()
README = (REPO / "README.md").read_text()

# Names the guide uses that belong to other programs, not to this repository.
FOREIGN_VARIABLES = {"NODE_EXTRA_CA_CERTS", "NO_PROXY", "HTTP_PROXY", "HTTPS_PROXY",
                     "ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_MODEL"}
FOREIGN_FLAGS = {"--transport", "--header"}  # `claude mcp add`
# Shell variables the guide sets for itself, and names it mentions only to rule them out.
SHELL_VARIABLES = {"ENV_FILE", "CORPORA_FILE", "RAG_VALUE", "RAG_APPEND", "MIRROR_URL",
                   "NO_PROXY_LIST", "CA_PEM"}
NOT_SETTINGS = {"CONFLUENCE_URL", "CONFLUENCE_TOKEN"}


def sources() -> str:
    """Everything the guide may name: the code, the deployment files and the examples."""
    return "\n".join(path.read_text(errors="ignore")
                     for folder in ("src", "deploy", "examples")
                     for path in (REPO / folder).rglob("*") if path.is_file())


def code_spans() -> list[str]:
    return re.findall(r"`([^`\n]+)`", GUIDE) + re.findall(r"```\w*\n(.*?)```", GUIDE, re.DOTALL)


def test_every_variable_the_guide_names_is_read_by_the_code_or_shipped_in_a_file():
    known = sources()
    names = set(re.findall(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b", GUIDE))
    names -= FOREIGN_VARIABLES | SHELL_VARIABLES | NOT_SETTINGS

    assert [name for name in sorted(names) if name not in known] == []


def test_the_guide_asks_for_every_input_the_ticket_lists():
    asked = ["LITELLM_BASE_URL", "LITELLM_API_KEY", "RAG_EMBED_MODEL", "RAG_RERANK_MODEL",
             "CONFLUENCE_DOMAIN", "CONFLUENCE_API_TOKEN", "corpora.toml", "kind = \"folder\"",
             "MCP_HOST", "MCP_PORT", "MCP_TLS_CERT", "MCP_TLS_KEY", "NODE_EXTRA_CA_CERTS",
             "NO_PROXY", "MCP_TOKENS"]

    assert [name for name in asked if name not in GUIDE] == []
    assert "no `CONFLUENCE_URL` or `CONFLUENCE_TOKEN`" in GUIDE
    assert not any(name in sources() for name in NOT_SETTINGS)


def test_every_absolute_path_in_the_guide_is_one_the_code_or_units_use():
    known = sources()
    paths = {path.rstrip(".,;:)") for path in re.findall(r"/(?:etc|opt|var)/[\w./-]+", GUIDE)}

    assert paths
    assert [path for path in sorted(paths) if path not in known] == []


def test_every_repository_path_the_guide_points_at_exists():
    paths = set(re.findall(r"`((?:src|deploy|examples|docs|tests)/[\w./-]+?)`", GUIDE))

    assert paths
    assert [path for path in sorted(paths) if not (REPO / path).exists()] == []


def test_every_unit_the_guide_names_is_a_file_in_deploy():
    deployed = {path.name for folder in ("systemd", "container")
                for path in (REPO / "deploy" / folder).iterdir()}
    units = set(re.findall(r"\b(?:local-rag|qdrant)[\w-]*\.(?:service|timer)\b", GUIDE))

    assert units
    assert [unit for unit in sorted(units) if unit not in deployed] == []
    assert set(smoke.SYSTEMD_UNITS) <= units


def test_every_module_the_guide_runs_exists():
    modules = set(re.findall(r"-m local_rag\.(\w+)", GUIDE))

    assert modules == {"installer", "container", "smoke", "acceptance"}
    assert [m for m in modules if not (REPO / "src/local_rag" / f"{m}.py").exists()] == []


def parsers() -> dict[str, argparse.ArgumentParser]:
    def subcommands(parser):
        action = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
        return action.choices

    return {
        **{f"local-rag {name}": sub for name, sub in subcommands(cli._parser()).items()},
        **{f"container {name}": sub for name, sub in subcommands(container._parser()).items()},
        "installer": installer._parser(), "smoke": smoke._parser(),
        "acceptance": acceptance._parser()}


def flags_of(parser: argparse.ArgumentParser) -> set[str]:
    return {flag for action in parser._actions for flag in action.option_strings}


def test_every_local_rag_command_and_flag_in_the_guide_exists():
    known = parsers()
    commands = set(re.findall(r"local-rag (sync|rebuild|forget|switch-back|search|status)\b", GUIDE))
    assert commands == {name.split()[1] for name in known if name.startswith("local-rag")}

    for line in "\n".join(code_spans()).splitlines():
        for name in ("sync", "rebuild", "forget", "search", "status", "switch-back"):
            if re.search(rf"local-rag {name}\b", line):
                unknown = set(re.findall(r"(?<![\w-])--?[a-z][\w-]*", line.split(f"local-rag {name}")[1]))
                allowed = flags_of(known[f"local-rag {name}"]) | {"-m"}
                assert unknown <= allowed | {"--index-url"}, (line, unknown - allowed)


def test_every_flag_of_the_installer_container_and_smoke_commands_exists():
    known = parsers()
    for line in "\n".join(code_spans()).splitlines():
        for module, parser in (("installer", "installer"), ("container start", "container start"),
                               ("smoke", "smoke"), ("acceptance", "acceptance")):
            word = module.split()[0]
            if f"-m local_rag.{word}" in line:
                flags = set(re.findall(r"(?<![\w-])--[a-z][\w-]*",
                                       line.split(f"-m local_rag.{word}")[1]))
                assert flags <= flags_of(known[parser]), (line, flags - flags_of(known[parser]))


def test_the_acceptance_groups_and_check_names_the_guide_gives_exist():
    groups = set(re.findall(r"`(gateway|qdrant|index|mcp|install|confluence|gitlab)`", GUIDE))
    names = set(re.findall(r"`((?:gateway|qdrant|index|mcp|install|confluence|gitlab)\.[\w-]+)`", GUIDE))
    source = "\n".join((REPO / "src/local_rag" / f"acceptance_{m}.py").read_text()
                       for m in ("stack", "index", "mcp", "host"))

    assert groups == set(acceptance.GROUPS)
    assert names and [n for n in sorted(names) if f'"{n}"' not in source] == []


def test_the_claude_commands_use_only_the_flags_of_claude_mcp_add():
    flags = {flag for line in "\n".join(code_spans()).splitlines() if "claude mcp add" in line
             for flag in re.findall(r"(?<![\w-])--[a-z][\w-]*", line)}

    assert flags <= FOREIGN_FLAGS


def helper(text: str, name: str) -> str:
    """A shell function's definition, whitespace collapsed."""
    found = re.search(rf"^{name}\(\) \{{.*?\"\$@\"; \}}", text, re.DOTALL | re.MULTILINE)
    assert found, name
    return " ".join(found.group(0).replace("\\\n", " ").split())


@pytest.mark.parametrize("name", ["run_rag", "compose"])
def test_the_helpers_are_the_readmes(name):
    assert helper(GUIDE, name) == helper(README, name)


def test_the_guide_states_what_it_must_not_contradict():
    flat = " ".join(GUIDE.split())

    assert "runs the same `local-rag sync` as the hourly one" in flat
    assert "plain HTTP, no TLS pass-through" in flat
    assert "builds from the working tree but archives `HEAD`" in flat
    assert "one Passage per Document" not in flat and "issue #29" not in flat
    assert "Every MCP token sees every indexed Document" in flat
    for term in ("Knowledge index", "Corpus", "Model gateway", "Mirror", "Binary drop",
                 "Installer", "Rebuild", "Sync"):
        assert term in GUIDE


def test_the_guide_is_reachable_from_claude_md_and_the_readme():
    assert "docs/agents/closed-network-setup.md" in (REPO / "CLAUDE.md").read_text()
    assert "closed-network-setup.md" in README


def set_env_script(env_file, calls: list[str]) -> str:
    """The guide's set_env, defined and called on ``env_file``, with a stand-in for sudo."""
    shell = re.search(r"^```bash\n(set_env\(\).*?)^```", GUIDE, re.DOTALL | re.MULTILINE).group(1)
    return "sudo() { shift; \"$@\"; }\n" + shell + f"\nENV_FILE={env_file}\n" + "\n".join(calls)


def run_set_env(tmp_path, template: str, calls: list[str]) -> tuple[str, str]:
    """Run the guide's set_env on a copy of an env file; returns the file and what it said."""
    env_file = tmp_path / "env"
    env_file.write_text(template)
    done = subprocess.run(["bash", "-c", set_env_script(env_file, calls)],
                          capture_output=True, text=True, check=True)
    return env_file.read_text(), done.stdout


@pytest.mark.parametrize("template", [installer.ENV_FILE, container.ENV_FILE])
def test_set_env_fills_in_a_setting_once_and_touches_nothing_else(tmp_path, template):
    fresh = template.format(qdrant_key="Q", mcp_token="T")
    call = "RAG_VALUE=http://gateway.example.test:4000/v1 set_env LITELLM_BASE_URL"

    once, said = run_set_env(tmp_path, fresh, [call])
    twice, said_again = run_set_env(tmp_path, fresh, [call, call])

    assert once == twice
    assert said == "set       LITELLM_BASE_URL\n"
    assert said_again == "set       LITELLM_BASE_URL\nunchanged LITELLM_BASE_URL\n"
    changed = [line for line in once.splitlines() if line not in fresh.splitlines()]
    assert changed == ["LITELLM_BASE_URL=http://gateway.example.test:4000/v1"]
    assert "# LITELLM_BASE_URL=" not in once


def test_set_env_replaces_a_live_setting_and_appends_tokens_once(tmp_path):
    fresh = installer.ENV_FILE.format(qdrant_key="Q", mcp_token="T")

    text, said = run_set_env(tmp_path, fresh, [
        "RAG_VALUE=9000 set_env MCP_PORT",
        "RAG_VALUE=second RAG_APPEND=1 set_env MCP_TOKENS",
        "RAG_VALUE=second RAG_APPEND=1 set_env MCP_TOKENS",
        "RAG_VALUE=127.0.0.1,localhost set_env NO_PROXY"])

    lines = text.splitlines()
    assert "MCP_PORT=9000" in lines and "MCP_PORT=8765" not in lines
    assert "MCP_TOKENS=T,second" in lines
    assert lines[-1] == "NO_PROXY=127.0.0.1,localhost"
    assert said.count("unchanged MCP_TOKENS") == 1


def test_set_env_refuses_a_value_the_env_file_syntax_would_misread(tmp_path):
    fresh = installer.ENV_FILE.format(qdrant_key="Q", mcp_token="T")
    for bad in ("two words", "has#hash", "has$dollar", "'quoted'"):
        env_file = tmp_path / "env"
        env_file.write_text(fresh)
        done = subprocess.run(
            ["bash", "-c", set_env_script(env_file, ['RAG_VALUE="$BAD" set_env LITELLM_API_KEY'])],
            capture_output=True, text=True, env={"BAD": bad, "PATH": "/usr/bin:/bin"})

        assert done.returncode != 0 and "refused" in done.stderr
        assert env_file.read_text() == fresh


def test_both_install_sections_of_the_readme_send_the_operator_to_the_acceptance_suite():
    for heading, next_heading in (("## 3. Install, systemd route", "## 4."),
                                  ("## 4. Install, container route", "## 5.")):
        section = README.split(heading)[1].split(next_heading)[0]

        assert "local_rag.acceptance" in section and "ACCEPTANCE PASSED" in section, heading
