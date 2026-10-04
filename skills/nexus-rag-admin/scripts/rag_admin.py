#!/usr/bin/env python3
"""rag-admin: day-to-day administration of an installed nexus-rag, for an agent or an operator.

Run as root on the nexus-rag host:  sudo python3.12 rag_admin.py COMMAND ...

It knows both delivery routes (systemd and container) and hides the difference: which env file,
how a `nexus-rag` command runs, how the MCP server is restarted, which timer units exist. It
never prints a secret. Standard library only, so it runs before and without the venv.

    route                                  the route, its files and units
    env show | set KEY | unset KEY         the env file, masked; a value is read from stdin
    token list | add LABEL | revoke WHO | name FINGERPRINT LABEL
                                           MCP tokens, named in a fingerprint ledger
    corpora stage | add-repo | check | diff | apply | discard
                                           edit corpora.toml through a staged copy
    run ARGS...                            a command as the services run it (nexus-rag, python)
    sync NAME [--detach]                   Sync one Corpus
    smoke | acceptance [ARGS...]           the health checks, with the route's flags
    restart                                apply env file changes to the MCP server
    timer enable|disable NAME | timer list the per-Corpus Sync timers

RAG_ADMIN_ROOT (default /) prefixes every host path, and RAG_ADMIN_DRY_RUN=1 prints the
commands instead of running them; both exist for tests and rehearsals.
"""

import argparse
import datetime
import difflib
import getpass
import hashlib
import json
import os
import pwd
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(os.environ.get("RAG_ADMIN_ROOT", "/"))
DRY_RUN = os.environ.get("RAG_ADMIN_DRY_RUN") == "1"

ETC = ROOT / "etc/nexus-rag"
CORPORA = ETC / "corpora.toml"
LEDGER = ETC / "mcp-tokens.labels"
BACKUPS = ETC / "backups"
VENV_BIN = "/opt/nexus-rag/venv/bin"
COMPOSE_FILE = "/opt/nexus-rag/container/compose.yaml"

KEY = re.compile(r"[A-Z][A-Z0-9_]*")
SECRET_KEY = re.compile(r"(KEY|TOKENS?|PASSWORD|SECRET)$")
UNSAFE_VALUE = re.compile(r"[\s'\"#$\\]")
LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._@-]*")
CORPUS_NAME = re.compile(r"[A-Za-z0-9_-]+")
# Settings recorded in the index: changing one makes the index read-only until a Rebuild.
REBUILD_SETTINGS = {
    "RAG_RETRIEVAL",
    "RAG_EMBED_MODEL",
    "RAG_CHUNK_TOKENS",
    "RAG_CHUNK_OVERLAP",
    "RAG_CODE_CHUNK_TOKENS",
    "RAG_BM25_K1",
    "RAG_BM25_B",
    "RAG_BM25_AVG_DL",
    "RAG_COLLECTION",
}
MANAGED = {
    "MCP_TOKENS": "use `token add` and `token revoke`, which keep the ledger in step",
    "QDRANT_API_KEY": "generated at install; Qdrant's two keys must stay equal",
    "QDRANT__SERVICE__API_KEY": "generated at install; Qdrant's two keys must stay equal",
}

# Builds every declared Corpus the way `sync` does, so a check finds what a Sync would refuse.
VALIDATOR = """\
import os, sys
from nexus_rag.declaration import Declaration, DeclarationError
from nexus_rag.environment import environment_from
try:
    declaration = Declaration.read(sys.argv[1])
    environment = environment_from(os.environ)
    for name in declaration.names:
        declaration.corpus(name, environment)
        print(f"ok        corpus {name}")
except DeclarationError as error:
    print(f"invalid   {error}")
    sys.exit(1)
"""


class Refused(Exception):
    """The command cannot be carried out; nothing was changed."""


# --- the route -------------------------------------------------------------------------------


class Route:
    def __init__(self, name: str):
        self.name = name
        self.systemd = name == "systemd"
        self.env_file = ETC / ("nexus-rag.env" if self.systemd else "container.env")

    @staticmethod
    def detect() -> "Route":
        forced = os.environ.get("RAG_ADMIN_ROUTE")
        if forced in ("systemd", "container"):
            return Route(forced)
        found = [
            name
            for name, file in (("systemd", "nexus-rag.env"), ("container", "container.env"))
            if (ETC / file).exists()
        ]
        if len(found) == 1:
            return Route(found[0])
        if not found:
            raise Refused(f"no nexus-rag installation here: neither env file exists in {ETC}")
        raise Refused(
            "both env files exist; set RAG_ADMIN_ROUTE=systemd or container to say which "
            "installation to manage"
        )

    def command(self, args: list[str], mounts: dict[str, str] | None = None) -> list[str]:
        """``args`` run as the services run them: the venv's `nexus-rag` and `python`, as the
        service user, with the env file's settings."""
        if self.systemd:
            if args and args[0] in ("nexus-rag", "python", "lit"):
                args = [f"{VENV_BIN}/{args[0]}", *args[1:]]
            return [
                "systemd-run", "--quiet", "--pipe", "--wait", "--collect",
                "-p", "User=ragsvc",
                "-p", f"EnvironmentFile={self.env_file}",
                "-p", "Environment=HOME=/var/lib/ragsvc",
                "-p", "Environment=PATH=/usr/local/bin:/usr/bin:/bin",
                *args,
            ]  # fmt: skip
        volumes = [
            arg for host, inside in (mounts or {}).items() for arg in ("-v", f"{host}:{inside}:ro")
        ]
        return [*self.compose(), "run", "--rm", "-T", *volumes, "sync", *args]

    def compose(self) -> list[str]:
        return [
            "docker", "compose", "--project-name", "nexus-rag",
            "--env-file", str(self.env_file), "-f", COMPOSE_FILE,
        ]  # fmt: skip

    def sync_unit(self, corpus: str) -> str:
        prefix = "nexus-rag-sync" if self.systemd else "nexus-rag-container-sync"
        return f"{prefix}@{corpus}"

    def restart(self) -> list[str]:
        if self.systemd:
            return ["systemctl", "restart", "nexus-rag-mcp"]
        return [*self.compose(), "up", "-d"]


def execute(command: list[str]) -> int:
    if DRY_RUN:
        print("would run: " + shlex.join(command))
        return 0
    return subprocess.run(command).returncode


# --- files -----------------------------------------------------------------------------------


def operator() -> pwd.struct_passwd:
    """The account that called sudo, who owns the staged and token files."""
    name = os.environ.get("SUDO_USER") or getpass.getuser()
    return pwd.getpwnam(name)


def operator_dir(name: str) -> Path:
    account = operator()
    folder = Path(os.environ.get("RAG_ADMIN_HOME", account.pw_dir)) / name
    folder.mkdir(mode=0o700, exist_ok=True)
    _give(folder, account)
    return folder


def _give(path: Path, account: pwd.struct_passwd) -> None:
    if os.geteuid() == 0:
        os.chown(path, account.pw_uid, account.pw_gid)


def backup(path: Path) -> Path:
    """A copy of ``path`` under the backups folder, as private as the original."""
    BACKUPS.mkdir(mode=0o700, parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    target = BACKUPS / f"{path.name}.{stamp}"
    shutil.copy2(path, target)
    return target


def rewrite(path: Path, text: str) -> None:
    """Replaces the content in place, so the file keeps its owner and mode."""
    with open(path, "w", encoding="utf-8") as file:
        file.write(text)


# --- the env file ----------------------------------------------------------------------------


class EnvFile:
    def __init__(self, path: Path):
        if not path.exists():
            raise Refused(f"{path} does not exist")
        self.path = path
        self.lines = path.read_text(encoding="utf-8").split("\n")

    def get(self, key: str) -> str | None:
        for line in self.lines:
            if line.startswith(key + "="):
                return line.split("=", 1)[1]
        return None

    def set(self, key: str, value: str) -> bool:
        """Sets ``key``; False if it already held exactly that."""
        new = f"{key}={value}"
        active = [i for i, line in enumerate(self.lines) if line.startswith(key + "=")]
        if active:
            if self.lines[active[0]] == new and len(active) == 1:
                return False
            self.lines[active[0]] = new
            for i in reversed(active[1:]):
                del self.lines[i]
            return True
        template = [i for i, line in enumerate(self.lines) if re.match(rf"#\s*{key}=", line)]
        if template:
            self.lines[template[0]] = new
        else:
            if self.lines and self.lines[-1] == "":
                self.lines.pop()
            self.lines += [new, ""]
        return True

    def unset(self, key: str) -> bool:
        active = [i for i, line in enumerate(self.lines) if line.startswith(key + "=")]
        for i in active:
            self.lines[i] = f"# {key}="
        return bool(active)

    def save(self) -> None:
        backup(self.path)
        rewrite(self.path, "\n".join(self.lines))

    def masked(self) -> list[str]:
        shown = []
        for line in self.lines:
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition("=")
            if SECRET_KEY.search(key) and value:
                count = len(split_tokens(value)) if key == "MCP_TOKENS" else 1
                value = "<hidden>" if count == 1 else f"<hidden: {count} tokens>"
            shown.append(f"{key}={value}")
        return shown


def split_tokens(listed: str) -> list[str]:
    return [token.strip() for token in listed.split(",") if token.strip()]


def check_value(key: str, value: str) -> None:
    if not KEY.fullmatch(key):
        raise Refused(f"{key!r} is not a setting name (capitals, digits and _)")
    if not value:
        raise Refused(f"{key} needs a value; use `env unset {key}` to remove it")
    if UNSAFE_VALUE.search(value):
        raise Refused(
            f"the value for {key} holds whitespace, a quote, #, $ or a backslash, which systemd "
            "or compose would misread; ask the operator to edit the env file by hand"
        )


def read_secret(key: str) -> str:
    if sys.stdin.isatty():
        return getpass.getpass(f"{key} (not echoed): ").strip()
    return sys.stdin.read().strip()


def after_env_change(keys: list[str], route: Route) -> None:
    rebuild = [k for k in keys if k in REBUILD_SETTINGS]
    if rebuild:
        print(
            f"warning   {', '.join(rebuild)} is recorded in the index: the index is read-only "
            "until a Rebuild promotes. Do not restart the MCP server yet; follow 'Changing "
            "RAG_EMBED_MODEL' in references/closed-network-setup.md, section 9."
        )
    else:
        print("next      `restart` to apply it to the MCP server (a Sync reads it at its start)")


# --- the token ledger ------------------------------------------------------------------------


def fingerprint(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()[:12]


class Ledger:
    """Label of each MCP token, by fingerprint: never the token itself."""

    def __init__(self) -> None:
        self.entries: dict[str, tuple[str, str]] = {}  # fingerprint -> (label, created)
        if LEDGER.exists():
            for line in LEDGER.read_text(encoding="utf-8").splitlines():
                if line.strip() and not line.startswith("#"):
                    fp, label, created = (line.split("\t") + ["", ""])[:3]
                    self.entries[fp] = (label, created)

    def label_of(self, fp: str) -> str | None:
        return self.entries.get(fp, (None, ""))[0]

    def find(self, label: str) -> str | None:
        return next((fp for fp, (lb, _) in self.entries.items() if lb == label), None)

    def put(self, fp: str, label: str) -> None:
        for stale in [f for f, (lb, _) in self.entries.items() if lb == label and f != fp]:
            del self.entries[stale]
        self.entries[fp] = (label, datetime.date.today().isoformat())

    def save(self) -> None:
        lines = ["# nexus-rag MCP token labels: fingerprint (sha256, 12 hex), label, date added"]
        lines += [f"{fp}\t{label}\t{created}" for fp, (label, created) in self.entries.items()]
        old = os.umask(0o077)
        try:
            rewrite(LEDGER, "\n".join(lines) + "\n")
        finally:
            os.umask(old)


def tokens(route: Route) -> tuple[EnvFile, list[str]]:
    env = EnvFile(route.env_file)
    return env, split_tokens(env.get("MCP_TOKENS") or "")


def resolve(who: str, listed: list[str], ledger: Ledger) -> str:
    """The token in ``listed`` that ``who`` names: a label, a fingerprint or a prefix of one."""
    by_label = ledger.find(who)
    matches = [
        t
        for t in listed
        if fingerprint(t) == by_label or (len(who) >= 6 and fingerprint(t).startswith(who))
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise Refused(
            f"no token in MCP_TOKENS is labelled or fingerprinted {who!r}; see `token list`"
        )
    raise Refused(f"{who!r} matches {len(matches)} tokens; give more of the fingerprint")


def cmd_token(args: argparse.Namespace, route: Route) -> int:
    env, listed = tokens(route)
    ledger = Ledger()
    if args.action == "list":
        for token in listed:
            fp = fingerprint(token)
            label = ledger.label_of(fp) or "(unlabelled: `token name` it)"
            note = "  placeholder, replace it" if token.startswith("CHANGE_ME") else ""
            print(f"{fp}  {label}  {ledger.entries.get(fp, ('', ''))[1]}{note}")
        gone = [fp for fp in ledger.entries if fp not in {fingerprint(t) for t in listed}]
        for fp in gone:
            print(f"{fp}  {ledger.label_of(fp)}  not in MCP_TOKENS (revoked by hand?)")
        print(f"{len(listed)} tokens; the running server has those of its last restart")
        return 0
    if args.action == "add":
        label = args.label
        if not label or not LABEL.fullmatch(label):
            raise Refused("a label is letters, digits, and . _ @ - (one per person or team)")
        if ledger.find(label) in {fingerprint(t) for t in listed}:
            raise Refused(f"{label!r} already has a token; revoke it first to replace it")
        token = secrets.token_hex(24)
        env.set("MCP_TOKENS", ",".join([*listed, token]))
        out = operator_dir("nexus-rag-tokens") / f"{label}.token"
        old = os.umask(0o077)
        try:
            out.write_text(token + "\n", encoding="utf-8")
        finally:
            os.umask(old)
        _give(out, operator())
        env.save()
        ledger.put(fingerprint(token), label)
        ledger.save()
        print(f"added     {label}  fingerprint {fingerprint(token)}")
        print(f"token in  {out}  (mode 0600; hand it over, then delete the file)")
        print("next      `restart`, then give the user their client snippet")
        return 0
    if args.action == "revoke":
        if args.who == "-":  # the token itself, from the operator: never echo it
            token = sys.stdin.read().strip()
            if token not in listed:
                raise Refused("the token given on stdin is not one of MCP_TOKENS")
        else:
            token = resolve(args.who or "", listed, ledger)
        remaining = [t for t in listed if t != token]
        if not remaining:
            raise Refused("that is the last token; the MCP server refuses to start without one")
        fp = fingerprint(token)
        env.set("MCP_TOKENS", ",".join(remaining))
        env.save()
        holder = ledger.label_of(fp)
        ledger.entries.pop(fp, None)
        ledger.save()
        print(f"revoked   {holder or '(unlabelled)'}  fingerprint {fp}")
        print("next      `restart`; the token works until the MCP server restarts")
        return 0
    if args.action == "name":
        if not args.label or not LABEL.fullmatch(args.label):
            raise Refused("usage: token name FINGERPRINT LABEL")
        token = resolve(args.who, listed, ledger)
        holder = ledger.find(args.label)
        if holder and holder != fingerprint(token) and holder in {fingerprint(t) for t in listed}:
            raise Refused(f"{args.label!r} already names another token")
        ledger.put(fingerprint(token), args.label)
        ledger.save()
        print(f"named     {fingerprint(token)}  {args.label}")
        return 0
    raise Refused(f"unknown token action {args.action!r}")


# --- the Corpus declaration ------------------------------------------------------------------


def staged() -> Path:
    return operator_dir("nexus-rag-staging") / "corpora.toml"


def parse(path: Path) -> dict:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8")).get("corpus", {})
    except tomllib.TOMLDecodeError as error:
        raise Refused(f"{path} is not valid TOML: {error}") from None


def summary(corpora: dict) -> dict[str, set[str]]:
    """What each Corpus covers, as comparable items: Roots, a path, Repository@ref."""
    covered = {}
    for name, table in corpora.items():
        kind = table.get("kind")
        if kind == "gitlab":
            items = set()
            for repo in table.get("repositories", []):
                refs = [
                    r if isinstance(r, str) else r.get("name", "?") for r in repo.get("refs", [])
                ]
                items |= {f"{repo.get('path', '?')}@{ref}" for ref in refs}
        elif kind == "confluence":
            items = {f"root {root}" for root in table.get("roots", [])}
        else:
            items = {f"path {table.get('path')}"}
        covered[name] = items
    return covered


def validate(path: Path, route: Route) -> int:
    parse(path)
    # The service user must read the file, and inside the stack it is mounted, so check a copy.
    with tempfile.TemporaryDirectory(prefix="rag-admin-") as folder:
        os.chmod(folder, 0o755)
        copy = Path(folder) / "corpora.toml"
        shutil.copyfile(path, copy)
        os.chmod(copy, 0o644)
        if route.systemd:
            command = route.command(["python", "-c", VALIDATOR, str(copy)])
        else:
            inside = "/tmp/rag-admin-corpora.toml"
            command = route.command(["python", "-c", VALIDATOR, inside], {str(copy): inside})
        return execute(command)


def cmd_corpora(args: argparse.Namespace, route: Route) -> int:
    stage = staged()
    if args.action == "stage":
        if stage.exists() and stage.read_text() != CORPORA.read_text() and not args.fresh:
            raise Refused(
                f"{stage} holds changes not applied yet: `corpora diff` shows them; `corpora "
                "apply` or `corpora discard` them first"
            )
        shutil.copyfile(CORPORA, stage)
        _give(stage, operator())
        print(f"staged    {stage}  (edit it, then `corpora apply`)")
        return 0
    if args.action == "discard":
        stage.unlink(missing_ok=True)
        print("discarded the staged copy")
        return 0
    if args.action == "add-repo":
        return add_repository(args, stage, route)
    target = Path(args.file) if args.file else stage
    if not target.exists():
        raise Refused(f"nothing staged: run `corpora stage` first ({target} does not exist)")
    if args.action == "check":
        return validate(target, route)
    if args.action == "diff":
        show_diff(target)
        return 0
    if args.action == "apply":
        return apply(target, route)
    raise Refused(f"unknown corpora action {args.action!r}")


def show_diff(target: Path) -> bool:
    diff = list(
        difflib.unified_diff(
            CORPORA.read_text().splitlines(),
            target.read_text().splitlines(),
            str(CORPORA),
            "staged",
            lineterm="",
        )
    )
    print("\n".join(diff) if diff else "no difference from the live declaration")
    return bool(diff)


def add_repository(args: argparse.Namespace, stage: Path, route: Route) -> int:
    corpus, path, refs = args.corpus, (args.path or "").strip("/"), args.refs
    if not corpus or not CORPUS_NAME.fullmatch(corpus):
        raise Refused("usage: corpora add-repo CORPUS GROUP/PROJECT REF [REF ...]")
    if "/" not in path or " " in path or path.startswith("http"):
        raise Refused(f"{path!r} is not a full project path such as platform/payments")
    if not refs:
        raise Refused("name at least one branch or tag; the first is the Primary ref")
    if not stage.exists():
        shutil.copyfile(CORPORA, stage)
        _give(stage, operator())
    corpora = parse(stage)
    block = []
    if corpus not in corpora:
        if not args.new_corpus:
            raise Refused(
                f"no Corpus named {corpus!r} is declared ({', '.join(corpora) or 'none'}); add "
                "--new-corpus to declare a GitLab Corpus by that name"
            )
        block += [f"[corpus.{corpus}]", 'kind = "gitlab"', ""]
    elif corpora[corpus].get("kind") != "gitlab":
        raise Refused(f"corpus {corpus!r} is a {corpora[corpus].get('kind')} Corpus, not gitlab")
    elif any(r.get("path", "").strip("/") == path for r in corpora[corpus].get("repositories", [])):
        raise Refused(f"{path} is already in {corpus!r}: edit its refs in {stage}")
    block += [
        f"[[corpus.{corpus}.repositories]]",
        f"path = {json.dumps(path)}",
        f"refs = [{', '.join(json.dumps(r) for r in refs)}]   # the first is the Primary ref",
    ]
    text = stage.read_text(encoding="utf-8").rstrip("\n")
    stage.write_text(f"{text}\n\n" + "\n".join(block) + "\n", encoding="utf-8")
    print(f"staged    {path} ({', '.join(refs)}) in corpus {corpus}; next `corpora apply`")
    return 0


def apply(target: Path, route: Route) -> int:
    if not show_diff(target):
        return 0
    if validate(target, route) != 0:
        print("refused   the live declaration is unchanged; fix the staged copy and apply again")
        return 1
    before, after = summary(parse(CORPORA)), summary(parse(target))
    saved = backup(CORPORA)
    rewrite(CORPORA, target.read_text(encoding="utf-8"))
    if target == staged():
        target.unlink()
    print(f"applied   {CORPORA}  (previous copy: {saved})")
    for name in after.keys() - before.keys():
        print(
            f"new       corpus {name}: run `sync {name}`; for a schedule, `timer enable {name}`"
            if name != "confluence"
            else "new       corpus confluence: the hourly timer syncs it"
        )
    for name in before.keys() - after.keys():
        print(
            f"removed   corpus {name}: its Documents stay searchable until `run nexus-rag "
            f"forget {name}`; `timer disable {name}` if it had a timer"
        )
    for name in after.keys() & before.keys():
        added, dropped = after[name] - before[name], before[name] - after[name]
        for item in sorted(added):
            print(f"added     {name}: {item}")
        for item in sorted(dropped):
            print(f"dropped   {name}: {item} (its Documents go at the next Sync of {name})")
        if added or dropped:
            print(f"next      `sync {name}` to bring the index in step now")
    return 0


# --- commands that run on the host ----------------------------------------------------------


def cmd_sync(args: argparse.Namespace, route: Route) -> int:
    if not CORPUS_NAME.fullmatch(args.corpus):
        raise Refused(f"{args.corpus!r} is not a Corpus name")
    if args.detach:
        unit = route.sync_unit(args.corpus)
        code = execute(["systemctl", "start", "--no-block", f"{unit}.service"])
        print(f"started   {unit}.service; follow: journalctl -fu {unit}.service")
        return code
    return execute(route.command(["nexus-rag", "sync", args.corpus]))


def cmd_timer(args: argparse.Namespace, route: Route) -> int:
    if args.action == "list":
        pattern = "nexus-rag-*" if route.systemd else "nexus-rag-container-*"
        return execute(["systemctl", "list-timers", "--all", pattern])
    if not args.corpus or not CORPUS_NAME.fullmatch(args.corpus):
        raise Refused("usage: timer enable|disable CORPUS")
    unit = f"{route.sync_unit(args.corpus)}.timer"
    verb = ["enable", "--now"] if args.action == "enable" else ["disable", "--now"]
    return execute(["systemctl", *verb, unit])


def cmd_env(args: argparse.Namespace, route: Route) -> int:
    env = EnvFile(route.env_file)
    if args.action == "show":
        print(f"# {route.env_file} (secrets hidden)")
        print("\n".join(env.masked()))
        return 0
    key = args.key or ""
    if key in MANAGED:
        raise Refused(f"{key} is not set by hand here: {MANAGED[key]}")
    if args.action == "set":
        if SECRET_KEY.search(key) and args.value is not None:
            raise Refused(f"{key} is a secret: give it on stdin, never as an argument")
        value = read_secret(key) if args.value is None else args.value
        check_value(key, value)
        if env.set(key, value):
            env.save()
            print(f"set       {key}")
            after_env_change([key], route)
        else:
            print(f"unchanged {key}")
        return 0
    if args.action == "unset":
        if not KEY.fullmatch(key):
            raise Refused("usage: env unset KEY")
        if env.unset(key):
            env.save()
            print(f"unset     {key}")
            after_env_change([key], route)
        else:
            print(f"unchanged {key} (not set)")
        return 0
    raise Refused(f"unknown env action {args.action!r}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="rag-admin", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("route")
    env = sub.add_parser("env")
    env.add_argument("action", choices=["show", "set", "unset"])
    env.add_argument("key", nargs="?")
    env.add_argument("value", nargs="?", help="for settings that are not secrets; else stdin")
    token = sub.add_parser("token")
    token.add_argument("action", choices=["list", "add", "revoke", "name"])
    token.add_argument(
        "who", nargs="?", help="add: the LABEL; revoke: label, fingerprint or - for stdin"
    )
    token.add_argument("label", nargs="?", help="name: the LABEL")
    corpora = sub.add_parser("corpora")
    corpora.add_argument(
        "action", choices=["stage", "add-repo", "check", "diff", "apply", "discard"]
    )
    corpora.add_argument("corpus", nargs="?")
    corpora.add_argument("path", nargs="?")
    corpora.add_argument("refs", nargs="*")
    corpora.add_argument("--new-corpus", action="store_true")
    corpora.add_argument("--fresh", action="store_true", help="stage: discard staged changes")
    corpora.add_argument("--file", help="check, diff, apply: this file instead of the staged one")
    run = sub.add_parser("run")
    run.add_argument("args", nargs=argparse.REMAINDER)
    sync = sub.add_parser("sync")
    sync.add_argument("corpus")
    sync.add_argument("--detach", action="store_true")
    sub.add_parser("smoke")
    acceptance = sub.add_parser("acceptance")
    acceptance.add_argument("args", nargs=argparse.REMAINDER)
    sub.add_parser("restart")
    timer = sub.add_parser("timer")
    timer.add_argument("action", choices=["enable", "disable", "list"])
    timer.add_argument("corpus", nargs="?")
    if argv[:1] in (["run"], ["acceptance"]) and argv[1:2] not in (["-h"], ["--help"]):
        # Their arguments belong to the command they run (`acceptance --only gitlab`).
        args = argparse.Namespace(command=argv[0], args=argv[1:])
    else:
        args = parser.parse_args(argv)
    if args.command == "token" and args.action == "add":
        args.label = args.who
    try:
        route = Route.detect()
        if args.command == "route":
            print(f"route     {route.name}")
            print(f"env file  {route.env_file}")
            print(f"corpora   {CORPORA}")
            print(f"sync unit {route.sync_unit('NAME')}.service / .timer")
            return 0
        if args.command == "env":
            return cmd_env(args, route)
        if args.command == "token":
            return cmd_token(args, route)
        if args.command == "corpora":
            return cmd_corpora(args, route)
        if args.command == "run":
            if not args.args:
                raise Refused("usage: run nexus-rag status (or python -m ...)")
            return execute(route.command(args.args))
        if args.command == "sync":
            return cmd_sync(args, route)
        if args.command == "smoke":
            flags = ["--systemd"] if route.systemd else []
            return execute(route.command(["python", "-m", "nexus_rag.smoke", *flags]))
        if args.command == "acceptance":
            flags = (
                ["--systemd"]
                if route.systemd
                else ["--skip", "install", "--mcp-url", "http://mcp:8765"]
            )
            return execute(
                route.command(["python", "-m", "nexus_rag.acceptance", *flags, *args.args])
            )
        if args.command == "restart":
            return execute(route.restart())
        if args.command == "timer":
            return cmd_timer(args, route)
    except Refused as refusal:
        print(f"refused   {refusal}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
