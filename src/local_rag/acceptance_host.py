"""The acceptance checks of the installation on the host (systemd or container route) and of
Confluence. The install group needs the standard library only, so it runs on a host that has no
Python packages: with ``PYTHONPATH=<unpacked code archive>/src``, as the Installer does."""
import json
from pathlib import Path

from local_rag.acceptance_report import Context, Failure, Skipped

SYSTEMD_ENV_FILE = "/etc/local-rag/local-rag.env"
CONTAINER_ENV_FILE = "/etc/local-rag/container.env"
SYSTEMD_UNITS = ("qdrant.service", "local-rag-mcp.service",
                 "local-rag-sync.timer", "local-rag-full.timer")
CONTAINER_TIMERS = ("local-rag-container-sync.timer", "local-rag-container-full.timer")
CONTAINER_SERVICES = ("qdrant", "mcp")
COMPOSE_PROJECT = "local-rag"
DEFAULT_ROOTS_FILE = "/etc/local-rag/confluence_roots.txt"


def run_install(ctx: Context) -> None:
    check = ctx.report.check
    if ctx.args.systemd:
        env_file, timers = SYSTEMD_ENV_FILE, [u for u in SYSTEMD_UNITS if u.endswith(".timer")]
        check("install.units", lambda: _units_exist(ctx, SYSTEMD_UNITS))
        check("install.active", lambda: _units_active(ctx, SYSTEMD_UNITS), needs=["install.units"])
        check("install.enabled", lambda: _units_enabled(ctx, SYSTEMD_UNITS),
              needs=["install.units"])
    elif ctx.args.container:
        env_file, timers = CONTAINER_ENV_FILE, list(CONTAINER_TIMERS)
        check("install.units", lambda: _units_exist(ctx, CONTAINER_TIMERS))
        check("install.stack", lambda: _stack_up(ctx))
        check("install.enabled", lambda: _units_enabled(ctx, CONTAINER_TIMERS),
              needs=["install.units"])
    else:
        check("install.units", lambda: _skip("neither --systemd nor --container was given"))
        return
    check("install.env-file", lambda: _env_file_private(ctx, env_file))
    check("install.timers", lambda: _timers_scheduled(ctx, timers), needs=["install.units"])


def _skip(why: str) -> str:
    raise Skipped(why)


def _each(ctx: Context, units, command: list[str]) -> list[str]:
    """The units for which ``systemctl <command> <unit>`` fails."""
    bad = []
    for unit in units:
        try:
            ctx.run(["systemctl", *command, unit])
        except Exception:
            bad.append(unit)
    return bad


def _units_exist(ctx: Context, units) -> str:
    missing = _each(ctx, units, ["cat", "--no-pager"])
    if missing:
        raise Failure(f"no unit file: {', '.join(missing)}",
                      cause="the Installer (or `container start`) did not run on this host, or "
                            "it was interrupted: run it again, it is idempotent",
                      see="section 4 (Install)")
    return f"{len(units)} unit files exist"


def _units_active(ctx: Context, units) -> str:
    inactive = _each(ctx, units, ["is-active", "--quiet"])
    if inactive:
        raise Failure(f"not active: {', '.join(inactive)}",
                      cause="the unit failed or was stopped; `systemctl status` and `journalctl "
                            "-u` name why", see='section 10 (row "/healthz 503, Qdrant is '
                                                'unreachable")')
    return f"{len(units)} units active"


def _units_enabled(ctx: Context, units) -> str:
    disabled = _each(ctx, units, ["is-enabled", "--quiet"])
    if disabled:
        raise Failure(f"not enabled: {', '.join(disabled)}",
                      cause="they would not come back after a reboot: `systemctl enable --now` "
                            "them, or run the Installer again", see="section 4 (Install)")
    return f"{len(units)} units enabled"


def _stack_up(ctx: Context) -> str:
    try:
        listing = ctx.run(["docker", "ps", "--filter",
                           f"label=com.docker.compose.project={COMPOSE_PROJECT}", "--format",
                           '{{.Label "com.docker.compose.service"}} {{.Status}}'])
    except Exception as error:
        raise Failure(f"cannot list the stack's containers: {error}",
                      cause="this command needs the docker command and its socket: run it on the "
                            "host with sudo, not inside the container", see="section 7 (Verify)"
                      ) from None
    status = {line.split(" ", 1)[0]: line.split(" ", 1)[1] if " " in line else ""
              for line in listing.splitlines() if line.strip()}
    down = [s for s in CONTAINER_SERVICES if not status.get(s, "").startswith("Up")]
    if down:
        raise Failure(f"not running: {', '.join(down)}",
                      cause="the stack is down: `compose ps`, then `docker logs` of the service",
                      see='section 10 (row "/healthz 503, Qdrant is unreachable")')
    if "unhealthy" in status["mcp"] or "starting" in status["mcp"]:
        raise Failure(f"mcp is {status['mcp']}",
                      cause="the MCP server cannot reach Qdrant", see="section 10")
    return f"{' and '.join(CONTAINER_SERVICES)} are up"


def _env_file_private(ctx: Context, path: str) -> str:
    try:
        mode = int(ctx.run(["stat", "-c", "%a", path]).strip(), 8)
    except Exception as error:
        raise Failure(f"cannot read the mode of {path}: {error}",
                      cause="the env file is missing or this user cannot see its directory",
                      see="section 5 (Configure)") from None
    if mode & 0o007:
        raise Failure(f"{path} is readable by everyone (mode {mode:o})",
                      cause="it holds the Qdrant key and the MCP tokens; a hand edit or a copy "
                            "changed its mode", see="section 0 (Ground rules): `chmod o-rwx` it")
    return f"{path} is not world-readable (mode {mode:o})"


def _timers_scheduled(ctx: Context, timers: list[str]) -> str:
    try:
        listing = ctx.run(["systemctl", "list-timers", "--all", "--no-pager", "--no-legend"])
    except Exception as error:
        raise Failure(f"cannot list the timers: {error}", cause="systemctl is not usable here",
                      see="section 7 (Verify)") from None
    lines = listing.splitlines()
    idle = []
    for timer in timers:
        line = next((l for l in lines if timer in l.split()), None)
        if line is None or line.split()[0] == "n/a":
            idle.append(timer)
    if idle:
        raise Failure(f"no next run is scheduled for {', '.join(idle)}",
                      cause="the timer is not started: `systemctl enable --now` it",
                      see="section 7 (Verify)")
    return f"{len(timers)} timers are scheduled"


def run_confluence(ctx: Context) -> None:
    check = ctx.report.check
    check("confluence.read", lambda: _read(ctx))
    check("confluence.roots", lambda: _roots(ctx), needs=["confluence.read"])


def _command(ctx: Context) -> str:
    return ctx.environ.get("CONFLUENCE_BIN") or "confluence"


def _read(ctx: Context) -> str:
    domain = ctx.environ.get("CONFLUENCE_DOMAIN")
    if not domain:
        raise Skipped("CONFLUENCE_DOMAIN is not set, so Confluence is not this Corpus")
    command = _command(ctx)
    try:
        ctx.run([command, "spaces"])  # one authenticated, read-only call
    except Exception as error:
        raise Failure(f"{command} spaces failed: {error}",
                      cause="the CLI is not on this user's PATH (set CONFLUENCE_BIN), or the "
                            "domain, token or auth type is wrong",
                      see='section 10 (rows "Sync: `confluence not found`" and "Sync: '
                          '`confluence ... failed`")') from None
    return f"{command} reads {domain}"


def _roots(ctx: Context) -> str:
    from local_rag.confluence import read_roots  # not at import: the install group runs without it
    path = Path(ctx.args.roots_file or DEFAULT_ROOTS_FILE)
    try:
        roots = read_roots(path)
    except (OSError, UnicodeDecodeError):
        raise Skipped(f"no roots file at {path} (use --roots-file)") from None
    if not roots:
        raise Skipped(f"{path} lists no Root")
    command, unresolved = _command(ctx), {}
    for root in roots:
        try:
            if "id" not in json.loads(ctx.run([command, "info", root, "--json"])):
                raise ValueError("the answer has no page id")
        except Exception as error:
            unresolved[root] = str(error) or type(error).__name__
    if unresolved:
        shown = "; ".join(f"{root}: {why}" for root, why in unresolved.items())
        raise Failure(f"{len(unresolved)} of {len(roots)} Roots do not resolve ({shown})",
                      cause="a page id or URL that is wrong, deleted, or not visible to the "
                            "Confluence account", see='section 10 (row "Sync: `failed root <id>`")')
    return f"all {len(roots)} Roots resolve"
