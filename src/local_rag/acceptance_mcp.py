"""The acceptance checks of the running MCP server, reached over HTTP the way Claude Code does.

These are read-only against the live Knowledge index the server serves: a search, a list of
Spaces and a page lookup. Tokens come from ``MCP_TOKENS`` or ``--token``; they are never printed.
"""

import re
import secrets
import ssl
import urllib.error
import urllib.request

import anyio
import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

from local_rag.acceptance_report import Context
from local_rag.checks import QDRANT_DOWN, Failure
from local_rag.environment import split_tokens

TIMEOUT = 20.0
_DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # a local server, no proxy
_SERVER_SECTION = 'section 10 (row "The MCP server will not start, exit 2")'
_TOKEN_SECTION = 'section 10 (row "401 from /mcp")'


def server_url(ctx: Context) -> str:
    """Where the MCP server listens, as this process's environment says (or --mcp-url)."""
    if ctx.args.mcp_url:
        return ctx.args.mcp_url.rstrip("/")
    environment = ctx.environment
    host = environment.mcp_host
    if host in ("0.0.0.0", "::"):  # listening everywhere includes here
        host = "127.0.0.1"
    scheme = "https" if environment.mcp_tls_cert else "http"
    return f"{scheme}://{host}:{environment.mcp_port}"


def tokens(ctx: Context) -> list[str]:
    return list(split_tokens(ctx.args.token) if ctx.args.token else ctx.environment.mcp_tokens)


def run_mcp(ctx: Context) -> None:
    check = ctx.report.check
    url = server_url(ctx)
    check("mcp.healthz", lambda: _healthz(url))
    check(
        "mcp.no-token",
        lambda: _rejected(url, None, "a request without a token"),
        needs=["mcp.healthz"],
    )
    check(
        "mcp.wrong-token",
        lambda: _rejected(url, f"wrong-{secrets.token_hex(8)}", "a request with a wrong token"),
        needs=["mcp.healthz"],
    )
    for name, tool in (
        ("mcp.search_docs", _search),
        ("mcp.list_spaces", _spaces),
        ("mcp.get_page", _page),
    ):
        check(name, lambda tool=tool: tool(ctx, url), needs=["mcp.healthz", "mcp.wrong-token"])


def _request(url: str, token: str | None, method: str = "GET") -> tuple[int, str]:
    # No body: the server answers a refused request without reading one and closes the
    # connection, which resets a client still sending.
    request = urllib.request.Request(url, method=method)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with _DIRECT.open(request, timeout=TIMEOUT) as answer:
            return answer.status, answer.read(2000).decode("utf-8", "replace")
    except urllib.error.HTTPError as error:
        return error.code, error.read(2000).decode("utf-8", "replace")
    except (urllib.error.URLError, OSError) as error:
        reason = getattr(error, "reason", error)
        certificate = isinstance(reason, ssl.SSLError)
        raise Failure(
            f"{url} cannot be reached: {reason}",
            cause=(
                "this host does not trust the server's certificate (an internal CA: "
                "set SSL_CERT_FILE to its bundle)"
                if certificate
                else "the MCP server is not running, or MCP_HOST / MCP_PORT here differ "
                "from where it listens (container route: --mcp-url "
                "http://mcp:PORT from inside the stack)"
            ),
            see=_SERVER_SECTION,
        ) from None


def _healthz(url: str) -> str:
    status, body = _request(f"{url}/healthz", None)
    if status != 200:
        raise Failure(
            f"/healthz answered {status}: {' / '.join(body.split())[:200]}",
            cause="Qdrant or the Knowledge index is not reachable from the server",
            see=QDRANT_DOWN,
        )
    return "/healthz answers 200 without a token"


def _rejected(url: str, token: str | None, what: str) -> str:
    status, _ = _request(f"{url}/mcp", token, method="POST")
    if status != 401:
        raise Failure(
            f"{what} to /mcp was answered {status}, not 401",
            cause="the server does not check tokens, or another server answers on this port",
            see=_TOKEN_SECTION,
        )
    return f"{what} is rejected (401)"


def _call(ctx: Context, url: str, tool: str, arguments: dict) -> str:
    candidates = tokens(ctx)
    if not candidates:
        raise Failure(
            "no token to call the server with",
            cause="MCP_TOKENS is empty in this environment and --token was not given",
            see="section 5 (Configure)",
        )

    status, _ = _request(f"{url}/mcp", candidates[0], method="POST")
    if status == 401:  # the client's own error does not say why
        raise Failure(
            f"{tool} was refused (401) with the first token of MCP_TOKENS",
            cause="this environment's MCP_TOKENS differs from the running server's "
            "(a token added without a restart)",
            see=_TOKEN_SECTION,
        )

    async def call() -> str:
        headers = {"Authorization": f"Bearer {candidates[0]}"}
        async with httpx2.AsyncClient(headers=headers, trust_env=False, timeout=TIMEOUT) as http:
            transport = streamable_http_client(f"{url}/mcp", http_client=http)
            async with Client(transport) as client:
                result = await client.call_tool(tool, arguments)
        text = "\n".join(getattr(part, "text", "") for part in result.content)
        if result.is_error:
            raise Failure(
                f"{tool} returned an error: {' '.join(text.split())[:300]}",
                cause="the server could not answer: its Model gateway or Qdrant is "
                "down (read the server's own log)",
                see="section 10",
            )
        return text

    try:
        return anyio.run(call)
    except Failure:
        raise
    except Exception as error:  # includes the exception group of the client's task group
        text = " ".join(f"{type(e).__name__}: {e}" for e in _leaves(error))
        raise Failure(
            f"{tool} failed: {text[:300]}",
            cause="the server answered something that is not MCP at /mcp",
            see=_SERVER_SECTION,
        ) from None


def _leaves(error: BaseException) -> list[BaseException]:
    inner = getattr(error, "exceptions", None)
    return [leaf for e in inner for leaf in _leaves(e)] if inner else [error]


def _search(ctx: Context, url: str) -> str:
    text = _call(ctx, url, "search_docs", {"query": "acceptance probe", "top_k": 1})
    found = len(re.findall(r"^\[\d+\] ", text, re.MULTILINE))
    return f"search_docs answers ({found} result{'s' if found != 1 else ''})"


def _spaces(ctx: Context, url: str) -> str:
    text = _call(ctx, url, "list_spaces", {})
    return f"list_spaces answers ({text.strip() or 'empty'})"


def _page(ctx: Context, url: str) -> str:
    text = _call(ctx, url, "search_docs", {"query": "acceptance probe", "top_k": 1})
    found = re.search(r"page_id: (\S+)", text)
    if found:
        page = _call(ctx, url, "get_page", {"page_id": found.group(1), "max_chars": 200})
        if "is not in the index" in page:
            raise Failure(
                f"get_page does not know page {found.group(1)}, which a search just gave",
                cause="the Live index changed between the calls, or its records are "
                "damaged: run `local-rag status`",
                see="section 10",
            )
        return f"get_page returns the text of page {found.group(1)}"
    missing = f"acceptance-{secrets.token_hex(4)}"
    page = _call(ctx, url, "get_page", {"page_id": missing})
    if "is not in the index" not in page:
        raise Failure(
            f"get_page for a page that does not exist answered: {page[:100]}",
            cause="the server is not this system's MCP server",
            see=_SERVER_SECTION,
        )
    return "get_page answers (nothing indexed to read, a missing page is reported as such)"
