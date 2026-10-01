"""MCP server adapter over the Knowledge index: search_docs, list_spaces, get_page.

It talks only to the Knowledge index and reads no data directory. Settings come from the
environment (see ``environment``), plus:

    MCP_TOKENS    comma-separated bearer tokens; every valid token sees all indexed content
    MCP_HOST      address to listen on (default 127.0.0.1)
    MCP_PORT      port to listen on (default 8765)
    MCP_TLS_CERT, MCP_TLS_KEY   serve HTTPS with this certificate and key

Tokens are read once, at start: adding or revoking one is an edit of the env file and a
restart. ``--stdio`` serves one local client over stdin/stdout instead, with no tokens.
"""
import hmac
import logging
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import TextIO

import anyio.to_thread
from mcp.server.mcpserver import MCPServer
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response

from local_rag.environment import SettingsError, settings_from_environment
from local_rag.gateway_litellm import gateway_from_settings
from local_rag.index import KnowledgeIndex, Passage, SearchResults
from local_rag.settings import RetrievalMode

MAX_TOP_K = 15  # most Passages one search_docs call may ask for
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

type Serve = Callable[..., None]  # (app, *, host, port, certfile, keyfile), blocks until stopped


_CODE_PARAGRAPH = """Code results show the file, its line range, the enclosing symbol and their `ref`. Searches see each
Repository's Primary ref only: pass `ref` to search another Tracked ref (a branch or tag), and
`repository` (its path) to search one Repository."""

_SEARCH_DESCRIPTION = {
    RetrievalMode.HYBRID: f"""Hybrid (semantic + keyword) search over indexed Documents: pages, attachments and code files from GitLab Repositories.
Returns the best Passages with title, link, page_id and space.
Exact identifiers (error codes, hostnames, ticket ids) work well in the query.
Optional `space` restricts the search to one space (see list_spaces); spaces include GitLab groups.
{_CODE_PARAGRAPH}
For the whole Document use get_page with the returned page_id.""",
    RetrievalMode.KEYWORD: f"""Keyword search over indexed Documents: pages, attachments and code files from GitLab Repositories. It matches the
words you give, not their meaning: use the terms the Document itself would use.
Returns the best Passages with title, link, page_id and space.
Exact identifiers (error codes, hostnames, ticket ids) work well in the query.
Optional `space` restricts the search to one space (see list_spaces); spaces include GitLab groups.
{_CODE_PARAGRAPH}
For the whole Document use get_page with the returned page_id.""",
}


def build_server(index: KnowledgeIndex) -> MCPServer:
    server = MCPServer("local-rag", instructions=(
        "Search the team's indexed internal documentation."))

    @server.tool(description=_SEARCH_DESCRIPTION[index.retrieval])
    async def search_docs(query: str, top_k: int = 6, space: str = "",
                          repository: str = "", ref: str = "") -> str:
        limit = max(1, min(top_k, MAX_TOP_K))
        results = await anyio.to_thread.run_sync(
            lambda: index.search(query, k=limit, space=space or None, repository=repository or None,
                                ref=ref or None))
        return _format_results(results)

    @server.tool()
    async def list_spaces() -> str:
        """List the spaces that are indexed, GitLab groups included (usable as the `space` filter of search_docs)."""
        spaces = await anyio.to_thread.run_sync(index.list_spaces)
        return ", ".join(spaces) or "none"

    @server.tool()
    async def get_page(page_id: str, max_chars: int = 20000) -> str:
        """Return the full indexed text of a page or a file (a Document) by the page_id
        search_docs gave (or of an attachment by its attachment id), as of the last Sync."""
        text = await anyio.to_thread.run_sync(index.get_document, page_id)
        if text is None:
            return f"Page {page_id} is not in the index."
        note = "\n\n[truncated]" if len(text) > max_chars else ""
        return text[:max_chars] + note

    @server.custom_route("/healthz", methods=["GET"])
    async def healthz(request: Request) -> Response:
        health = await anyio.to_thread.run_sync(index.health)
        body = "\n".join(["ok" if health.ok else "unavailable", *health.notes])
        return PlainTextResponse(body, status_code=200 if health.ok else 503)

    return server


def build_app(index: KnowledgeIndex, tokens: Sequence[str]):
    """The ASGI app: MCP at /mcp behind bearer tokens."""
    return BearerAuth(build_server(index).streamable_http_app(stateless_http=True), tokens)


def main(argv: Sequence[str], environ: Mapping[str, str], *,
         index: KnowledgeIndex | None = None, serve: Serve | None = None,
         stderr: TextIO | None = None) -> int:
    """Serve until stopped; returns a process exit code (2 for a configuration that cannot be used).

    ``index`` and ``serve`` replace the Knowledge index the environment names and uvicorn,
    for callers (tests) that have stand-ins.
    """
    err = stderr or sys.stderr
    stdio = "--stdio" in argv
    unknown = [a for a in argv if a != "--stdio"]
    if unknown:
        print(f"error: unknown arguments: {' '.join(unknown)}", file=err)
        return 2
    tokens = _split(environ.get("MCP_TOKENS", ""))
    if not stdio and not tokens:
        print("error: MCP_TOKENS is empty; set at least one token "
              "(refusing to serve without authentication)", file=err)
        return 2
    try:
        port = _port(environ)
        if index is None:
            settings = settings_from_environment(environ)
            index = KnowledgeIndex(settings, gateway_from_settings(settings))
    except (SettingsError, ValueError) as error:
        print(f"error: {error}", file=err)
        return 2
    if stdio:
        build_server(index).run("stdio")  # stdout belongs to the protocol
        return 0
    (serve or _serve_with_uvicorn)(
        build_app(index, tokens), host=environ.get("MCP_HOST") or DEFAULT_HOST, port=port,
        certfile=environ.get("MCP_TLS_CERT") or None, keyfile=environ.get("MCP_TLS_KEY") or None)
    return 0


def run() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(levelname)s: %(message)s")
    sys.exit(main(sys.argv[1:], os.environ))


def _split(tokens: str) -> list[str]:
    return [token.strip() for token in tokens.split(",") if token.strip()]


def _port(environ: Mapping[str, str]) -> int:
    value = environ.get("MCP_PORT") or str(DEFAULT_PORT)
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"MCP_PORT must be a number, got {value!r}") from None


def _serve_with_uvicorn(app, *, host: str, port: int,
                        certfile: str | None, keyfile: str | None) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info",
                ssl_certfile=certfile, ssl_keyfile=keyfile)


class BearerAuth:
    """ASGI middleware that lets a request through only with one of the tokens.

    Pure ASGI, so streamed responses are not buffered. /healthz stays open for monitors.
    """

    def __init__(self, app, tokens: Sequence[str]):
        self._app = app
        self._tokens = [token.encode() for token in tokens]

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] == "/healthz" or self._allows(scope):
            return await self._app(scope, receive, send)
        await send({"type": "http.response.start", "status": 401, "headers": [
            (b"content-type", b"text/plain"), (b"www-authenticate", b"Bearer")]})
        await send({"type": "http.response.body", "body": b"unauthorized"})

    def _allows(self, scope) -> bool:
        authorization = dict(scope["headers"]).get(b"authorization", b"")
        scheme, _, presented = authorization.partition(b" ")
        if scheme.lower() != b"bearer":
            return False
        presented = presented.strip()
        # Compare against every token, so the time taken does not say which one matched.
        return any([hmac.compare_digest(presented, token) for token in self._tokens])


def _format_results(results: SearchResults) -> str:
    text = _format_passages(results)
    return f"Note: {results.notice}\n\n{text}" if results.notice else text


def _format_passages(passages: list[Passage]) -> str:
    if not passages:
        return "No results."
    blocks = []
    for number, passage in enumerate(passages, 1):
        page_id = passage.parent_id or passage.document_id
        header = f"{passage.link} | page_id: {page_id} | space: {passage.space}"
        if passage.ref:
            header += f" | ref: {passage.ref}"
        if passage.parent_id:
            header += f" | attachment: {passage.document_id}"
        for name, value in passage.code_location:
            header += f" | {name}: {value}"
        blocks.append(f"[{number}] {passage.title}\n{header}\n{passage.text.strip()}")
    return "\n\n---\n\n".join(blocks)


if __name__ == "__main__":
    run()
