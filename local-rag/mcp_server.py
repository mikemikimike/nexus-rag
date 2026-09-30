"""Shared MCP server (streamable HTTP, bearer-token auth) for the team.
  python mcp_server.py           -> HTTP on MCP_HOST:MCP_PORT (endpoint /mcp, health /healthz)
  python mcp_server.py --stdio   -> stdio (single-user local use)
Never print to stdout in stdio mode; logs go to stderr."""
import hmac
import sys

import config  # noqa: F401  (must be first)

try:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server
    _V2 = False
except ModuleNotFoundError:  # mcp 2.x renamed it
    from mcp.server.mcpserver import MCPServer as _Server
    _V2 = True

import search as rag


def _security():
    """We authenticate with bearer tokens; Host-header rebinding checks would block VM hostnames."""
    try:
        from mcp.server.transport_security import TransportSecuritySettings
        return TransportSecuritySettings(enable_dns_rebinding_protection=False)
    except ImportError:
        return None


def _make_mcp():
    if _V2:
        return _Server("local-rag")
    kw = dict(host=config.MCP_HOST, port=config.MCP_PORT, stateless_http=True)
    ts = _security()
    if ts:
        kw["transport_security"] = ts
    try:
        return _Server("local-rag", **kw)
    except TypeError:
        kw.pop("transport_security", None)
        return _Server("local-rag", **kw)


def _http_app():
    if _V2:
        kw = dict(stateless_http=True, host=config.MCP_HOST)
        ts = _security()
        if ts:
            kw["transport_security"] = ts
        return mcp.streamable_http_app(**kw)
    return mcp.streamable_http_app()


mcp = _make_mcp()


@mcp.tool()
def search_docs(query: str, top_k: int = 6, space: str = "") -> str:
    """Hybrid (semantic + keyword) search over indexed Confluence pages and attachments.
    Returns best passages with title, URL, page_id and space key.
    Exact identifiers (error codes, hostnames, ticket IDs) work well in the query.
    Optional `space` restricts to one Confluence space key (see list_spaces).
    For the full page use get_page(page_id) or the Confluence CLI (`confluence read <page_id>`)."""
    top_k = max(1, min(int(top_k), 15))
    try:
        return rag.format_results(rag.search(query, top_k, space or None))
    except Exception as e:
        return f"Search failed: {e}. If the index is empty, an admin must run ingest."


@mcp.tool()
def list_spaces() -> str:
    """List Confluence space keys that are indexed (usable as the `space` filter)."""
    if not config.DATA_DIR.exists():
        return "No content exported yet."
    return ", ".join(sorted(p.name for p in config.DATA_DIR.iterdir()
                            if p.is_dir() and not p.name.startswith("."))) or "none"


@mcp.tool()
def get_page(page_id: str, max_chars: int = 20000) -> str:
    """Return the exported Markdown of a page by page_id (snapshot as of the last sync)."""
    if not page_id.isdigit():
        return "page_id must be numeric."
    for f in config.DATA_DIR.glob(f"*/{page_id}/**/page.md"):
        text = f.read_text(encoding="utf-8", errors="replace")
        note = "\n\n[truncated]" if len(text) > max_chars else ""
        return text[:max_chars] + note
    return f"Page {page_id} not found in the index snapshot."


class BearerAuth:
    """Pure-ASGI bearer check (safe for streaming/SSE). /healthz stays open."""

    def __init__(self, app, tokens):
        self.app, self.tokens = app, [t.encode() for t in tokens]

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] == "/healthz":
            return await self.app(scope, receive, send)
        auth = dict(scope["headers"]).get(b"authorization", b"")
        ok = auth[:7].lower() == b"bearer " and any(
            hmac.compare_digest(auth[7:].strip(), t) for t in self.tokens)
        if not ok:
            await send({"type": "http.response.start", "status": 401,
                        "headers": [(b"content-type", b"text/plain"),
                                    (b"www-authenticate", b"Bearer")]})
            await send({"type": "http.response.body", "body": b"unauthorized"})
            return
        await self.app(scope, receive, send)


def build_app():
    from starlette.responses import PlainTextResponse

    async def healthz(request):
        try:
            rag._client().get_collections()
            return PlainTextResponse("ok")
        except Exception as e:
            return PlainTextResponse(f"qdrant unreachable: {e}", status_code=503)

    app = _http_app()
    app.add_route("/healthz", healthz)
    return BearerAuth(app, config.MCP_TOKENS)


def main():
    if "--stdio" in sys.argv:
        rag.warmup()
        mcp.run()
        return
    if not config.MCP_TOKENS:
        sys.exit("MCP_TOKENS is empty. Set at least one token in .env (refusing to serve unauthenticated).")
    try:
        rag.warmup()
    except Exception as e:
        print(f"WARNING: warmup failed ({e}); will retry on first request", file=sys.stderr)
    import uvicorn
    uvicorn.run(build_app(), host=config.MCP_HOST, port=config.MCP_PORT, log_level="info",
                ssl_certfile=config.MCP_TLS_CERT or None, ssl_keyfile=config.MCP_TLS_KEY or None)


if __name__ == "__main__":
    main()
