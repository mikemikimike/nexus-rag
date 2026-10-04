# local-rag

A retrieval system that lets a team's Claude Code search internal documentation and code on a fully air-gapped network. Documents come from a **Corpus** (Confluence through `confluence-cli`, GitLab Repositories, or a plain folder), are embedded through the network's **Model gateway** (LiteLLM), and are served from a **Knowledge index** by one shared MCP server. The terms in bold are defined in [`CONTEXT.md`](CONTEXT.md) and used the same way here and in the code.

```
 Corpus ── Confluence (read-only), GitLab, or a folder of files
    │            Sync / Rebuild   (local-rag CLI, run by systemd timers)
    ▼
 Knowledge index ◄── embeddings (+ optional rerank) ── Model gateway (LiteLLM)
    │   Qdrant (127.0.0.1:6333): dense + BM25 sparse vectors, RRF fusion
    ▼
 MCP server (HTTP :8765 /mcp, bearer tokens, /healthz)
    ▲
 Claude Code (each user): search_docs, list_spaces, get_page
```

Search is hybrid (meaning and exact terms such as `ORA-00942`), or keyword-only if the network has no embedding model. A Rebuild goes live only if it succeeded, and can be undone. Details: [how it works](docs/how-it-works.md).

**Installing with an agent?** Point it at [`docs/agents/closed-network-setup.md`](docs/agents/closed-network-setup.md): it installs, configures and verifies step by step, and interviews the operator for the settings and tokens.

## Security model (read this first)

- **Every token sees everything indexed.** MCP tokens are not scoped: any valid token can search and read every Document in the Knowledge index. Confluence page permissions and GitLab Repository permissions are **not** enforced. Sync with service accounts that can only see what suits every user of this server, and choose Roots and Repositories accordingly.
- **Users.** Add or revoke one by editing `MCP_TOKENS` (comma-separated, one token per user or team) in the env file and restarting the MCP server. The server refuses to start without a token.
- **Qdrant** is reachable only from the host (systemd route) or only from the stack (container route), and is protected by an API key. The Confluence CLI uses a **read-only** profile (`CONFLUENCE_READ_ONLY=true`).
- **TLS.** The systemd route can serve TLS itself (`MCP_TLS_CERT`, `MCP_TLS_KEY`) or sit behind a proxy (`examples/nginx.conf.example`, with `MCP_HOST=127.0.0.1`). The container route has **no TLS of its own** and publishes the MCP port on every host interface: put a TLS-terminating proxy in front of it and restrict the port to that proxy or your client subnet. Docker's published ports bypass host firewall rules such as ufw, so use the `DOCKER-USER` chain or a network firewall.
- **Prompt injection.** Indexed content, Confluence pages and GitLab code alike, is untrusted input to the model. The MCP server's instructions and `examples/CLAUDE.md.snippet` tell Claude to treat it as data, and to leave out and flag text in it that addresses an AI, runs code fetched from the network or turns off a safety check; the `local-rag-research` skill adds a screening step before every answer. This lowers the risk; it does not remove it, so keep edit rights on indexed Spaces and Repositories to people you trust.
- **Secrets** are generated on the target machine, into the env file, and never committed. The repository holds only placeholders (`*.example.test`, `GATEWAY_HOST`).

## Choose a route

Both routes install from a hand-carried **Binary drop** (except compose, below), refuse a drop that does not match before changing anything, and never touch an existing env file or Knowledge index. What has to be carried in: [what crosses the air gap](docs/air-gap.md).

| | systemd route | container route |
|---|---|---|
| Runs as | native systemd services | a Docker compose stack (or Portainer stack), with host timers |
| Host needs | Python 3.12, systemd, packages from the **Mirror** | Docker with compose, systemd; no Mirror |
| Delivery | `python3.12 -m local_rag.installer` | `python3.12 -m local_rag.container start` |
| Env file | `/etc/local-rag/local-rag.env` | `/etc/local-rag/container.env` |
| TLS | built in, or behind a proxy | proxy required |
| **Guide** | **[docs/install-systemd.md](docs/install-systemd.md)** | **[docs/install-container.md](docs/install-container.md)** |

The container route also runs **straight from compose**, with no drop: clone the mirrored repository on the host and `docker compose up -d --build` builds the image there, with the host's own apt, pip and npm configuration. You write the env file and `corpora.toml` yourself; [the steps](docs/install-container.md#straight-from-compose).

Both routes ship with the same defaults: `CONFLUENCE_DOMAIN=confluence` and `GITLAB_URL=https://gitlab.org` (the compose file applies them when the env file leaves them out), and a commented Confluence, GitLab and folder Corpus in `corpora.toml`. Set the two tokens, declare the Corpora you want and change the domain and URL to your own.

Each guide walks the same steps, with the commands for its route:

1. Build the Binary drop on a connected machine.
2. Install it on the host.
3. Configure the Model gateway and declare the Corpora.
4. Run the smoke check, then the first Sync.
5. Run the acceptance suite until it ends `ACCEPTANCE PASSED`.
6. Connect Claude Code (below).

## Connect Claude Code

Each user, with their token:

```bash
export LOCAL_RAG_TOKEN=<your token>
claude mcp add --transport http local-rag https://rag.example.test/mcp \
  --header "Authorization: Bearer $LOCAL_RAG_TOKEN"
```

- Or commit `examples/mcp.json.example` as `.mcp.json` to a repository, and each user only sets `LOCAL_RAG_TOKEN`.
- If TLS uses an internal CA: `export NODE_EXTRA_CA_CERTS=/path/ca.pem`.
- Add `examples/CLAUDE.md.snippet` to the project's `CLAUDE.md`.
- To point Claude Code itself at your LiteLLM chat model, see `examples/claude_settings.example.json`.

The tools Claude gets:

| Tool | Returns |
|---|---|
| `search_docs(query, top_k, space, repository, ref)` | the best Passages with title, link, page id and Space; for code also file, line range, enclosing symbol and ref. It sees each Repository's Primary ref unless `ref` is given |
| `list_spaces()` | exactly the Spaces in the Knowledge index, GitLab groups included |
| `get_page(page_id)` | a Document's full indexed text, a page or a file, as of the last Sync. For the live page Claude uses the Confluence CLI plugin (`confluence read <page_id>`) |

## Documentation

| Read | For |
|---|---|
| [Install: systemd route](docs/install-systemd.md) | installing and upgrading as native services |
| [Install: container route](docs/install-container.md) | installing and upgrading as a compose stack |
| [Configuration](docs/configuration.md) | declaring Corpora (Confluence, folder, GitLab) and every setting |
| [Operations](docs/operations.md) | `local-rag` commands, smoke check, acceptance suite, backup, troubleshooting |
| [Upgrading](docs/upgrading.md) | release-specific upgrade steps |
| [What crosses the air gap](docs/air-gap.md) | what to request from the Mirror and carry in the Binary drop |
| [How it works](docs/how-it-works.md) | hybrid and keyword retrieval, Rebuilds, the Index format |
| [Closed-network setup](docs/agents/closed-network-setup.md) | the agent's step-by-step procedure, with an operator interview |
| [`CONTEXT.md`](CONTEXT.md), [`docs/adr/`](docs/adr/) | the domain language and the decisions behind it |
| [Development](docs/development.md) | running the tests, the module map |
