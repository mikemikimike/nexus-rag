# local-rag

A retrieval system that lets a team's Claude Code search internal documentation on a fully air-gapped network. Documents come from a **Corpus** (Confluence through `confluence-cli`, or a plain folder), are embedded through the network's **Model gateway** (LiteLLM), and are served from a **Knowledge index** by one shared MCP server. The terms in bold are defined in [`CONTEXT.md`](CONTEXT.md) and used the same way here and in the code.

An agent can install, configure, verify and operate it from [`docs/agents/closed-network-setup.md`](docs/agents/closed-network-setup.md), which also interviews the operator for the settings and tokens.

```
 Corpus ── Confluence (confluence-cli, read-only)  or  a folder of files
    │            Sync / Rebuild   (local-rag CLI, run by systemd timers)
    ▼
 Knowledge index ◄── embeddings (+ optional rerank) ── Model gateway (LiteLLM)
    │   Qdrant (127.0.0.1:6333): dense + BM25 sparse vectors, RRF fusion
    ▼
 MCP server (HTTP :8765 /mcp, bearer tokens, /healthz)
    ▲
 Claude Code (each user): search_docs, list_spaces, get_page
```

**Hybrid retrieval.** A Document is split into Passages of about `RAG_CHUNK_TOKENS` tokens, each prefixed with the Document's title when it is embedded, so a long page is searchable by any part of it; results still point at the Document. Every Passage gets a dense vector (your embedding model) and a BM25-style sparse vector. A query runs both, Qdrant fuses the rankings with Reciprocal Rank Fusion, and optionally the Model gateway's reranker reorders the best candidates. Exact identifiers such as `ORA-00942`, `OPS-4711` or hostnames match reliably. No more than a few Passages of one Document are returned per search.

**Safe Rebuilds.** A **Rebuild** re-embeds every Document into a fresh index and makes it the **Live index** only if its failure rate is within `RAG_REBUILD_FAILURE_THRESHOLD` (default `0`); otherwise the current Live index stays and the refusal says why. The **Previous index** is kept for one generation and `local-rag switch-back` returns to it. Searches keep working during a Sync or Rebuild.

## 1. Security model (read this first)

- **Every token sees everything indexed.** MCP tokens are not scoped: any valid token can search and read every Document in the Knowledge index, and Confluence page permissions are **not** enforced. Sync with a Confluence service account that can only see Spaces suitable for every user of this server, and choose Roots accordingly. Per-token or per-Space access control is out of scope.
- Add or revoke a user by editing `MCP_TOKENS` (comma-separated, one token per user or team) in the env file and restarting the MCP server. The server refuses to start without a token.
- Qdrant is reachable only from the host (systemd route) or only from the stack (container route), and is protected by an API key. The Confluence CLI uses a **read-only** profile (`CONFLUENCE_READ_ONLY=true`).
- **TLS.** The systemd route can serve TLS itself (`MCP_TLS_CERT`, `MCP_TLS_KEY`) or sit behind a proxy (`examples/nginx.conf.example`, with `MCP_HOST=127.0.0.1`). The container route has **no TLS of its own**: the compose file publishes the MCP port on every host interface and does not pass `MCP_TLS_*` to the container. Put a TLS-terminating proxy in front of it and restrict the port to that proxy or to your client subnet. Docker's published ports bypass host firewall rules such as ufw, so use the `DOCKER-USER` chain or a network firewall.
- Confluence content is untrusted input to the model (prompt injection). `examples/CLAUDE.md.snippet` tells Claude to treat it as data.
- The tracked repository holds only placeholders (`*.example.test`, `GATEWAY_HOST`). Secrets are generated on the target machine, into the env file, and never committed.

## 2. What crosses the air gap

The air-gapped network has a **Mirror** of public package sources, populated on request, and receives a hand-carried **Binary drop**. The **Installer** (systemd route) and `local_rag.container` (container route) install or upgrade from those two and refuse a drop that does not match what the repository expects, before they change anything. Neither ever touches an existing env file, roots file, documents or Knowledge index.

| What | Where from |
|---|---|
| Python packages, pinned in `requirements.lock` (the list to request) | Mirror, through pip (`--index-url`, or pip's own configuration) |
| Node.js and `confluence-cli` (Confluence Corpus only) | Mirror (npm); the service user needs the `confluence` command on its `PATH` |
| Qdrant server binary, version and checksum recorded in `src/local_rag/binary_drop.py` (systemd route) | Binary drop |
| This system's own code, `git archive` of this repository (while it is private) | Binary drop |
| Saved system and Qdrant images (container route) | Binary drop |
| Python 3.12 with `venv`, `systemd`, `useradd`, `chown` (systemd route); Docker with the compose plugin (container route) | OS packages: checked, never carried |
| LibreOffice or ImageMagick, only if `lit` reports missing converters for Office or image files; OCR language data, only with `RAG_OCR=1` | OS packages (see `lit parse --help`) |
| Claude Code on each client, plus your internal CA certificate if TLS uses one | your usual channel |

There are no embedding model files and no torch: embeddings and rerank come from the Model gateway.

The Qdrant release the system is tested against is recorded as `QDRANT` in `src/local_rag/binary_drop.py` (version and sha256 of the release tarball) and, for the container route, in the image tag of `deploy/container/compose.yaml`. To move to another release change them together.

## 3. Install, systemd route (Binary drop plus Mirror)

**On a connected machine**, from a clean checkout of the commit to ship, build the Binary drop:

```bash
VERSION=0.1.0          # `version` in pyproject.toml
QDRANT_VERSION=1.19.1  # `QDRANT.version` in src/local_rag/binary_drop.py
mkdir drop
git archive --prefix=local-rag/ -o drop/local-rag-$VERSION.tar.gz HEAD
curl -fL -o drop/qdrant-x86_64-unknown-linux-gnu.tar.gz \
  https://github.com/qdrant/qdrant/releases/download/v$QDRANT_VERSION/qdrant-x86_64-unknown-linux-gnu.tar.gz
(cd drop && sha256sum local-rag-$VERSION.tar.gz > SHA256SUMS)
```

The Qdrant tarball is verified against the checksum recorded in the repository, the code archive against `SHA256SUMS`. Carry `drop/` into the network.

**On the host** (x86_64 Linux with systemd and Python 3.12), unpack the code archive anywhere and run its Installer, which is the same command for the first install and every upgrade:

```bash
tar -xzf drop/local-rag-$VERSION.tar.gz -C /tmp
sudo PYTHONPATH=/tmp/local-rag/src python3.12 -m local_rag.installer --drop drop \
  [--index-url https://mirror.example.test/simple]
```

It checks the prerequisites and the drop, then creates (only what is missing): the `ragsvc` user, `/opt/qdrant/qdrant`, `/var/lib/qdrant`, the env file `/etc/local-rag/local-rag.env` with a **generated** Qdrant key and MCP token (mode 0640, `root:ragsvc`), an empty `/etc/local-rag/confluence_roots.txt`, the venv `/opt/local-rag/venv` (packages from the Mirror, then the code), and the units from `deploy/systemd/`. It enables `qdrant.service`, `local-rag-mcp.service` and the two timers. A second run changes nothing; an upgrade replaces the Qdrant binary, the code and the unit files (keep local changes to a unit in a drop-in, `systemctl edit`) and restarts what changed.

Then configure and start working:

1. Edit `/etc/local-rag/local-rag.env`: `LITELLM_BASE_URL` (ends with `/v1`), `LITELLM_API_KEY`, `RAG_EMBED_MODEL` (the `model_name` registered in LiteLLM, see `examples/litellm_config.yaml`), optionally `RAG_RERANK_MODEL`, and the `CONFLUENCE_*` block (bearer token, `CONFLUENCE_READ_ONLY=true`). Settings are read at start: `sudo systemctl restart local-rag-mcp` afterwards.
2. List the Confluence Roots, one page id or URL per line, in `/etc/local-rag/confluence_roots.txt`. A Space's home page as a Root covers the whole Space.
3. Run the smoke check (section 6), then the first Sync: `sudo systemctl start local-rag-sync.service` (it runs until done; follow it with `journalctl -fu local-rag-sync`).
4. Health: `curl http://HOST:8765/healthz` answers `ok`. Your first token: `sudo grep '^MCP_TOKENS=' /etc/local-rag/local-rag.env`.
5. When the first Sync is done, run the acceptance suite (section 6, "Acceptance suite"): `run_rag /opt/local-rag/venv/bin/python -m local_rag.acceptance --systemd` (the helper is defined below). The installation is accepted when it ends `ACCEPTANCE PASSED`.

The hourly timer runs an incremental Sync; the weekly timer runs the same Sync as a safety net. Both read the roots file. Until step 1 and 2 are done they fail harmlessly.

Commands below use this helper, which runs a program as the service user with the services' own settings:

```bash
run_rag() { sudo systemd-run --quiet --pipe --wait --collect -p User=ragsvc \
  -p EnvironmentFile=/etc/local-rag/local-rag.env -p Environment=HOME=/var/lib/ragsvc \
  -p Environment=PATH=/usr/local/bin:/usr/bin:/bin "$@"; }
run_rag /opt/local-rag/venv/bin/local-rag status
```

## 4. Install, container route (Binary drop of saved images)

For hosts that run Docker. The system and Qdrant run as a compose stack (also usable as a Portainer stack); the timers are host systemd units that run one Sync in the same image.

**On a connected machine** (Docker, git, Python 3.12), from a **clean checkout of the commit to ship**: `save` builds the system image from the working tree but archives `HEAD`, so uncommitted changes would end up in the image and not in the archive.

```bash
PYTHONPATH=src python3.12 -m local_rag.container save --out drop
```

`drop/` then holds `qdrant-image-<version>.tar`, `local-rag-image-<version>.tar`, `local-rag-<version>.tar.gz` (the compose file and timers) and `SHA256SUMS`. Carry it into the network.

**On the host** (Docker with the compose plugin, systemd), unpack the code archive (`VERSION` as above) and run its copy of the command; it is also the upgrade:

```bash
tar -xzf drop/local-rag-$VERSION.tar.gz -C /tmp
sudo PYTHONPATH=/tmp/local-rag/src python3.12 -m local_rag.container start --drop drop
```

It verifies the checksums, the image tags and the versions the images report, loads the images, writes (only if missing) the env file `/etc/local-rag/container.env` with generated secrets (mode 0600) and the roots file, installs the compose file in `/opt/local-rag/container/` and the container timers, and runs `docker compose up -d`. Data and the index live in Docker volumes, which an upgrade does not touch.

Configure as on the systemd route (step 1 and 2 of section 3, with `container.env`), then apply it with `up -d`. The container route needs no Mirror for packages, since the images carry them. Commands below use:

```bash
compose() { sudo docker compose --project-name local-rag --env-file /etc/local-rag/container.env \
  -f /opt/local-rag/container/compose.yaml "$@"; }
compose up -d                              # apply an env file edit
compose run --rm -T sync local-rag status  # run any local-rag command
compose run --rm -T sync                   # one Sync (what the timers do)
```

After the first Sync, run the acceptance suite (section 6, "Acceptance suite"): `compose run --rm -T sync python -m local_rag.acceptance --skip install --mcp-url http://mcp:8765` inside the stack, then its `install` group on the host, which needs `docker` and `systemctl` (the command is in section 6). The installation is accepted when both end `ACCEPTANCE PASSED`.

Mind the missing TLS (section 1).

## 5. Connect Claude Code (each user)

```bash
export LOCAL_RAG_TOKEN=<your token>
claude mcp add --transport http local-rag https://rag.example.test/mcp \
  --header "Authorization: Bearer $LOCAL_RAG_TOKEN"
```

Or commit `examples/mcp.json.example` as `.mcp.json` to a repository, and each user only sets `LOCAL_RAG_TOKEN`. If TLS uses an internal CA: `export NODE_EXTRA_CA_CERTS=/path/ca.pem`. Add `examples/CLAUDE.md.snippet` to the project's `CLAUDE.md`. To point Claude Code itself at your LiteLLM chat model, see `examples/claude_settings.example.json`.

Tools: `search_docs(query, top_k, space)` returns the best Passages with title, link, page id and Space; `list_spaces()` lists exactly the Spaces in the Knowledge index; `get_page(page_id)` returns a Document's full indexed text as of the last Sync. For the live page Claude uses the Confluence CLI plugin (`confluence read <page_id>`).

## 6. Smoke check and operations

The smoke check is the manual test against the real Model gateway, Qdrant, LiteParse, Confluence CLI and (systemd route) units. It prints `[ OK ]`, `[WARN]`, `[SKIP]` or `[FAIL]` per check, never a secret, and changes nothing:

```bash
run_rag /opt/local-rag/venv/bin/python -m local_rag.smoke --systemd   # systemd route
compose run --rm -T sync python -m local_rag.smoke                    # container route
```

### Acceptance suite

The smoke check looks at each component; the acceptance suite proves the whole stack works end to end, with assertions, a clear pass or fail and an exit code. Run it once the infrastructure is finished and after any change to it. It is part of the package (no extra dependency, nothing to download), so it runs from the installed environment like the smoke check:

```bash
run_rag /opt/local-rag/venv/bin/python -m local_rag.acceptance --systemd   # systemd route
compose run --rm -T sync python -m local_rag.acceptance --skip install --mcp-url http://mcp:8765   # container route, inside the stack (MCP_PORT if changed)
sudo PYTHONPATH=/tmp/local-rag/src python3.12 -m local_rag.acceptance --container --only install   # container route, on the host
```

It runs these groups, each check on its own line with a stable name (`--only GROUP`, repeatable, runs just some):

| Group | Proves |
|---|---|
| `gateway` | `/embeddings` answers and reports its dimension; a text far longer than one Passage is split into Passages the model accepts; rerank, only if `RAG_RERANK_MODEL` is set |
| `qdrant` | Qdrant answers with the configured key and refuses a wrong one |
| `index` | on a throwaway collection and a temp folder of generated Documents: Sync; a distinctive term (keyword) and a paraphrase sharing no word (meaning) find their Documents; text from the tail of a long Document is found by term and by meaning, which shows the real embedding model reads whole Passages; `status` counts; a changed Document is replaced and a removed one disappears; Rebuild and `switch-back` |
| `mcp` | the running MCP server, over HTTP with a token from `MCP_TOKENS` (or `--token`): `search_docs`, `list_spaces`, `get_page` answer; no token and a wrong token get 401; `/healthz` answers 200 without a token. It listens where `MCP_HOST` and `MCP_PORT` say, or `--mcp-url` (inside the container stack the server is another container: `--mcp-url http://mcp:8765`, as above) |
| `install` | needs `--systemd` or `--container`. Units exist, are active and enabled, the env file is not world-readable, the sync timers are scheduled; container route: the stack's `qdrant` and `mcp` containers are up and healthy. The container route's checks need `docker` and `systemctl`, so run them on the host (third command above, where only the standard library is needed), not inside the stack |
| `confluence` | only when `CONFLUENCE_DOMAIN` is set: the read-only `confluence spaces` call works and every Root of the roots file (`--roots-file`, default `/etc/local-rag/confluence_roots.txt`) resolves |

Reading the output: `[ OK ]`, `[FAIL]` or `[SKIP]` per check, with a reason. A `[FAIL]` line says what was seen, the likely cause and the section of `docs/agents/closed-network-setup.md` to read. A `[SKIP]` is never a pass: it says why (an optional component that is not configured, or an earlier check that failed, so one broken gateway shows as one failure and not a dozen). It starts by printing the settings it uses (never a secret; keys and tokens are only said to be set) and ends with a summary. Exit code `0`: nothing failed; `1`: a check failed; `2`: the command line or the settings cannot be used.

It never touches the live index: the `index` group works in its own collection `acceptance-<random>` and a temp folder, and removes both at the end, also after a failed check (the `cleanup.*` lines say so). If the process is killed, collections with that prefix stay behind and are safe to delete; they hold no live data. The `mcp` and `confluence` groups only read. It reads the same settings as the services, so run it with `run_rag` / `compose` as shown.

| Task | Command (prefix with `run_rag` / `compose run --rm -T sync`) |
|---|---|
| Status: Live index generation, Documents, Passages, Index format, last Sync | `local-rag status` |
| Try a query | `local-rag search "ORA-00942" [--space ENG] [-k 5]` |
| Sync now | `local-rag sync --roots-file /etc/local-rag/confluence_roots.txt`, or `systemctl start local-rag-sync.service` |
| Rebuild (new embedding model) | `local-rag rebuild --roots-file /etc/local-rag/confluence_roots.txt` |
| Undo a bad Rebuild | `local-rag switch-back` |
| Logs | `journalctl -u local-rag-mcp -f`, `journalctl -u local-rag-sync` (container route: `docker logs`, journal of the `local-rag-container-*` units) |
| Add or revoke a user | edit `MCP_TOKENS`, restart the MCP server (`compose up -d` on the container route) |

Exit codes of `local-rag`: `0` success; `1` failure or refusal (a Sync with failed Documents, a refused Rebuild, another run in progress, an unusable index); `2` a command line that cannot be used. Only one Sync or Rebuild runs at a time; the Knowledge index enforces it.

A folder Corpus works the same way (`local-rag sync --folder DIR`; the Space is the top-level folder), but one Knowledge index follows one Corpus: a Sync of another Corpus removes every Document it does not list. Try a folder with its own `RAG_COLLECTION`.

**Index format.** The index records how it was built (schema version, sparse encoder parameters, chunk size and overlap, embedding model name and dimension). After an upgrade, a model change or a change of `RAG_CHUNK_TOKENS` or `RAG_CHUNK_OVERLAP`, `local-rag status` tells whether it still matches. A mismatching index that can still be searched is served as a **Read-only index** with a warning and refuses Sync until you run a Rebuild; an unusable one is refused. Nothing is ever rebuilt automatically. After changing the embedding model or the chunk settings, run a Rebuild. An index built before Documents were split into Passages did not record its chunk settings, so it shows up as a Read-only index until you run one.

**Settings** (the env file; unset means the default): `RAG_TOP_K` (6), `RAG_HYBRID_CANDIDATES` per retriever (40), `RAG_MAX_PER_DOC` (2), `RAG_CHUNK_TOKENS` (400, the target Passage size in tokens), `RAG_CHUNK_OVERLAP` (60, tokens shared by neighbouring Passages; must be smaller than the size), `RAG_EMBED_BATCH` (16), `RAG_GATEWAY_TIMEOUT` (60 s), `RAG_RERANK_MODEL` (off), `RAG_REBUILD_FAILURE_THRESHOLD` (0, a share between 0 and 1), `RAG_COLLECTION`, `RAG_BM25_*`, `RAG_OCR`, `LIT_BIN`, `CONFLUENCE_BIN`. The authoritative list is `src/local_rag/environment.py`.

**Backup.** The Knowledge index can be rebuilt from the Corpus and the Model gateway; back up the env file (the settings and secrets), and optionally Qdrant snapshots.

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `/healthz` answers 503 | Qdrant is down or `QDRANT_API_KEY` differs from `QDRANT__SERVICE__API_KEY` (the systemd env file keeps them equal), or the Live index is unusable: `local-rag status` |
| `search_docs` says "No results" | Nothing indexed yet: run a Sync and read its output or the journal |
| Embedding errors, 401 or 404 | `LITELLM_BASE_URL` must end with `/v1`; `RAG_EMBED_MODEL` is the LiteLLM `model_name`; the gateway is always contacted directly, ignoring proxy variables. The smoke check shows the gateway's answer |
| `status` warns about the embedding dimension or Index format | The model, the chunk settings or the code changed since the index was built: run a Rebuild |
| A Rebuild is refused | The failure rate exceeded the threshold, or a listing was incomplete (a Root or an Attachment listing failed): read the reason, fix it, retry; the Live index is untouched |
| 401 from clients | The header must be `Authorization: Bearer <token>`, and the token one of `MCP_TOKENS` at the last restart |
| Streaming timeouts through nginx | Use `examples/nginx.conf.example` (`proxy_buffering off`) |
| Sync fails with 401/403 or `confluence: not found` | Token and auth type; the service user needs the CLI on its `PATH` (or `CONFLUENCE_BIN`) and `HOME=/var/lib/ragsvc` |
| Rerank ignored | Look for "rerank failed" in the MCP log; the Model gateway must serve `/v1/rerank` for `RAG_RERANK_MODEL`; unset it to disable |
| `lit: command not found` | `LIT_BIN=/opt/local-rag/venv/bin/lit` (the Installer's venv is found automatically) |
| Scanned PDFs are empty (counted as failed Documents) | Enable OCR (`RAG_OCR=1`) once the language data is available offline |
| The Installer or `container start` refuses | Read the message: wrong Qdrant version or checksum, a missing or mismatching file in the Binary drop, or a missing prerequisite. Nothing was changed |

## 8. Development

Python 3.12. The package is in `src/local_rag`.

```
pip install -e . --group dev   # or: uv pip install -e . --group dev
pytest
```

Tests use an in-memory Corpus, a fake Model gateway and in-process Qdrant, plus fake `lit`, `confluence`, `docker` and host commands, so they need no external service. `local_rag.smoke` and `local_rag.acceptance` check the real ones (the acceptance suite's own logic is tested against the fakes, with a deliberately broken fake for each failure path).

| Module | Role |
|---|---|
| `index`, `generations`, `index_format`, `exclusive`, `journal`, `ranking`, `sparse`, `chunker`, `extraction`, `settings` | the Knowledge index and what sits behind it |
| `corpus`, `confluence`, `folder`, `drawio` | the Corpus seam and its two adapters |
| `gateway`, `gateway_litellm`, `fakes` | the Model gateway seam, its LiteLLM adapter and the deterministic fake |
| `cli`, `mcp_server`, `environment` | adapters: `local-rag`, `local-rag-mcp`, and settings from environment variables |
| `installer`, `container`, `binary_drop` | delivery: the Installer, the container route and Binary drop verification |
| `smoke` | the manual smoke check |
| `acceptance`, `acceptance_report`, `acceptance_corpus`, `acceptance_stack`, `acceptance_index`, `acceptance_mcp`, `acceptance_host` | the live acceptance suite: its entry point, how it reports, its generated Documents, and one module per check group |

`deploy/systemd/` and `deploy/container/` hold the unit files, the compose file and the Dockerfile; `examples/` holds the client and proxy examples. To regenerate `requirements.lock` after a dependency change, follow the comment at its top.
