# local-rag (shared, hybrid) — Confluence → LiteParse → Qdrant → MCP → Claude Code

One VM inside the closed network runs everything. Team members' Claude Code connects to it over HTTPS.

```
                         ┌────────────────────────── VM ──────────────────────────┐
 Confluence ─ confluence-cli export ─► data/confluence_export/ (page.md, pdf, docx…)
                         │        ingest.py: LiteParse + chunking (LlamaIndex)      │
                         │        embeddings ◄── LiteLLM gateway (/v1/embeddings)   │
                         │                 ▼                                        │
                         │  Qdrant (127.0.0.1:6333)  dense + BM25 sparse vectors    │
                         │                 ▲  RRF fusion (+ optional LiteLLM rerank)│
                         │  mcp_server.py  (HTTP :8765 /mcp, bearer tokens)         │
                         └─────────────────────────▲───────────────────────────────┘
                              Claude Code (each user) ── tools: search_docs, get_page, list_spaces
                                        └── Confluence CLI plugin (live pages, read-only)
```

**Hybrid retrieval** (borrowed idea, implemented natively): every chunk gets a dense vector (your embedding model) and a BM25-style sparse vector (IDF computed by Qdrant). A query runs both, Qdrant fuses with Reciprocal Rank Fusion, then (optional) your LiteLLM reranker reorders the top candidates. Exact tokens like `ORA-00942`, `OPS-4711`, hostnames now match reliably.

**Zero-downtime rebuilds:** `ingest.py --rebuild` builds a new collection and swaps a Qdrant alias atomically.

## 1. Security model (read this first)

- **Everyone with a token sees everything indexed.** Confluence page permissions are NOT enforced. Export with a service account that can only see spaces suitable for all users of this MCP server, and choose roots accordingly.
- One **bearer token per user/team** in `MCP_TOKENS` (comma-separated); remove one to revoke. Put TLS in front (`MCP_TLS_CERT/KEY`, or `deploy/nginx.conf.example`). Restrict the port to your client subnet.
- Qdrant listens on **localhost only**, with an API key. The exporter uses a **read-only** Confluence profile.
- Confluence content is untrusted input to the model (prompt injection). `CLAUDE.md.snippet` tells Claude to treat it as data.

## 2. What to MIRROR (on an internet-connected machine, same OS/arch/Python minor as the VM)

| # | Item | How |
|---|------|-----|
| 1 | Python wheels | `bash scripts/download_wheels.sh` → `wheelhouse/` + `requirements.lock` (skip if you have an internal PyPI) |
| 2 | Qdrant server binary (x86_64 Linux) | `bash scripts/download_qdrant.sh` → `vendor/qdrant` |
| 3 | Python 3 + `python3-venv`, `flock` (util-linux), `nginx` (optional) | OS repos |
| 4 | Node.js + `confluence-cli` + its Claude plugin/skill | **already have**. On the VM the exporter needs the `confluence` binary as the service user |
| 5 | LibreOffice / ImageMagick — only if `lit` reports missing converters for Office/image files | OS packages |
| 6 | OCR language data — only if you enable `RAG_OCR=1` | see `lit parse --help` |
| 7 | Client machines: Claude Code (+ your internal CA cert if TLS uses one) | your usual channel |

No embedding model files and no torch: embeddings (and rerank) come from your LiteLLM gateway.

## 3. Install on the VM

```bash
# copy the project (incl. wheelhouse/, vendor/qdrant, requirements.lock) to /opt/local-rag, then:
sudo bash /opt/local-rag/deploy/install_vm.sh
```
It creates user `ragsvc`, the venv (from `wheelhouse/`), `.env` with **generated** Qdrant key and MCP token, Qdrant + MCP systemd services, and hourly/weekly sync timers. It prints your first token. Then:

1. Edit `/opt/local-rag/.env`: `LITELLM_BASE_URL`, `LITELLM_API_KEY`, `RAG_EMBED_MODEL`, optionally `RAG_RERANK_MODEL`, and the `CONFLUENCE_*` block (bearer PAT, `CONFLUENCE_READ_ONLY=true`). Verify the embedding model exists on LiteLLM (`litellm_config.yaml` shows the entry).
2. `cp confluence_roots.txt.example confluence_roots.txt` and list root page IDs/URLs (a space's home page covers the whole space).
3. Check everything: `sudo -u ragsvc /opt/local-rag/.venv/bin/python /opt/local-rag/scripts/smoke_test.py`
4. First export + index: `sudo -u ragsvc /opt/local-rag/scripts/export_confluence.sh`
5. `sudo systemctl start local-rag-mcp local-rag-sync.timer local-rag-full.timer`
6. Health: `curl http://VM:8765/healthz` → `ok`

TLS: either set `MCP_TLS_CERT`/`MCP_TLS_KEY` in `.env`, or run nginx with `deploy/nginx.conf.example` and set `MCP_HOST=127.0.0.1`.

## 4. Connect Claude Code (each user)

```bash
export LOCAL_RAG_TOKEN=<your token>
claude mcp add --transport http local-rag https://rag.corp.local/mcp \
  --header "Authorization: Bearer $LOCAL_RAG_TOKEN"
```
Or commit `.mcp.json.example` (as `.mcp.json`) to a repo; each user just sets `LOCAL_RAG_TOKEN`. If TLS uses an internal CA: `export NODE_EXTRA_CA_CERTS=/path/ca.pem`. Add `CLAUDE.md.snippet` to the project's `CLAUDE.md`. Point Claude Code at your LiteLLM chat model with `claude_settings.example.json` if you are air-gapped.

Tools exposed: `search_docs(query, top_k, space)`, `list_spaces()`, `get_page(page_id)`. For live pages Claude uses the Confluence CLI plugin (`confluence read <page_id>`).

## 5. Operations

| Task | Command |
|------|---------|
| Status / logs | `systemctl status qdrant local-rag-mcp`; `journalctl -u local-rag-mcp -f`; `journalctl -u local-rag-sync` |
| Sync now | `sudo systemctl start local-rag-sync.service` |
| Full refresh (attachments) | `sudo systemctl start local-rag-full.service` (weekly by timer) |
| Rebuild index (new model/chunking) | `sudo -u ragsvc /opt/local-rag/.venv/bin/python /opt/local-rag/ingest.py --rebuild` (searches keep working during it) |
| Try a query | `sudo -u ragsvc .venv/bin/python search.py "ORA-00942" -k 5 [--mode sparse\|dense] [--space ENG]` |
| Add/revoke a user | edit `MCP_TOKENS` in `.env`, `sudo systemctl restart local-rag-mcp` |
| Backup | Optional: the index is rebuildable from `data/` + the gateway. Qdrant snapshots: `POST /snapshots` on the collection |

Tuning (`.env`): `RAG_HYBRID_CANDIDATES` (per retriever), `RAG_MAX_PER_DOC`, `RAG_CHUNK_TOKENS` (then `--rebuild`), `RAG_RERANK_MODEL` + `RAG_RERANK_CANDIDATES`. After changing the embedding model always `--rebuild` (vector size changes).

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `healthz` 503 | Qdrant down or wrong `QDRANT_API_KEY` (must match `/etc/local-rag/qdrant.env`; re-run `install_vm.sh`) |
| `search_docs` says index empty | No successful ingest yet; run the sync, check `journalctl -u local-rag-sync` |
| Embedding errors | `LITELLM_BASE_URL` must end with `/v1`; model name = LiteLLM `model_name`; `NO_PROXY` for the gateway |
| 401 from clients | Token mismatch; header must be `Authorization: Bearer <token>` |
| Streaming/timeouts via nginx | Use the provided config (`proxy_buffering off`) |
| Export 401/403 or `confluence: not found` | Token/auth type; the service user needs the CLI on `PATH` (`CONFLUENCE_BIN`), `HOME=/var/lib/ragsvc` for `confluence init` configs |
| Rerank ignored | Look for "rerank failed" in the MCP log; the endpoint must support `/v1/rerank`; unset `RAG_RERANK_MODEL` to disable |
| `lit: command not found` | `LIT_BIN=/opt/local-rag/.venv/bin/lit` |
| Scanned PDFs empty | Enable OCR once language data is available offline |

## Files

`config.py` · `parsing.py` LiteParse+HTML · `embed.py` LiteLLM embeddings/tokenizer · `sparse.py` BM25 encoder · `store.py` Qdrant+alias · `ingest.py` · `search.py` hybrid+rerank · `mcp_server.py` HTTP MCP · `scripts/export_confluence.py` confluence-cli exporter · `scripts/` install/mirror/smoke · `deploy/` systemd, nginx, installer
