# Operations

How to check, run and repair an installation. Commands are route-neutral: prefix `local-rag` and `python -m` commands with the helper of your route, defined in its install guide:

| Route | Prefix |
|---|---|
| [systemd](install-systemd.md#2-run-the-installer) | `run_rag /opt/local-rag/venv/bin/` |
| [container](install-container.md#2-start-the-stack) | `compose run --rm -T sync ` |

- [Commands](#commands)
- [Smoke check](#smoke-check)
- [Acceptance suite](#acceptance-suite)
- [Backup](#backup)
- [Troubleshooting](#troubleshooting)

## Commands

| Task | Command |
|---|---|
| Status: Live index generation, Documents, Passages, Index format, last Sync of each Corpus | `local-rag status` |
| Try a query | `local-rag search "ORA-00942" [--space ENG] [-k 5]` |
| Sync now | `local-rag sync confluence` (any declared Corpus by its name), or `systemctl start local-rag-sync.service` |
| Rebuild (new embedding model, `RAG_RETRIEVAL` or chunk settings changed) | `local-rag rebuild` (re-embeds every declared Corpus) |
| Remove a Corpus no longer wanted | `local-rag forget NAME` |
| Undo a bad Rebuild | `local-rag switch-back` |
| Add or revoke a user | edit `MCP_TOKENS`, restart the MCP server |

Exit codes of `local-rag`: `0` success; `1` failure or refusal (a Sync with failed Documents, a refused Rebuild, another run in progress, an unusable index); `2` a command line that cannot be used. Only one Sync or Rebuild runs at a time.

A Rebuild can run for hours: detach it as in section 9 of [the closed-network setup guide](agents/closed-network-setup.md#9-operate), which also covers changing the embedding model without downtime.

## Smoke check

The manual test against the real Model gateway, Qdrant, LiteParse, Confluence CLI and (systemd route) units. It changes nothing and is safe before the first Sync.

```bash
python -m local_rag.smoke --systemd   # systemd route
python -m local_rag.smoke             # container route
```

Each check prints one line:

- `[ OK ]` it works;
- `[FAIL]` names the likely cause and the guide section to read;
- `[SKIP]` says why it did not run;
- `[WARN]` something works but needs acting on, such as a read-only index.

It never prints a secret, and ends with `SMOKE CHECK PASSED` or `SMOKE CHECK FAILED: <checks>`.

## Acceptance suite

The smoke check looks at each component; the acceptance suite proves the whole stack works end to end, with assertions, a clear pass or fail and an exit code. Run it once the infrastructure is finished and after any change to it. It is part of the package (nothing to download) and reads the same settings as the services, so run it with your route's prefix. The exact commands per route are in step 5 of each install guide.

### Groups

Each check prints on its own line with a stable name. `--only GROUP` (repeatable) runs just some, `--skip GROUP` leaves some out.

| Group | Proves |
|---|---|
| `gateway` | `/embeddings` answers and reports its dimension; a text far longer than one Passage is split into Passages the model accepts; rerank, only if `RAG_RERANK_MODEL` is set. Skipped in `keyword` mode |
| `qdrant` | Qdrant answers with the configured key and refuses a wrong one |
| `index` | on a throwaway collection and a temp folder of generated Documents: Sync; a distinctive term (keyword) and a paraphrase sharing no word (meaning) find their Documents; text from the tail of a long Document is found by term and by meaning, which shows the real embedding model reads whole Passages (meaning checks skipped in `keyword` mode); `status` counts; a changed Document is replaced and a removed one disappears; Rebuild and `switch-back` |
| `mcp` | the running MCP server, over HTTP with a token from `MCP_TOKENS` (or `--token`): `search_docs`, `list_spaces`, `get_page` answer; no token and a wrong token get 401; `/healthz` answers 200 without a token. It listens where `MCP_HOST` and `MCP_PORT` say, or `--mcp-url` |
| `install` | needs `--systemd` or `--container`. Units exist, are active and enabled, the env file is not world-readable, the sync timers are scheduled; container route: the stack's `qdrant` and `mcp` containers are up and healthy. On the container route run it on the host, not inside the stack |
| `confluence` | only when `CONFLUENCE_DOMAIN` is set: the read-only `confluence spaces` call works and every Root of the Confluence Corpora in the declaration file (`--corpora`) resolves |
| `gitlab` | only when the declaration file holds a GitLab Corpus: `git --version` runs and every Tracked ref of every GitLab Corpus can be read with `GITLAB_TOKEN` (it fetches the mirrors) |

### Reading the output

- It starts by printing the settings it uses (never a secret; keys and tokens are only said to be set) and ends with a summary.
- `[ OK ]`, `[FAIL]` or `[SKIP]` per check, with a reason. A `[FAIL]` line says what was seen, the likely cause and the section of [the closed-network setup guide](agents/closed-network-setup.md) to read.
- A `[SKIP]` is never a pass: it says why (an optional component that is not configured, or an earlier check that failed, so one broken gateway shows as one failure and not a dozen).
- Exit code `0`: nothing failed; `1`: a check failed; `2`: the command line or the settings cannot be used.

It never touches the live index: the `index` group works in its own collection `acceptance-<random>` and a temp folder, and removes both at the end, also after a failed check (the `cleanup.*` lines say so). If the process is killed, collections with that prefix stay behind and are safe to delete. The `mcp` and `confluence` groups only read.

## Backup

The Knowledge index can be rebuilt from the Corpora and the Model gateway. Back up the env file (the settings and secrets) and `corpora.toml`, and optionally Qdrant snapshots.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `/healthz` answers 503 | Qdrant is down or `QDRANT_API_KEY` differs from `QDRANT__SERVICE__API_KEY` (the systemd env file keeps them equal), or the Live index is unusable: `local-rag status` |
| `search_docs` says "No results" | Nothing indexed yet: run a Sync and read its output or the journal |
| Embedding errors, 401 or 404 | `LITELLM_BASE_URL` must end with `/v1`; `RAG_EMBED_MODEL` is the LiteLLM `model_name`; the gateway is always contacted directly, ignoring proxy variables. The smoke check shows the gateway's answer |
| `status` warns about the embedding dimension or Index format | The model, `RAG_RETRIEVAL`, the chunk settings or the code changed since the index was built: run a Rebuild ([why](how-it-works.md#index-format)) |
| A command stops with `RAG_RETRIEVAL is hybrid but RAG_EMBED_MODEL is not set` | Name the embedding model, or set `RAG_RETRIEVAL=keyword` to run without one |
| A command stops with `... but LITELLM_BASE_URL is not set` | `hybrid` and a named `RAG_RERANK_MODEL` need the Model gateway named: set `LITELLM_BASE_URL` (ending in `/v1`), or use `RAG_RETRIEVAL=keyword` with no reranker |
| Search results begin with `Note: Results are keyword-only` | The embedding model could not be reached for that search (see the smoke check), or the index was built without one and `status` says `retrieval mode differs`: run a Rebuild |
| A Rebuild is refused | The failure rate exceeded the threshold, or a listing was incomplete (a Root or an Attachment listing failed): read the reason, fix it, retry; the Live index is untouched |
| 401 from clients | The header must be `Authorization: Bearer <token>`, and the token one of `MCP_TOKENS` at the last restart |
| Streaming timeouts through nginx | Use `examples/nginx.conf.example` (`proxy_buffering off`) |
| Sync fails with 401/403 or `confluence: not found` | Token and auth type; the service user needs the CLI on its `PATH` (or `CONFLUENCE_BIN`) and `HOME=/var/lib/ragsvc` |
| Rerank ignored | Look for "rerank failed" in the MCP log; the Model gateway must serve `/v1/rerank` for `RAG_RERANK_MODEL`; unset it to disable |
| `lit: command not found` | `LIT_BIN=/opt/local-rag/venv/bin/lit` (the Installer's venv is found automatically) |
| Scanned PDFs are empty (counted as failed Documents) | Enable OCR (`RAG_OCR=1`) once the language data is available offline |
| The Installer or `container start` refuses | Read the message: wrong Qdrant version or checksum, a missing or mismatching file in the Binary drop, or a missing prerequisite. Nothing was changed |

Logs: `journalctl -u local-rag-mcp -f` and `journalctl -u local-rag-sync` (systemd route); `compose logs -f mcp` and the journal of the `local-rag-container-*` units (container route).
