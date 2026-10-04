# Upgrading

An upgrade is the first install's command again with a new Binary drop: the Installer on the [systemd route](install-systemd.md#upgrades), `container start` on the [container route](install-container.md#upgrades). Neither touches the env file, the Corpus declaration file or the Knowledge index. Some releases also need the steps below; read the ones newer than your installed release before upgrading.

## The weekly Sync timer is gone

`nexus-rag-full.timer` (container route: `nexus-rag-container-full.timer`) ran the same `nexus-rag sync confluence` as the hourly timer, collided with it on Sundays and left a failed unit. The Installer and `container start` now stop, disable and remove it, so nothing is needed from you (if `systemctl --failed` still lists the old `*-full.service` from an earlier collision, clear it with `systemctl reset-failed`); a Sync that finds another run holding the lock now exits `75`, which the sync units accept as success. If you wrote your own unit around `nexus-rag sync`, add `SuccessExitStatus=75` to it, and change any monitoring that treated exit `1` as "another run in progress".

## The Model gateway must be named

An unnamed Model gateway is now a settings error, as an unnamed embedding model is. A `hybrid` install, or one with `RAG_RERANK_MODEL` set, that has no `LITELLM_BASE_URL` (or an empty one) no longer falls back to `http://localhost:4000/v1`: the services, the CLI, the smoke check and the acceptance suite refuse to start and say what to set.

1. Set `LITELLM_BASE_URL` in the env file. A LiteLLM on the same host is `LITELLM_BASE_URL=http://localhost:4000/v1`, written out. Or run `RAG_RETRIEVAL=keyword` without a reranker, which needs no gateway.
2. Restart the MCP server.

## Code search

This release changes the Index format and replaces `/etc/nexus-rag/confluence_roots.txt` (no longer read) with `corpora.toml`. The old index is served read-only until one Rebuild. The full procedure, with the checks and the way back, is "Upgrade runbook" in [the closed-network setup guide](agents/closed-network-setup.md#upgrade-runbook-from-a-release-without-code-search). In short:

1. Request the wheels `tree-sitter`, `tree-sitter-python` and `tree-sitter-c` from the Mirror (all pins of `requirements.lock`), and have `git` on the host.
2. Write the Roots of `confluence_roots.txt` into `[corpus.confluence]` of `corpora.toml` ([format](configuration.md#declaring-corpora)); the Corpus must be named `confluence`, as the timers sync it by that name.
3. Run the Installer (or `container start`) with the new drop.
4. `nexus-rag rebuild`, detached (it re-embeds every declared Corpus). It goes live only within `RAG_REBUILD_FAILURE_THRESHOLD`.
5. `nexus-rag status` shows `access: read-write`, one line per Corpus, no warnings; restart the MCP server; run the smoke check and the acceptance suite.
6. If the Rebuild is bad: `nexus-rag switch-back` (the Previous index becomes the Live index again, read-only under the new code).
