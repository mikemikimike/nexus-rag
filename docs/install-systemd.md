# Install: systemd route

The system runs as native systemd services: Qdrant, the MCP server and the Sync timers. Packages come from the **Mirror**, the code and the Qdrant binary from a **Binary drop**.

**You need:** a connected machine with git and curl to build the drop; on the host, x86_64 Linux with systemd, Python 3.12 with `venv`, `useradd`, `chown`, `git`, and access to the Mirror. For a Confluence Corpus also Node.js and `confluence-cli` from the Mirror, on the service user's `PATH`. The full list is in [what crosses the air gap](air-gap.md).

Steps:

1. [Build the Binary drop](#1-build-the-binary-drop) (connected machine)
2. [Run the Installer](#2-run-the-installer) (host)
3. [Configure](#3-configure)
4. [Smoke check and first Sync](#4-smoke-check-and-first-sync)
5. [Accept the installation](#5-accept-the-installation)
6. [Connect Claude Code](#6-connect-claude-code)

Then: [day to day](#day-to-day) and [upgrades](#upgrades).

## 1. Build the Binary drop

On a connected machine, from a clean checkout of the commit to ship:

```bash
VERSION=0.1.0          # `version` in pyproject.toml
QDRANT_VERSION=1.19.1  # `QDRANT.version` in src/nexus_rag/binary_drop.py
mkdir drop
git archive --prefix=nexus-rag/ -o drop/nexus-rag-$VERSION.tar.gz HEAD
curl -fL -o drop/qdrant-x86_64-unknown-linux-gnu.tar.gz \
  https://github.com/qdrant/qdrant/releases/download/v$QDRANT_VERSION/qdrant-x86_64-unknown-linux-gnu.tar.gz
(cd drop && sha256sum nexus-rag-$VERSION.tar.gz > SHA256SUMS)
```

The Qdrant tarball is verified against the checksum recorded in the repository, the code archive against `SHA256SUMS`. Carry `drop/` into the network.

## 2. Run the Installer

On the host, unpack the code archive anywhere and run its **Installer**. It is the same command for the first install and every upgrade:

```bash
tar -xzf drop/nexus-rag-$VERSION.tar.gz -C /tmp
sudo PYTHONPATH=/tmp/nexus-rag/src python3.12 -m nexus_rag.installer --drop drop \
  [--index-url https://mirror.example.test/simple]
```

It checks the prerequisites and the drop, and refuses before changing anything if either is wrong. Then it creates only what is missing:

| What | Where |
|---|---|
| Service user | `ragsvc` |
| Qdrant binary and data | `/opt/qdrant/qdrant`, `/var/lib/qdrant` |
| Env file, with a **generated** Qdrant key and MCP token (0640, `root:ragsvc`) | `/etc/nexus-rag/nexus-rag.env` |
| Corpus declaration file, commented | `/etc/nexus-rag/corpora.toml` |
| venv: packages from the Mirror, then the code | `/opt/nexus-rag/venv` |
| Units, from `deploy/systemd/` | `qdrant.service`, `nexus-rag-mcp.service`, the hourly timer (enabled); `nexus-rag-sync@.timer` (installed, not enabled) |

A second run changes nothing. An upgrade replaces the Qdrant binary, the code and the unit files, and restarts what changed; keep local changes to a unit in a drop-in (`systemctl edit`). It also stops and removes the weekly timer that earlier releases installed. It never touches an existing env file, declaration file, documents or Knowledge index.

Define this helper; every command below uses it. It runs a program as the service user with the services' own settings:

```bash
run_rag() { sudo systemd-run --quiet --pipe --wait --collect -p User=ragsvc \
  -p EnvironmentFile=/etc/nexus-rag/nexus-rag.env -p Environment=HOME=/var/lib/ragsvc \
  -p Environment=PATH=/usr/local/bin:/usr/bin:/bin "$@"; }
run_rag /opt/nexus-rag/venv/bin/nexus-rag status
```

## 3. Configure

1. **Model gateway and Confluence.** Edit `/etc/nexus-rag/nexus-rag.env`:
   - `LITELLM_BASE_URL` (ends with `/v1`) and `LITELLM_API_KEY`
   - `RAG_EMBED_MODEL`, the `model_name` registered in LiteLLM (see `examples/litellm_config.yaml`). If the network has no embedding model, set `RAG_RETRIEVAL=keyword` instead ([why](how-it-works.md#keyword-only-mode))
   - optionally `RAG_RERANK_MODEL`
   - the `CONFLUENCE_*` block: bearer token, `CONFLUENCE_READ_ONLY=true`

   Other settings are in [configuration](configuration.md#settings).
2. **Corpora.** Declare them in `/etc/nexus-rag/corpora.toml` ([format](configuration.md#declaring-corpora)). Name the Confluence Corpus `confluence`: the timers sync it by that name. A Space's home page as a Root covers the whole Space.
3. **TLS.** Either let the MCP server serve it (`MCP_TLS_CERT`, `MCP_TLS_KEY`), or put a proxy in front (`examples/nginx.conf.example`) and set `MCP_HOST=127.0.0.1`.
4. **Apply.** Settings are read at start:

   ```bash
   sudo systemctl restart nexus-rag-mcp
   ```

## 4. Smoke check and first Sync

The smoke check tests each real component and changes nothing ([reading its output](operations.md#smoke-check)):

```bash
run_rag /opt/nexus-rag/venv/bin/python -m nexus_rag.smoke --systemd
```

When it ends `SMOKE CHECK PASSED`, run the first Sync. It runs until done:

```bash
sudo systemctl start nexus-rag-sync.service
journalctl -fu nexus-rag-sync            # follow it
curl http://HOST:8765/healthz            # answers ok
sudo grep '^MCP_TOKENS=' /etc/nexus-rag/nexus-rag.env   # your first token
```

From now on the hourly timer runs `nexus-rag sync confluence`, which compares every Document with the index each time. Any other Corpus syncs on its own templated timer, every 15 minutes, once you enable it. For a Corpus named `gitlab`:

```bash
sudo systemctl enable --now nexus-rag-sync@gitlab.timer
```

## 5. Accept the installation

When the first Sync is done, run the acceptance suite ([groups and output](operations.md#acceptance-suite)):

```bash
run_rag /opt/nexus-rag/venv/bin/python -m nexus_rag.acceptance --systemd
```

The installation is accepted when it ends `ACCEPTANCE PASSED`.

## 6. Connect Claude Code

Give each user a token and follow [connect Claude Code](../README.md#connect-claude-code).

## Day to day

Prefix every `nexus-rag` command with `run_rag /opt/nexus-rag/venv/bin/`:

```bash
run_rag /opt/nexus-rag/venv/bin/nexus-rag status
run_rag /opt/nexus-rag/venv/bin/nexus-rag search "ORA-00942"
sudo systemctl start nexus-rag-sync.service   # Sync now
sudo systemctl restart nexus-rag-mcp          # apply an env file edit, or a new MCP_TOKENS
journalctl -u nexus-rag-mcp -f                # MCP server log
journalctl -u nexus-rag-sync                  # Sync log
```

All commands, and what to do when something breaks: [operations](operations.md).

## Upgrades

Build a new drop (step 1) and run the Installer again (step 2). Read [upgrading](upgrading.md) first: some releases need a Rebuild or a settings change.
