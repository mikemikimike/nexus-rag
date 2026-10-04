# Install: container route

The system and Qdrant run as a Docker compose stack (also usable as a Portainer stack). The Sync timers are host systemd units that run one Sync in the same image. The images carry every package, so this route needs no Mirror.

**You need:** a connected machine with Docker, git and Python 3.12 to build the drop; on the host, Docker with the compose plugin, systemd and Python 3.12.

> **No TLS of its own.** The compose file publishes the MCP port on every host interface over plain HTTP. Put a TLS-terminating proxy in front and restrict the port to that proxy or your client subnet. Docker's published ports bypass host firewall rules such as ufw: use the `DOCKER-USER` chain or a network firewall.

Steps:

1. [Build the Binary drop](#1-build-the-binary-drop) (connected machine)
2. [Start the stack](#2-start-the-stack) (host)
3. [Configure](#3-configure)
4. [Smoke check and first Sync](#4-smoke-check-and-first-sync)
5. [Accept the installation](#5-accept-the-installation)
6. [Connect Claude Code](#6-connect-claude-code)

Then: [day to day](#day-to-day) and [upgrades](#upgrades).

## 1. Build the Binary drop

On a connected machine, from a **clean checkout of the commit to ship**. `save` builds the system image from the working tree but archives `HEAD`, so uncommitted changes would end up in the image and not in the archive.

```bash
PYTHONPATH=src python3.12 -m local_rag.container save --out drop
```

`drop/` then holds `qdrant-image-<version>.tar`, `local-rag-image-<version>.tar`, `local-rag-<version>.tar.gz` (the compose file and timers) and `SHA256SUMS`. Carry it into the network.

## 2. Start the stack

On the host, unpack the code archive and run its copy of the command. It is also the upgrade:

```bash
VERSION=0.1.0          # `version` in pyproject.toml
tar -xzf drop/local-rag-$VERSION.tar.gz -C /tmp
sudo PYTHONPATH=/tmp/local-rag/src python3.12 -m local_rag.container start --drop drop
```

It verifies the checksums, the image tags and the versions the images report, and refuses before changing anything if one is wrong. Then it:

- loads the images;
- writes, only if missing, the env file `/etc/local-rag/container.env` with generated secrets (0600) and the Corpus declaration file `/etc/local-rag/corpora.toml`;
- installs the compose file in `/opt/local-rag/container/` and the container timers from `deploy/container/`;
- runs `docker compose up -d`.

Data and the index live in Docker volumes, which an upgrade does not touch.

Define this helper; every command below uses it:

```bash
compose() { sudo docker compose --project-name local-rag --env-file /etc/local-rag/container.env \
  -f /opt/local-rag/container/compose.yaml "$@"; }
compose up -d                              # apply an env file edit
compose run --rm -T sync local-rag status  # run any local-rag command
compose run --rm -T sync                   # one Sync (what the timers do)
```

## 3. Configure

1. **Model gateway and Confluence.** Edit `/etc/local-rag/container.env`:
   - `LITELLM_BASE_URL` (ends with `/v1`) and `LITELLM_API_KEY`
   - `RAG_EMBED_MODEL`, the `model_name` registered in LiteLLM (see `examples/litellm_config.yaml`). If the network has no embedding model, set `RAG_RETRIEVAL=keyword` instead ([why](how-it-works.md#keyword-only-mode))
   - optionally `RAG_RERANK_MODEL`
   - the `CONFLUENCE_*` block: bearer token, `CONFLUENCE_READ_ONLY=true`
   - if an internal CA signed Confluence's certificate (`unable to verify the first certificate`): copy its PEM, with any intermediate certificates, to `/etc/local-rag/certs/` and set `NODE_EXTRA_CA_CERTS=/etc/local-rag/certs/ca.pem`

   Other settings are in [configuration](configuration.md#settings).
2. **Corpora.** Declare them in `/etc/local-rag/corpora.toml` ([format](configuration.md#declaring-corpora)). Name the Confluence Corpus `confluence`: the timers sync it by that name.
3. **Apply:**

   ```bash
   compose up -d
   ```

## 4. Smoke check and first Sync

The smoke check tests each real component and changes nothing ([reading its output](operations.md#smoke-check)):

```bash
compose run --rm -T sync python -m local_rag.smoke
```

When it ends `SMOKE CHECK PASSED`, run the first Sync and find your first token:

```bash
compose run --rm -T sync
sudo grep '^MCP_TOKENS=' /etc/local-rag/container.env
```

From now on the container timers sync the `confluence` Corpus hourly. Any other Corpus syncs on its own templated timer, every 15 minutes, once you enable it. For a Corpus named `gitlab`:

```bash
sudo systemctl enable --now local-rag-container-sync@gitlab.timer
```

## 5. Accept the installation

After the first Sync, run the acceptance suite ([groups and output](operations.md#acceptance-suite)) in two parts. Inside the stack (set the port if you changed `MCP_PORT`):

```bash
compose run --rm -T sync python -m local_rag.acceptance --skip install --mcp-url http://mcp:8765
```

Then its `install` group on the host, which needs `docker` and `systemctl`:

```bash
sudo PYTHONPATH=/tmp/local-rag/src python3.12 -m local_rag.acceptance --container --only install
```

The installation is accepted when both end `ACCEPTANCE PASSED`.

## 6. Connect Claude Code

Give each user a token and follow [connect Claude Code](../README.md#connect-claude-code). Clients reach the server through your TLS proxy.

## Day to day

Prefix every `local-rag` command with `compose run --rm -T sync`:

```bash
compose run --rm -T sync local-rag status
compose run --rm -T sync local-rag search "ORA-00942"
compose run --rm -T sync        # Sync now
compose up -d                   # apply an env file edit, or a new MCP_TOKENS
compose logs -f mcp             # MCP server log
journalctl -u local-rag-container-sync   # Sync log
```

All commands, and what to do when something breaks: [operations](operations.md).

## Upgrades

Build a new drop (step 1) and run `container start` again (step 2). Read [upgrading](upgrading.md) first: some releases need a Rebuild or a settings change.
