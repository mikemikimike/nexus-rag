# Install: container route

The system and Qdrant run as a Docker compose stack (also usable as a Portainer stack). The Sync timers are host systemd units that run one Sync in the same image. The images carry every package, so this route needs no Mirror, unless the network [builds its own image](#build-on-the-closed-network).

**You need:** a connected machine with Docker, git and Python 3.12 to build the drop; on the host, Docker with the compose plugin, systemd and Python 3.12.

> **No TLS of its own.** The compose file publishes the MCP port on every host interface over plain HTTP. Put a TLS-terminating proxy in front and restrict the port to that proxy or your client subnet. Docker's published ports bypass host firewall rules such as ufw: use the `DOCKER-USER` chain or a network firewall.

Steps:

1. [Build the Binary drop](#1-build-the-binary-drop) (connected machine)
2. [Start the stack](#2-start-the-stack) (host)
3. [Configure](#3-configure)
4. [Smoke check and first Sync](#4-smoke-check-and-first-sync)
5. [Accept the installation](#5-accept-the-installation)
6. [Connect Claude Code](#6-connect-claude-code)

Then: [build on the closed network](#build-on-the-closed-network), [day to day](#day-to-day) and [upgrades](#upgrades).

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

## Build on the closed network

Instead of carrying the system image in, the host can build it from the network's own mirrors: either [straight from compose](#straight-from-compose), from a clone of a mirrored repository, or with `start --build` from a code archive. Use this when the network has a Debian mirror, a PyPI index, an npm registry and a registry that holds `python:3.12-slim` and Qdrant's image, and you would rather carry only the code.

**You need,** besides step 2's prerequisites: those four sources, reachable from the host's Docker daemon (a `docker build` runs on the daemon, so its proxy settings apply, not your shell's). The PyPI index must hold every pin of `requirements.lock`. An `https://` mirror signed by an internal CA fails the build, because the base image does not know the CA: use `http://`, or make a base image that carries the CA and pass it with `--base-image`.

On the connected machine (or anywhere with the repository: `git archive` needs no network), from a clean checkout of the commit to ship, save only the code:

```bash
PYTHONPATH=src python3.12 -m local_rag.container save --code-only --out drop
```

`drop/` holds `local-rag-<version>.tar.gz` and `SHA256SUMS`. On the host, unpack the archive as in step 2 and start with `--build`:

```bash
VERSION=0.1.0
tar -xzf drop/local-rag-$VERSION.tar.gz -C /tmp
sudo PYTHONPATH=/tmp/local-rag/src python3.12 -m local_rag.container start --drop drop --build \
  --base-image registry.example.test/library/python:3.12-slim \
  --qdrant-image registry.example.test/qdrant/qdrant:v1.19.1 \
  --apt-mirror http://apt.example.test \
  --pip-index-url http://pypi.example.test/simple \
  --npm-registry http://npm.example.test/
```

Each option is optional: what you leave out is the public source, which a Docker daemon with a registry mirror configured can still reach. The command:

- verifies the code archive against `SHA256SUMS`, as before, and refuses before touching Docker if it differs;
- pulls the Qdrant image and tags it as the compose file names it (`qdrant/qdrant:v<version>`);
- builds `local-rag:<version>` from the verified code with `deploy/container/Dockerfile`, passing the options as its build arguments (`BASE_IMAGE`, `APT_MIRROR`, `PIP_INDEX_URL`, `NPM_CONFIG_REGISTRY`);
- checks the versions the two images report, which must be the ones the repository records, and then continues exactly as step 2 does.

`--apt-mirror` is the base URL of a mirror that serves `/debian` and `/debian-security`. The build arguments are visible in `docker history`: give URLs without credentials. An upgrade is the same command with a new code archive; the Qdrant image and the packages are pulled again, so keep the mirrors' contents in step with `requirements.lock`.

### Straight from compose

If the network mirrors this repository, skip the drop: clone it and let compose build the image on the host, with the host's own apt, pip and npm configuration. You write what `start` would have written (the env file, the Corpus declaration file); none of the drop's checksums apply, so the clone is only as trustworthy as the mirror.

```bash
git clone https://git.example.test/mirror/local-rag.git && cd local-rag    # a clean checkout of the commit to ship
sudo install -d -m 0755 /etc/local-rag && sudo touch /etc/local-rag/corpora.toml   # the sync service mounts it: it must be a file
sudo install -m 0600 /dev/null /etc/local-rag/container.env
```

Put `QDRANT_API_KEY=<secret>` and `MCP_TOKENS=<token>` in `/etc/local-rag/container.env` (generate each with `python3 -c 'import secrets; print(secrets.token_hex(24))'`), then the settings of [step 3](#3-configure). The compose file defaults `CONFLUENCE_DOMAIN` to `confluence` and `GITLAB_URL` to `https://gitlab.org`; set them in the env file to override. The images the build and the stack start from must be present under their public names, `python:3.12-slim` and `qdrant/qdrant:v1.19.1`; if your registry names them otherwise, `docker pull` and `docker tag` them. Then build and start from the checkout:

```bash
sudo docker compose --project-name local-rag --env-file /etc/local-rag/container.env \
  -f deploy/container/compose.yaml up -d --build
```

Define `compose()` as in step 2, with `-f` pointing at this checkout's `deploy/container/compose.yaml`. [Upgrading](#upgrades) is `git pull` and the same command.

This path does not check the image's version, write the Corpus template or install the timers; `start --build` does those. For the timers, copy `deploy/container/*.service` and `*.timer` to `/etc/systemd/system/` and the compose file to `/opt/local-rag/container/compose.yaml` (the units name that path; the image is already built, so the build context is not used there), then `systemctl daemon-reload` and `systemctl enable --now local-rag-container-sync.timer`.

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

Read [upgrading](upgrading.md) first: some releases need a Rebuild or a settings change.

With a drop (step 1), build a new one and run `container start` again (step 2). When the network [builds its own image](#build-on-the-closed-network), build the drop with `save --code-only` and run `start` with `--build` and the same options.

**Straight from compose,** in the checkout, after the mirror has the new version:

```bash
git pull                                       # or check out the tag you ship
git diff ORIG_HEAD HEAD -- deploy/container/compose.yaml | grep qdrant/qdrant   # a new Qdrant release?
sudo docker compose --project-name local-rag --env-file /etc/local-rag/container.env \
  -f deploy/container/compose.yaml up -d --build
```

1. If the `grep` prints a changed Qdrant tag, `docker pull` the new release from your registry and `docker tag` it to the name in the compose file first (see above); otherwise compose cannot find it.
2. `up -d --build` rebuilds the system image and recreates the `mcp` container, and `qdrant` only when its image changed. The env file, the volumes and the index are not touched. The Sync timers run the new image from their next run.
3. If you installed the timers by hand, refresh their files too, since the units and the compose file they name are copies: `sudo cp deploy/container/compose.yaml /opt/local-rag/container/` and `sudo cp deploy/container/*.service deploy/container/*.timer /etc/systemd/system/`, then `sudo systemctl daemon-reload`.
4. Run the smoke check ([step 4](#4-smoke-check-and-first-sync)). A release that changes the Index format makes the old index read-only until a Rebuild: `upgrading.md` says when, and `local-rag status` shows `access: read-only`.
