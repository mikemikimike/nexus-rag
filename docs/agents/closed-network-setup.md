# Closed-network setup (agent guide)

Install, configure, verify and operate local-rag on an air-gapped network, interviewing the operator for the inputs only they hold. Use the terms of `CONTEXT.md`: **Knowledge index**, **Corpus**, **Model gateway**, **Mirror**, **Binary drop**, **Installer**, **Sync**, **Rebuild**, **Live index**, and for code **Repository**, **Tracked ref**, **Primary ref**. The README is the reference for what the system is; this guide is the procedure. Every command below exists in this repository's code at the time of writing; anything that needs a real host to confirm is listed in section 11.

Work the sections in order. Each ends with a **Gate**: do not start the next section until the gate holds, and tell the operator which gate you passed.

## 0. Ground rules

- **Secrets stay out of output.** Never `cat` an env file, echo a token or paste one into a log line. Show an env file masked:
  `sudo sed -E 's/^([A-Z0-9_]*(KEY|TOKENS?))=.*/\1=<hidden>/' "$ENV_FILE" | grep -v '^#'`
  Values the operator gives you pass through this session; offer them the alternative of editing the env file themselves and letting you verify with the smoke check.
- **Existing state is never overwritten.** The Installer and `local_rag.container start` create the env file, the Corpus declaration file, the volumes and the Knowledge index once and never touch them again. Do not delete or regenerate them, do not clear Qdrant's storage, and do not run a **Rebuild** unless the operator asked for one: a Rebuild is never automatic, and the weekly timer runs the same `local-rag sync` as the hourly one.
- **Every MCP token sees every indexed Document.** Page permissions in Confluence and Repository permissions in GitLab are not enforced: every MCP token can read every listed Repository, whoever the GitLab token's owner is. Say this to the operator in section 3 and get an explicit yes.
- **Stop and ask** when: a command fails and the troubleshooting table (section 10) does not name the cause; the Installer or `container start` refuses (it changes nothing when it refuses, so read the message to the operator and ask); an answer cannot pass its validation after one retry; you would need to leave the paths this guide uses.
- **Where commands run.** Everything runs on the target host, as a user with `sudo`, except section 8 (the operators' clients) and the connected-machine build, which is not part of this guide (the README, sections 3 and 4, has it, and its ship-from-a-clean-checkout rule: `container save` builds from the working tree but archives `HEAD`).

## 1. Route and helpers

Ask the first question group now (A1 to A3 in section 3) so that you know the route, then set the helpers for it. The two routes differ in where settings live and how commands run:

| | systemd route | container route |
|---|---|---|
| Delivery | Installer, `python3.12 -m local_rag.installer` | `python3.12 -m local_rag.container start` |
| Needs | Python 3.12 with `venv`, systemd, the Mirror, `confluence` on the service user's PATH | Docker with the compose plugin, systemd (for the timers) |
| Env file `ENV_FILE` | `/etc/local-rag/local-rag.env` (root:ragsvc, 0640) | `/etc/local-rag/container.env` (0600) |
| Run a `local-rag` command | `run_rag /opt/local-rag/venv/bin/local-rag ...` | `compose run --rm -T sync local-rag ...` |
| Apply an env file edit | `sudo systemctl restart local-rag-mcp` | `compose up -d` |
| TLS | the MCP server can serve it (`MCP_TLS_CERT`, `MCP_TLS_KEY`) or sit behind a proxy | none: plain HTTP, no TLS pass-through; a TLS proxy in front is required |

Define the helpers for your route. `run_rag` and `compose` are the README's (section 3 and 4), kept equal by a test; `set_env` edits one setting idempotently and never prints the value.

```bash
ENV_FILE=/etc/local-rag/local-rag.env     # container route: /etc/local-rag/container.env

run_rag() { sudo systemd-run --quiet --pipe --wait --collect -p User=ragsvc \
  -p EnvironmentFile=/etc/local-rag/local-rag.env -p Environment=HOME=/var/lib/ragsvc \
  -p Environment=PATH=/usr/local/bin:/usr/bin:/bin "$@"; }

compose() { sudo docker compose --project-name local-rag --env-file /etc/local-rag/container.env \
  -f /opt/local-rag/container/compose.yaml "$@"; }
```

```bash
set_env() {  # usage: RAG_VALUE=... [RAG_APPEND=1] set_env KEY      (edits $ENV_FILE)
  sudo --preserve-env=RAG_VALUE,RAG_APPEND python3 - "$ENV_FILE" "$1" <<'PY'
import os, re, sys
path, key = sys.argv[1:3]
value, append = os.environ["RAG_VALUE"], os.environ.get("RAG_APPEND") == "1"
if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key) or not value or re.search(r"[\s'\"#$\\]", value):
    sys.exit(f"refused: {key} needs a non-empty value without whitespace, quotes, # $ or backslash")
lines = open(path, encoding="utf-8").read().split("\n")
active = [i for i, line in enumerate(lines) if line.startswith(key + "=")]
if append and active:
    current = lines[active[0]].split("=", 1)[1]
    if value in current.split(","):
        print(f"unchanged {key}"); sys.exit(0)
    value = f"{current},{value}" if current else value
new = f"{key}={value}"
if active:
    if lines[active[0]] == new and len(active) == 1:
        print(f"unchanged {key}"); sys.exit(0)
    lines[active[0]] = new
    for i in reversed(active[1:]):
        del lines[i]
else:
    template = [i for i, line in enumerate(lines) if re.match(rf"#\s*{key}=", line)]
    if template:
        lines[template[0]] = new
    else:
        if lines and lines[-1] == "":
            lines.pop()
        lines += [new, ""]
open(path, "w", encoding="utf-8").write("\n".join(lines))
print(f"set       {key}")
PY
}
```

`set_env` replaces the active `KEY=` line, else fills in the commented template line the Installer wrote, else appends; it keeps the file's owner and mode, and `RAG_APPEND=1` adds a comma-separated item (used for `MCP_TOKENS`). It refuses values that systemd or compose would misread (whitespace, quotes, `#`, `$`, backslash): when an operator's secret contains one, stop and ask them to edit the file by hand.

**Gate:** the route is chosen and the helpers are defined in your shell.

## 2. Pre-flight

Run the checks of your route. Report each result; a failed check is fixed by the operator (these are OS packages: checked, never carried).

systemd route:

```bash
uname -m                                  # x86_64 (the only supported machine)
python3.12 --version                      # Python 3.12.x
python3.12 -c 'import venv, ensurepip'    # silent; Debian family: install python3-venv
command -v systemctl useradd chown git   # four paths
sudo -n true && echo sudo-ok              # sudo works without a prompt, or ask the operator to run the Installer
```

Container route:

```bash
docker --version && docker compose version
command -v systemctl python3.12           # the timers are systemd units; `container start` runs on host Python 3.12
sudo -n true && echo sudo-ok
```

Both routes, only for a Confluence Corpus: the service user finds the Confluence CLI with the PATH the units give it (the container image carries its own, skip this on the container route):

```bash
PATH=/usr/local/bin:/usr/bin:/bin command -v confluence   # a path; else the Mirror's npm: `npm install --global confluence-cli`
```

LibreOffice or ImageMagick are only needed if `lit` later reports missing converters for Office or image files; OCR language data only with `RAG_OCR=1`. Locking is done by the Knowledge index, not `flock`, so `flock` is not a prerequisite.

**The Binary drop** (`DROP` is the directory the operator carried in, `VERSION` is `version` in `pyproject.toml`; the Installer and `container start` verify everything again and refuse a mismatch before they change anything, so this is an early warning, not the safeguard):

```bash
DROP=/path/to/drop VERSION=0.1.0
ls "$DROP"                                      # see below
(cd "$DROP" && sha256sum -c SHA256SUMS)         # every listed file: OK
tar -xzf "$DROP/local-rag-$VERSION.tar.gz" -C /tmp
PYTHONPATH=/tmp/local-rag/src python3.12 -c 'from local_rag.binary_drop import QDRANT; print(QDRANT.version, QDRANT.sha256)'
```

- systemd route: `local-rag-$VERSION.tar.gz`, `SHA256SUMS` (lists the code archive) and `qdrant-x86_64-unknown-linux-gnu.tar.gz`, whose `sha256sum` must equal the `QDRANT.sha256` printed above (the Installer also runs the unpacked binary and requires `qdrant --version` to report `QDRANT.version`).
- container route: `local-rag-$VERSION.tar.gz`, `local-rag-image-$VERSION.tar`, `qdrant-image-<QDRANT.version>.tar` and `SHA256SUMS` listing all three. `container start` also checks the image tags and the versions the images report.

**The Mirror** (systemd route; the container route's images carry their packages). Request from the Mirror every package in `requirements.lock` (it is the list to request). Prove they are all there, in a throwaway venv, before the Installer starts changing the machine:

```bash
python3.12 -m venv /tmp/mirrorcheck
/tmp/mirrorcheck/bin/pip install --dry-run --ignore-installed --only-binary :all: --no-deps \
  -r /tmp/local-rag/requirements.lock --index-url "$MIRROR_URL"     # omit --index-url if pip is configured for the Mirror
rm -rf /tmp/mirrorcheck
```

`Would install ...` for every pin means the Mirror has them; `No matching distribution found for X` names a package to request from the Mirror's operators. Node.js and `confluence-cli` also come from the Mirror (npm), for a Confluence Corpus. The pins include the tree-sitter wheels that split Python and C code at function and class boundaries (`tree-sitter`, `tree-sitter-python`, `tree-sitter-c`); an upgrade from a release without code search must request them too, and pip names the missing one. `git` is an OS package (checked above), needed on the systemd route for a GitLab Corpus.

**Gate:** every pre-flight command passed, the drop's checksums verify, and the Mirror has every pin (systemd route).

## 3. Interview the operator

Ask one group per message, quote the question text, wait for the answers, validate each answer, and re-ask only the item that failed (say why it failed). Keep answers in shell variables for the validations; write them to the env file in section 5, not before. Skip questions the operator says do not apply.

### A. Delivery

> **A1.** Is this host going to run local-rag directly under systemd, or as a Docker compose stack? (Docker is needed for the second.)
> **A2.** Which directory holds the Binary drop you carried in?
> **A3.** (systemd route) What is the Mirror's Python package index URL, or is pip on this host already configured for it?

Validate: A1 is one of the two routes and its pre-flight passes. A2 contains the files listed in section 2. A3 starts with `http://` or `https://`; `python3.12 -m pip config list` shows an `index-url` when pip is already configured, in which case `--index-url` is left out of the Installer command.

### B. Model gateway

> **B1.** What is the LiteLLM Model gateway's base URL, including `/v1`? (`LITELLM_BASE_URL`; example `http://gateway.example.test:4000/v1`.)
> **B2.** What API key does the gateway want? Leave it blank if it needs none. (`LITELLM_API_KEY`)
> **B3.** Does the Model gateway serve an embedding model: yes, no, or not sure? If yes, what is its name as registered in LiteLLM, its `model_name`? (`RAG_EMBED_MODEL`; see `examples/litellm_config.yaml`.) Changing it later means a Rebuild.
> **B4.** Optional: the reranker's `model_name`? Leave it blank for no reranking. (`RAG_RERANK_MODEL`)

Optional, not asked unless the embedding model is Qwen3-Embedding: `RAG_QUERY_INSTRUCTION`, an instruction put in front of each *query* embedding in the format Qwen3-Embedding was trained with (`Instruct: <instruction>\nQuery: <query>`). Suggested value: `Given a question, retrieve the passages that answer it`. Documents are embedded unchanged and it is not part of the Index format, so setting or changing it needs a restart of the MCP server but no Rebuild. Unset (the default) leaves queries as they are.

B3 decides the **Retrieval mode** (`RAG_RETRIEVAL`). With an embedding model the mode is `hybrid`, the default: searches match by meaning and by keywords. Without one it is `keyword`: searches match by keywords alone and nothing needs the gateway, so B1 and B2 matter only if B4 is answered. The mode is a deliberate choice: the system never switches by itself and refuses to start in `hybrid` without a named model. It also refuses to start without a named gateway (B1) in `hybrid`, or in `keyword` with B4 answered: there is no default address, so an unset or empty `LITELLM_BASE_URL` is a settings error. A gateway on the same host is written out, `http://localhost:4000/v1`.

- **yes:** validate the name with the request below. A pass means `hybrid`. A failed request is reported to the operator, with its cause, and the **operator** chooses between fixing it and `keyword`; never pick `keyword` for them because a request failed.
- **no or not sure:** set `RAG_RETRIEVAL=keyword` and tell the operator that adding an embedding model later means setting `RAG_RETRIEVAL=hybrid`, naming the model, and running a **Rebuild** (section 9). Until then the index is searched by keywords alone, which finds exact terms (error codes, hostnames) but not paraphrases.

Validate B1 to B3, for `hybrid`, by asking the gateway itself, from this host, with the request the adapter sends. `--noproxy '*'` matches the adapter, which never uses a proxy for the gateway:

```bash
curl -sS --noproxy '*' -m 30 -o /tmp/embed.json -w 'HTTP %{http_code}\n' "$LITELLM_BASE_URL/embeddings" \
  -H "Authorization: Bearer $LITELLM_API_KEY" -H 'Content-Type: application/json' \
  -d "{\"model\": \"$RAG_EMBED_MODEL\", \"input\": [\"smoke test\"]}"
python3 -c 'import json; print(len(json.load(open("/tmp/embed.json"))["data"][0]["embedding"]), "dimensions")'; rm -f /tmp/embed.json
```

`HTTP 200` and a dimension are the pass; tell the operator the dimension (the index is tied to it). `401` is the key, `404` is the URL (missing `/v1`) or the model name, a connection error is the address or a firewall. B4, if given, the same way against `"$LITELLM_BASE_URL/rerank"` with `{"model": "$RAG_RERANK_MODEL", "query": "smoke test", "documents": ["first text", "second text"], "top_n": 2}`; the answer must hold `results` with an `index` and a `relevance_score` for both texts. Naming a rerank model turns reranking on; a rerank failure at search time falls back to the unreranked order.

### C. Corpus

> **C1.** Is the Corpus Confluence, a folder of files (Markdown, PDF, DOCX and others), GitLab code, or several of these?
> **C2.** (Confluence) What is the Confluence host name, as in `confluence.example.test`? (`CONFLUENCE_DOMAIN`)
> **C3.** (Confluence) What is the REST API path? Press enter for `/rest/api`. (`CONFLUENCE_API_PATH`)
> **C4.** (Confluence) What is the service account's personal access token? It should be a **read-only** account that can see only Spaces suitable for every user of this server. (`CONFLUENCE_API_TOKEN`, with `CONFLUENCE_AUTH_TYPE=bearer` and `CONFLUENCE_READ_ONLY=true`)
> **C5.** (Confluence) Which pages are the Roots? Give page ids or page URLs, one per line; a Space's home page covers the whole Space. Each Root's whole subtree is indexed, attachments included.
> **C6.** (folder) Which absolute path holds the files? The first-level folders become the Spaces.
> **C7.** (GitLab) What is GitLab's base URL, without a project path, as in `https://gitlab.example.test`? (`GITLAB_URL`)
> **C8.** (GitLab) What is the token of a **read-only service account** with the `read_repository` scope and no other? It must be a member of only Repositories suitable for every user of this server. (`GITLAB_TOKEN`)
> **C9.** (GitLab) Which Repositories, by full path (`platform/payments`), and which branch or tag of each? The first ref is the **Primary ref** that searches see by default; others are found by naming their `ref`. Optional per ref: `exclude` globs for paths not worth indexing.
> **C10.** (GitLab) Where may the mirror cache live? Press enter for the default, `$HOME/.cache/local-rag/gitlab` of the service user (the `rag-data` volume on the container route).

Validate: the Confluence variable names are exactly the five above (there is no `CONFLUENCE_URL` or `CONFLUENCE_TOKEN`). C2 has no scheme and no path; `curl -sS -o /dev/null -w '%{http_code}\n' "https://$CONFLUENCE_DOMAIN/"` returns a status (a certificate error means the CA question in group D; try `http://` if the instance is plain HTTP). C3 starts with `/`. C4 is non-empty and passes the `set_env` character rule. C5 each line is all digits or starts with `http://` or `https://`; after section 5, prove every Root with `run_rag confluence info ROOT --json` (container route: `compose run --rm -T sync confluence info ROOT --json`), which must print JSON with `id`, `title` and `spaceKey`. C6 `test -d` passes and `sudo -u ragsvc test -r` on it passes (the service user must read it). C7 starts with `http://` or `https://` and has no path. C8 is non-empty and passes the `set_env` character rule; prove it after section 5 with the first Sync or `python -m local_rag.acceptance --only gitlab`, which reads every declared Tracked ref. C9 each path is `group/project` form; the paths come from the operator, there is no discovery (see the note below). C10 is a directory the service user can create (`sudo -u ragsvc mkdir -p DIR`); the cache holds bare git mirrors, is rebuilt by the next Sync when deleted, and needs disk for every listed Repository's history.

Corpora are declared by name in one file, `/etc/local-rag/corpora.toml` (section 5 writes it). Each Corpus owns its Documents: `local-rag sync NAME` adds, replaces and removes only that Corpus's Documents, so a folder and Confluence share one Knowledge index. The scheduled timers run `local-rag sync confluence`, so name the Confluence Corpus `confluence`; any other Corpus is synced by hand (`local-rag sync NAME`), or on its own timer through the templated units (section 5): name the GitLab Corpus `gitlab` and it syncs every 15 minutes, so feature branches stay fresher than Confluence. A Document's id in the index is its Corpus's name, a colon, then its own id (`confluence:12345`); `get_page` takes that id. The container stack mounts no folder, so a folder Corpus there needs a change to the compose file, which this repository does not provide: tell the operator and stop.

### D. Network and host

> **D1.** What host name or IP will the users' Claude Code use to reach this server? (It goes into the client URL.)
> **D2.** Which port should the MCP server listen on? Press enter for `8765`. (`MCP_PORT`)
> **D3.** Should the server accept connections from the whole network, or only from this host because a proxy sits in front? (`MCP_HOST`: `0.0.0.0` or `127.0.0.1`; systemd route.)
> **D4.** How will clients get TLS? (a) this server serves it, give the certificate and key paths (`MCP_TLS_CERT`, `MCP_TLS_KEY`; systemd route); (b) a proxy terminates it (see `examples/nginx.conf.example`); (c) plain HTTP.
> **D5.** Is any certificate (the MCP endpoint's, or Confluence's) signed by an internal CA? If so, where is the CA PEM file? (`NODE_EXTRA_CA_CERTS`)
> **D6.** Is `HTTP_PROXY` or `HTTPS_PROXY` set on this host's services or on the users' machines? If so, which host names must bypass it? (`NO_PROXY`)

Validate: D1 resolves (`getent hosts "$HOST"`) or is an IP address. D2 is an integer from 1 to 65535 and nothing listens on it yet (`ss -ltn "sport = :$PORT"` prints no listener); on a proxy setup it is the port the proxy's `proxy_pass` names. D3 is one of the two values; `127.0.0.1` only together with D4 (b). D4 (a): both files readable by the service user (`sudo -u ragsvc test -r FILE`), the certificate not expired and covering the D1 name (`openssl x509 -in CERT -noout -enddate -ext subjectAltName`), and the key belonging to it (`openssl x509 -in CERT -noout -pubkey | sha256sum` equals `openssl pkey -in KEY -pubout | sha256sum`). D4 (c): warn that every bearer token then crosses the network in clear text, and record the operator's explicit yes. The container route supports only (b) and (c): the compose file does not pass `MCP_TLS_*` and the MCP port is plain HTTP on every host interface, so for (b) also tell the operator to restrict the port to the proxy or the client subnet, with the `DOCKER-USER` chain or a network firewall (Docker's published ports bypass ufw). D5 `openssl x509 -in CA.pem -noout -subject -enddate` succeeds. D6 is a comma-separated list for `NO_PROXY`.

`NODE_EXTRA_CA_CERTS` and `NO_PROXY` serve three places: each user's Claude Code (section 8), and on the systemd route this host's services (the Confluence CLI is a Node program; the Qdrant client talks to `127.0.0.1`), set in the env file in section 5. The container route's compose file lists the variables it passes into the containers and neither is among them, so on that route they cannot be set without changing the compose file: tell the operator.

### E. Access control

> **E1.** Who needs access? Give one label per person or team; I will generate one bearer token for each. (`MCP_TOKENS`)
> **E2.** Confirm: every valid token can search and read every Document in the Knowledge index. Confluence page permissions and GitLab Repository permissions are not enforced, so the service account in C4 and the Roots in C5, and the GitLab service account in C8 and the Repositories in C9, must contain only content suitable for everyone with a token. Source code is often more sensitive than wiki pages: ask about it separately. Do you accept that?

Validate: E2 is an explicit yes; otherwise stop, and suggest narrowing the Roots, the Repositories or the service accounts. Per-token Space allow-lists do not exist yet. Generate each token on the host: `python3 -c 'import secrets; print(secrets.token_hex(24))'`. A token has no comma or whitespace, differs from the others and does not start with `CHANGE_ME` (the smoke check flags that). Keep the label-to-token map for section 8, and do not store it in the repository.

**Gate:** every applicable question has a validated answer, E2 is a yes, and you can read the whole set back to the operator, secrets masked, and they confirm it.

## 4. Install

Run the route's command. It is the first install and every later upgrade, it is idempotent, and it prints `created`, `kept`, `installed` or `unchanged` per item. Exit 1 with `refused:` or `prerequisites are not met:` or `failed:` means stop and read the message to the operator.

systemd route (adds `--index-url "$MIRROR_URL"` unless pip is configured for the Mirror):

```bash
sudo PYTHONPATH=/tmp/local-rag/src python3.12 -m local_rag.installer --drop "$DROP" --index-url "$MIRROR_URL"
```

It creates the `ragsvc` user, `/opt/qdrant/qdrant`, `/var/lib/qdrant`, `/etc/local-rag/local-rag.env` (with a generated Qdrant key and one MCP token, `kept` if it exists), a commented `/etc/local-rag/corpora.toml`, the venv `/opt/local-rag/venv` (packages from the Mirror), and the units, and it enables `qdrant.service`, `local-rag-mcp.service`, `local-rag-sync.timer` and `local-rag-full.timer`. A Mirror failure happens after the user, the Qdrant binary and the env file exist: fix the Mirror and run the same command again.

Container route:

```bash
sudo PYTHONPATH=/tmp/local-rag/src python3.12 -m local_rag.container start --drop "$DROP"
```

It loads the images, writes `/etc/local-rag/container.env` (generated secrets, `kept` if it exists) and the Corpus declaration file, installs `/opt/local-rag/container/compose.yaml` and the timers `local-rag-container-sync.timer` and `local-rag-container-full.timer`, and runs `docker compose up -d`. Data lives in Docker volumes that an upgrade does not touch.

**Gate:** the command exited 0, and the env file and the Corpus declaration file exist (`sudo ls -l /etc/local-rag/`).

## 5. Configure

Apply the section 3 answers with `set_env` (set `ENV_FILE` for your route). Only the settings the operator gave; leave the rest at the Installer's defaults.

```bash
RAG_VALUE="$LITELLM_BASE_URL"  set_env LITELLM_BASE_URL
RAG_VALUE="$LITELLM_API_KEY"   set_env LITELLM_API_KEY          # skip if the gateway needs none
RAG_VALUE="$RAG_EMBED_MODEL"   set_env RAG_EMBED_MODEL          # skip when B3 chose keyword
RAG_VALUE=keyword              set_env RAG_RETRIEVAL            # only when B3 chose keyword
RAG_VALUE="$RAG_RERANK_MODEL"  set_env RAG_RERANK_MODEL         # skip for no rerank
# Confluence Corpus
RAG_VALUE="$CONFLUENCE_DOMAIN"    set_env CONFLUENCE_DOMAIN
RAG_VALUE="/rest/api"             set_env CONFLUENCE_API_PATH   # C3
RAG_VALUE=bearer                  set_env CONFLUENCE_AUTH_TYPE
RAG_VALUE="$CONFLUENCE_API_TOKEN" set_env CONFLUENCE_API_TOKEN
RAG_VALUE=true                    set_env CONFLUENCE_READ_ONLY
# GitLab Corpus
RAG_VALUE="$GITLAB_URL"           set_env GITLAB_URL
RAG_VALUE="$GITLAB_TOKEN"         set_env GITLAB_TOKEN
# Network, systemd route (the container route: only MCP_PORT)
RAG_VALUE="$MCP_PORT"  set_env MCP_PORT
RAG_VALUE="$MCP_HOST"  set_env MCP_HOST
RAG_VALUE="$MCP_TLS_CERT" set_env MCP_TLS_CERT                  # D4 (a): both, or neither
RAG_VALUE="$MCP_TLS_KEY"  set_env MCP_TLS_KEY
RAG_VALUE="$CA_PEM"       set_env NODE_EXTRA_CA_CERTS           # systemd route, D5
RAG_VALUE="$NO_PROXY_LIST" set_env NO_PROXY                     # systemd route, D6
# Access: one token per person (the Installer's first token stays unless the operator removes it)
RAG_VALUE="$TOKEN" RAG_APPEND=1 set_env MCP_TOKENS
```

The Corpora, declared by name (the declaration is TOML; the table's key is the Corpus's name). The Confluence Corpus must be named `confluence`, because the timers sync it by that name. Write the file once with what the operator answered, quoting each Root:

```bash
CORPORA_FILE=/etc/local-rag/corpora.toml
sudo tee "$CORPORA_FILE" >/dev/null <<EOF
[corpus.confluence]
kind = "confluence"
roots = ["12345", "https://confluence.example.test/pages/viewpage.action?pageId=67890"]
# optional: max_depth = 0, attachments = true, exclude_attachments = ["*.png"]

[corpus.docs]
kind = "folder"
path = "/srv/docs"

[corpus.gitlab]
kind = "gitlab"
# optional: cache = "/var/cache/local-rag/gitlab"   (the mirrors; safe to delete)
[[corpus.gitlab.repositories]]
path = "platform/payments"
refs = ["main", {name = "release/1.2", exclude = ["docs/**"]}]   # the first is the Primary ref
EOF
```

Leave out the tables that do not apply. The GitLab Corpus must be named `gitlab` to match the timer below. A Repository's `refs` may mix plain names and `{name, primary, exclude}` tables; a ref that no longer exists (a merged and deleted branch) fails alone, and a Sync holds back removals only under that ref. If the operator already edited the file, show it to them instead of overwriting it.

GitLab Corpus on its own 15-minute timer (the hourly and weekly timers sync only `confluence`). The Installer and `container start` install the templated units but enable no instance, so opt in once the Corpus is declared:

```bash
sudo systemctl enable --now local-rag-sync@gitlab.timer              # systemd route
sudo systemctl enable --now local-rag-container-sync@gitlab.timer    # container route
```

The unit instance name is the Corpus name, so another Corpus gets its own timer the same way (`local-rag-sync@NAME.timer`).

Apply the settings: systemd route `sudo systemctl restart qdrant local-rag-mcp` (only `local-rag-mcp` if Qdrant's settings did not change); container route `compose up -d`. The settings reach a Sync from the env file each time it starts, but the MCP server reads them only at its start.

Then show the masked env file and the Corpus declaration file to the operator.

**Gate:** each `set_env` printed `set` or `unchanged`, the masked env file shows what the operator answered, and each Root passed `confluence info ROOT --json` (Confluence Corpus), and `local-rag status` runs without a usage error.

## 6. First Sync

Run the Sync the timers run, in the foreground, so its output and exit code are yours (it can take a long time; if your tool times out, start it detached with `sudo systemctl start --no-block local-rag-sync.service` and follow `journalctl -fu local-rag-sync`; container route `sudo systemctl start --no-block local-rag-container-sync.service` and `journalctl -fu local-rag-container-sync`):

```bash
run_rag /opt/local-rag/venv/bin/local-rag sync confluence   # systemd route
compose run --rm -T sync                                    # container route
run_rag /opt/local-rag/venv/bin/local-rag sync docs         # another Corpus, by the name declared for it
run_rag /opt/local-rag/venv/bin/local-rag sync gitlab       # a GitLab Corpus
```

Container route: `compose run --rm -T sync local-rag sync gitlab`. The first Sync of a GitLab Corpus fetches every listed Repository in full into the mirror cache and can take long; later Syncs fetch only what changed.

Read the result: `Sync: N added, ...` and exit 0 is complete. Exit 1 lists `failed <id>: <why>` per Document, and the rest is indexed: show the list to the operator (a scanned PDF without OCR is a typical cause; see the troubleshooting table). Exit 2 is a command line that cannot be used (a Corpus name that is not declared, or a declaration that cannot be read, names what is wrong). `sync` on an empty Knowledge index makes the index live with the first Document that succeeds. A **Rebuild** is not part of a first install; the operator asks for one when the embedding model or chunking changes.

**Gate:** exit 0, or exit 1 with a failure list the operator has seen and accepted.

## 7. Verify

Work through all of these; each has an expected result. Run the smoke check first, it covers most:

```bash
run_rag /opt/local-rag/venv/bin/python -m local_rag.smoke --systemd   # systemd route
compose run --rm -T sync python -m local_rag.smoke                    # container route
```

Pass is exit code 0 and the last line `SMOKE CHECK PASSED`; on a failure it is `SMOKE CHECK FAILED:` and the failed checks, and each `[FAIL]` line states the likely cause and the section of this guide to read. Its checks: `MCP_TOKENS`, `Model gateway (embeddings)` (model answers, with its dimension; `[SKIP]` in `keyword` mode), `Rerank` (ranks a relevant text first; `[SKIP]` when `RAG_RERANK_MODEL` is unset, or when the embedding check did not pass), `Knowledge index` (Qdrant reachable, document and passage counts), `LiteParse`, `Confluence CLI` (`[SKIP]` for a folder Corpus), `GitLab (git)` (`[SKIP]` unless `GITLAB_URL` is set), and with `--systemd` the four units (`qdrant.service`, `local-rag-mcp.service`, `local-rag-sync.timer`, `local-rag-full.timer`). A `[WARN]` does not fail the check, but it is not something to pass over: read it and tell the operator (a read-only index, for example).

Then run the acceptance suite, the end-to-end proof that the stack works (it builds its own throwaway collection and temp folder and removes them, so it is safe next to the Live index; the MCP and Confluence groups only read):

```bash
run_rag /opt/local-rag/venv/bin/python -m local_rag.acceptance --systemd   # systemd route
compose run --rm -T sync python -m local_rag.acceptance --skip install --mcp-url http://mcp:8765   # container route, inside the stack
sudo PYTHONPATH=/tmp/local-rag/src python3.12 -m local_rag.acceptance --container --only install   # container route, on the host
```

Pass is exit code 0 and the last line `ACCEPTANCE PASSED`. Every line is `[ OK ]`, `[FAIL]` or `[SKIP]` with a stable check name (`gateway.embeddings`, `qdrant.wrong-key`, `index.tail-meaning`, `mcp.no-token`, `install.timers`, `confluence.roots`, ...); a `[FAIL]` line states the likely cause and the section of this guide to read, so go there before changing anything. A `[SKIP]` states why (no `RAG_RERANK_MODEL`, no `CONFLUENCE_DOMAIN`, `RAG_RETRIEVAL=keyword` for the checks that need an embedding model, or an earlier check failed): report skips to the operator, they are not passes. In `hybrid` mode a dead embedding model is a `[FAIL]`, never a skip. Exit code 1 is a failed check, 2 a command line or setting it cannot use. The `gitlab` group (`gitlab.git`, `gitlab.repositories`) is skipped unless the declaration file holds a GitLab Corpus; with one, it proves `git` runs and every declared Tracked ref can be read with the token (it fetches the mirrors, and changes nothing else). `--only GROUP` (repeatable; groups `gateway`, `qdrant`, `index`, `mcp`, `install`, `confluence`, `gitlab`) reruns one group after a fix. `index.tail-meaning` failing while `index.tail-keyword` passes means the embedding model reads less than a whole Passage: lower `RAG_CHUNK_TOKENS`, then a Rebuild (when the operator asks for one). The `install` group is not optional: without `--systemd` or `--container` it fails (`install.route`). Inside the stack, which has no `docker` or `systemctl`, leave it out with `--skip install` and run it on the host, as the second and third commands show.

Then the rest:

| Check | Command | Expected |
|---|---|---|
| Status | `run_rag /opt/local-rag/venv/bin/local-rag status` | a generation, `documents` above 0, `access: read-write`, a `last sync` line, no `warning:` lines |
| A real query | `run_rag /opt/local-rag/venv/bin/local-rag search "<a phrase from an indexed page>" -k 3` | numbered Passages with `link:`, `space:` and `page id:`; ask the operator for a phrase they know is indexed, and for an exact identifier such as an error code to prove keyword matching |
| A code query | `run_rag /opt/local-rag/venv/bin/local-rag search "<a function name>" --repository platform/payments -k 3` | Passages with `file:`, `lines:`, `symbol:` and `ref:` lines; a feature branch appears only with `--ref NAME` |
| Health | `curl -sS -w '\nHTTP %{http_code}\n' http://127.0.0.1:$MCP_PORT/healthz` | `ok`, `HTTP 200` (`unavailable` with notes and 503 means Qdrant or the Knowledge index is not reachable) |
| Auth | `curl -s -o /dev/null -w '%{http_code}\n' -X POST http://127.0.0.1:$MCP_PORT/mcp` | `401` |
| A token | the same with `-H "Authorization: Bearer $TOKEN"` | any code other than `401` |
| Timers | systemd: `systemctl list-timers 'local-rag-*'`; container: `systemctl list-timers 'local-rag-container-*'` | the sync timer hourly, the full timer Sundays 03:00; both run `local-rag sync confluence`; with a GitLab Corpus also the `@gitlab` instance every 15 minutes |
| Containers | `compose ps` | `qdrant` and `mcp` up, `mcp` healthy |
| From a client | `curl -sS --noproxy '*' https://$HOST/healthz` (the URL clients will use) | `ok`; a certificate error is group D5 |

`/healthz` needs no token and asks Qdrant only, so it does not depend on the Model gateway; `status` and the smoke check do.

**Gate:** the smoke check has no `[FAIL]`, the acceptance suite ends `ACCEPTANCE PASSED` (or its failures are understood and accepted by the operator), `status` is `read-write`, `search` returns a Passage the operator recognises, `/healthz` answers `ok` from the clients' side, and an unauthenticated `/mcp` answers 401. Give the operator the result of each.

## 8. Client provisioning

Each user needs the URL (`https://HOST[:PORT]/mcp`, or `http://` for plain HTTP; the path is `/mcp`), their own token, and the CA if one signed the certificate. Hand each user their snippet by the operator's secure channel; do not put tokens in the repository.

```bash
export LOCAL_RAG_TOKEN=<their token>
claude mcp add --transport http local-rag https://rag.example.test/mcp \
  --header "Authorization: Bearer $LOCAL_RAG_TOKEN"
```

Or commit `examples/mcp.json.example` as `.mcp.json` to a repository (it reads `${LOCAL_RAG_TOKEN}`), so each user only sets `LOCAL_RAG_TOKEN`. Add `examples/CLAUDE.md.snippet` to the project's `CLAUDE.md`: it tells Claude to use `search_docs` first, for code questions as well as documentation, how to search another Tracked ref or one Repository, to cite links, to use the Confluence CLI read-only, and to treat indexed text, pages and code alike, as untrusted data. Replace `rag.example.test` with the D1 name.

If an internal CA signed the certificate: `export NODE_EXTRA_CA_CERTS=/path/ca.pem` before starting Claude Code. If the client has `HTTPS_PROXY` set, add the server's host name to `NO_PROXY`.

If Claude Code itself should use the network's LiteLLM chat model, `examples/claude_settings.example.json` holds the `env` block (`ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_MODEL` and the default model names); fill in the Model gateway's host and key. That is the chat model, independent of the embedding model.

The tools users get are `search_docs(query, top_k, space, repository, ref)`, `list_spaces()` (Confluence Spaces, folders and GitLab groups) and `get_page(page_id)`, which returns a Document's full text, a page or a file, as of the last Sync. Their descriptions tell Claude what code results are.

**Gate:** from one client, `claude mcp list` shows `local-rag` connected and a `search_docs` call returns Passages.

## 9. Operate

Prefix `local-rag` commands with `run_rag /opt/local-rag/venv/bin/` (systemd) or `compose run --rm -T sync ` (container).

| Task | How |
|---|---|
| State of the system | `local-rag status`; `journalctl -u local-rag-mcp -f`, `journalctl -u local-rag-sync` (container: `docker logs`, journal of `local-rag-container-sync`) |
| Add a user | generate a token (section 3 E), `RAG_VALUE=... RAG_APPEND=1 set_env MCP_TOKENS`, restart the MCP server (section 5), give the user their snippet |
| Revoke a user | edit `MCP_TOKENS` by hand to drop that token, restart the MCP server |
| Upgrade | a new Binary drop, then section 2's drop checks and section 4's command again; from a release without code search, the runbook below |
| Remove a Corpus that is no longer wanted | `local-rag forget NAME` (dropping it from `corpora.toml` removes nothing: `local-rag status` warns while its Documents remain, and a **Rebuild** also leaves them out) |
| Undo a bad Rebuild | `local-rag switch-back` (the **Previous index** becomes the Live index) |
| Change the embedding model | a **Rebuild**, only when the operator asks, see below |
| Switch between `hybrid` and `keyword` | set `RAG_RETRIEVAL` (and `RAG_EMBED_MODEL` for `hybrid`), then a **Rebuild**, as below |

Changing `RAG_RETRIEVAL`: the index records its Retrieval mode. A `keyword` index run under `hybrid` settings, or the reverse, is a **Read-only index**: searches work by keywords alone (a search under `hybrid` settings starts with a note saying so), Sync is refused, and nothing is rebuilt automatically. Follow the same sequence as for a new embedding model, below.

When the embedding model cannot be reached while a `hybrid` index is running, searches fall back to keywords for that query and start with `Note: Results are keyword-only ...`; `/healthz` stays `200` and says so in its body, and `local-rag status` reports the dimension check as unreachable. During a Sync or Rebuild, an embedding call that failed for a reason that may pass (a timeout, a busy gateway, a 5xx) is tried `RAG_EMBED_ATTEMPTS` times (default 3), waiting `RAG_EMBED_RETRY_BACKOFF` seconds (default 0.5) before the second and twice as long before each later one; a wrong key or model name is not retried. A Document that still fails counts as failed: `sync` exits 1 and lists it, and a Rebuild is promoted only within `RAG_REBUILD_FAILURE_THRESHOLD`. If the gateway will be down for long, switch to `keyword` and run a Rebuild.

Changing `RAG_EMBED_MODEL` (or its dimension): the index records the model and its dimension, and a mismatch makes it read-only (Sync refused) or, for a different dimension, refused (search too) until a Rebuild promotes. A running MCP server keeps the settings it started with, so edit the env file with the new model but **do not restart the MCP server yet**; run the Rebuild, which reads the new settings, and restart the server once it reports `promoted`. The hourly and weekly Syncs fail harmlessly in between. A Rebuild can run for hours: detach it,

```bash
sudo systemd-run --unit=local-rag-rebuild -p User=ragsvc -p EnvironmentFile=/etc/local-rag/local-rag.env \
  -p Environment=HOME=/var/lib/ragsvc -p Environment=PATH=/usr/local/bin:/usr/bin:/bin \
  /opt/local-rag/venv/bin/local-rag rebuild
journalctl -fu local-rag-rebuild
```

(container route: the same `systemd-run --unit=local-rag-rebuild`, running `/usr/bin/docker compose --project-name local-rag --env-file /etc/local-rag/container.env -f /opt/local-rag/container/compose.yaml run --rm -T sync local-rag rebuild`; a Rebuild re-embeds every declared Corpus), then read `local-rag status`. A Rebuild goes live only when its failure rate is within `RAG_REBUILD_FAILURE_THRESHOLD` (default `0`) and the Corpus was listed completely; otherwise the Live index is untouched and the refusal says why. Only one Sync or Rebuild runs at a time; the other exits 1 at once.

### Upgrade runbook: from a release without code search

Code search changes the Index format: named Corpora, code Passages, ref and repository fields. The new code serves the old index **read-only** (searches work, Sync is refused) until one **Rebuild** promotes a new one. Do this only when the operator asks for the upgrade, in this order, and tell them the Rebuild re-embeds every Document through the Model gateway (hours on a large index).

1. **Prerequisites.** Request the wheels `tree-sitter`, `tree-sitter-python` and `tree-sitter-c` from the Mirror (section 2 proves it with the pins of the new `requirements.lock`); install `git` on the host (systemd route; the container image has it). Verify the new Binary drop as in section 2.
2. **Declare the Corpora.** Nothing reads the old `/etc/local-rag/confluence_roots.txt` any more. Show it to the operator (`sudo cat /etc/local-rag/confluence_roots.txt`; Roots are not secrets), and write each line as a Root of `[corpus.confluence]` in `/etc/local-rag/corpora.toml` (section 5). The name must be `confluence`: the shipped units run `local-rag sync confluence`. A folder that used to be a `--folder` flag in a drop-in becomes `[corpus.NAME]` with `kind = "folder"`; remove the flag from the drop-in. The Installer keeps an existing `corpora.toml`, so edit it by hand; do not rely on it.
3. **Install.** Section 4's command with the new drop. It replaces the code and the units, restarts what changed, and installs the templated `@` units without enabling any instance. A Rebuild is not started by it.
4. **Add GitLab** (optional): C7 to C10 of section 3 and the GitLab part of section 5, but do not enable the `@gitlab` timer yet.
5. **Check the starting point.** `local-rag status`: `access: read-only` with a warning that the Index format differs is expected. Searches still answer.
6. **Rebuild** every declared Corpus, detached (section 9, "Changing `RAG_EMBED_MODEL`"). It prints `promoted`, or refuses and leaves the Live index untouched (read the reason; a failed GitLab ref or Confluence Root blocks it: fix it and run again). Only one Sync or Rebuild runs at a time, and the hourly Syncs fail harmlessly meanwhile.
7. **Check Status.** `local-rag status`: a new generation, `access: read-write`, one `corpus NAME:` line per declared Corpus with a last sync, no `warning:` lines (a warning names a Corpus whose Documents remain without being declared: `local-rag forget NAME` if the operator wants them gone). Then restart the MCP server (the running one keeps its judgement of the old index), and run `local-rag search`, the smoke check and the acceptance suite of section 7, including `gitlab` when a GitLab Corpus is declared.
8. **Turn on the GitLab timer:** `systemctl enable --now local-rag-sync@gitlab.timer` (container route: `local-rag-container-sync@gitlab.timer`).
9. **If the Rebuild is bad** (promoted but wrong: missing Documents, a poor search): `local-rag switch-back` makes the **Previous index** the Live index again. It is the old release's index, so the new code serves it read-only: searches work and Sync stays refused. To run the old release fully, install its Binary drop too; to go forward again, fix the cause and Rebuild. The Previous index lives for one generation only: do not Rebuild again before the operator is satisfied.

## 10. Troubleshooting

Diagnose from the symptom; check the cause with the command before changing anything.

| Symptom | Likely cause | Check and fix |
|---|---|---|
| `/healthz` 503, `Qdrant is unreachable` | Qdrant down, or the key differs | systemd: `systemctl status qdrant`, `journalctl -u qdrant`; `QDRANT_API_KEY` and `QDRANT__SERVICE__API_KEY` must be equal (the Installer writes them equal; a hand edit of one breaks it). Container: `compose ps`, `docker logs` of `qdrant`. If Qdrant is up but the client says unreachable, a proxy: see the proxy row |
| `/healthz` 503, `cannot be read` | corrupt or unusable index records | `local-rag status` for the reason; a Rebuild replaces corrupt records. Never delete Qdrant data to "fix" it |
| `gateway ... unreachable` | address, firewall or DNS to the Model gateway | the `curl --noproxy '*'` of group B from this host, or from inside the stack on the container route (`compose run --rm -T sync python -m local_rag.smoke`) |
| `gateway ... answered 401` | `LITELLM_API_KEY` wrong or missing | re-ask B2, `set_env`, no restart needed for a Sync; restart the MCP server |
| `gateway ... answered 404` (or 400) | `LITELLM_BASE_URL` lacks `/v1`, or `RAG_EMBED_MODEL` is not a LiteLLM `model_name` | group B request, URL then model name |
| `embedding dimension differs: index has X, the Model gateway returns Y`; access `refused` | the embedding model changed since the index was built | the Rebuild sequence in section 9; to go back, restore the old `RAG_EMBED_MODEL` |
| `RAG_RETRIEVAL is hybrid but RAG_EMBED_MODEL is not set`, a command exits 1 or 2 | hybrid is the default and needs an embedding model named | name it (group B, B3), or `set_env RAG_RETRIEVAL keyword` to run without one |
| `RAG_RETRIEVAL is hybrid but LITELLM_BASE_URL is not set`, or `RAG_RERANK_MODEL is set but LITELLM_BASE_URL is not set`; the smoke check prints it as `[FAIL] settings`, the acceptance suite exits 2 before any check | the Model gateway is not named, and there is no default address (an empty value counts as unset) | name it (group B, B1), or `set_env RAG_RETRIEVAL keyword` and leave `RAG_RERANK_MODEL` unset to run without a gateway |
| search results start with `Note: Results are keyword-only` | the embedding model could not be reached for that search, or the index was built without one | the first: the `curl --noproxy '*'` of group B; the second: `local-rag status` says `retrieval mode differs`, so run a Rebuild when the operator asks |
| `retrieval mode differs`, `embedding model differs`, `sparse encoder parameters differ`, access `read-only` | model or code changed; Sync is refused | a Rebuild, when the operator asks |
| Sync: `confluence not found` | the service user's PATH lacks the CLI | install it where `/usr/local/bin:/usr/bin:/bin` finds it, or set `CONFLUENCE_BIN` |
| Sync: `confluence ... failed (...)`, 401 or 403 | token, `CONFLUENCE_AUTH_TYPE`, or the account cannot see the page | `run_rag confluence spaces` and `confluence info ROOT --json`; re-ask C4 |
| Sync of a GitLab Corpus: `failed <repository>@<ref>`; `gitlab.repositories` `[FAIL]` | the token cannot see the Repository or lacks `read_repository`, a typo in the path or ref, a deleted branch, or `GITLAB_URL` is unreachable | check the path and ref in `corpora.toml`; `GITLAB_TOKEN` is not echoed anywhere, so re-ask C8. Other refs and Repositories are synced and keep their Documents; a missing ref removes nothing and blocks a Rebuild until it is fixed or dropped from the declaration |
| Sync of a GitLab Corpus: `git` not found, or `corpus 'gitlab': ... must be set in the environment` | git missing on the host (systemd route), or the settings are not in the env file | `command -v git` with the units' PATH (section 2); set both in the env file (section 5) |
| Sync: `failed root <id>` (an incomplete listing) | a Root or attachment listing failed | nothing was removed (a Sync removes nothing after an incomplete listing, and a Rebuild is refused); fix the Root, run again |
| `unable to verify the first certificate`, `self signed certificate` | an internal CA the client or the Confluence CLI does not know | client: `NODE_EXTRA_CA_CERTS`; host: `NODE_EXTRA_CA_CERTS` in the env file (systemd route). Prove the CA with `curl --cacert CA.pem https://HOST/healthz` |
| Calls to Qdrant or Confluence time out or return a proxy's error page, though the target is up | `HTTP_PROXY`/`HTTPS_PROXY` in the process environment | systemd services do not inherit a shell's proxy variables, only what the env file or systemd sets; put the local names in `NO_PROXY` (`127.0.0.1,localhost` and the Confluence host). The Model gateway adapter never uses a proxy. Client side: the MCP host in the user's `NO_PROXY` |
| 401 from `/mcp` | wrong header or a token not loaded | header is `Authorization: Bearer <token>`, exactly one of `MCP_TOKENS` at the last restart (adding one needs a restart); the smoke check's `MCP_TOKENS` line shows the count and any `CHANGE_ME` placeholder |
| The MCP server will not start, exit 2 | `MCP_TOKENS` empty or `MCP_PORT` not a number | `journalctl -u local-rag-mcp -n 20`; the message names it |
| Clients stall on streaming through a proxy | proxy buffering | `examples/nginx.conf.example` (`proxy_buffering off`) |
| `search_docs` says `No results.` | nothing indexed, or a Space filter that does not exist | `local-rag status` and `list_spaces`; run a Sync |
| A Rebuild is refused | failure rate above the threshold, no Documents, or an incomplete listing | the printed reason; the Live index is untouched; fix the cause, retry |
| Sync exit 1, another run | `Sync` or `Rebuild` already in progress | wait; do not delete the lock |
| Documents fail: empty text, scanned PDFs | no OCR | `RAG_OCR=1` once OCR language data is installed; `lit: command not found`: `LIT_BIN=/opt/local-rag/venv/bin/lit` |
| `refused:` from the Installer or `container start` | wrong Qdrant version or checksum, a missing or mismatched file, a missing prerequisite | the message names it; nothing was changed; fix the Binary drop |
| `set QDRANT_API_KEY in the env file` from compose | the container env file lacks a required variable | it is generated by `container start`; do not recreate it, restore it from backup |

## 11. Not verified without a real host

This guide was checked against the code and the files in the repository, and its structure is held by `tests/test_agent_guide.py`; it has not been run end to end on an air-gapped host. Treat these as unconfirmed, and say so to the operator when one of them matters:

- The real Qdrant release tarball, the Mirror, the Model gateway, Confluence and `confluence-cli` were not exercised. How `confluence-cli` handles the `CONFLUENCE_DOMAIN` scheme, proxies and `NODE_EXTRA_CA_CERTS` is its own behaviour.
- GitLab was exercised only against local git remotes: the Basic-auth header with `GITLAB_TOKEN` over HTTPS, an internal CA for GitLab (git reads its own CA settings, such as `http.sslCAInfo`, not `NODE_EXTRA_CA_CERTS`), and the cost of a first fetch of large Repositories are unconfirmed.
- Out of scope, for later: discovering the Repositories of a GitLab group with the `glab` CLI (a declared prerequisite), and per-token Space allow-lists. Until then the operator lists each Repository and every token sees all of them.
- That `NO_PROXY` makes the Qdrant client skip a proxy (standard HTTP-client behaviour, not tested here).
- The detached Rebuild with `systemd-run`, and a unit drop-in for a folder Corpus, are standard systemd, not run here. The Rebuild-before-restart order for a model change is derived from reading the code (the running server keeps its settings and its cached judgement of the Live index).
- `claude mcp add`, `claude mcp list` and where Claude Code reads its settings are Claude Code's own interface, not this repository's.
- The `openssl`, `ss`, `curl` and `docker compose` invocations are standard usage.
