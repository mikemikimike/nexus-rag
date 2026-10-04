---
name: local-rag-admin
description: Install, upgrade and administer a local-rag server on its host. Use when the operator wants to install local-rag or apply a new Binary drop, add or remove a GitLab Repository or branch, add a Confluence Root or another Corpus, give or revoke someone's MCP access, rotate the GitLab, Confluence or gateway token, change a setting, run a Sync, or find out why search or a Sync is failing.
---

# local-rag administration

You run on the local-rag host, as a user with `sudo`.

**A first install or an upgrade** (a new Binary drop) is a different path: follow `references/closed-network-setup.md` from section 1, in order, and stop reading here. Everything below is for a running installation, and that guide is its full reference too (section 9 operate, section 10 troubleshooting).

Speak the system's words: **Corpus**, **Repository**, **Tracked ref**, **Primary ref**, **Root**, **Sync**, **Rebuild**, **Knowledge index**.

## The helper

Every change goes through one script. Below, `rag-admin ...` means:

```bash
sudo python3.12 <this skill's directory>/scripts/rag_admin.py ...
```

Write the full command each time: shell functions and variables do not survive between your commands. Start every session with `rag-admin route`: it finds the installation (systemd or container) and every later command adapts to it. If it refuses, read the message to the operator and stop.

`rag-admin --help` and `rag-admin COMMAND --help` list everything. Exit 0 is done, 1 is a failure the output explains, 2 is `refused` and changed nothing.

## Ground rules

- **Secrets stay out of your output.** Use `rag-admin env show` (masked), never `cat` the env file or a token file. A secret reaches `env set` on stdin. The best way is for the operator to type it themselves at Claude Code's `!` prompt: `! sudo python3.12 <path>/scripts/rag_admin.py env set GITLAB_TOKEN`. That prompts without echoing, and the value never enters this conversation. In anything you write to the operator, name a secret and never quote it ("`GITLAB_TOKEN` is updated"), even a value they gave you: chat logs and reports outlive the token.
- **Every MCP token sees every indexed Document.** GitLab and Confluence permissions are not enforced. Before adding a Repository or Root, ask the operator to confirm everyone who holds a token may read it, and get an explicit yes. Source code is often more sensitive than wiki pages.
- **A Rebuild only when the operator asks for one.** It re-embeds everything and can take hours. Never delete Qdrant data, a lock or the mirror cache to "fix" something.
- **Confirm before changing.** Say what you will change and get a yes. Then do it, then show the result.

## The change loop

Every task is **look → change → apply → verify**:

1. **Look**: `rag-admin run local-rag status` and the relevant `list`/`show`, so you and the operator start from the real state.
2. **Change**: the declaration through a staged copy (`corpora ...`), settings through `env`, tokens through `token`.
3. **Apply**: `corpora apply` validates and installs the declaration. Settings and tokens reach the MCP server only with `rag-admin restart`. A Sync reads the files fresh each time.
4. **Verify**: the task's check below. The task is done when that check passes, not when the edit succeeded.

## Corpora: corpora.toml

`/etc/local-rag/corpora.toml` declares every Corpus. You never edit it directly:

1. `rag-admin corpora stage` copies it to `~/local-rag-staging/corpora.toml`, which you can edit with your editing tools.
2. `rag-admin corpora apply` shows the diff, then builds every Corpus the way a Sync would (the GitLab settings, folder paths, the syntax). It installs the file only if that passes, keeps a backup in `/etc/local-rag/backups/`, and prints what changed and what to do next. Do what it prints.

`corpora diff` shows the staged changes, `corpora check` validates without installing, and `corpora discard` drops them.

Every key a Corpus table takes, and every setting of the env file, is in `references/configuration.md`.

### Add a Repository

Ask for the full project path (`platform/payments`) and the branches or tags to track. The first ref is the **Primary ref**, the one searches see by default. The others are found only by naming their ref.

```bash
rag-admin corpora add-repo gitlab platform/payments main release/2.0
rag-admin corpora apply
rag-admin sync gitlab
rag-admin run local-rag search "<a function name from that Repository>" --repository platform/payments -k 3
```

`add-repo` writes into the staged copy. With no GitLab Corpus yet, add `--new-corpus`, check that `GITLAB_URL` and `GITLAB_TOKEN` are set (`env show`), and after the first Sync turn on its 15-minute timer with `rag-admin timer enable gitlab`. The first Sync of a large Repository fetches its whole history. If it might take long, use `sync gitlab --detach` and follow the journal command it prints.

**Done when** the Sync output has no `failed platform/payments@...` line and the search returns Passages with `file:` and `ref:` lines from the Repository.

### Change a Repository's refs, Primary ref or excludes, or remove a Repository

`corpora stage`, then edit that Repository's table in the staged file:

```toml
[[corpus.gitlab.repositories]]
path = "platform/payments"
refs = ["main", {name = "release/2.0", exclude = ["docs/**", "*.min.js"]}]
# Primary ref: the first in the list, or mark one: {name = "develop", primary = true}
```

To remove a Repository, delete its whole `[[corpus.gitlab.repositories]]` table. Then run `corpora apply`, then `sync gitlab`. A dropped ref or Repository loses its Documents at that Sync.

**Done when** `apply` lists the change you meant and nothing else, and the Sync exits 0.

### Confluence Roots, folders, and whole Corpora

- **A Confluence Root**: stage, add the page id or URL to `roots` of `[corpus.confluence]`, apply. Prove it with `rag-admin run confluence info ROOT --json`, which prints its `id`, `title` and `spaceKey`. Then `sync confluence`. A Space's home page covers the whole Space.
- **A folder Corpus** (systemd route only; the container stack mounts no folder): stage, add `[corpus.NAME]` with `kind = "folder"` and `path = "/abs/path"`, apply, `sync NAME`, `timer enable NAME`. The `ragsvc` user must be able to read the folder.
- **Remove a Corpus**: dropping its table removes nothing from the index. After `apply`, run `rag-admin run local-rag forget NAME` (only after the operator confirms) and `rag-admin timer disable NAME`. The `confluence` Corpus is synced by the built-in hourly and weekly timers, by that name.

**Done when** `rag-admin run local-rag status` shows the Corpus as expected and has no `warning:` line about undeclared Corpora.

## Access: MCP tokens

Tokens are named in a ledger (`/etc/local-rag/mcp-tokens.labels`). It holds a label and a fingerprint per token, never the token itself.

| Task | Commands |
|---|---|
| Who has access | `rag-admin token list` |
| Give access | `rag-admin token add LABEL` (one label per person or team), then `rag-admin restart` |
| Revoke | `rag-admin token revoke LABEL` (or a fingerprint), then `rag-admin restart` |
| Name an old, unlabelled token | `rag-admin token name FINGERPRINT LABEL` |
| Revoke a token the operator holds but cannot name | the operator pipes it: `! printf %s 'TOKEN' \| sudo python3.12 <path>/scripts/rag_admin.py token revoke -` |

`token add` writes the new token to `~/local-rag-tokens/LABEL.token` (mode 0600). Give the operator the path, not the token. The user then needs the URL (`https://HOST[:PORT]/mcp`), their token, and the client setup of section 8 of the setup guide (`claude mcp add --transport http local-rag ... --header "Authorization: Bearer $LOCAL_RAG_TOKEN"`). Tell the operator to delete the token file once it has been handed over.

A token is valid or revoked only from the next `restart`. **Done when** `token list` shows the change and, after the restart, the health check answers (see Health below).

## Settings and secrets

`rag-admin env show` shows the env file with the secrets hidden. `rag-admin env set KEY VALUE` sets a plain setting. A secret (any `*_KEY`, `*_TOKEN`) goes in on stdin, best typed by the operator at `!` as described in the ground rules. `rag-admin env unset KEY` comments a setting out. `MCP_TOKENS` and the Qdrant keys are refused: use `token`, and the Qdrant keys are generated at install.

- **Rotate `GITLAB_TOKEN` or `CONFLUENCE_API_TOKEN`**: `env set`, then prove it. GitLab: `rag-admin acceptance --only gitlab`. Confluence: `rag-admin run confluence spaces`. A Sync picks up the new token by itself, and `restart` is for the MCP server's settings.
- **Rotate `LITELLM_API_KEY`**: `env set`, `restart`, then `rag-admin smoke`.
- **`RAG_EMBED_MODEL`, `RAG_RETRIEVAL`, the chunk and BM25 settings** are recorded in the index. `env set` warns about them. Changing one needs a Rebuild, in the order given in section 9 of the setup guide ("Changing `RAG_EMBED_MODEL`"). Do it only when the operator asked for that change, and restart only after the Rebuild reports `promoted`.

## Syncs, health and trouble

| Need | Command |
|---|---|
| State of the index | `rag-admin run local-rag status` |
| Sync one Corpus now | `rag-admin sync NAME` (`--detach` for a long one: it starts the unit and prints the journal command) |
| Try a search | `rag-admin run local-rag search "TEXT" [--space S] [--repository P] [--ref R] -k 3` |
| Component health | `rag-admin smoke` (ends `SMOKE CHECK PASSED`) |
| End-to-end proof | `rag-admin acceptance [--only GROUP]` (ends `ACCEPTANCE PASSED`) |
| Schedules | `rag-admin timer list` |
| Undo a bad Rebuild | `rag-admin run local-rag switch-back` (only when the operator asks) |

Read a Sync's result: exit 0 is complete. Exit 1 lists `failed <id>: <why>`, and the rest was indexed, so show the list to the operator. Exit 1 with no list can mean another Sync is running: wait, and leave the lock alone. Exit 2 means the Corpus name or the declaration is unusable, and the message says which.

For any failure, match the symptom in the troubleshooting table, section 10 of `references/closed-network-setup.md`, and run its check before you change anything. A `[FAIL]` line from `smoke` or `acceptance` names the setup guide section to read. A `[SKIP]` is not a pass: report it.

## Reporting back

End each task by telling the operator:

- what changed: the `apply`/`set`/`token` output lines;
- what you verified, and its result;
- anything left for them: a token file to hand over and delete, a Sync still running, a warning you saw.
