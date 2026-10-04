# Configuration

Two files, both created once by the Installer or `container start` and never overwritten:

| File | Holds |
|---|---|
| Env file: `/etc/nexus-rag/nexus-rag.env` (systemd route) or `/etc/nexus-rag/container.env` (container route) | settings and secrets |
| `/etc/nexus-rag/corpora.toml` (override with `RAG_CORPORA_FILE` or `--corpora FILE`) | the Corpora to index |

Settings are read at start: after an edit, restart the MCP server (`sudo systemctl restart nexus-rag-mcp`, or `compose up -d` on the container route).

## Declaring Corpora

Every Corpus is declared by name in the TOML file. The table's key is the Corpus's name (letters, digits, `-`, `_`). Name the Confluence Corpus `confluence`: the hourly timer syncs it by that name. Any other Corpus syncs on the templated `@` timer of its name (see your route's install guide).

### Confluence

```toml
[corpus.confluence]
kind = "confluence"
roots = ["12345", "https://wiki.example.test/pages/viewpage.action?pageId=678"]
max_depth = 0                       # optional, 0 = the whole subtree
attachments = true                  # optional
exclude_attachments = ["*.png"]     # optional file name globs; the default leaves out images, video, archives
```

The `CONFLUENCE_*` settings in the env file hold the token; keep `CONFLUENCE_READ_ONLY=true`. A Space's home page as a Root covers the whole Space.

### Folder

```toml
[corpus.docs]
kind = "folder"
path = "/srv/docs"                  # relative paths are relative to the file; the Space is the top-level folder
```

### GitLab

```toml
[corpus.gitlab]                     # GITLAB_URL and GITLAB_TOKEN (env file): a read-only service account, scope read_repository
kind = "gitlab"
cache = "/var/cache/nexus-rag/gitlab"   # optional: bare git mirrors, safe to delete (default: $HOME/.cache/nexus-rag/gitlab)

[[corpus.gitlab.repositories]]
path = "platform/payments"          # the project's full path; the Space is its group, platform
refs = ["main", {name = "release/1.2", exclude = ["docs/**"]}]   # Tracked refs; the first is the Primary ref
```

A GitLab Corpus indexes the files of each listed Repository at each Tracked ref, split along functions and classes for Python and C, by lines for other code.

> **GitLab permissions are not enforced: every MCP token can read every listed Repository**, whoever the GitLab token belongs to. List only Repositories suitable for everyone who holds a token. Per-token Space allow-lists and discovering a group's Repositories with the `glab` CLI are later work.

## Settings

The env file; unset means the default. The authoritative list is `src/nexus_rag/environment.py`.

### Model gateway

| Setting | Default | Meaning |
|---|---|---|
| `LITELLM_BASE_URL` | none | the Model gateway, ending with `/v1`; required for `hybrid` or a reranker |
| `LITELLM_API_KEY` | none | its key |
| `RAG_RETRIEVAL` | `hybrid` | or `keyword` to run without an embedding model ([how](how-it-works.md#keyword-only-mode)); a change needs a Rebuild |
| `RAG_EMBED_MODEL` | none | the LiteLLM `model_name`; required for `hybrid`; a change needs a Rebuild |
| `RAG_RERANK_MODEL` | off | a reranker served at `/v1/rerank` |
| `RAG_QUERY_INSTRUCTION` | empty | put in front of query embeddings only, as `Instruct: <it>\nQuery: <query>`, e.g. `Given a question, retrieve the passages that answer it` for Qwen3-Embedding; needs no Rebuild |
| `RAG_EMBED_BATCH` | 16 | |
| `RAG_EMBED_ATTEMPTS` | 3 | tries at one Document's embeddings in a Sync or Rebuild; a search never retries |
| `RAG_EMBED_RETRY_BACKOFF` | 0.5 s | before the second try, doubling |
| `RAG_GATEWAY_TIMEOUT` | 60 s | |

### Search and Passages

| Setting | Default | Meaning |
|---|---|---|
| `RAG_TOP_K` | 6 | results per search |
| `RAG_HYBRID_CANDIDATES` | 40 | candidates per retriever |
| `RAG_MAX_PER_DOC` | 2 | Passages of one Document per search |
| `RAG_CHUNK_TOKENS` | 400 | target Passage size in tokens; a change needs a Rebuild |
| `RAG_CHUNK_OVERLAP` | 60 | tokens shared by neighbouring Passages, smaller than the size; a change needs a Rebuild |
| `RAG_CODE_CHUNK_TOKENS` | 800 | target size of a code Passage, split by lines; a change needs a Rebuild |
| `RAG_BM25_K1`, `RAG_BM25_B`, `RAG_BM25_AVG_DL` | 1.2, 0.75, 250 | sparse encoder parameters; a change needs a Rebuild |

### Index, tools and access

| Setting | Default | Meaning |
|---|---|---|
| `RAG_REBUILD_FAILURE_THRESHOLD` | 0 | share of failed Documents, 0 to 1, a Rebuild may have and still go live |
| `QDRANT_URL`, `QDRANT_API_KEY` | `http://127.0.0.1:6333`, generated | the Qdrant server |
| `RAG_COLLECTION` | `knowledge` | the Qdrant collection name |
| `RAG_OCR` | off | `1` to OCR scanned PDFs (needs the language data offline) |
| `LIT_BIN`, `CONFLUENCE_BIN` | the venv's `lit`, `confluence` on `PATH` | the `lit` and `confluence` commands |
| `CONFLUENCE_DOMAIN` and the other `CONFLUENCE_*` | `confluence` (the domain, a host name without `https://`) | the Confluence CLI's profile |
| `GITLAB_URL`, `GITLAB_TOKEN` | URL `https://gitlab.org`, no token | the GitLab Corpus: base URL and a read-only token, scope `read_repository` |
| `MCP_TOKENS` | one generated | comma-separated, one per user or team |
| `MCP_HOST`, `MCP_PORT` | `127.0.0.1`, 8765 | where the MCP server listens (the Installer's env file sets `0.0.0.0`) |
| `MCP_TLS_CERT`, `MCP_TLS_KEY` | | systemd route only: TLS served by the MCP server |
