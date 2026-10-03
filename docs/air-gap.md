# What crosses the air gap

The air-gapped network has a **Mirror** of public package sources, populated on request, and receives a hand-carried **Binary drop**. The **Installer** (systemd route) and `local_rag.container` (container route) install or upgrade from those two and refuse a drop that does not match what the repository expects, before they change anything. Neither ever touches an existing env file, Corpus declaration file, documents or Knowledge index. The reasons are in [ADR 0001](adr/0001-air-gapped-delivery.md).

| What | Where from |
|---|---|
| Python packages, pinned in `requirements.lock` (the list to request) | Mirror, through pip (`--index-url`, or pip's own configuration) |
| Node.js and `confluence-cli` (Confluence Corpus only) | Mirror (npm); the service user needs the `confluence` command on its `PATH` |
| Qdrant server binary, version and checksum recorded in `src/local_rag/binary_drop.py` (systemd route) | Binary drop |
| This system's own code, `git archive` of this repository (while it is private) | Binary drop |
| Saved system and Qdrant images (container route) | Binary drop |
| Python 3.12 with `venv`, `systemd`, `useradd`, `chown`, `git` (systemd route; `git` also in the container image, GitLab Corpus); Docker with the compose plugin (container route) | OS packages: checked, never carried |
| LibreOffice or ImageMagick, only if `lit` reports missing converters for Office or image files; OCR language data, only with `RAG_OCR=1` | OS packages (see `lit parse --help`) |
| Claude Code on each client, plus your internal CA certificate if TLS uses one | your usual channel |

There are no embedding model files and no torch: embeddings and rerank come from the Model gateway, and with `RAG_RETRIEVAL=keyword` nothing needs the gateway but an optional reranker.

The Qdrant release the system is tested against is recorded as `QDRANT` in `src/local_rag/binary_drop.py` (version and sha256 of the release tarball) and, for the container route, in the image tag of `deploy/container/compose.yaml`. To move to another release change them together.
