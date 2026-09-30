# local-rag

A retrieval system that lets a team's Claude Code search internal documentation on a fully air-gapped network, using models served by a gateway on that network.

## Language

**Knowledge index**:
The searchable body of indexed Documents, and the single thing that accepts a Corpus to be synced in and answers search queries.
_Avoid_: RAG, vector store, database, Qdrant collection

**Corpus**:
A source of Documents, described independently of where they come from (a Confluence export, a folder of files).
_Avoid_: export, data dir, dataset

**Document**:
One unit of content from a Corpus with a stable identity, title, origin link and version, and the unit that is added, replaced or removed as a whole in the Knowledge index.
_Avoid_: file, page (a page is one kind of Document from Confluence), source

**Attachment**:
A Document that belongs to a page Document, indexed and replaced separately but citing its parent page.
_Avoid_: file, child document

**Media type**:
The kind of content a Corpus supplies for a Document (PDF, HTML, plain text), which decides how it is turned into text before it is embedded.
_Avoid_: mime, file type, extension

**Space**:
A named grouping of Documents (a Confluence space, or a top-level folder) that a search can be narrowed to.
_Avoid_: project, namespace, category

**Passage**:
A chunk of a Document that is embedded and returned by search.
_Avoid_: chunk, node, snippet

**Sync**:
Bringing the Knowledge index into agreement with a Corpus, adding, replacing and removing Documents as they changed.
_Avoid_: ingest, refresh, update

**Rebuild**:
A Sync that re-embeds every Document into a fresh index, needed when the embedding model or chunking changes, and promoted to live only if it is complete enough.
_Avoid_: reindex, full refresh

**Live index**:
The generation of the Knowledge index that search currently reads from.
_Avoid_: active collection, alias target

**Previous index**:
The one generation kept from before the last promotion, which an operator can switch back to if a Rebuild turns out to be bad; switching back makes it the Live index and the former Live index the Previous index.
_Avoid_: backup, snapshot, old collection

**Index format**:
What an index records about how it was built (schema version, sparse encoder parameters, embedding model name and dimension), compared with the running code after an upgrade. An index whose format differs is served as a Read-only index if it can still be searched and refused if it cannot; it is never rebuilt automatically. An index whose records are corrupt or unreadable is reported, never treated as empty.
_Avoid_: version, schema

**Read-only index**:
A Live index whose Index format differs from the running code but can still be searched. It is served with a warning and refuses Sync until an operator runs a Rebuild.
_Avoid_: degraded, stale, legacy index

**Status**:
What an operator can ask of the Knowledge index: the Live index generation, Document and Passage counts, Index format, whether it is writable, and the result of the last Sync or Rebuild.
_Avoid_: info, stats

**Health**:
Whether Qdrant and the Knowledge index can be reached, for monitoring.
_Avoid_: liveness, ping

**Model gateway**:
The network-local service (LiteLLM) through which embeddings, reranking and the chat model are reached.
_Avoid_: LLM server, backend

**Mirror**:
The air-gapped network's internal copy of public package sources (Python packages, npm, GitHub repositories), populated on request.
_Avoid_: proxy, registry, PyPI server

**Binary drop**:
The verified set of files that are not available from the Mirror and are carried into the air-gapped network by hand: the Qdrant server binary and, while the source repository is private, an archive of this system's own code, each with a version and checksum.
_Avoid_: bundle, vendor directory, release bundle

**Installer**:
The single idempotent command that installs or upgrades the system on the air-gapped network from the Mirror and the Binary drop, and never touches settings, secrets, documents or the Knowledge index that already exist.
_Avoid_: setup script, deploy script
