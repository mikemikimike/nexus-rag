# local-rag

A retrieval system that lets a team's Claude Code search internal documentation on a fully air-gapped network, using models served by a gateway on that network.

## Language

**Knowledge index**:
The searchable body of indexed Documents, and the single thing that accepts a Corpus to be synced in and answers search queries.
_Avoid_: RAG, vector store, database, Qdrant collection

**Corpus**:
A named source of Documents, described independently of where they come from (Confluence Roots, a folder of files, Tracked refs of GitLab Repositories), declared by the operator. Every Document in the Knowledge index belongs to exactly one Corpus.
_Avoid_: export, data dir, dataset

**Document**:
One unit of content from a Corpus with a stable identity, title, origin link and version, and the unit that is added, replaced or removed as a whole in the Knowledge index.
_Avoid_: file, page (a page is one kind of Document from Confluence), source

**Attachment**:
A Document that belongs to a page Document, indexed and replaced separately but citing its parent page (its title and link, and a reference to the page's id).
_Avoid_: file, child document

**Root**:
A Confluence page, given by id or URL, whose whole subtree of pages a Confluence Corpus contributes; a Space's home page as a Root covers the Space.
_Avoid_: seed, entry point, start page

**Repository**:
A GitLab project's git repository, with all its branches and tags.
_Avoid_: repo, project

**Tracked ref**:
A Repository at one branch or tag that the operator listed, whose files a GitLab Corpus contributes as Documents; a Repository may have several.
_Avoid_: checkout, snapshot, Root

**Primary ref**:
The one Tracked ref of a Repository that a search sees unless the caller asks for a ref; the others are found only by naming their ref.
_Avoid_: default branch (GitLab's setting, which may differ), main

**Incomplete listing**:
A Corpus's list of Documents that is missing a part the Corpus could not list (a Root, a page's Attachments, a Tracked ref that no longer exists); since a missing Document may still exist, a Sync removes nothing from that part, though it still removes from the parts it listed, and a Rebuild is refused while a listing is incomplete.
_Avoid_: partial export, listing error

**Media type**:
The kind of content a Corpus supplies for a Document (PDF, HTML, plain text), which decides how it is turned into text before it is embedded.
_Avoid_: mime, file type, extension

**Space**:
A named grouping of Documents (a Confluence space, a top-level folder, or the full path of the GitLab group directly holding a Repository) that a search can be narrowed to.
_Avoid_: project, namespace, category

**Passage**:
A chunk of a Document that is indexed, and embedded too in a Hybrid index, and returned by search.
_Avoid_: chunk, node, snippet

**Sync**:
Bringing the Knowledge index into agreement with one Corpus, adding, replacing and removing that Corpus's Documents as they changed, and leaving every other Corpus's Documents alone.
_Avoid_: ingest, refresh, update

**Rebuild**:
A Sync of every declared Corpus at once that adds every Document again into a fresh index (embedding it, in a Hybrid index), needed when the Retrieval mode, embedding model or chunking changes, and promoted to live only if it is complete enough. A Document whose embedding keeps failing counts as failed, and the Rebuild is promoted only if failures stay within the threshold.
_Avoid_: reindex, full refresh

**Live index**:
The generation of the Knowledge index that search currently reads from.
_Avoid_: active collection, alias target

**Previous index**:
The one generation kept from before the last promotion, which an operator can switch back to if a Rebuild turns out to be bad; switching back makes it the Live index and the former Live index the Previous index.
_Avoid_: backup, snapshot, old collection

**Index format**:
What an index records about how it was built (schema version, Retrieval mode, sparse encoder parameters, chunk size and overlap for prose and for code, the languages whose code is split by its syntax, and for a Hybrid index the embedding model name and dimension), compared with the running code after an upgrade. An index whose format differs is served as a Read-only index if it can still be searched and refused if it cannot; it is never rebuilt automatically. An index whose records are corrupt or unreadable is reported, never treated as empty.
_Avoid_: version, schema

**Retrieval mode**:
How an index finds Passages, chosen by the operator and recorded in the Index format: _Hybrid_ (meaning and keywords, which needs an embedding model) or _Keyword-only_ (keywords alone, which needs none). Choosing Hybrid without naming an embedding model, or without naming the Model gateway, is a settings error, never a silent downgrade or a guessed address.
_Avoid_: degraded mode, BM25 mode, no-embedding mode

**Keyword fallback**:
A search served by keywords alone because the embedding model could not be reached for that query, which the caller is told.
_Avoid_: degraded mode, silent fallback

**Read-only index**:
A Live index whose Index format differs from the running code but can still be searched. It is served with a warning and refuses Sync until an operator runs a Rebuild.
_Avoid_: degraded, stale, legacy index

**Status**:
What an operator can ask of the Knowledge index: the Live index generation, Document and Passage counts, Index format, whether it is writable, and the result of the last Sync or Rebuild.
_Avoid_: info, stats

**Health**:
Whether Qdrant and the Knowledge index can be reached, for monitoring. Whether the embedding model worked the last time it was used is noted, but an unreachable embedding model does not make Health fail, since search still answers by Keyword fallback.
_Avoid_: liveness, ping

**Model gateway**:
The network-local service (LiteLLM) through which embeddings, reranking and the chat model are reached.
_Avoid_: LLM server, backend

**Mirror**:
The air-gapped network's internal copy of public package sources (Python packages, npm, GitHub repositories), populated on request.
_Avoid_: proxy, registry, PyPI server

**Binary drop**:
The verified set of files that are not available from the Mirror and are carried into the air-gapped network by hand: the Qdrant server binary and, while the source repository is private, an archive of this system's own code, each with a version and checksum. On the container route it carries the saved system and Qdrant images instead of the Qdrant binary.
_Avoid_: bundle, vendor directory, release bundle

**Installer**:
The single idempotent command that installs or upgrades the system on the air-gapped network from the Mirror and the Binary drop, and never touches settings, secrets, documents or the Knowledge index that already exist.
_Avoid_: setup script, deploy script
