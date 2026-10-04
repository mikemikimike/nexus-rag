# How it works

The terms in bold are defined in [`CONTEXT.md`](../CONTEXT.md).

## Hybrid retrieval

A **Document** is split into **Passages** of about `RAG_CHUNK_TOKENS` tokens, each prefixed with the Document's title when it is embedded, so a long page is searchable by any part of it; results still point at the Document. Every Passage gets a dense vector (your embedding model) and a BM25-style sparse vector.

A query runs both. Qdrant fuses the rankings with Reciprocal Rank Fusion, and optionally the **Model gateway**'s reranker reorders the best candidates. Exact identifiers such as `ORA-00942`, `OPS-4711` or hostnames match reliably. No more than a few Passages of one Document are returned per search (`RAG_MAX_PER_DOC`).

## Keyword-only mode

With `RAG_RETRIEVAL=keyword` the system needs no embedding model at all: Passages get only the BM25-style vector, search runs only the keyword side, and nothing reaches the Model gateway unless a reranker is named. It finds exact terms but not paraphrases, so the search tool says so to Claude.

- The mode is a choice you make, recorded in the Index format. `hybrid` is the default and refuses to start without `RAG_EMBED_MODEL`.
- Moving between the two is a **Rebuild**, like changing the embedding model.
- If the embedding model cannot be reached for a search in `hybrid` mode, that search falls back to keywords and its results begin with a note saying so; `/healthz` keeps answering 200 and `status` reports it.
- During a Sync or Rebuild a failed embedding call is retried (`RAG_EMBED_ATTEMPTS`). A Document that still fails counts as failed: `sync` exits 1, and a Rebuild is promoted only within the failure threshold.

The reasons are in [ADR 0003](adr/0003-retrieval-mode-is-explicit-and-never-mixed.md).

## Safe Rebuilds

A **Rebuild** re-embeds every Document into a fresh index and makes it the **Live index** only if its failure rate is within `RAG_REBUILD_FAILURE_THRESHOLD` (default `0`); otherwise the current Live index stays and the refusal says why. The **Previous index** is kept for one generation and `local-rag switch-back` returns to it. Searches keep working during a Sync or Rebuild. Only one Sync or Rebuild runs at a time; the Knowledge index enforces it.

## Corpora share one index

`local-rag sync NAME` syncs one **Corpus** and touches only its Documents, so Corpora share one Knowledge index. `local-rag rebuild` re-embeds every declared Corpus. A Document's id is its Corpus's name and its own id (`confluence:12345`), and Status shows the last Sync of each Corpus.

Removing a Corpus from the declaration does not remove its Documents: they stay searchable and Status warns about them until `local-rag forget NAME` removes them (a Rebuild leaves them out too). A Corpus named that is not declared is a usage error (exit 2). See [ADR 0004](adr/0004-each-corpus-owns-its-documents-and-syncs-on-its-own.md).

## Index format

The index records how it was built: schema version, Retrieval mode, sparse encoder parameters, chunk size and overlap for prose and for code, the languages whose code is split by its syntax, the version of the token counter that sized the Passages, and for a hybrid index the embedding model name and dimension. After an upgrade, a model change, a change of `RAG_RETRIEVAL` or a change of `RAG_CHUNK_TOKENS` or `RAG_CHUNK_OVERLAP`, `local-rag status` tells whether it still matches.

- A mismatching index that can still be searched is served as a **Read-only index** with a warning, and refuses Sync until you run a Rebuild.
- An unusable one is refused.
- Nothing is ever rebuilt automatically.

An index built before Documents were split into Passages, or before named Corpora, shows up as a Read-only index until you run a Rebuild.
