# The Retrieval mode is chosen by the operator and an index is never mixed

The system must not depend on the network having an embedding model: it may not exist, and it may be down. An index can be built in one of two **Retrieval modes**, _Hybrid_ (dense and BM25 vectors) or _Keyword-only_ (BM25 only). The operator chooses with `RAG_RETRIEVAL=keyword|hybrid`, and the mode is recorded in the **Index format**.

- **Explicit choice.** `hybrid` is the default, so existing installs are unchanged. `hybrid` without `RAG_EMBED_MODEL` is a settings error, never a silent downgrade.
- **Never mixed.** In a Hybrid index every Passage has both vectors. A Document whose embedding keeps failing, after retries on transient errors, counts as failed, and the Rebuild threshold decides promotion. No Passage is stored without its dense vector.
- **Mismatch is a Read-only index.** A Keyword-only index under `hybrid` settings, or the reverse, is searched by BM25 only and refuses Sync until an operator runs a Rebuild. Nothing is rebuilt automatically.
- **Fallback is search-time only.** If the embedding model cannot be reached for a query in hybrid mode, that search is a **Keyword fallback**, and the result tells the caller. This never changes what an index is. Health notes it and keeps answering 200.

## Considered options

- **Auto-detect the mode at Sync:** rejected. An embedding outage during a Rebuild would silently promote a weaker index over a good one, against the rule that nothing about an index's format changes without an operator.
- **Store BM25-only Passages when embedding fails, backfill later:** rejected. It makes "partially embedded index" a state that Status, Rebuild and search must all reason about, and it hides a broken model behind a complete-looking index.
- **Fail search when the model is down:** rejected. Keyword results with a notice are more useful than an error, and Claude can tell the user the results are keyword-only.

## Consequences

- Moving between modes is always a Rebuild, the same path as changing the embedding model.
- A long embedding outage shows as a failed Sync or Rebuild, not as a quietly weaker index; the operator can switch to Keyword-only and Rebuild.
- `smoke` and `acceptance` skip the embedding checks in Keyword-only mode and fail them in Hybrid mode.
