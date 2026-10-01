# The Knowledge index owns change detection, locking and Document text

Before the restructuring, the exporter, ingest scripts, shell glue and MCP server each held part of the picture: sidecar files recorded what had been indexed, `flock` in the systemd units prevented overlap, and the MCP server read exported files from a data directory. These had to agree through an undocumented on-disk layout. The **Knowledge index** now owns all three.

- **Change detection.** A **Corpus** supplies an opaque version per **Document**. The index stores it and replaces a Document only when it differs. Corpora hold no state about what has been indexed.
- **Locking.** The index allows one **Sync** or **Rebuild** at a time and enforces it itself, so overlapping timers or an operator's manual run cannot corrupt it. The units contain no `flock`.
- **Document text and Spaces.** The index stores each Document's full text and its **Space**. `get_page` and `list_spaces` read from the index, so the MCP server needs no data directory and `list_spaces` cannot disagree with what search can return.

## Consequences

- The Knowledge index can be built from an in-memory Corpus and a fake Model gateway and tested without any external service; the Corpus adapters stay thin and stateless.
- The index grows by the size of the stored text. Accepted in exchange for removing the shared on-disk layout.
- Corrupt or unreadable state is reported, never treated as empty, so it cannot silently trigger a full re-embed.
