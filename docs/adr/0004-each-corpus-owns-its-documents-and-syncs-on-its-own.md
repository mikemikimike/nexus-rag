# Each Corpus owns its Documents and syncs on its own

The Knowledge index used to hold one Corpus: a Sync removed every Document its Corpus did not list, so a second Corpus synced into the same index would have removed the first one's. To search Confluence pages and GitLab code together, with feature branches kept fresher than Confluence, the index now holds several **named Corpora** declared by the operator in one place, and every Document belongs to exactly one of them.

- **A Sync is of one Corpus.** It adds, replaces and removes only that Corpus's Documents. Corpora sync on their own timers.
- **Locking is per Corpus for Syncs.** Syncs of different Corpora touch disjoint Documents and may run at once; a Sync of the same Corpus is refused as in progress. A Rebuild still holds the whole index. This amends the locking in ADR 0002.
- **An Incomplete listing holds back removals only in the part that failed** (a Root, a page's Attachments, a Tracked ref), so a deleted feature branch does not freeze removals everywhere. A missing ref is a failure, never an empty Repository.
- **A Rebuild covers every declared Corpus.** A fresh generation built from one Corpus would drop the others' Documents on promotion.
- **Removing a Corpus is explicit.** Documents of a Corpus no longer declared stay searchable, with a Status warning, until `nexus-rag forget <corpus>` or the next Rebuild.

## Considered options

- **One composite Corpus listed by every Sync:** rejected. Code freshness would be tied to the cost of listing Confluence.
- **A separate index per kind of content:** rejected. Claude would have to search twice and could not rank a design page against the code that implements it.
- **Per-Corpus Rebuild that copies the other Corpora's Passages:** rejected. It only works when the embedding model has not changed, which is the main reason to Rebuild.
- **Sync removes Documents of undeclared Corpora:** rejected. A typo in the declaration would wipe a Corpus.

## Consequences

- Corpora are declared in one file rather than by per-run `--folder` and `--roots-file` flags, so that `sync <corpus>` and `rebuild` read the same definitions.
- A change that needs a Rebuild for one kind of content (code chunking, the embedding model) re-embeds every Corpus.
- The record of index generations is no longer written only under the index-wide lock: the first Syncs of different Corpora into an empty index each create a generation, and one of them promotes it. Every change to that record therefore holds a short lock of its own, a first promotion never prunes (the other first Syncs are still writing theirs, and each discards its own), and the next Rebuild cleans up after any that crashed.
