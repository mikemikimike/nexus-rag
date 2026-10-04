"""The names of everything one Knowledge index keeps in Qdrant, all derived from one name.

``<name>`` is the plain collection earlier versions kept as the whole index. Since then each
generation is a collection ``<name>.gen-<12 hex digits>``, the Live and previous index are the
aliases ``<name>.live`` and ``<name>.previous``, and four kinds of small record collections sit
beside them: the generation registry, the last Sync, the index-wide lock with one lock per
Corpus, and the lock that guards a first promotion.
"""

import re
import uuid

from nexus_rag.corpus import check_corpus_name


class IndexNames:
    def __init__(self, name: str):
        self.name = name
        self.plain = name  # the whole index, as earlier versions kept it
        self.live = f"{name}.live"  # aliases, not collections
        self.previous = f"{name}.previous"
        self.registry = f"{name}.generations"
        self.last_sync = f"{name}.last-sync"
        self.lock = f"{name}.sync-lock"  # held by a Rebuild, a switch-back or a Forget
        self.promote_lock = f"{name}.promote-lock"
        self._generation = re.compile(re.escape(f"{name}.gen-") + "[0-9a-f]{12}")

    def corpus_lock(self, corpus: str) -> str:
        """The lock a Sync of ``corpus`` holds."""
        return f"{self.lock}.{corpus}"

    def corpus_of_lock(self, collection: str) -> str | None:
        """The Corpus whose Sync lock ``collection`` is; None if it is not one of this index's."""
        prefix = f"{self.lock}."
        if not collection.startswith(prefix):
            return None
        corpus = collection.removeprefix(prefix)
        try:
            return check_corpus_name(corpus)
        except ValueError:
            return None

    def fresh_generation(self) -> str:
        """A name for a new generation, not yet recorded or created."""
        return f"{self.name}.gen-{uuid.uuid4().hex[:12]}"

    def is_generation(self, collection: str) -> bool:
        """Whether ``collection`` is named like one of this index's generations."""
        return self._generation.fullmatch(collection) is not None

    def owns(self, collection: str) -> bool:
        """Whether ``collection`` has a name this index gives its collections. A collection
        named like a generation may still be another's: only the registry says it is ours."""
        return (
            collection in (self.plain, self.registry, self.last_sync, self.lock, self.promote_lock)
            or self.corpus_of_lock(collection) is not None
            or self.is_generation(collection)
        )
