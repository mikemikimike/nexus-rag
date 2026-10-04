"""The Knowledge index and its public names.

The names are imported on first use (PEP 562), so that importing a sibling module that needs
only the standard library, the Installer, works on a machine where the package's
dependencies are not installed yet.
"""

from importlib import import_module

_EXPORTS = {
    "nexus_rag.exclusive": ["SyncInProgress"],
    "nexus_rag.generations": ["CorruptIndex", "NoPreviousIndex"],
    "nexus_rag.index": [
        "Health",
        "IndexStatus",
        "KnowledgeIndex",
        "RebuildReport",
        "SearchResults",
        "SyncReport",
    ],
    "nexus_rag.index_format": [
        "Access",
        "IncompatibleIndex",
        "IndexFormat",
        "IndexReadOnly",
        "IndexRefused",
    ],
    "nexus_rag.journal": ["LastSync"],
    "nexus_rag.passage_store": ["Passage"],
    "nexus_rag.settings": ["Settings"],
}
_MODULE_OF = {name: module for module, names in _EXPORTS.items() for name in names}

__all__ = sorted(_MODULE_OF)


def __getattr__(name: str):
    try:
        module = _MODULE_OF[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *__all__])
