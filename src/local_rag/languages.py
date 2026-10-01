"""The languages whose code is split along its syntax, by Media type.

Kept free of the parser itself so the Index format can record the set without loading it.
Adding a language changes ``syntax_languages()``, which makes an existing index Read-only
until a Rebuild.
"""

MEDIA_TYPE_LANGUAGES = {"text/x-python": "python", "text/x-c": "c"}


def syntax_languages() -> tuple[str, ...]:
    """The syntax-split languages, in a stable order."""
    return tuple(sorted(set(MEDIA_TYPE_LANGUAGES.values())))
