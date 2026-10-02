"""Pins what a Media type is: prose or code, and the language a code type is split as."""
from local_rag.corpus import PROSE_MEDIA_TYPES, is_code
from local_rag.extraction import MEDIA_TYPES
from local_rag.gitlab import _NAMES, _SUFFIXES
from local_rag.languages import MEDIA_TYPE_LANGUAGES


def test_the_prose_types_are_the_text_types_that_are_not_code():
    assert PROSE_MEDIA_TYPES == {"text/plain", "text/markdown", "text/x-rst", "text/html"}
    assert not any(is_code(media_type) for media_type in PROSE_MEDIA_TYPES)
    assert not is_code("application/pdf")


def test_a_gitlab_file_is_code_unless_its_type_is_prose():
    gitlab_types = set(_NAMES.values()) | set(_SUFFIXES.values())
    prose = {t for t in gitlab_types if not is_code(t)}
    assert prose == {"text/markdown", "text/x-rst", "text/plain"}
    assert prose <= PROSE_MEDIA_TYPES


def test_gitlab_never_yields_html_as_a_page():
    assert "text/html" not in set(_NAMES.values()) | set(_SUFFIXES.values())
    assert is_code(_SUFFIXES[".html"]) and is_code(_SUFFIXES[".htm"])


def test_a_folder_suffix_keeps_its_prose_decision():
    assert not any(is_code(t) for t in MEDIA_TYPES.values())
    assert {s for s, t in MEDIA_TYPES.items() if t in PROSE_MEDIA_TYPES} == {
        ".md", ".txt", ".rst", ".html", ".htm"}


def test_only_code_types_are_split_by_syntax():
    assert MEDIA_TYPE_LANGUAGES
    assert all(is_code(media_type) for media_type in MEDIA_TYPE_LANGUAGES)
    assert set(MEDIA_TYPE_LANGUAGES) <= set(_SUFFIXES.values())
