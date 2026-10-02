"""The Corpus declaration file: every Corpus the operator syncs, by name, in one TOML file.

    [corpus.docs]                       # the table's key is the Corpus's name
    kind = "folder"
    path = "/srv/docs"                  # relative paths are relative to this file

    [corpus.wiki]
    kind = "confluence"
    roots = ["12345", "https://wiki.example.test/pages/viewpage.action?pageId=678"]
    max_depth = 0                       # optional: 0 (the default) = the whole subtree
    attachments = true                  # optional
    exclude_attachments = ["*.png"]     # optional: file name globs; defaults to images, video, archives

    [corpus.code]
    kind = "gitlab"                     # GITLAB_URL and GITLAB_TOKEN come from the environment
    cache = "/var/cache/local-rag/gitlab"   # optional: where the mirrors live; safe to delete
                                        # (default: $HOME/.cache/local-rag/gitlab)
    [[corpus.code.repositories]]        # one table per Repository
    path = "platform/payments"          # the project's full path in GitLab
    refs = ["main", {name = "release/1.2", primary = true, exclude = ["docs/**"]}]
                                        # its Tracked refs, a branch or tag each. The first is
                                        # the Primary ref unless another sets primary = true;
                                        # exclude globs (a path, or a file name) are added to
                                        # the defaults for that ref

Each Corpus has a ``kind`` and that kind's parameters. Unknown names, kinds and parameters
are rejected when the file is read; a wrong value (or a folder that is not there) when the
Corpus is built, which ``sync`` does only for the Corpus it names. A kind is added by
registering a builder in ``_KINDS``: it validates its table and builds the Corpus from it.
The file is read by ``sync <corpus>``, ``rebuild`` and Status.
"""
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from local_rag.confluence import DEFAULT_EXCLUDED_ATTACHMENTS, ConfluenceCorpus
from local_rag.corpus import Corpus, check_corpus_name
from local_rag.folder import FolderCorpus
from local_rag.gitlab import GitLabCorpus, Repository

DEFAULT_PATH = "/etc/local-rag/corpora.toml"
DEFAULT_GITLAB_CACHE = Path(".cache/local-rag/gitlab")  # below HOME


class DeclarationError(ValueError):
    """The declaration file is missing, unreadable or wrong, or a Corpus is not declared in it."""


@dataclass(frozen=True)
class _Builder:
    parameters: frozenset[str]  # the keys a table of this kind may hold besides ``kind``
    build: Callable[[str, "_Table", Mapping[str, str]], Corpus]


class _Table:
    """One Corpus's table, with typed reads that name the Corpus when a value is wrong."""

    def __init__(self, name: str, table: Mapping[str, Any], base: Path):
        self.name, self._table, self.base = name, table, base

    def _fail(self, key: str, wanted: str) -> DeclarationError:
        return DeclarationError(f"corpus {self.name!r}: {key} must be {wanted}")

    def __contains__(self, key: str) -> bool:
        return key in self._table

    def tables(self, key: str) -> list["_Table"]:
        """The array of tables under ``key``, each readable like a Corpus's own table."""
        value = self._table.get(key)
        if (not isinstance(value, list) or not value
                or not all(isinstance(v, Mapping) for v in value)):
            raise self._fail(key, "one or more [[tables]]")
        return [_Table(self.name, v, self.base) for v in value]

    def items(self, key: str) -> list["str | _Table"]:
        """The list under ``key`` of strings and inline tables (the tables readable like a
        Corpus's own)."""
        value = self._table.get(key)
        if (not isinstance(value, list)
                or not all(isinstance(v, (str, Mapping)) for v in value)):
            raise self._fail(key, "a list of names or {name = ...} tables")
        return [v.strip() if isinstance(v, str) else _Table(self.name, v, self.base)
                for v in value]

    def optional_flag(self, key: str) -> bool | None:
        return self.flag(key, False) if key in self._table else None

    def only(self, *keys: str) -> None:
        """Raises unless the table holds no key besides ``keys``."""
        unknown = sorted(set(self._table) - set(keys))
        if unknown:
            raise DeclarationError(
                f"corpus {self.name!r}: unknown {', '.join(unknown)} (takes {', '.join(keys)})")

    def text(self, key: str) -> str:
        value = self._table.get(key)
        if not isinstance(value, str) or not value.strip():
            raise self._fail(key, "a non-empty string")
        return value

    def texts(self, key: str, default: Sequence[str] | None = None) -> list[str]:
        value = self._table.get(key, default)
        if value is None:
            raise self._fail(key, "given")
        if (not isinstance(value, (list, tuple)) or not all(isinstance(v, str) and v.strip()
                                                           for v in value)):
            raise self._fail(key, "a list of non-empty strings")
        return [v.strip() for v in value]

    def whole(self, key: str, default: int) -> int:
        value = self._table.get(key, default)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise self._fail(key, "a whole number, 0 or more")
        return value

    def flag(self, key: str, default: bool) -> bool:
        value = self._table.get(key, default)
        if not isinstance(value, bool):
            raise self._fail(key, "true or false")
        return value


def _folder(name: str, table: _Table, environ: Mapping[str, str]) -> Corpus:
    path = table.base / table.text("path")
    if not path.is_dir():
        # An absent folder (an unmounted share) must not look like an empty Corpus,
        # which a Sync would answer by removing every Document.
        raise DeclarationError(f"corpus {name!r}: {path} is not a folder")
    return FolderCorpus(path, name=name)


def _confluence(name: str, table: _Table, environ: Mapping[str, str]) -> Corpus:
    roots = table.texts("roots")
    if not roots:
        raise DeclarationError(f"corpus {name!r}: roots must name at least one Root")
    return ConfluenceCorpus(
        roots, name=name, confluence_bin=environ.get("CONFLUENCE_BIN") or "confluence",
        max_depth=table.whole("max_depth", 0), attachments=table.flag("attachments", True),
        exclude_attachments=table.texts("exclude_attachments", DEFAULT_EXCLUDED_ATTACHMENTS))


def _repositories(table: _Table) -> list[Repository]:
    """One ``Repository`` per Tracked ref of the Repository the table declares."""
    table.only("path", "refs")
    path = table.text("path").strip("/")
    entries = table.items("refs")
    if not entries:
        raise DeclarationError(f"corpus {table.name!r}: refs must name at least one ref")
    refs = []
    for entry in entries:
        if isinstance(entry, str):
            if not entry:
                raise table._fail("refs", "names of branches or tags")
            refs.append((entry, None, ()))
            continue
        entry.only("name", "primary", "exclude")
        refs.append((entry.text("name").strip(), entry.optional_flag("primary"),
                     tuple(entry.texts("exclude", ()))))
    names = [name for name, _, _ in refs]
    if len(set(names)) != len(names):
        raise DeclarationError(f"corpus {table.name!r}: {path} lists a ref twice")
    marked = [name for name, primary, _ in refs if primary]
    if len(marked) > 1:
        raise DeclarationError(
            f"corpus {table.name!r}: {path} marks more than one ref primary ({', '.join(marked)})")
    primary_name = marked[0] if marked else names[0]
    return [Repository(path, name, name == primary_name, exclude) for name, _, exclude in refs]


def _gitlab(name: str, table: _Table, environ: Mapping[str, str]) -> Corpus:
    repositories = [r for t in table.tables("repositories") for r in _repositories(t)]
    missing = [v for v in ("GITLAB_URL", "GITLAB_TOKEN") if not environ.get(v, "").strip()]
    if missing:
        raise DeclarationError(
            f"corpus {name!r}: {' and '.join(missing)} must be set in the environment")
    home = Path(environ.get("HOME") or Path.home())
    cache = table.base / table.text("cache") if "cache" in table else home / DEFAULT_GITLAB_CACHE
    return GitLabCorpus(repositories, name=name, url=environ["GITLAB_URL"].strip(),
                        token=environ["GITLAB_TOKEN"].strip(), cache=cache)


_KINDS: dict[str, _Builder] = {
    "gitlab": _Builder(frozenset({"cache", "repositories"}), _gitlab),
    "folder": _Builder(frozenset({"path"}), _folder),
    "confluence": _Builder(
        frozenset({"roots", "max_depth", "attachments", "exclude_attachments"}), _confluence),
}


class Declaration:
    """The Corpora an operator declared: their names, and a way to build each from its table."""

    def __init__(self, corpora: Mapping[str, Mapping[str, Any]], base: Path | str = "."):
        """Raises DeclarationError if a name, a kind or a parameter is not valid."""
        self._base = Path(base)
        self._tables: dict[str, _Table] = {}
        self._kinds: dict[str, str] = {}
        for name, table in corpora.items():
            try:
                check_corpus_name(name)
            except ValueError as error:
                raise DeclarationError(str(error)) from None
            if not isinstance(table, Mapping):
                raise DeclarationError(f"corpus {name!r} must be a table with a kind")
            kind = table.get("kind")
            builder = _KINDS.get(kind) if isinstance(kind, str) else None
            if builder is None:
                raise DeclarationError(
                    f"corpus {name!r}: kind must be one of {', '.join(sorted(_KINDS))}, "
                    f"got {kind!r}")
            unknown = sorted(set(table) - builder.parameters - {"kind"})
            if unknown:
                raise DeclarationError(
                    f"corpus {name!r}: unknown {', '.join(unknown)} for a {kind} Corpus "
                    f"(it takes {', '.join(sorted(builder.parameters))})")
            self._tables[name] = _Table(name, table, self._base)
            self._kinds[name] = kind

    @classmethod
    def read(cls, path: Path | str) -> "Declaration":
        """Raises DeclarationError if the file cannot be read or is not a valid declaration."""
        path = Path(path)
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except OSError as error:
            raise DeclarationError(f"cannot read the Corpus declaration file: {error}") from None
        except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
            raise DeclarationError(f"{path} is not valid TOML: {error}") from None
        unknown = sorted(set(data) - {"corpus"})
        if unknown:
            raise DeclarationError(f"{path}: unknown top-level {', '.join(unknown)}; "
                                   f"declare each Corpus as [corpus.NAME]")
        corpora = data.get("corpus", {})
        if not isinstance(corpora, Mapping):
            raise DeclarationError(f"{path}: corpus must hold one [corpus.NAME] table per Corpus")
        try:
            return cls(corpora, path.parent)
        except DeclarationError as error:
            raise DeclarationError(f"{path}: {error}") from None

    @property
    def names(self) -> list[str]:
        """The declared Corpus names, in the order the file lists them."""
        return list(self._tables)

    def names_of(self, kind: str) -> list[str]:
        """The names of the declared Corpora of one kind (``folder``, ``confluence``, ``gitlab``)."""
        return [name for name, declared in self._kinds.items() if declared == kind]

    def corpus(self, name: str, environ: Mapping[str, str] | None = None) -> Corpus:
        """The Corpus declared under ``name``; raises DeclarationError if there is none or its
        parameters cannot be used (for example a folder that is not there)."""
        if name not in self._tables:
            declared = ", ".join(self.names) or "none"
            raise DeclarationError(f"no Corpus named {name!r} is declared (declared: {declared})")
        return _KINDS[self._kinds[name]].build(name, self._tables[name], environ or {})

    def corpora(self, environ: Mapping[str, str] | None = None) -> list[Corpus]:
        """Every declared Corpus, built."""
        return [self.corpus(name, environ) for name in self.names]

    def confluence_roots(self) -> list[str]:
        """The Roots of every Confluence Corpus declared, for checks that need them resolved."""
        return [root for name, kind in self._kinds.items() if kind == "confluence"
                for root in self._tables[name].texts("roots")]
