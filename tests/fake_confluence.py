"""A stand-in `confluence` executable, so Confluence Corpus tests need no Confluence.

It speaks the commands the adapter uses (`info`, `children`, `attachments`, `export`),
answering from a small site kept in a JSON file that the test edits through `FakeSite`,
and logs every call so a test can tell what was (not) fetched. A page can be told to fail
each command; a failing command exits non-zero with a message on stderr, like the CLI.
"""
import json
import stat
import sys
from pathlib import Path

SCRIPT = '''#!{python}
import json
import re
import sys
from pathlib import Path

SITE = Path({site!r})
args = sys.argv[1:]
with open(str(SITE) + ".log", "a") as log:
    log.write(json.dumps(args) + "\\n")
site = json.loads(SITE.read_text())
pages = site["pages"]


def fail(message):
    sys.exit(message)


def option(name):
    return args[args.index(name) + 1] if name in args else None


def page(reference):
    match = re.search(r"pageId=(\\d+)", reference)
    found = pages.get(match.group(1) if match else reference)
    if found is None:
        fail("Error: Page not found: " + reference)
    return found


def summary(p):
    return {{key: p[key] for key in ("id", "title", "type", "spaceKey", "url", "version")}}


def descendants(page_id):
    for p in pages.values():
        if p["parent"] == page_id:
            yield p
            yield from descendants(p["id"])


command, target = args[0], args[1]
p = page(target)
if command in p["fail"]:
    fail("Error: " + command + " " + p["id"] + " failed (HTTP 500)")
if command == "info":
    print(json.dumps(summary(p)))
elif command == "children":
    print(json.dumps({{"children": [summary(c) for c in descendants(p["id"])]}}))
elif command == "attachments" and "--download" not in args:
    print(json.dumps({{"attachments": [
        {{"id": "att-" + name, "title": name, "version": a["version"]}}
        for name, a in p["attachments"].items()]}}))
elif command == "attachments":
    name = option("--pattern")
    if "download" in p["fail"] or name not in p["attachments"]:
        fail("Error: could not download " + name)
    Path(option("--dest"), name).write_bytes(p["attachments"][name]["content"].encode())
elif command == "export":
    Path(option("--dest"), option("--file")).write_text(p["body"])
else:
    fail("fake confluence: unsupported command " + command)
'''


class FakeSite:
    def __init__(self, directory: Path):
        self._file = directory / "site.json"
        self._pages: dict[str, dict] = {}
        self._save()
        self.bin = directory / "confluence"
        self.bin.write_text(SCRIPT.format(python=sys.executable, site=str(self._file)))
        self.bin.chmod(self.bin.stat().st_mode | stat.S_IEXEC)

    def page(self, page_id: str, title: str, *, space: str = "ENG", parent: str | None = None,
             version: int = 1, body: str = "", type: str = "page") -> None:
        """Add a page, or change it: a changed page keeps its attachments and failures."""
        old = self._pages.get(page_id, {})
        self._pages[page_id] = {
            "id": page_id, "title": title, "type": type, "spaceKey": space,
            "url": f"https://wiki.example.test/pages/viewpage.action?pageId={page_id}",
            "version": version, "parent": parent, "body": body,
            "attachments": old.get("attachments", {}), "fail": old.get("fail", [])}
        self._save()

    def attach(self, page_id: str, name: str, content: str, *, version: int = 1) -> None:
        self._pages[page_id]["attachments"][name] = {"content": content, "version": version}
        self._save()

    def detach(self, page_id: str, name: str) -> None:
        del self._pages[page_id]["attachments"][name]
        self._save()

    def fail(self, page_id: str, *commands: str) -> None:
        """Make these commands (info, children, attachments, download, export) fail for the page."""
        self._pages[page_id]["fail"] = list(commands)
        self._save()

    def delete(self, page_id: str) -> None:
        del self._pages[page_id]
        self._save()

    def calls(self, command: str | None = None) -> list[list[str]]:
        """Every call so far, as argument lists, optionally only one command's."""
        log = Path(str(self._file) + ".log")
        calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        return [c for c in calls if command is None or c[0] == command]

    def _save(self) -> None:
        self._file.write_text(json.dumps({"pages": self._pages}))
