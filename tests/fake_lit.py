"""A stand-in `lit` executable, so extraction tests need no LiteParse install.

It speaks the one command the extractor uses, `lit parse <file> [--no-ocr]`: the first
line of the file is a format header that is dropped, the rest is the "extracted" text.
A file whose body says CORRUPT makes it fail the way a bad PDF does.
"""
import stat
import sys
from pathlib import Path

SCRIPT = '''#!{python}
import sys

args = sys.argv[1:]
if args[:1] != ["parse"]:
    sys.exit("fake lit: only `parse` is supported")
body = open(args[1], "rb").read().decode("utf-8").split("\\n", 1)[1]
if "CORRUPT" in body:
    sys.exit("Error: could not read the document")
print("[liteparse] total: 1.0ms", file=sys.stderr)
sys.stdout.write(body)
'''


def install(directory: Path) -> Path:
    path = directory / "lit"
    path.write_text(SCRIPT.format(python=sys.executable))
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def pdf(body: str) -> bytes:
    return b"%PDF-fake\n" + body.encode()


def docx(body: str) -> bytes:
    return b"PK-fake-docx\n" + body.encode()
