"""The text of a draw.io diagram: the page names and every shape and connector label."""
import base64
import re
import urllib.parse
import zlib
from xml.etree import ElementTree

from bs4 import BeautifulSoup

MEDIA_TYPE = "application/vnd.jgraph.mxfile"


class DrawioError(Exception):
    pass


def text(content: bytes) -> str:
    """Labels one per line under a heading per page; raises DrawioError if not a diagram."""
    try:
        root = ElementTree.fromstring(content)
    except ElementTree.ParseError as error:
        raise DrawioError(f"not a readable drawio diagram: {error}") from None
    if root.tag == "mxGraphModel":
        return "\n".join(_labels(root))
    if root.tag != "mxfile":
        raise DrawioError(f"not a drawio diagram: unexpected <{root.tag}>")
    pages = []
    for diagram in root.iter("diagram"):
        lines = _labels(_model(diagram))
        name = (diagram.get("name") or "").strip()
        if lines and name:
            lines.insert(0, f"## {name}")
        pages.append("\n".join(lines))
    return "\n\n".join(page for page in pages if page)


def _model(diagram: ElementTree.Element) -> ElementTree.Element:
    """The page's graph model: nested in the page, or stored compressed as its text."""
    nested = diagram.find("mxGraphModel")
    if nested is not None:
        return nested
    encoded = "".join((diagram.text or "").split())
    if not encoded:
        return diagram
    try:
        # Raw deflate, then URL-encoded XML, is how draw.io compresses a page.
        xml = urllib.parse.unquote(
            zlib.decompress(base64.b64decode(encoded, validate=True), -15).decode("utf-8"))
        return ElementTree.fromstring(xml)
    except (ValueError, zlib.error, ElementTree.ParseError) as error:
        raise DrawioError(f"not a readable drawio page: {error}") from None


def _labels(model: ElementTree.Element) -> list[str]:
    found = []
    for element in model.iter():
        # A plain cell keeps its label in `value`; a cell wrapped in object/UserObject
        # (custom properties) has it on the wrapper as `label`.
        label = element.get("label") if element.tag in ("object", "UserObject") else (
            element.get("value") if element.tag == "mxCell" else None)
        if label and (label := _plain(label)):
            found.append(label)
    return found


def _plain(label: str) -> str:
    # Labels are HTML whenever the shape says html=1; stripping tags from plain text
    # that has none changes nothing.
    if "<" in label or "&" in label:
        label = BeautifulSoup(label, "html.parser").get_text(" ")
    return re.sub(r"\s+", " ", label).strip()
