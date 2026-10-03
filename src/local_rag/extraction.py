"""Turns a RawDocument's content into text, by media type.

PDF, Office and image files go through LiteParse (its `lit` command), HTML becomes
Markdown, a draw.io diagram yields its labels, plain text is read as it is. Anything that cannot be turned into text raises
ExtractionError, which a Sync counts as a failed Document.
"""

import re
import subprocess
import tempfile
from pathlib import Path

from bs4 import BeautifulSoup
from markdownify import markdownify

from local_rag import drawio
from local_rag.corpus import RawDocument, is_code
from local_rag.settings import Settings

# The file types a folder Corpus recognises, with the media type each is yielded as.
MEDIA_TYPES = {
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".rst": "text/x-rst",
    ".html": "text/html",
    ".htm": "text/html",
    ".pdf": "application/pdf",
    ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".ppt": "application/vnd.ms-powerpoint",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".xls": "application/vnd.ms-excel",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".drawio": drawio.MEDIA_TYPE,
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}
# Everything that is not text, HTML or a draw.io diagram goes through LiteParse, which
# picks its parser by file suffix.
_LITEPARSE_SUFFIX = {
    media_type: suffix
    for suffix, media_type in MEDIA_TYPES.items()
    if not media_type.startswith("text/") and media_type != drawio.MEDIA_TYPE
}
_MAX_ERROR_LENGTH = 500


class ExtractionError(Exception):
    pass


class EmptyExtraction(ExtractionError):
    """The content was read but held no text (a scanned page with OCR off, a blank file)."""


class Extractor:
    def __init__(self, *, liteparse_bin: str = "lit", ocr: bool = False, timeout: float = 300.0):
        self._liteparse_bin = liteparse_bin
        self._ocr = ocr
        self._timeout = timeout

    @classmethod
    def from_settings(cls, settings: Settings) -> "Extractor":
        return cls(
            liteparse_bin=settings.liteparse_bin,
            ocr=settings.ocr,
            timeout=settings.liteparse_timeout,
        )

    def extract(self, document: RawDocument) -> str:
        """The Document's text; raises ExtractionError if it has none or cannot be read."""
        media_type = document.media_type
        if media_type == "text/html":
            text = _html_to_markdown(document.content)
        elif media_type == drawio.MEDIA_TYPE:
            try:
                text = drawio.text(document.content)
            except drawio.DrawioError as error:
                raise ExtractionError(str(error)) from None
        elif media_type.startswith("text/"):
            text = document.content.decode("utf-8-sig", errors="replace")
        elif media_type in _LITEPARSE_SUFFIX:
            text = self._liteparse(
                document.content,
                _LITEPARSE_SUFFIX[media_type],
                ocr=self._ocr or media_type.startswith("image/"),
            )
        else:
            raise ExtractionError(f"unsupported media type {media_type}")
        if is_code(media_type):
            # Code is cited by line number, so nothing but the line endings is touched.
            text = text.replace("\r\n", "\n")
            if not text.strip():
                text = ""
        else:
            text = _normalise_whitespace(text)
        if not text:
            raise EmptyExtraction("empty extraction: no text could be read")
        return text

    def _liteparse(self, content: bytes, suffix: str, *, ocr: bool) -> str:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / f"document{suffix}"
            path.write_bytes(content)
            command = [self._liteparse_bin, "parse", str(path)]
            if not ocr:
                command.append("--no-ocr")
            try:
                result = subprocess.run(
                    command, capture_output=True, text=True, errors="replace", timeout=self._timeout
                )
            except FileNotFoundError:
                raise ExtractionError(
                    f"LiteParse not found (looked for {self._liteparse_bin!r})"
                ) from None
            except subprocess.TimeoutExpired:
                raise ExtractionError(f"LiteParse timed out after {self._timeout:g}s") from None
        if result.returncode != 0:
            raise ExtractionError(
                f"LiteParse failed ({result.returncode}): "
                f"{result.stderr.strip()[:_MAX_ERROR_LENGTH]}"
            )
        return result.stdout


def _html_to_markdown(content: bytes) -> str:
    soup = BeautifulSoup(content.decode("utf-8", errors="replace"), "html.parser")
    for element in soup(["script", "style"]):
        element.decompose()
    return markdownify(str(soup), heading_style="ATX")


def _normalise_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()
