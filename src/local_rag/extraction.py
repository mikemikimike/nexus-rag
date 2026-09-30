"""Turns a RawDocument's content into text, by media type.

PDF, Office and image files go through LiteParse (its `lit` command), HTML becomes
Markdown, plain text is read as it is. Anything that cannot be turned into text raises
ExtractionError, which a Sync counts as a failed Document.
"""
import re
import subprocess
import tempfile
from pathlib import Path

from bs4 import BeautifulSoup
from markdownify import markdownify

from local_rag.corpus import RawDocument
from local_rag.settings import Settings

# File suffix LiteParse uses to pick a parser, per media type it is given.
_LITEPARSE_SUFFIX = {
    "application/pdf": ".pdf",
    "application/msword": ".doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.ms-powerpoint": ".ppt",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "application/vnd.ms-excel": ".xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/tiff": ".tiff",
}
_MAX_ERROR_LENGTH = 500


class ExtractionError(Exception):
    pass


class Extractor:
    def __init__(self, *, liteparse_bin: str = "lit", ocr: bool = False,
                 timeout: float = 300.0):
        self._liteparse_bin = liteparse_bin
        self._ocr = ocr
        self._timeout = timeout

    @classmethod
    def from_settings(cls, settings: Settings) -> "Extractor":
        return cls(liteparse_bin=settings.liteparse_bin, ocr=settings.ocr,
                   timeout=settings.liteparse_timeout)

    def extract(self, document: RawDocument) -> str:
        """The Document's text; raises ExtractionError if it has none or cannot be read."""
        media_type = document.media_type
        if media_type == "text/html":
            text = _html_to_markdown(document.content)
        elif media_type.startswith("text/"):
            text = document.content.decode("utf-8", errors="replace")
        elif media_type in _LITEPARSE_SUFFIX:
            text = self._liteparse(document.content, _LITEPARSE_SUFFIX[media_type])
        else:
            raise ExtractionError(f"unsupported media type {media_type}")
        text = _tidy(text)
        if not text:
            raise ExtractionError("empty extraction: no text could be read")
        return text

    def _liteparse(self, content: bytes, suffix: str) -> str:
        command = [self._liteparse_bin, "parse"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / f"document{suffix}"
            path.write_bytes(content)
            command.append(str(path))
            if not self._ocr:
                command.append("--no-ocr")
            try:
                result = subprocess.run(
                    command, capture_output=True, text=True, errors="replace",
                    timeout=self._timeout)
            except FileNotFoundError:
                raise ExtractionError(
                    f"LiteParse not found (looked for {self._liteparse_bin!r})") from None
            except subprocess.TimeoutExpired:
                raise ExtractionError(
                    f"LiteParse timed out after {self._timeout:g}s") from None
        if result.returncode != 0:
            raise ExtractionError(
                f"LiteParse failed ({result.returncode}): "
                f"{result.stderr.strip()[:_MAX_ERROR_LENGTH]}")
        return result.stdout


def _html_to_markdown(content: bytes) -> str:
    soup = BeautifulSoup(content.decode("utf-8", errors="replace"), "html.parser")
    for element in soup(["script", "style"]):
        element.decompose()
    return markdownify(str(soup), heading_style="ATX")


def _tidy(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()
