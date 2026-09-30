"""File -> (title, text). LiteParse for binary docs; direct handling for html/md/txt."""
import json
import re
import subprocess
from pathlib import Path

from bs4 import BeautifulSoup
from markdownify import markdownify as to_md

import config

LIT_EXT = {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls",
           ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
TEXT_EXT = {".md", ".txt", ".rst"}
HTML_EXT = {".html", ".htm"}
SUPPORTED = LIT_EXT | TEXT_EXT | HTML_EXT


def sidecar_path(path: Path) -> Path:
    return path.with_name(path.name + ".meta.json")


def read_sidecar(path: Path) -> dict:
    sc = sidecar_path(path)
    if sc.exists():
        try:
            return json.loads(sc.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _parse_html(path: Path):
    raw = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw, "html.parser")
    for t in soup(["script", "style"]):
        t.decompose()
    title = soup.title.get_text(strip=True) if soup.title else None
    return title, _clean(to_md(str(soup), heading_style="ATX"))


def _parse_lit(path: Path) -> str:
    cmd = [config.LIT_BIN, "parse", str(path)]
    if not config.OCR:
        cmd.append("--no-ocr")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=config.LIT_TIMEOUT)
    if r.returncode != 0:
        raise RuntimeError(f"lit failed ({r.returncode}): {r.stderr.strip()[:500]}")
    return _clean(r.stdout)


def load(path: Path) -> dict:
    ext = path.suffix.lower()
    meta = read_sidecar(path)
    title = None
    if ext in HTML_EXT:
        title, text = _parse_html(path)
    elif ext in TEXT_EXT:
        text = _clean(path.read_text(encoding="utf-8", errors="replace"))
    elif ext in LIT_EXT:
        text = _parse_lit(path)
    else:
        raise ValueError(f"unsupported: {path}")
    return {
        "title": meta.get("title") or title or path.stem,
        "url": meta.get("url", ""),
        "space": meta.get("space", ""),
        "page_id": str(meta.get("page_id", "")),
        "text": text,
    }
