"""Export Confluence pages via pchuri/confluence-cli into data/confluence_export/.

Per page:  <out>/<SPACE>/<pageId>/page.md (+ attachments/...) and a <file>.meta.json sidecar
           carrying title, url, space, page_id, version (used for citations + `confluence read <id>`).
Incremental: a page is re-exported only when its version changed (or with --full).
Deleted pages are removed from disk (only if all root listings succeeded).

Usage: python scripts/export_confluence.py [--roots FILE] [--full] [--max-depth N]
                                            [--no-attachments] [--exclude-attachments GLOBS] [--delay S]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402,F401  (loads .env so CONFLUENCE_* reach the CLI)
import parsing  # noqa: E402

CLI = os.environ.get("CONFLUENCE_BIN", "confluence")
# Images are excluded by default because OCR is off; remove *.png etc. when you enable RAG_OCR=1.
DEFAULT_EXCLUDE = ("*.mp4,*.mov,*.avi,*.mkv,*.zip,*.gz,*.tar,*.7z,"
                   "*.png,*.jpg,*.jpeg,*.gif,*.svg,*.bmp,*.webp")
MAIN_EXT = parsing.TEXT_EXT | parsing.HTML_EXT


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def cli_json(args):
    r = subprocess.run([CLI, *args, "--json"], capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"confluence {' '.join(args)} -> {r.returncode}: "
                           f"{(r.stderr or r.stdout).strip()[:400]}")
    return json.loads(r.stdout)


def read_roots(path: Path):
    roots = []
    if not path.exists():
        return roots
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        roots.append(line.split(" #")[0].strip())
    return roots


def list_pages(root, max_depth):
    info = cli_json(["info", root])
    root_id = str(info["id"])
    pages = []
    if info.get("type", "page") == "page":
        pages.append(info)
    args = ["children", root_id, "--recursive"]
    if max_depth:
        args += ["--max-depth", str(max_depth)]
    kids = cli_json(args)
    for c in kids.get("children", []):
        if c.get("type", "page") == "page":
            pages.append(c)
    for p in pages:
        p["id"] = str(p["id"])
    return pages


def page_dir(out: Path, p) -> Path:
    return out / (p.get("spaceKey") or "_") / p["id"]


def export_page(p, out, skip_att, exclude):
    d = page_dir(out, p)
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    cmd = [CLI, "export", p["id"], "--format", "markdown", "--dest", str(d), "--file", "page.md"]
    if skip_att:
        cmd.append("--skip-attachments")
    elif exclude:
        cmd += ["--exclude-attachments", exclude]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        shutil.rmtree(d, ignore_errors=True)
        raise RuntimeError((r.stderr or r.stdout).strip()[:400])
    files = [f for f in d.rglob("*")
             if f.is_file() and f.suffix.lower() in parsing.SUPPORTED
             and not f.name.endswith(".meta.json")]
    if not files:
        shutil.rmtree(d, ignore_errors=True)
        raise RuntimeError("export produced no supported files")
    for f in files:
        rel = f.relative_to(d).parts
        is_main = f.suffix.lower() in MAIN_EXT and "attachments" not in rel[:-1]
        title = p["title"] if is_main else f"{p['title']} / {f.name}"
        parsing.sidecar_path(f).write_text(json.dumps({
            "title": title, "url": p.get("url", ""), "space": p.get("spaceKey", ""),
            "page_id": p["id"], "version": p.get("version"),
        }, ensure_ascii=False), encoding="utf-8")
    return len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", default=str(config.ROOT / "confluence_roots.txt"))
    ap.add_argument("--out", default=str(config.DATA_DIR))
    ap.add_argument("--full", action="store_true", help="re-export everything (also refreshes attachments)")
    ap.add_argument("--max-depth", type=int, default=0)
    ap.add_argument("--no-attachments", action="store_true")
    ap.add_argument("--exclude-attachments", default=DEFAULT_EXCLUDE)
    ap.add_argument("--delay", type=float, default=0.2, help="seconds between page exports")
    a = ap.parse_args()

    roots = read_roots(Path(a.roots))
    if not roots:
        sys.exit(f"No roots in {a.roots}. Copy confluence_roots.txt.example and add page IDs/URLs.")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    state_f = out / ".export_state.json"
    state = json.loads(state_f.read_text()) if state_f.exists() else {}

    pages, listing_ok = {}, True
    for root in roots:
        try:
            for p in list_pages(root, a.max_depth):
                pages[p["id"]] = p
            log(f"root {root}: {len(pages)} pages so far")
        except Exception as e:
            listing_ok = False
            log(f"LIST FAIL {root}: {e}")

    exported = skipped = 0
    failed = []
    for pid, p in pages.items():
        prev = state.get(pid)
        d = page_dir(out, p)
        if not a.full and prev and prev.get("version") == p.get("version") and d.exists():
            skipped += 1
            continue
        try:
            n = export_page(p, out, a.no_attachments, a.exclude_attachments)
            state[pid] = {"version": p.get("version"), "dir": str(d.relative_to(out))}
            exported += 1
            log(f"OK   {pid} {p['title'][:60]} ({n} files)")
            time.sleep(a.delay)
        except Exception as e:
            failed.append(pid)
            state.pop(pid, None)
            log(f"FAIL {pid} {p['title'][:60]}: {e}")
        state_f.write_text(json.dumps(state))

    removed = 0
    if listing_ok:
        for pid in list(state):
            if pid not in pages:
                shutil.rmtree(out / state[pid]["dir"], ignore_errors=True)
                del state[pid]
                removed += 1
    else:
        log("Skipping stale-page cleanup because a root listing failed.")
    state_f.write_text(json.dumps(state))
    log(f"done: {exported} exported, {skipped} unchanged, {removed} removed, {len(failed)} failed")
    sys.exit(1 if failed or not listing_ok else 0)


if __name__ == "__main__":
    main()
