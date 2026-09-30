"""Incremental ingest into Qdrant (dense + BM25 sparse vectors).
--rebuild builds a NEW collection and swaps the alias atomically (zero downtime for the MCP server).
Usage: python ingest.py [--rebuild] [--dir PATH]"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import config  # noqa: F401  (must be first)
from llama_index.core import Document, Settings
from llama_index.core.schema import MetadataMode
from qdrant_client import models

import embed
import parsing
import sparse
import store


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def file_hash(p: Path) -> str:
    h = hashlib.sha256()
    for f in (p, parsing.sidecar_path(p)):
        if f.exists():
            with open(f, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
    return h.hexdigest()


def load_state() -> dict:
    if config.MANIFEST.exists():
        try:
            return json.loads(config.MANIFEST.read_text())
        except Exception:
            pass
    return {"collection": None, "files": {}}


def save_state(s: dict) -> None:
    tmp = config.MANIFEST.with_suffix(".tmp")
    tmp.write_text(json.dumps(s))
    tmp.replace(config.MANIFEST)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="re-embed everything into a new collection")
    ap.add_argument("--dir", default=str(config.DATA_DIR))
    args = ap.parse_args()
    data_dir = Path(args.dir).resolve()
    config.STATE_DIR.mkdir(parents=True, exist_ok=True)

    embed.init_settings()
    model = embed.get_embed_model()
    splitter = Settings.node_parser
    client = store.get_client()

    live = store.resolve(client)
    prev_state = load_state()
    rebuild = args.rebuild or live is None or prev_state.get("collection") != live
    if rebuild:
        dim = len(model.get_text_embedding("dimension probe"))
        target = f"{config.COLLECTION}_{int(time.time())}"
        store.create_collection(client, target, dim)
        files_state = {}
        log(f"REBUILD into {target} (dim={dim}; live={live})")
    else:
        target, files_state = live, dict(prev_state["files"])
    state = {"collection": target, "files": files_state}

    files = [p for p in sorted(data_dir.rglob("*"))
             if p.is_file() and p.suffix.lower() in parsing.SUPPORTED]
    log(f"found {len(files)} supported files in {data_dir}")

    seen, done, skipped, failed = set(), 0, 0, []
    for p in files:
        sid = p.relative_to(data_dir).as_posix()
        seen.add(sid)
        h = file_hash(p)
        if files_state.get(sid) == h:
            skipped += 1
            continue
        try:
            d = parsing.load(p)
        except Exception as e:
            log(f"FAIL {sid}: {e}")
            failed.append(sid)
            continue
        if not d["text"].strip():
            log(f"EMPTY {sid} (no text extracted; scanned? enable RAG_OCR=1)")
            continue
        try:
            doc = Document(text=d["text"], metadata={"title": d["title"]})
            doc.excluded_embed_metadata_keys = []
            nodes = splitter.get_nodes_from_documents([doc])
            texts = [n.get_content(metadata_mode=MetadataMode.EMBED) for n in nodes]
            vecs = model.get_text_embedding_batch(texts)
            points = []
            for i, (n, t, v) in enumerate(zip(nodes, texts, vecs)):
                idx, val = sparse.encode_doc(t)
                points.append(models.PointStruct(
                    id=store.point_id(sid, i),
                    vector={"dense": v, "bm25": models.SparseVector(indices=idx, values=val)},
                    payload={"source_id": sid, "title": d["title"], "url": d["url"],
                             "space": d["space"], "page_id": d["page_id"], "chunk": i,
                             "text": n.get_content(metadata_mode=MetadataMode.NONE)},
                ))
            if not rebuild:
                store.delete_source(client, target, sid)
            store.upsert(client, target, points)
        except Exception as e:
            log(f"FAIL {sid}: {e}")
            failed.append(sid)
            continue
        files_state[sid] = h
        save_state(state)
        done += 1
        log(f"OK   {sid} ({len(points)} chunks)")

    removed = 0
    if not rebuild:
        for sid in [s for s in files_state if s not in seen]:
            store.delete_source(client, target, sid)
            del files_state[sid]
            removed += 1
    else:
        if live and done == 0:
            log("nothing indexed; keeping the existing live collection")
            client.delete_collection(target)
            save_state(prev_state)
            sys.exit(1)
        store.swap_alias(client, target, had_alias=live is not None)
        for c in client.get_collections().collections:
            if c.name.startswith(config.COLLECTION + "_") and c.name != target:
                client.delete_collection(c.name)
    save_state(state)

    log(f"done: {done} indexed, {skipped} unchanged, {removed} removed, {len(failed)} failed; "
        f"{client.count(target).count} chunks in {target}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
