"""Hybrid retrieval: dense + BM25 sparse, fused with RRF inside Qdrant, optional rerank via LiteLLM.
CLI: python search.py "question" [-k 6] [--mode hybrid|dense|sparse] [--space ENG] [--json]"""
import argparse
import json
import sys
from functools import lru_cache

import config  # noqa: F401  (must be first)
import requests
from qdrant_client import models

import embed
import sparse
import store


@lru_cache(maxsize=1)
def _client():
    return store.get_client()


def warmup():
    _client().get_collections()
    embed.get_embed_model()


def _rerank(query: str, hits: list[dict], top_n: int) -> list[dict]:
    try:
        r = requests.post(
            f"{config.LITELLM_BASE_URL.rstrip('/')}/rerank",
            headers={"Authorization": f"Bearer {config.LITELLM_API_KEY}"},
            json={"model": config.RERANK_MODEL, "query": query,
                  "documents": [h["text"][:2000] for h in hits], "top_n": top_n},
            timeout=60)
        r.raise_for_status()
        out = []
        for x in r.json()["results"]:
            h = dict(hits[x["index"]])
            h["score"] = round(float(x["relevance_score"]), 4)
            out.append(h)
        return out
    except Exception as e:
        print(f"rerank failed, using fused order: {e}", file=sys.stderr)
        return hits


def search(query: str, k: int = None, space: str = None, mode: str = "hybrid") -> list[dict]:
    k = k or config.TOP_K
    n = config.HYBRID_CANDIDATES
    flt = None
    if space:
        flt = models.Filter(must=[models.FieldCondition(
            key="space", match=models.MatchValue(value=space))])
    pre = []
    if mode in ("hybrid", "dense"):
        q = embed.get_embed_model().get_query_embedding(query)
        pre.append(models.Prefetch(query=q, using="dense", limit=n, filter=flt))
    if mode in ("hybrid", "sparse"):
        idx, val = sparse.encode_query(query)
        if idx:
            pre.append(models.Prefetch(
                query=models.SparseVector(indices=idx, values=val), using="bm25", limit=n, filter=flt))
    if not pre:
        return []
    want = max(config.RERANK_CANDIDATES if config.RERANK_MODEL else k * 3, k)
    client = _client()
    if len(pre) > 1:
        res = client.query_points(config.COLLECTION, prefetch=pre,
                                  query=models.FusionQuery(fusion=models.Fusion.RRF),
                                  limit=want, with_payload=True).points
    else:
        res = client.query_points(config.COLLECTION, query=pre[0].query, using=pre[0].using,
                                  query_filter=flt, limit=want, with_payload=True).points
    hits = [{
        "score": round(float(r.score), 4),
        "title": r.payload.get("title", ""), "url": r.payload.get("url", ""),
        "space": r.payload.get("space", ""), "page_id": r.payload.get("page_id", ""),
        "source": r.payload.get("source_id", ""), "text": r.payload.get("text", ""),
    } for r in res]
    if config.RERANK_MODEL and len(hits) > 1:
        hits = _rerank(query, hits, top_n=len(hits))
    out, per_doc = [], {}
    for h in hits:  # cap chunks per page so one long page can't crowd out the rest
        if per_doc.get(h["source"], 0) >= config.MAX_PER_DOC:
            continue
        per_doc[h["source"]] = per_doc.get(h["source"], 0) + 1
        out.append(h)
        if len(out) == k:
            break
    return out


def format_results(results: list[dict]) -> str:
    if not results:
        return "No results."
    blocks = []
    for i, r in enumerate(results, 1):
        ref = r["url"] or r["source"]
        pid = f" | page_id: {r['page_id']}" if r["page_id"] else ""
        sp = f" | space: {r['space']}" if r["space"] else ""
        blocks.append(f"[{i}] {r['title']}\n{ref}{pid}{sp}\n{r['text']}")
    return "\n\n---\n\n".join(blocks)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=config.TOP_K)
    ap.add_argument("--mode", default="hybrid", choices=["hybrid", "dense", "sparse"])
    ap.add_argument("--space", default=None)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    res = search(a.query, a.k, a.space, a.mode)
    print(json.dumps(res, ensure_ascii=False, indent=2) if a.json else format_results(res))
