"""Run on the VM as the service user: .venv/bin/python scripts/smoke_test.py"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config  # noqa: E402

ok = True


def check(name, fn):
    global ok
    try:
        print(f"[ OK ] {name}: {fn()}")
    except Exception as e:
        ok = False
        print(f"[FAIL] {name}: {e}")


check("lit CLI", lambda: subprocess.run([config.LIT_BIN, "--version"], capture_output=True,
                                        text=True, check=True).stdout.strip() or "runs")


def _embed():
    import embed
    v = embed.get_embed_model().get_text_embedding("smoke test")
    return f"model={config.EMBED_MODEL} dim={len(v)}"


check("embeddings via LiteLLM", _embed)


def _rerank():
    if not config.RERANK_MODEL:
        return "disabled (RAG_RERANK_MODEL empty)"
    import search
    out = search._rerank("q", [{"text": "a", "score": 0}, {"text": "b", "score": 0}], 2)
    return "responded"


check("rerank", _rerank)


def _qdrant():
    import store
    c = store.get_client()
    return f"reachable, collections={[x.name for x in c.get_collections().collections]}, alias->{store.resolve(c)}"


check("qdrant", _qdrant)


def _sparse():
    import sparse
    i, v = sparse.encode_query("error ORA-00942 in payments_db")
    assert i and len(i) == len(v)
    return f"{len(i)} query terms"


check("sparse encoder", _sparse)
check("mcp sdk", lambda: __import__("mcp").__name__)
check("MCP_TOKENS set", lambda: f"{len(config.MCP_TOKENS)} token(s)" if config.MCP_TOKENS
      and not any(t.startswith("CHANGE_ME") for t in config.MCP_TOKENS)
      else (_ for _ in ()).throw(RuntimeError("missing or still a placeholder")))
sys.exit(0 if ok else 1)
