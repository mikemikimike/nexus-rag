"""Central config. Import FIRST so env vars are set before other libs load."""
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(os.environ.get("RAG_HOME", Path(__file__).resolve().parent)).resolve()


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.split(" #")[0].strip().strip('"').strip("'")
        os.environ.setdefault(k.strip(), v)


_load_dotenv(ROOT / ".env")


def _p(env: str, default: str) -> Path:
    p = Path(os.environ.get(env, default))
    return p if p.is_absolute() else (ROOT / p).resolve()


def _i(env: str, default: int) -> int:
    return int(os.environ.get(env, default))


DATA_DIR = _p("RAG_DATA_DIR", "data/confluence_export")
STATE_DIR = _p("RAG_STATE_DIR", "state")
MANIFEST = STATE_DIR / "manifest.json"

# Embeddings
EMBED_BACKEND = os.environ.get("RAG_EMBED_BACKEND", "litellm")  # "litellm" | "local"
LITELLM_BASE_URL = os.environ.get("LITELLM_BASE_URL", "http://localhost:4000/v1")
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY", "sk-local")
EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "embed")
EMBED_BATCH = _i("RAG_EMBED_BATCH", 16)
MODEL_PATH = str(_p("RAG_EMBED_MODEL_PATH", "models/embed"))  # backend=local only
DEVICE = os.environ.get("RAG_DEVICE", "cpu")
MAX_SEQ = _i("RAG_MAX_SEQ", 512)

# Chunking
CHUNK_TOKENS = _i("RAG_CHUNK_TOKENS", 400)
CHUNK_OVERLAP = _i("RAG_CHUNK_OVERLAP", 60)

# Qdrant
QDRANT_URL = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
COLLECTION = os.environ.get("RAG_COLLECTION", "confluence")  # alias name

# Retrieval
TOP_K = _i("RAG_TOP_K", 6)
HYBRID_CANDIDATES = _i("RAG_HYBRID_CANDIDATES", 40)
MAX_PER_DOC = _i("RAG_MAX_PER_DOC", 2)
RERANK_MODEL = os.environ.get("RAG_RERANK_MODEL", "")
RERANK_CANDIDATES = _i("RAG_RERANK_CANDIDATES", 30)
BM25_K1 = float(os.environ.get("RAG_BM25_K1", "1.2"))
BM25_B = float(os.environ.get("RAG_BM25_B", "0.75"))
BM25_AVG_DL = float(os.environ.get("RAG_BM25_AVG_DL", "250"))

# MCP server
MCP_HOST = os.environ.get("MCP_HOST", "0.0.0.0")
MCP_PORT = _i("MCP_PORT", 8765)
MCP_TOKENS = [t.strip() for t in os.environ.get("MCP_TOKENS", "").split(",") if t.strip()]
MCP_TLS_CERT = os.environ.get("MCP_TLS_CERT", "")
MCP_TLS_KEY = os.environ.get("MCP_TLS_KEY", "")

# Parsing
OCR = os.environ.get("RAG_OCR", "0") == "1"
LIT_TIMEOUT = _i("RAG_LIT_TIMEOUT", 300)
_sibling = Path(sys.executable).parent / "lit"
LIT_BIN = os.environ.get("LIT_BIN") or (str(_sibling) if _sibling.exists() else "lit")
