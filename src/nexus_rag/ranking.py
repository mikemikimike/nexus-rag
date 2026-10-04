"""Result shaping applied to fused search candidates: per-Document cap and rerank."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

from nexus_rag.gateway import ModelGateway

if TYPE_CHECKING:
    from nexus_rag.passage_store import Passage

log = logging.getLogger(__name__)


def cap_per_document(passages: Sequence[Passage], limit: int) -> list[Passage]:
    """Keep at most `limit` Passages of each Document, preserving rank order."""
    seen: Counter[str] = Counter()
    kept = []
    for passage in passages:
        seen[passage.document_id] += 1
        if seen[passage.document_id] <= limit:
            kept.append(passage)
    return kept


def rerank(gateway: ModelGateway, query: str, passages: Sequence[Passage]) -> list[Passage]:
    """Reorder Passages by the gateway's relevance scores; on any failure keep the order given."""
    if len(passages) < 2:
        return list(passages)
    try:
        scores = gateway.rerank(query, [p.text for p in passages])
        if len(scores) != len(passages):
            raise ValueError(f"expected {len(passages)} scores, got {len(scores)}")
    except Exception:
        log.warning("rerank failed; keeping the fused order", exc_info=True)
        return list(passages)
    order = sorted(range(len(passages)), key=lambda i: -scores[i])
    return [replace(passages[i], score=float(scores[i])) for i in order]
