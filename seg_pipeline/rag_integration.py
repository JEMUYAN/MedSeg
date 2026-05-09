from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .protocol import RetrievalItem


@dataclass(frozen=True)
class RAGSearchMeta:
    k: int
    raw_result_count: int


def rag_search_topk(
    query_image_path: str,
    k: int,
    rag_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[List[RetrievalItem], RAGSearchMeta]:
    rag_kwargs = {} if rag_kwargs is None else dict(rag_kwargs)

    from rag.rag_system import RAGSystem

    rag_system = RAGSystem(**rag_kwargs)
    results = rag_system.search(query_image=query_image_path, k=k)

    items: List[RetrievalItem] = []
    for img_path, mask_path, distance in results:
        items.append(
            RetrievalItem(
                image_path=img_path,
                mask_path=mask_path,
                distance=float(distance),
            )
        )

    return items, RAGSearchMeta(k=int(k), raw_result_count=len(results))
