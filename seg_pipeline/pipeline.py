from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

from .protocol import (
    RetrievalItem,
    SegmentResult,
    TopKInput,
    filter_usable_retrieval_items,
    normalize_retrieval_topk,
    validate_query_image_path,
)
from .rag_integration import rag_search_topk
from .sam3_memory_segmenter import Sam3BuildConfig, Sam3MemoryAttentionSegmenter


def segment_image(
    *,
    query_image_path: str,
    retrieval_topk: Optional[TopKInput] = None,
    k: Optional[int] = None,
    rag_kwargs: Optional[Dict[str, Any]] = None,
    output_mask_path: Optional[str] = None,
    output_meta_path: Optional[str] = None,
    sam3_build_config: Optional[Sam3BuildConfig] = None,
    sam3_extra_build_kwargs: Optional[Dict[str, Any]] = None,
    output_prob_thresh: float = 0.5,
    lock_memory: bool = True,
) -> SegmentResult:
    validate_query_image_path(query_image_path)

    meta: Dict[str, Any] = {"query_image_path": query_image_path}

    if retrieval_topk is None:
        if k is None:
            raise ValueError("必须提供 retrieval_topk 或 k（启用RAG检索）")
        rag_items, rag_meta = rag_search_topk(
            query_image_path=query_image_path,
            k=int(k),
            rag_kwargs=rag_kwargs,
        )
        meta["rag"] = {"k": rag_meta.k, "raw_result_count": rag_meta.raw_result_count}
        normalized = rag_items
        parse_skipped = []
    else:
        normalized, parse_skipped = normalize_retrieval_topk(retrieval_topk)

    usable, skipped = filter_usable_retrieval_items(normalized)
    meta["retrieval"] = {
        "input_count": len(normalized),
        "usable_count": len(usable),
        "usable": [it.to_meta() for it in usable],
        "skipped": [*parse_skipped, *skipped],
    }

    if len(usable) == 0:
        raise ValueError("没有可用的 Top-k 条目（mask 不存在或不可读）")

    out_mask_path = _resolve_output_mask_path(
        query_image_path=query_image_path,
        output_mask_path=output_mask_path,
    )
    out_meta_path = output_meta_path or f"{out_mask_path}.json"

    seg = Sam3MemoryAttentionSegmenter(
        build_config=sam3_build_config or Sam3BuildConfig(),
        extra_build_kwargs=sam3_extra_build_kwargs,
    )
    mask_img, sam3_meta = seg.segment(
        query_image_path=query_image_path,
        retrieval_items=usable,
        output_prob_thresh=output_prob_thresh,
        lock_memory=lock_memory,
    )

    Path(out_mask_path).parent.mkdir(parents=True, exist_ok=True)
    mask_img.save(out_mask_path, format="PNG")

    meta.update(
        {
            "output_mask_path": out_mask_path,
            "sam3_meta": sam3_meta,
        }
    )
    _write_json(out_meta_path, meta)

    return SegmentResult(
        output_mask_path=out_mask_path,
        meta=meta,
        meta_path=out_meta_path,
    )


def _resolve_output_mask_path(*, query_image_path: str, output_mask_path: Optional[str]) -> str:
    if output_mask_path:
        return output_mask_path
    q = Path(query_image_path)
    out_dir = q.parent / "outputs"
    return str(out_dir / f"{q.stem}_mask.png")


def _write_json(path: Union[str, os.PathLike], data: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
