from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


class InputValidationError(ValueError):
    pass


@dataclass(frozen=True)
class RetrievalItem:
    image_path: str
    mask_path: str
    score: Optional[float] = None
    distance: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_meta(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "image_path": self.image_path,
            "mask_path": self.mask_path,
        }
        if self.score is not None:
            out["score"] = self.score
        if self.distance is not None:
            out["distance"] = self.distance
        if self.extra:
            out["extra"] = dict(self.extra)
        return out


@dataclass(frozen=True)
class SegmentResult:
    output_mask_path: str
    meta: Dict[str, Any]
    meta_path: Optional[str] = None


TopKInput = Union[
    Sequence[Tuple[str, str, float]],
    Sequence[Mapping[str, Any]],
    Sequence[RetrievalItem],
]


def normalize_retrieval_topk(
    retrieval_topk: TopKInput,
) -> Tuple[List[RetrievalItem], List[Dict[str, Any]]]:
    items: List[RetrievalItem] = []
    skipped: List[Dict[str, Any]] = []

    for idx, raw in enumerate(retrieval_topk):
        try:
            item = _parse_one_retrieval_item(raw)
        except Exception as e:
            skipped.append(
                {
                    "index": idx,
                    "reason": "parse_error",
                    "error": str(e),
                }
            )
            continue
        items.append(item)

    return items, skipped


def filter_usable_retrieval_items(
    items: Iterable[RetrievalItem],
) -> Tuple[List[RetrievalItem], List[Dict[str, Any]]]:
    usable: List[RetrievalItem] = []
    skipped: List[Dict[str, Any]] = []

    for idx, it in enumerate(items):
        if not it.image_path or not isinstance(it.image_path, str):
            skipped.append({"index": idx, "reason": "missing_image_path"})
            continue
        if not os.path.exists(it.image_path):
            skipped.append(
                {
                    "index": idx,
                    "reason": "image_not_found",
                    "image_path": it.image_path,
                }
            )
            continue

        if not it.mask_path or not isinstance(it.mask_path, str):
            skipped.append(
                {
                    "index": idx,
                    "reason": "missing_mask_path",
                    "image_path": it.image_path,
                }
            )
            continue
        if not os.path.exists(it.mask_path):
            skipped.append(
                {
                    "index": idx,
                    "reason": "mask_not_found",
                    "image_path": it.image_path,
                    "mask_path": it.mask_path,
                }
            )
            continue

        usable.append(it)

    return usable, skipped


def validate_query_image_path(query_image_path: str) -> None:
    if not query_image_path or not isinstance(query_image_path, str):
        raise InputValidationError("query_image_path 必须为非空字符串")
    if not os.path.exists(query_image_path):
        raise InputValidationError(f"query_image_path 不存在: {query_image_path}")


def _parse_one_retrieval_item(raw: Any) -> RetrievalItem:
    if isinstance(raw, RetrievalItem):
        return raw

    if isinstance(raw, (list, tuple)):
        if len(raw) < 2:
            raise InputValidationError("tuple/list retrieval item 至少需要 (image_path, mask_path)")
        image_path = str(raw[0]) if raw[0] is not None else ""
        mask_path = str(raw[1]) if raw[1] is not None else ""
        distance: Optional[float] = None
        if len(raw) >= 3 and raw[2] is not None:
            distance = float(raw[2])
        extra: Dict[str, Any] = {}
        if len(raw) > 3:
            extra["raw_tail"] = list(raw[3:])
        return RetrievalItem(
            image_path=image_path,
            mask_path=mask_path,
            distance=distance,
            extra=extra,
        )

    if isinstance(raw, Mapping):
        image_path = raw.get("image_path", raw.get("img_path", raw.get("image", "")))
        mask_path = raw.get("mask_path", raw.get("mask", raw.get("mask_image_path", "")))
        score = raw.get("score", None)
        distance = raw.get("distance", raw.get("dist", None))

        extra = dict(raw)
        for k in ["image_path", "img_path", "image", "mask_path", "mask", "mask_image_path", "score", "distance", "dist"]:
            extra.pop(k, None)

        return RetrievalItem(
            image_path=str(image_path) if image_path is not None else "",
            mask_path=str(mask_path) if mask_path is not None else "",
            score=float(score) if score is not None else None,
            distance=float(distance) if distance is not None else None,
            extra=extra,
        )

    raise InputValidationError(f"不支持的 retrieval item 类型: {type(raw)}")
