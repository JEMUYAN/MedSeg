from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from rag.dataset_importers.base import DatasetImporter


class COCOImporter(DatasetImporter):
    def discover_pairs(
        self, dataset_path: str, **kwargs: Any
    ) -> List[Tuple[str, str]]:
        if not os.path.isfile(dataset_path):
            raise FileNotFoundError(f"COCO 标注文件不存在: {dataset_path}")

        image_dir: Optional[str] = kwargs.get("image_dir", None)
        category_ids: Optional[List[int]] = kwargs.get("category_ids", None)
        save_masks_dir: Optional[str] = kwargs.get("save_masks_dir", None)

        with open(dataset_path, "r", encoding="utf-8") as f:
            coco_data = json.load(f)

        images_by_id: Dict[int, Dict[str, Any]] = {
            img["id"]: img for img in coco_data.get("images", [])
        }

        annotations_by_image: Dict[int, List[Dict[str, Any]]] = {}
        for ann in coco_data.get("annotations", []):
            cat_id = ann.get("category_id")
            if category_ids is not None and cat_id not in category_ids:
                continue
            img_id = ann["image_id"]
            if img_id not in images_by_id:
                continue
            annotations_by_image.setdefault(img_id, []).append(ann)

        mask_dir: str
        tmp_dir: Optional[tempfile.TemporaryDirectory] = None
        if save_masks_dir:
            mask_dir = save_masks_dir
            os.makedirs(mask_dir, exist_ok=True)
        else:
            tmp_dir = tempfile.TemporaryDirectory(prefix="coco_masks_")
            mask_dir = tmp_dir.name

        pairs: List[Tuple[str, str]] = []
        for img_id, img_info in images_by_id.items():
            img_filename: str = img_info["file_name"]
            img_path = self._resolve_image_path(img_filename, image_dir, dataset_path)
            if img_path is None or not self._validate_image(img_path):
                self._log_skip(img_filename, "图片文件不存在")
                continue

            anns = annotations_by_image.get(img_id, [])
            if not anns:
                self._log_skip(img_filename, "无可用的标注")
                continue

            mask_filename = self._mask_name(img_filename)
            mask_path = os.path.join(mask_dir, mask_filename)

            img_w = img_info.get("width", 0)
            img_h = img_info.get("height", 0)

            if save_masks_dir and os.path.exists(mask_path):
                pairs.append((img_path, mask_path))
                continue

            try:
                self._render_combined_mask(anns, img_w, img_h, mask_path)
            except Exception as e:
                self._log_skip(img_filename, f"mask 渲染失败: {e}")
                continue

            pairs.append((img_path, mask_path))

        _suppress_tmp_cleanup = save_masks_dir or pairs
        if tmp_dir is not None:
            if save_masks_dir:
                tmp_dir.cleanup()
            else:
                self._tmp_dir = tmp_dir

        return self._dedup_pairs(pairs)

    def _resolve_image_path(
        self,
        filename: str,
        image_dir: Optional[str],
        annotation_path: str,
    ) -> Optional[str]:
        """Resolve the absolute path to an image file."""
        if image_dir:
            candidate = os.path.join(image_dir, filename)
            if os.path.isfile(candidate):
                return self._resolve_path(candidate)

        candidates = [
            os.path.join(os.path.dirname(annotation_path), filename),
            os.path.join(os.path.dirname(annotation_path), "images", filename),
        ]
        for c in candidates:
            if os.path.isfile(c):
                return self._resolve_path(c)
        return None

    def _mask_name(self, image_filename: str) -> str:
        stem = Path(image_filename).stem
        return f"{stem}_mask.png"

    def _render_combined_mask(
        self,
        annotations: List[Dict[str, Any]],
        img_w: int,
        img_h: int,
        output_path: str,
    ) -> None:
        combined = Image.new("L", (img_w, img_h), 0)
        draw = ImageDraw.Draw(combined)

        for ann in annotations:
            seg = ann.get("segmentation", None)
            if seg is None:
                continue

            if isinstance(seg, list):
                self._draw_polygons(draw, seg, img_w, img_h)
            elif isinstance(seg, dict):
                self._draw_rle(combined, seg, img_w, img_h)

        if img_w == 0 or img_h == 0:
            bbox_union = [float("inf"), float("inf"), 0, 0]
            for ann in annotations:
                bbox = ann.get("bbox", None)
                if bbox is None:
                    continue
                x, y, w, h = bbox
                bbox_union[0] = min(bbox_union[0], x)
                bbox_union[1] = min(bbox_union[1], y)
                bbox_union[2] = max(bbox_union[2], x + w)
                bbox_union[3] = max(bbox_union[3], y + h)
            canvas_w = int(max(bbox_union[2] + 10, 1))
            canvas_h = int(max(bbox_union[3] + 10, 1))
            combined = combined.crop((0, 0, canvas_w, canvas_h))

        combined.save(output_path, format="PNG")

    def _draw_polygons(
        self,
        draw: ImageDraw.Draw,
        segmentation: list,
        img_w: int,
        img_h: int,
    ) -> None:
        if not segmentation:
            return

        is_rle_format = isinstance(segmentation[0], dict) and "counts" in segmentation[0]
        if is_rle_format:
            mask = np.zeros((img_h, img_w), dtype=np.uint8)
            for rle_dict in segmentation:
                counts = rle_dict["counts"]
                size = rle_dict.get("size", [img_h, img_w])
                rle_h, rle_w = int(size[0]), int(size[1])
                _decode_coco_rle(counts, rle_h, rle_w, mask)
            draw.bitmap((0, 0), Image.fromarray(mask * 255), fill=255)
            return

        if isinstance(segmentation[0], (int, float)):
            points = []
            for i in range(0, len(segmentation), 2):
                x = segmentation[i]
                y = segmentation[i + 1]
                points.append((x, y))
            draw.polygon(points, fill=255)
            return

        for poly in segmentation:
            if not poly:
                continue
            points = []
            for i in range(0, len(poly), 2):
                x = poly[i]
                y = poly[i + 1]
                points.append((x, y))
            if len(points) >= 3:
                draw.polygon(points, fill=255)

    def _draw_rle(
        self,
        combined: Image.Image,
        seg: dict,
        img_w: int,
        img_h: int,
    ) -> None:
        counts = seg["counts"]
        size = seg.get("size", [img_h, img_w])
        h, w = int(size[0]), int(size[1])

        mask = np.zeros((h, w), dtype=np.uint8)
        _decode_coco_rle(counts, h, w, mask)

        existing = np.array(combined, dtype=np.uint8)
        combined_mask = np.maximum(existing, mask * 255)
        combined.paste(Image.fromarray(combined_mask), (0, 0))


def _decode_coco_rle(
    counts: Any, h: int, w: int, mask: np.ndarray
) -> None:
    if isinstance(counts, list):
        _decode_uncompressed_rle(counts, mask)
    elif isinstance(counts, bytes):
        _decode_compressed_rle(counts, h, w, mask)
    elif isinstance(counts, str):
        _decode_compressed_rle(counts.encode("utf-8"), h, w, mask)


def _decode_uncompressed_rle(
    counts: List[int], mask: np.ndarray
) -> None:
    flat = mask.ravel()
    pos = 0
    val = 0
    for run in counts:
        if pos + run > len(flat):
            run = max(0, len(flat) - pos)
        flat[pos : pos + run] = val
        pos += run
        val = 1 - val


def _decode_compressed_rle(
    counts: bytes, h: int, w: int, mask: np.ndarray
) -> None:
    try:
        from pycocotools import mask as cocomask
        rle = {"counts": counts, "size": [h, w]}
        decoded = cocomask.decode(rle)
        mask[:] = np.where(decoded > 0, 1, 0).astype(np.uint8)
    except ImportError:
        raise RuntimeError(
            "COCO 压缩 RLE 解码需要 pycocotools，请运行: pip install pycocotools"
        )
