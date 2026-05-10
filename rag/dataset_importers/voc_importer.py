from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image

from rag.dataset_importers.base import DatasetImporter

VOC_CLASSES = [
    "__background__",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


class VOCImporter(DatasetImporter):

    IMAGE_DIR = "JPEGImages"
    MASK_DIR = "SegmentationClass"
    IMAGE_SET_FILE = "ImageSets/Segmentation/train.txt"

    def discover_pairs(
        self, dataset_path: str, **kwargs: Any
    ) -> List[Tuple[str, str]]:
        root = self._resolve_path(dataset_path)
        if not os.path.isdir(root):
            raise NotADirectoryError(f"VOC 数据集根目录不存在: {root}")

        class_name: Optional[str] = kwargs.get("class_name", None)
        class_id: Optional[int] = kwargs.get("class_id", None)
        image_dir: str = kwargs.get("image_dir", os.path.join(root, self.IMAGE_DIR))
        mask_dir: str = kwargs.get("mask_dir", os.path.join(root, self.MASK_DIR))

        if class_name is not None and class_name not in VOC_CLASSES:
            raise ValueError(
                f"未知的 VOC 类别: '{class_name}'，有效类别: {VOC_CLASSES[1:]}"
            )
        if class_id is not None and not (0 <= class_id < len(VOC_CLASSES)):
            raise ValueError(
                f"无效的类别 ID: {class_id}，有效范围: 0-{len(VOC_CLASSES) - 1}"
            )

        if class_name is not None:
            class_id = VOC_CLASSES.index(class_name)

        pairs: List[Tuple[str, str]] = []

        sample_ids = self._read_image_set(root)
        if sample_ids:
            pairs = self._pair_by_ids(
                sample_ids, image_dir, mask_dir, class_id
            )
        else:
            pairs = self._pair_by_mask_files(
                image_dir, mask_dir, class_id
            )

        return self._dedup_pairs(pairs)

    def _read_image_set(self, root: str) -> List[str]:
        set_path = os.path.join(root, self.IMAGE_SET_FILE)
        if not os.path.isfile(set_path):
            return []

        sample_ids: List[str] = []
        with open(set_path, "r", encoding="utf-8") as f:
            for line in f:
                sid = line.strip()
                if sid:
                    sample_ids.append(sid)
        return sample_ids

    def _pair_by_ids(
        self,
        sample_ids: List[str],
        image_dir: str,
        mask_dir: str,
        target_class_id: Optional[int],
    ) -> List[Tuple[str, str]]:
        pairs: List[Tuple[str, str]] = []
        img_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

        for sid in sample_ids:
            img_path = None
            for ext in img_exts:
                candidate = os.path.join(image_dir, f"{sid}{ext}")
                if os.path.isfile(candidate):
                    img_path = self._resolve_path(candidate)
                    break
            if img_path is None:
                self._log_skip(sid, "图片文件不存在")
                continue

            mask_path = self._find_mask_for_id(sid, mask_dir)
            if mask_path is None:
                self._log_skip(sid, "缺少对应 mask 文件")
                continue

            if target_class_id is not None:
                mask_path = self._maybe_extract_class_mask(
                    mask_path, target_class_id
                )

            pairs.append((img_path, mask_path))

        return pairs

    def _pair_by_mask_files(
        self,
        image_dir: str,
        mask_dir: str,
        target_class_id: Optional[int],
    ) -> List[Tuple[str, str]]:
        pairs: List[Tuple[str, str]] = []
        mask_files = self._find_files(
            mask_dir, extensions={".png"}, recursive=False
        )

        for mask_path in mask_files:
            mask_stem = Path(mask_path).stem
            sid = mask_stem

            img_path = None
            img_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
            for ext in img_exts:
                candidate = os.path.join(image_dir, f"{sid}{ext}")
                if os.path.isfile(candidate):
                    img_path = self._resolve_path(candidate)
                    break

            if img_path is None:
                self._log_skip(sid, "图片文件不存在（从 mask 逆推）")
                continue

            final_mask_path = mask_path
            if target_class_id is not None:
                final_mask_path = self._maybe_extract_class_mask(
                    mask_path, target_class_id
                )

            pairs.append((img_path, final_mask_path))

        return pairs

    def _find_mask_for_id(
        self, sample_id: str, mask_dir: str
    ) -> Optional[str]:
        candidate = os.path.join(mask_dir, f"{sample_id}.png")
        if os.path.isfile(candidate):
            return self._resolve_path(candidate)
        return None

    def _maybe_extract_class_mask(
        self, mask_path: str, class_id: int
    ) -> str:
        stem = Path(mask_path).stem
        class_mask_path = os.path.join(
            os.path.dirname(mask_path), f"{stem}_class{class_id}.png"
        )
        if os.path.exists(class_mask_path):
            return class_mask_path

        mask = self._load_voc_mask(mask_path)
        binary = (mask == class_id).astype(np.uint8) * 255
        Image.fromarray(binary, mode="L").save(class_mask_path, format="PNG")
        return class_mask_path

    def _load_voc_mask(self, mask_path: str) -> np.ndarray:
        img = Image.open(mask_path)
        if img.mode == "P":
            return np.array(img, dtype=np.uint8)
        return np.array(img.convert("P"), dtype=np.uint8)
