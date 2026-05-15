from __future__ import annotations

import os
import logging
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class DatasetImportError(Exception):
    """数据集导入过程中的错误。"""


class DatasetImporter(ABC):

    def __init__(self) -> None:
        self.skips: List[Tuple[str, str]] = []

    @abstractmethod
    def discover_pairs(
        self, dataset_path: str, **kwargs: Any
    ) -> List[Tuple[str, str]]:
        raise NotImplementedError

    def _resolve_path(self, path: str) -> str:
        return os.path.abspath(path)

    def _find_files(
        self,
        root: str,
        extensions: Optional[Set[str]] = None,
        recursive: bool = True,
    ) -> List[str]:
        if extensions is None:
            extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

        root_path = Path(root)
        if not root_path.is_dir():
            return []

        pattern = "**/*" if recursive else "*"
        files: List[str] = []
        for f in root_path.glob(pattern):
            if f.is_file() and f.suffix.lower() in extensions:
                files.append(str(f))
        return sorted(files)

    def _validate_image(self, image_path: str) -> bool:
        if not image_path or not os.path.isfile(image_path):
            return False
        ext = Path(image_path).suffix.lower()
        return ext in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

    def _validate_mask(self, mask_path: str) -> bool:
        if not mask_path or not os.path.isfile(mask_path):
            return False
        return True

    def _log_skip(self, image_path: str, reason: str) -> None:
        msg = f"  [SKIP] {image_path}: {reason}"
        self.skips.append((image_path, reason))
        logger.warning(msg)
        print(msg, file=sys.stderr)

    def _dedup_pairs(
        self, pairs: List[Tuple[str, str]]
    ) -> List[Tuple[str, str]]:
        seen: Set[str] = set()
        result: List[Tuple[str, str]] = []
        for img, mask in pairs:
            key = self._resolve_path(img)
            if key not in seen:
                seen.add(key)
                result.append((img, mask))
        return result
