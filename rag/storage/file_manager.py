import os
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from rag.config import SUPPORTED_IMAGE_EXTENSIONS, INDEX_DIR, METADATA_FILE


class FileManager:
    def __init__(self, index_dir: str = INDEX_DIR):
        self.index_dir = index_dir
        self.metadata_path = os.path.join(self.index_dir, METADATA_FILE)
        self._image_to_mask: Dict[str, str] = {}
        self._mask_to_image: Dict[str, str] = {}
        self._load_metadata()

    def _load_metadata(self):
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r") as f:
                data = json.load(f)
                self._image_to_mask = data.get("image_to_mask", {})
                self._mask_to_image = data.get("mask_to_image", {})

    def _save_metadata(self):
        os.makedirs(self.index_dir, exist_ok=True)
        data = {
            "image_to_mask": self._image_to_mask,
            "mask_to_image": self._mask_to_image,
        }
        with open(self.metadata_path, "w") as f:
            json.dump(data, f, indent=2)

    def add_pair(self, image_path: str, mask_path: str):
        image_path = os.path.abspath(image_path)
        mask_path = os.path.abspath(mask_path)

        self._image_to_mask[image_path] = mask_path
        self._mask_to_image[mask_path] = image_path
        self._save_metadata()

    def remove_image(self, image_path: str):
        image_path = os.path.abspath(image_path)

        if image_path in self._image_to_mask:
            mask_path = self._image_to_mask.pop(image_path)
            self._mask_to_image.pop(mask_path, None)
            self._save_metadata()
            return True
        return False

    def remove_mask(self, mask_path: str):
        mask_path = os.path.abspath(mask_path)

        if mask_path in self._mask_to_image:
            image_path = self._mask_to_image.pop(mask_path)
            self._image_to_mask.pop(image_path, None)
            self._save_metadata()
            return True
        return False

    def get_mask(self, image_path: str) -> Optional[str]:
        image_path = os.path.abspath(image_path)
        return self._image_to_mask.get(image_path)

    def get_image(self, mask_path: str) -> Optional[str]:
        mask_path = os.path.abspath(mask_path)
        return self._mask_to_image.get(mask_path)

    def get_all_pairs(self) -> List[Tuple[str, str]]:
        return [
            (img, mask)
            for img, mask in self._image_to_mask.items()
        ]

    def get_all_images(self) -> List[str]:
        return list(self._image_to_mask.keys())

    def get_all_masks(self) -> List[str]:
        return list(self._mask_to_image.keys())

    def is_image_file(self, path: str) -> bool:
        ext = Path(path).suffix.lower()
        return ext in SUPPORTED_IMAGE_EXTENSIONS

    def is_mask_file(self, path: str) -> bool:
        ext = Path(path).suffix.lower()
        return ext in SUPPORTED_IMAGE_EXTENSIONS

    def find_image_for_mask(self, mask_path: str) -> Optional[str]:
        mask_path = os.path.abspath(mask_path)
        return self._mask_to_image.get(mask_path)

    def find_mask_for_image(self, image_path: str) -> Optional[str]:
        image_path = os.path.abspath(image_path)
        return self._image_to_mask.get(image_path)
