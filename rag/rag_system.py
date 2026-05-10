import logging
import os
from typing import List, Tuple, Optional

from rag.config import INDEX_DIR, DEFAULT_TOP_K
from rag.embedding.dinov3_embedder import Dinov3Embedder
from rag.index.faiss_indexer import FaissIndexer
from rag.storage.file_manager import FileManager

logger = logging.getLogger(__name__)


class RAGSystem:
    def __init__(
        self,
        index_dir: str = INDEX_DIR,
        embedder_device: str = None,
        top_k: int = DEFAULT_TOP_K,
        embedder_model_name: str = None,
        embedder_local_files_only: bool = False,
    ):
        self.index_dir = index_dir
        self.top_k = top_k

        self.embedder = Dinov3Embedder(
            device=embedder_device,
            model_name=embedder_model_name,
            local_files_only=embedder_local_files_only,
        )
        self.indexer = FaissIndexer(
            index_dir=index_dir,
            embedding_dim=self.embedder.embedding_dim,
        )
        self.file_manager = FileManager(index_dir=index_dir)

        loaded = self.indexer.load()
        if loaded and self.indexer.embedding_dim != self.embedder.embedding_dim:
            logger.warning(
                "已存在的 FAISS 索引维度 (%d) 与当前 embedder (%d) 不匹配，将重建索引",
                self.indexer.embedding_dim,
                self.embedder.embedding_dim,
            )
            self.indexer.create_index()
            self.indexer.save()

    def index_images(self, image_paths: List[str], mask_paths: List[str] = None):
        if mask_paths is None:
            mask_paths = [None] * len(image_paths)

        if len(image_paths) != len(mask_paths):
            raise ValueError("Number of image_paths must match mask_paths")

        valid_pairs = []
        for img_path, mask_path in zip(image_paths, mask_paths):
            if not os.path.exists(img_path):
                continue
            valid_pairs.append((img_path, mask_path))

        if not valid_pairs:
            return

        valid_images = [pair[0] for pair in valid_pairs]
        embeddings = self.embedder.extract_embeddings(valid_images)

        paths_to_add = []
        masks_to_add = []
        for (img_path, mask_path), embedding in zip(valid_pairs, embeddings):
            if img_path not in self.file_manager._image_to_mask:
                paths_to_add.append(img_path)
                masks_to_add.append(mask_path)
                if mask_path:
                    self.file_manager.add_pair(img_path, mask_path)

        if paths_to_add:
            self.indexer.add(embeddings[:len(paths_to_add)], paths_to_add)
            self.indexer.save()

    def add_image(self, image_path: str, mask_path: str = None):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        if mask_path and not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        if image_path in self.file_manager._image_to_mask:
            return

        embedding = self.embedder.extract_embedding(image_path)

        self.indexer.add(embedding, [image_path])
        if mask_path:
            self.file_manager.add_pair(image_path, mask_path)
        else:
            self.file_manager.add_pair(image_path, "")
        self.indexer.save()

    def remove_image(self, image_path: str):
        if not os.path.exists(image_path):
            return False

        self.file_manager.remove_image(image_path)
        self._rebuild_index()
        return True

    def search(
        self, query_image: str, k: int = None
    ) -> List[Tuple[str, str, float]]:
        if k is None:
            k = self.top_k

        embedding = self.embedder.extract_embedding(query_image)
        results = self.indexer.search(embedding, k)

        search_results = []
        for img_path, distance in results:
            mask_path = self.file_manager.get_mask(img_path)
            if mask_path is None:
                mask_path = ""
            search_results.append((img_path, mask_path, distance))

        return search_results

    def _rebuild_index(self):
        all_pairs = self.file_manager.get_all_pairs()
        if not all_pairs:
            self.indexer.create_index()
            self.indexer.save()
            return

        all_images = [pair[0] for pair in all_pairs]
        embeddings = self.embedder.extract_embeddings(all_images)

        self.indexer.create_index()
        self.indexer.add(embeddings, all_images)
        self.indexer.save()

    def get_indexed_count(self) -> int:
        return len(self.indexer)

    def import_dataset(self, format: str, dataset_path: str, **kwargs) -> int:
        from rag.dataset_importers import get_importer

        importer = get_importer(format)
        pairs = importer.discover_pairs(dataset_path, **kwargs)

        if not pairs:
            return 0

        image_paths, mask_paths = zip(*pairs)
        self.index_images(list(image_paths), list(mask_paths))
        return len(image_paths)

    def clear_index(self):
        self.file_manager._image_to_mask = {}
        self.file_manager._mask_to_image = {}
        self.file_manager._save_metadata()
        self.indexer.create_index()
        self.indexer.save()
