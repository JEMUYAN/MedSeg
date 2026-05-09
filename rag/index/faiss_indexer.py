import faiss
import numpy as np
import os
from typing import List, Tuple, Optional

from rag.config import INDEX_DIR, INDEX_FILE, EMBEDDING_DIM


class FaissIndexer:
    def __init__(self, index_dir: str = INDEX_DIR, embedding_dim: int = EMBEDDING_DIM):
        self.index_dir = index_dir
        self.embedding_dim = embedding_dim
        self.index: Optional[faiss.Index] = None
        self._id_to_path: dict = {}
        self._path_to_id: dict = {}
        self._current_id = 0

        os.makedirs(self.index_dir, exist_ok=True)

    def create_index(self):
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self._id_to_path = {}
        self._path_to_id = {}
        self._current_id = 0

    def add(self, embeddings: np.ndarray, paths: List[str]):
        if self.index is None:
            self.create_index()

        if len(embeddings) != len(paths):
            raise ValueError("Number of embeddings must match number of paths")

        embeddings = np.asarray(embeddings).astype(np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        self.index.add(embeddings)

        for path in paths:
            if path not in self._path_to_id:
                self._id_to_path[self._current_id] = path
                self._path_to_id[path] = self._current_id
                self._current_id += 1

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        if self.index is None:
            return []

        query_embedding = np.asarray(query_embedding).astype(np.float32)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        distances, indices = self.index.search(query_embedding, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx in self._id_to_path:
                results.append((self._id_to_path[idx], float(dist)))

        return results

    def remove(self, paths: List[str]):
        raise NotImplementedError(
            "Faiss IndexFlatL2 does not support removal. "
            "Recreate index to remove entries."
        )

    def save(self, index_path: str = None, metadata_path: str = None):
        if self.index is None:
            return

        if index_path is None:
            index_path = os.path.join(self.index_dir, INDEX_FILE)
        if metadata_path is None:
            metadata_path = os.path.join(self.index_dir, "metadata.npy")

        faiss.write_index(self.index, index_path)

        metadata = {
            "id_to_path": self._id_to_path,
            "path_to_id": self._path_to_id,
            "current_id": self._current_id,
            "embedding_dim": self.embedding_dim,
        }
        np.save(metadata_path, metadata)

    def load(self, index_path: str = None, metadata_path: str = None):
        if index_path is None:
            index_path = os.path.join(self.index_dir, INDEX_FILE)
        if metadata_path is None:
            metadata_path = os.path.join(self.index_dir, "metadata.npy")

        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            return False

        self.index = faiss.read_index(index_path)

        metadata = np.load(metadata_path, allow_pickle=True).item()
        self._id_to_path = metadata["id_to_path"]
        self._path_to_id = metadata["path_to_id"]
        self._current_id = metadata["current_id"]
        self.embedding_dim = metadata["embedding_dim"]

        return True

    def get_all_paths(self) -> List[str]:
        return list(self._path_to_id.keys())

    def __len__(self) -> int:
        if self.index is None:
            return 0
        return self.index.ntotal
