import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from typing import Union, List
import torch.nn.functional as F

from rag.config import DINOV3_MODEL_NAME, EMBEDDING_DIM


class Dinov3Embedder:
    def __init__(self, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.processor = AutoImageProcessor.from_pretrained(DINOV3_MODEL_NAME)
        self.model = AutoModel.from_pretrained(DINOV3_MODEL_NAME)
        self.model = self.model.to(self.device)
        self.model.eval()

    def extract_embedding(self, image: Union[str, Image.Image]) -> np.ndarray:
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :]

            embedding = F.normalize(embedding, p=2, dim=1)

        return embedding.cpu().numpy().astype(np.float32)

    def extract_embeddings(self, images: List[Union[str, Image.Image]]) -> np.ndarray:
        if not images:
            return np.array([], dtype=np.float32).reshape(0, EMBEDDING_DIM)

        pil_images = []
        for img in images:
            if isinstance(img, str):
                pil_images.append(Image.open(img).convert("RGB"))
            else:
                pil_images.append(img)

        inputs = self.processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy().astype(np.float32)
