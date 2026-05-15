import os
from pathlib import Path

DINOV3_MODEL_NAME = "facebook/dinov3-vith16plus"

INDEX_DIR = os.environ.get("RAG_INDEX_DIR", "./rag_index")
INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "image_metadata.json"

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

DEFAULT_TOP_K = 5

EMBEDDING_DIM = 1024
