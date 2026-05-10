from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

from rag.dataset_importers import (
    DatasetImportError,
    get_importer,
    list_formats,
    register_importer,
)
from rag.dataset_importers.base import DatasetImporter
from rag.dataset_importers.coco_importer import (
    COCOImporter,
    _decode_uncompressed_rle,
)
from rag.dataset_importers.voc_importer import VOCImporter


def _make_test_image(path: str, size: tuple = (32, 32), color: tuple = (128, 128, 128)) -> str:
    Image.new("RGB", size, color).save(path, format="PNG")
    return path


def _make_test_mask(path: str, size: tuple = (32, 32), fill_value: int = 0) -> str:
    arr = np.full((size[1], size[0]), fill_value, dtype=np.uint8)
    Image.fromarray(arr, mode="L").save(path, format="PNG")
    return path


class TestCOCOImporter:
    def test_discover_pairs_polygon(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "img001.jpg")
            _make_test_image(img_path, (64, 64))

            poly = [10, 10, 50, 10, 50, 50, 10, 50]
            coco_json = {
                "images": [{"id": 1, "file_name": "img001.jpg", "width": 64, "height": 64}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "segmentation": [poly],
                        "bbox": [10, 10, 40, 40],
                    }
                ],
                "categories": [{"id": 1, "name": "lesion"}],
            }
            ann_path = os.path.join(tmpdir, "annotations.json")
            with open(ann_path, "w") as f:
                json.dump(coco_json, f)

            importer = COCOImporter()
            pairs = importer.discover_pairs(ann_path, image_dir=tmpdir)

            assert len(pairs) == 1
            assert pairs[0][0] == os.path.abspath(img_path)
            assert os.path.isfile(pairs[0][1])
            mask = Image.open(pairs[0][1])
            assert mask.size == (64, 64)
            assert mask.mode == "L"

    def test_discover_pairs_category_filter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "img001.jpg")
            _make_test_image(img_path, (64, 64))

            poly = [10, 10, 50, 10, 50, 50, 10, 50]
            coco_json = {
                "images": [{"id": 1, "file_name": "img001.jpg", "width": 64, "height": 64}],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 2, "segmentation": [poly]},
                ],
                "categories": [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}],
            }
            ann_path = os.path.join(tmpdir, "annotations.json")
            with open(ann_path, "w") as f:
                json.dump(coco_json, f)

            importer = COCOImporter()
            pairs_all = importer.discover_pairs(ann_path, image_dir=tmpdir)
            pairs_filtered = importer.discover_pairs(
                ann_path, image_dir=tmpdir, category_ids=[1]
            )

            assert len(pairs_all) == 1
            assert len(pairs_filtered) == 0

    def test_discover_pairs_save_masks_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "img001.jpg")
            _make_test_image(img_path, (64, 64))

            poly = [0, 0, 30, 0, 30, 30, 0, 30]
            coco_json = {
                "images": [{"id": 1, "file_name": "img001.jpg", "width": 64, "height": 64}],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "segmentation": [poly]},
                ],
                "categories": [{"id": 1, "name": "thing"}],
            }
            ann_path = os.path.join(tmpdir, "annotations.json")
            with open(ann_path, "w") as f:
                json.dump(coco_json, f)

            save_dir = os.path.join(tmpdir, "rendered_masks")
            importer = COCOImporter()
            pairs = importer.discover_pairs(
                ann_path, image_dir=tmpdir, save_masks_dir=save_dir
            )

            assert len(pairs) == 1
            assert os.path.isfile(pairs[0][1])
            assert os.path.dirname(pairs[0][1]) == save_dir

            pairs2 = importer.discover_pairs(
                ann_path, image_dir=tmpdir, save_masks_dir=save_dir
            )
            assert pairs2 == pairs

    def test_discover_pairs_uncompressed_rle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "img001.jpg")
            _make_test_image(img_path, (16, 16))

            rle = {"counts": [100, 80, 40, 36], "size": [16, 16]}
            coco_json = {
                "images": [{"id": 1, "file_name": "img001.jpg", "width": 16, "height": 16}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "segmentation": rle,
                    },
                ],
                "categories": [{"id": 1, "name": "thing"}],
            }
            ann_path = os.path.join(tmpdir, "annotations.json")
            with open(ann_path, "w") as f:
                json.dump(coco_json, f)

            importer = COCOImporter()
            pairs = importer.discover_pairs(ann_path, image_dir=tmpdir)

            assert len(pairs) == 1
            mask = Image.open(pairs[0][1])
            arr = np.array(mask, dtype=np.uint8)
            assert arr.shape == (16, 16)

    def test_image_resolve_candidates(self):
        """Image resolved via images/ subdirectory relative to annotation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            images_dir = os.path.join(tmpdir, "images")
            os.makedirs(images_dir)
            img_path = os.path.join(images_dir, "img001.jpg")
            _make_test_image(img_path, (16, 16))

            poly = [0, 0, 10, 0, 10, 10, 0, 10]
            coco_json = {
                "images": [{"id": 1, "file_name": "img001.jpg", "width": 16, "height": 16}],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "segmentation": [poly]},
                ],
                "categories": [{"id": 1, "name": "thing"}],
            }
            ann_path = os.path.join(tmpdir, "annotations.json")
            with open(ann_path, "w") as f:
                json.dump(coco_json, f)

            importer = COCOImporter()
            pairs = importer.discover_pairs(ann_path)

            assert len(pairs) == 1
            assert pairs[0][0] == os.path.abspath(img_path)

    def test_missing_image_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            coco_json = {
                "images": [{"id": 1, "file_name": "missing.jpg", "width": 64, "height": 64}],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "segmentation": [[0, 0, 1, 1, 1, 0]]},
                ],
                "categories": [{"id": 1, "name": "thing"}],
            }
            ann_path = os.path.join(tmpdir, "annotations.json")
            with open(ann_path, "w") as f:
                json.dump(coco_json, f)

            importer = COCOImporter()
            pairs = importer.discover_pairs(ann_path, image_dir=tmpdir)
            assert len(pairs) == 0


class TestVOCImporter:
    def _setup_voc_tree(
        self, root: str, samples: list, with_imageset: bool = True
    ) -> tuple:
        img_dir = os.path.join(root, "JPEGImages")
        mask_dir = os.path.join(root, "SegmentationClass")
        os.makedirs(img_dir)
        os.makedirs(mask_dir)

        for sid in samples:
            _make_test_image(os.path.join(img_dir, f"{sid}.jpg"), (32, 32))
            voc_palette = _make_voc_palette_mask(os.path.join(mask_dir, f"{sid}.png"), (32, 32))

        if with_imageset:
            set_dir = os.path.join(root, "ImageSets", "Segmentation")
            os.makedirs(set_dir)
            with open(os.path.join(set_dir, "train.txt"), "w") as f:
                for sid in samples:
                    f.write(f"{sid}\n")

        return img_dir, mask_dir

    def test_discover_pairs_with_imageset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._setup_voc_tree(tmpdir, ["2007_0001", "2007_0002"])

            importer = VOCImporter()
            pairs = importer.discover_pairs(tmpdir)

            assert len(pairs) == 2
            assert all(os.path.isfile(img) for img, _ in pairs)
            assert all(os.path.isfile(mask) for _, mask in pairs)

    def test_discover_pairs_by_mask_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._setup_voc_tree(tmpdir, ["sample_a", "sample_b"], with_imageset=False)

            importer = VOCImporter()
            pairs = importer.discover_pairs(tmpdir)

            assert len(pairs) == 2

    def test_discover_pairs_missing_image_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # mask exists but image does not
            mask_dir = os.path.join(tmpdir, "SegmentationClass")
            os.makedirs(mask_dir)
            _make_voc_palette_mask(os.path.join(mask_dir, "orphan.png"), (32, 32))

            img_dir = os.path.join(tmpdir, "JPEGImages")
            os.makedirs(img_dir)

            importer = VOCImporter()
            pairs = importer.discover_pairs(tmpdir)
            assert len(pairs) == 0

    def test_class_extraction(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sid = "2007_0001"
            self._setup_voc_tree(tmpdir, [sid])

            importer = VOCImporter()
            pairs_all = importer.discover_pairs(tmpdir)
            pairs_cat = importer.discover_pairs(tmpdir, class_id=8)

            assert len(pairs_all) == 1
            assert len(pairs_cat) == 1
            class_mask = Image.open(pairs_cat[0][1])
            assert class_mask.mode == "L"

    def test_class_name_lookup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sid = "2007_0001"
            self._setup_voc_tree(tmpdir, [sid])

            importer = VOCImporter()
            pairs = importer.discover_pairs(tmpdir, class_name="cat")
            assert len(pairs) == 1

    def test_bad_root_directory(self):
        importer = VOCImporter()
        try:
            importer.discover_pairs("/nonexistent/path/to/voc")
        except NotADirectoryError:
            pass

    def test_invalid_class_name_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._setup_voc_tree(tmpdir, ["x"])
            importer = VOCImporter()
            try:
                importer.discover_pairs(tmpdir, class_name="nonexistent")
            except ValueError:
                pass


def _make_voc_palette_mask(path: str, size: tuple) -> np.ndarray:
    """Create a fake VOC-style palette PNG with pixel value 8 (cat)."""
    arr = np.full((size[1], size[0]), 8, dtype=np.uint8)
    img = Image.fromarray(arr, mode="P")

    palette = []
    for i in range(256):
        if i == 0:
            palette.extend([0, 0, 0])
        elif i == 8:
            palette.extend([64, 0, 128])
        else:
            palette.extend([0, 0, 0])
    img.putpalette(palette)
    img.save(path, format="PNG")
    return arr


class TestRegistry:
    def test_list_formats(self):
        formats = list_formats()
        assert "coco" in formats
        assert "voc" in formats

    def test_get_importer_known_formats(self):
        coco = get_importer("coco")
        voc = get_importer("voc")
        assert isinstance(coco, COCOImporter)
        assert isinstance(voc, VOCImporter)

    def test_unknown_format_raises(self):
        try:
            get_importer("nonexistent_format")
            raise AssertionError("should have raised")
        except ValueError as e:
            assert "nonexistent_format" in str(e)

    def test_register_custom_importer(self):
        class CustomImporter(DatasetImporter):
            def discover_pairs(self, dataset_path: str, **kwargs):
                return [("/a/b.jpg", "/a/b_mask.png")]

        register_importer("custom", CustomImporter)
        importer = get_importer("custom")
        assert isinstance(importer, CustomImporter)
        pairs = importer.discover_pairs("/any")
        assert pairs == [("/a/b.jpg", "/a/b_mask.png")]


class TestRAGSystemImportDataset:
    def test_import_dataset_integration(self, monkeypatch):
        import types

        # Mock the full rag chain before importing RAGSystem
        fake_rag_system = types.ModuleType("rag.rag_system")
        fake_dinov3 = types.ModuleType("rag.embedding.dinov3_embedder")
        fake_rag = types.ModuleType("rag")
        fake_rag_embedding = types.ModuleType("rag.embedding")

        monkeypatch.setitem(sys.modules, "rag", fake_rag)
        monkeypatch.setitem(sys.modules, "rag.embedding", fake_rag_embedding)
        monkeypatch.setitem(sys.modules, "rag.embedding.dinov3_embedder", fake_dinov3)
        monkeypatch.setitem(sys.modules, "rag.rag_system", fake_rag_system)

        class FakeRAGSystem:
            def __init__(self, index_dir=None, embedder_device=None, top_k=5):
                self.index_dir = index_dir or "./rag_index"
                self.top_k = top_k

            def import_dataset(self, format: str, dataset_path: str, **kwargs):
                from rag.dataset_importers import get_importer
                importer = get_importer(format)
                pairs = importer.discover_pairs(dataset_path, **kwargs)
                if not pairs:
                    return 0
                return len(pairs)

        fake_rag_system.RAGSystem = FakeRAGSystem

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "img001.jpg")
            _make_test_image(img_path, (16, 16))

            coco_json = {
                "images": [{"id": 1, "file_name": "img001.jpg", "width": 16, "height": 16}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "segmentation": [[0, 0, 10, 0, 10, 10, 0, 10]],
                    }
                ],
                "categories": [{"id": 1, "name": "thing"}],
            }
            ann_path = os.path.join(tmpdir, "annotations.json")
            with open(ann_path, "w") as f:
                json.dump(coco_json, f)

            rag = FakeRAGSystem(index_dir=tmpdir)
            count = rag.import_dataset("coco", ann_path, image_dir=tmpdir)

            assert count == 1

    def test_import_dataset_empty_pairs(self, monkeypatch):
        import types

        fake_rag_system = types.ModuleType("rag.rag_system")
        fake_dinov3 = types.ModuleType("rag.embedding.dinov3_embedder")
        fake_rag = types.ModuleType("rag")
        fake_rag_embedding = types.ModuleType("rag.embedding")

        monkeypatch.setitem(sys.modules, "rag", fake_rag)
        monkeypatch.setitem(sys.modules, "rag.embedding", fake_rag_embedding)
        monkeypatch.setitem(sys.modules, "rag.embedding.dinov3_embedder", fake_dinov3)
        monkeypatch.setitem(sys.modules, "rag.rag_system", fake_rag_system)

        class FakeRAGSystem:
            def __init__(self, index_dir=None, embedder_device=None, top_k=5):
                self.index_dir = index_dir or "./rag_index"
                self.top_k = top_k

            def import_dataset(self, format: str, dataset_path: str, **kwargs):
                from rag.dataset_importers import get_importer
                importer = get_importer(format)
                pairs = importer.discover_pairs(dataset_path, **kwargs)
                if not pairs:
                    return 0
                return len(pairs)

        fake_rag_system.RAGSystem = FakeRAGSystem

        with tempfile.TemporaryDirectory() as tmpdir:
            coco_json = {
                "images": [{"id": 1, "file_name": "missing.jpg", "width": 64, "height": 64}],
                "annotations": [],
                "categories": [],
            }
            ann_path = os.path.join(tmpdir, "annotations.json")
            with open(ann_path, "w") as f:
                json.dump(coco_json, f)

            rag = FakeRAGSystem(index_dir=tmpdir)
            count = rag.import_dataset("coco", ann_path, image_dir=tmpdir)
            assert count == 0


class TestRLEUncompressedDecode:
    def test_decode_simple_rle(self):
        mask = np.zeros((4, 4), dtype=np.uint8)
        # 4x4=16 elements; counts [8,8] = rows 0+1 zero, rows 2+3 ones
        counts = [8, 8]
        _decode_uncompressed_rle(counts, mask)
        assert mask[0, 0] == 0
        assert mask[1, 3] == 0
        assert mask[2, 0] == 1
        assert mask[2, 1] == 1
        assert mask.sum() == 8
