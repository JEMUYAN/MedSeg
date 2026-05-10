from rag.dataset_importers.base import DatasetImporter, DatasetImportError
from rag.dataset_importers.coco_importer import COCOImporter
from rag.dataset_importers.voc_importer import VOCImporter

_registry: "dict[str, type[DatasetImporter]]" = {}


def register_importer(format: str, importer_cls: type[DatasetImporter]) -> None:
    _registry[format] = importer_cls


def get_importer(format: str) -> DatasetImporter:
    cls = _registry.get(format)
    if cls is None:
        raise ValueError(
            f"不支持的数据集格式: '{format}'。可用格式: {list(_registry.keys())}"
        )
    return cls()


def list_formats() -> list[str]:
    return sorted(_registry.keys())


register_importer("coco", COCOImporter)
register_importer("voc", VOCImporter)


__all__ = [
    "DatasetImporter",
    "DatasetImportError",
    "COCOImporter",
    "VOCImporter",
    "register_importer",
    "get_importer",
    "list_formats",
]
