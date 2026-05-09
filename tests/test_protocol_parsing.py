import os

import pytest

from seg_pipeline.protocol import InputValidationError, RetrievalItem
from seg_pipeline.protocol import filter_usable_retrieval_items, normalize_retrieval_topk, validate_query_image_path


def test_normalize_retrieval_topk_tuple_and_tail():
    items, skipped = normalize_retrieval_topk([("a.png", "b.png", 0.25, "tail1")])
    assert skipped == []
    assert len(items) == 1
    it = items[0]
    assert it.image_path == "a.png"
    assert it.mask_path == "b.png"
    assert it.distance == 0.25
    assert it.extra == {"raw_tail": ["tail1"]}


def test_normalize_retrieval_topk_mapping_and_extra_fields():
    raw = {"img_path": "a.png", "mask": "b.png", "score": "0.9", "foo": 1, "bar": "x"}
    items, skipped = normalize_retrieval_topk([raw])
    assert skipped == []
    assert len(items) == 1
    it = items[0]
    assert it.image_path == "a.png"
    assert it.mask_path == "b.png"
    assert it.score == 0.9
    assert it.distance is None
    assert it.extra == {"foo": 1, "bar": "x"}


def test_normalize_retrieval_topk_keeps_dataclass_instance():
    src = RetrievalItem(image_path="a.png", mask_path="b.png", distance=1.0)
    items, skipped = normalize_retrieval_topk([src])
    assert skipped == []
    assert items == [src]


def test_normalize_retrieval_topk_collects_parse_error():
    items, skipped = normalize_retrieval_topk([object()])
    assert items == []
    assert len(skipped) == 1
    assert skipped[0]["index"] == 0
    assert skipped[0]["reason"] == "parse_error"
    assert "error" in skipped[0]


def test_filter_usable_retrieval_items_honors_path_existence(tmp_path):
    img_ok = tmp_path / "img.png"
    mask_ok = tmp_path / "mask.png"
    img_ok.write_bytes(b"")
    mask_ok.write_bytes(b"")

    img_missing = tmp_path / "missing.png"
    mask_missing = tmp_path / "missing_mask.png"

    items = [
        RetrievalItem(image_path=str(img_ok), mask_path=str(mask_ok)),
        RetrievalItem(image_path=str(img_missing), mask_path=str(mask_ok)),
        RetrievalItem(image_path=str(img_ok), mask_path=str(mask_missing)),
    ]
    usable, skipped = filter_usable_retrieval_items(items)

    assert usable == [items[0]]
    assert [s["reason"] for s in skipped] == ["image_not_found", "mask_not_found"]


def test_validate_query_image_path_rejects_empty_and_missing(tmp_path):
    with pytest.raises(InputValidationError):
        validate_query_image_path("")

    missing = tmp_path / "q.png"
    with pytest.raises(InputValidationError):
        validate_query_image_path(str(missing))

    existing = tmp_path / "q2.png"
    existing.write_bytes(b"")
    validate_query_image_path(str(existing))
    assert os.path.exists(existing)
