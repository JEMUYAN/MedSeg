from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image

from seg_pipeline.protocol import RetrievalItem
from seg_pipeline.sam3_memory_segmenter import Sam3MemoryAttentionSegmenter


@dataclass
class TrackerCallLog:
    init_state_kwargs: Optional[Dict[str, Any]] = None
    preflight_kwargs: Optional[Dict[str, Any]] = None
    propagate_kwargs: Optional[Dict[str, Any]] = None
    added_masks: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.added_masks is None:
            self.added_masks = []


class FakeTracker:
    def __init__(self, log: TrackerCallLog):
        self._log = log

    def init_state(self, *, video_path: str, async_loading_frames: bool):
        self._log.init_state_kwargs = {"video_path": video_path, "async_loading_frames": async_loading_frames}
        return {"video_path": video_path}

    def add_new_mask(self, *, inference_state, frame_idx: int, obj_id: int, mask):
        self._log.added_masks.append(
            {"inference_state": inference_state, "frame_idx": frame_idx, "obj_id": obj_id, "mask": mask}
        )

    def propagate_in_video_preflight(self, inference_state, *, run_mem_encoder: bool):
        self._log.preflight_kwargs = {"inference_state": inference_state, "run_mem_encoder": run_mem_encoder}

    def propagate_in_video(
        self,
        *,
        inference_state,
        start_frame_idx: int,
        max_frame_num_to_track: int,
        reverse: bool,
        tqdm_disable: bool,
        run_mem_encoder: bool,
    ) -> Iterable[Tuple[int, List[int], Any, Any, Any]]:
        self._log.propagate_kwargs = {
            "inference_state": inference_state,
            "start_frame_idx": start_frame_idx,
            "max_frame_num_to_track": max_frame_num_to_track,
            "reverse": reverse,
            "tqdm_disable": tqdm_disable,
            "run_mem_encoder": run_mem_encoder,
        }
        yield (start_frame_idx, [0], None, object(), None)


class FakePredictor:
    def __init__(self, tracker: FakeTracker):
        self.model = type("M", (), {"tracker": tracker})()


def _write_rgb_image(path: Path) -> None:
    img = Image.new("RGB", (8, 8), color=(255, 0, 0))
    img.save(path, format="PNG")


def test_memory_lock_parameter_is_forwarded(monkeypatch, tmp_path: Path):
    cond_img = tmp_path / "cond.png"
    query_img = tmp_path / "query.png"
    mask_path = tmp_path / "mask.png"
    _write_rgb_image(cond_img)
    _write_rgb_image(query_img)
    mask_path.write_bytes(b"placeholder")

    log = TrackerCallLog()
    tracker = FakeTracker(log)
    predictor = FakePredictor(tracker)

    seg = Sam3MemoryAttentionSegmenter(predictor=predictor)

    monkeypatch.setattr(seg, "_load_mask_as_torch", lambda _p: ("MASK_TENSOR", {}))
    monkeypatch.setattr(seg, "_select_and_binarize_mask", lambda **_kw: Image.new("L", (8, 8), color=0))

    retrieval_items = [RetrievalItem(image_path=str(cond_img), mask_path=str(mask_path))]

    _mask, meta = seg.segment(
        query_image_path=str(query_img),
        retrieval_items=retrieval_items,
        lock_memory=True,
        obj_id=0,
    )

    assert log.preflight_kwargs == {"inference_state": {"video_path": log.init_state_kwargs["video_path"]}, "run_mem_encoder": True}
    assert log.propagate_kwargs["start_frame_idx"] == 1
    assert log.propagate_kwargs["max_frame_num_to_track"] == 0
    assert log.propagate_kwargs["tqdm_disable"] is True
    assert log.propagate_kwargs["run_mem_encoder"] is False

    assert meta["memory_locked"] is True
    assert meta["run_mem_encoder_on_target"] is False


def test_memory_unlock_parameter_is_forwarded(monkeypatch, tmp_path: Path):
    cond_img = tmp_path / "cond.png"
    query_img = tmp_path / "query.png"
    mask_path = tmp_path / "mask.png"
    _write_rgb_image(cond_img)
    _write_rgb_image(query_img)
    mask_path.write_bytes(b"placeholder")

    log = TrackerCallLog()
    tracker = FakeTracker(log)
    predictor = FakePredictor(tracker)

    seg = Sam3MemoryAttentionSegmenter(predictor=predictor)

    monkeypatch.setattr(seg, "_load_mask_as_torch", lambda _p: ("MASK_TENSOR", {}))
    monkeypatch.setattr(seg, "_select_and_binarize_mask", lambda **_kw: Image.new("L", (8, 8), color=0))

    retrieval_items = [RetrievalItem(image_path=str(cond_img), mask_path=str(mask_path))]

    _mask, meta = seg.segment(
        query_image_path=str(query_img),
        retrieval_items=retrieval_items,
        lock_memory=False,
        obj_id=0,
    )

    assert log.propagate_kwargs["run_mem_encoder"] is True
    assert meta["memory_locked"] is False
    assert meta["run_mem_encoder_on_target"] is True
