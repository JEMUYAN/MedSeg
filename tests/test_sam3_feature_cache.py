from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image

from seg_pipeline.protocol import RetrievalItem
from seg_pipeline.sam3_memory_segmenter import Sam3BuildConfig, Sam3MemoryAttentionSegmenter


@dataclass
class CacheLog:
    cached_frames: List[int]


class FakeInputBatch:
    def __init__(self, num_frames: int):
        self.img_batch = [f"IMG_{i}" for i in range(num_frames)]


class FakeModel:
    def __init__(self, tracker: Any, log: CacheLog, num_frames: int = 2):
        self.tracker = tracker
        self._log = log
        self._num_frames = num_frames

    def init_state(
        self,
        *,
        resource_path: str,
        offload_video_to_cpu: bool,
        offload_state_to_cpu: bool,
        async_loading_frames: bool,
    ):
        _ = (resource_path, offload_video_to_cpu, offload_state_to_cpu, async_loading_frames)
        return {
            "num_frames": self._num_frames,
            "orig_height": 8,
            "orig_width": 8,
            "offload_state_to_cpu": offload_state_to_cpu,
            "feature_cache": {},
            "input_batch": FakeInputBatch(self._num_frames),
            "constants": {"empty_geometric_prompt": object()},
        }

    def run_backbone_and_detection(
        self,
        *,
        frame_idx: int,
        num_frames: int,
        input_batch: Any,
        geometric_prompt: Any,
        feature_cache: Dict[int, Any],
        reverse: bool,
        allow_new_detections: bool,
    ):
        _ = (num_frames, input_batch, geometric_prompt, reverse, allow_new_detections)
        self._log.cached_frames.append(frame_idx)
        feature_cache[frame_idx] = ("CACHED_IMAGE", {"tracker_backbone_out": object()})
        feature_cache.pop(frame_idx - 1, None)


class FakeTracker:
    def __init__(self):
        self.added_masks: List[Dict[str, Any]] = []
        self.preflight_kwargs: Optional[Dict[str, Any]] = None
        self.propagate_kwargs: Optional[Dict[str, Any]] = None

    def init_state(
        self,
        *,
        cached_features: Dict[int, Any],
        video_height: int,
        video_width: int,
        num_frames: int,
        offload_state_to_cpu: bool,
    ):
        _ = (num_frames, offload_state_to_cpu)
        return {
            "cached_features": cached_features,
            "video_height": video_height,
            "video_width": video_width,
            "num_frames": num_frames,
        }

    def add_new_mask(self, *, inference_state, frame_idx: int, obj_id: int, mask):
        assert frame_idx in inference_state["cached_features"]
        self.added_masks.append(
            {"inference_state": inference_state, "frame_idx": frame_idx, "obj_id": obj_id, "mask": mask}
        )

    def propagate_in_video_preflight(self, inference_state, *, run_mem_encoder: bool):
        self.preflight_kwargs = {"inference_state": inference_state, "run_mem_encoder": run_mem_encoder}

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
        self.propagate_kwargs = {
            "inference_state": inference_state,
            "start_frame_idx": start_frame_idx,
            "max_frame_num_to_track": max_frame_num_to_track,
            "reverse": reverse,
            "tqdm_disable": tqdm_disable,
            "run_mem_encoder": run_mem_encoder,
        }
        yield (start_frame_idx, [0], None, object(), None)


class FakePredictor:
    def __init__(self, model: Any):
        self.model = model


def _write_rgb_image(path: Path) -> None:
    img = Image.new("RGB", (8, 8), color=(255, 0, 0))
    img.save(path, format="PNG")


def test_sam3_caches_features_before_adding_masks(monkeypatch, tmp_path: Path):
    cond_img = tmp_path / "cond.png"
    query_img = tmp_path / "query.png"
    mask_path = tmp_path / "mask.png"
    _write_rgb_image(cond_img)
    _write_rgb_image(query_img)
    mask_path.write_bytes(b"placeholder")

    log = CacheLog(cached_frames=[])
    tracker = FakeTracker()
    model = FakeModel(tracker=tracker, log=log, num_frames=2)
    predictor = FakePredictor(model=model)

    seg = Sam3MemoryAttentionSegmenter(build_config=Sam3BuildConfig(version="sam3"), predictor=predictor)
    monkeypatch.setattr(seg, "_load_mask_as_torch", lambda _p: ("MASK_TENSOR", {}))
    monkeypatch.setattr(seg, "_select_and_binarize_mask", lambda **_kw: Image.new("L", (8, 8), color=0))

    retrieval_items = [RetrievalItem(image_path=str(cond_img), mask_path=str(mask_path))]

    _mask, _meta = seg.segment(
        query_image_path=str(query_img),
        retrieval_items=retrieval_items,
        lock_memory=True,
        obj_id=0,
    )

    assert log.cached_frames == [0, 1]
    assert len(tracker.added_masks) == 1
    assert tracker.propagate_kwargs["start_frame_idx"] == 1
    assert tracker.propagate_kwargs["run_mem_encoder"] is False
